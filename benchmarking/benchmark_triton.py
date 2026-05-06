"""Benchmark Triton FIL inference vs local Python predict() and SHAP overhead.

Measures p50/p95 latency for each model at batch sizes 1, 32, 128, 1024.
Adds a separate SHAP block that times TreeExplainer.shap_values() for the
classifier and one health regressor per equipment type.

Usage:
    python benchmarking/benchmark_triton.py --triton-url localhost:8001
"""

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.triton_client import TritonInferenceClient
from src.inference.explainer import ShapExplainerManager
import shap


logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s %(levelname)s %(message)s"

BATCH_SIZES = (1, 32, 128, 1024)
N_TRIALS = 50


def load_python_model(model_dir: Path):
    """Reuse the explainer artifact for Python-side predictions."""
    import xgboost as xgb

    json_path = model_dir / "model_explainer.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    obj_name = (
        raw.get("learner", {}).get("objective", {}).get("name", "")
        if isinstance(raw, dict) else ""
    )
    if "softprob" in obj_name or "softmax" in obj_name or "binary" in obj_name:
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    model.load_model(str(json_path))
    return model


def time_calls(fn: Callable, n_trials: int) -> List[float]:
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def percentiles(samples: List[float]) -> Tuple[float, float, float]:
    samples = sorted(samples)
    n = len(samples)
    p50 = samples[n // 2]
    p95 = samples[max(0, int(0.95 * n) - 1)]
    return statistics.mean(samples), p50, p95


def benchmark_model(triton, model_dir: Path) -> None:
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    name = meta["model_name"]
    n_features = int(meta["n_features"])
    is_classifier = meta.get("model_purpose") == "classifier"
    py_model = load_python_model(model_dir)

    rng = np.random.default_rng(0)
    logger.info(f"\n=== {name} (n_features={n_features}, classifier={is_classifier}) ===")
    for batch in BATCH_SIZES:
        X = rng.normal(0, 1, size=(batch, n_features)).astype(np.float32)

        if py_model is not None:
            py_fn = (
                (lambda: py_model.predict_proba(X))
                if is_classifier else (lambda: py_model.predict(X))
            )
            py_times = time_calls(py_fn, N_TRIALS)
            mean_p, p50_p, p95_p = percentiles(py_times)
        else:
            mean_p = p50_p = p95_p = float("nan")

        if is_classifier:
            triton_fn = lambda: triton.predict_classifier(meta["equipment_type"], X)  # noqa: E731
        else:
            triton_fn = lambda: triton.predict_health(  # noqa: E731
                meta["equipment_type"], meta["model_purpose"], X
            )
        triton_times = time_calls(triton_fn, N_TRIALS)
        mean_t, p50_t, p95_t = percentiles(triton_times)

        logger.info(
            f"  batch={batch:>5}  python: mean={mean_p:7.2f}ms p50={p50_p:7.2f}ms p95={p95_p:7.2f}ms"
            f"   |   triton: mean={mean_t:7.2f}ms p50={p50_t:7.2f}ms p95={p95_t:7.2f}ms"
        )


def benchmark_shap(repo_root: Path) -> None:
    """Time TreeExplainer.shap_values for one classifier + one regressor per equipment type."""

    mgr = ShapExplainerManager(model_repo_root=repo_root)
    mgr.load_all()
    if mgr.num_loaded() == 0:
        logger.warning("No explainers loaded; skipping SHAP benchmark")
        return

    rng = np.random.default_rng(1)
    logger.info("\n=== SHAP TreeExplainer ===")
    seen_types = set()
    for name in mgr.loaded_models():
        # Pick one classifier + one regressor per equipment type to keep output manageable
        eq_type = name.split("_", 1)[0]
        purpose = name.split("_", 1)[1]
        key = (eq_type, "classifier" if purpose == "classifier" else "regressor")
        if key in seen_types:
            continue
        seen_types.add(key)

        entry = mgr._models[name]
        n_features = len(entry.feature_columns)
        for batch in (1, 32):
            X = rng.normal(0, 1, size=(batch, n_features)).astype(np.float32)
            times = time_calls(lambda: entry.explainer.shap_values(X), 20)
            mean_v, p50_v, p95_v = percentiles(times)
            logger.info(
                f"  {name} batch={batch:>3}  shap: mean={mean_v:7.2f}ms p50={p50_v:7.2f}ms p95={p95_v:7.2f}ms"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "triton" / "model_repository",
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip the SHAP benchmark block"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    if not args.repo_root.exists():
        logger.error(f"Model repository {args.repo_root} not found. Run export first.")
        return 2

    triton = TritonInferenceClient(url=args.triton_url, model_repo_root=args.repo_root)
    if not triton.is_healthy():
        logger.error(f"Triton at {args.triton_url} is not ready")
        return 2

    for model_dir in sorted(args.repo_root.iterdir()):
        if not (model_dir / "metadata.json").exists():
            continue
        try:
            benchmark_model(triton, model_dir)
        except Exception as e:
            logger.error(f"benchmark failed for {model_dir.name}: {e}")

    if not args.skip_shap:
        benchmark_shap(args.repo_root)

    return 0

if __name__ == "__main__":
    sys.exit(main())
