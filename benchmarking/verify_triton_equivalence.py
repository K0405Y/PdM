"""Verify that Triton FIL predictions match the original Python model output.

Loads each model from the Triton repository (XGBoost JSON / treelite checkpoint),
generates random feature vectors of the right shape, runs predictions through
both the local Python model and the running Triton server, and asserts
np.allclose(atol=1e-5) for both regressors and classifiers.

Usage:
    python benchmarking/verify_triton_equivalence.py
        [--triton-url localhost:8001]
        [--repo-root triton/model_repository]
        [--n-samples 64]
        [--atol 1e-5]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.triton_client import TritonInferenceClient
import xgboost as xgb


logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s %(levelname)s %(message)s"


def load_python_model(model_dir: Path):
    """Load the raw XGBoost Booster -- this is the source of truth FIL serves.

    We avoid the sklearn wrapper because XGBClassifier().load_model() leaves
    the wrapper without n_classes_/objective_type_, causing predict_proba to
    apply softmax a second time on top of an already-softmaxed multi:softprob
    booster. Booster.predict() returns the raw model output directly.
    """
    json_path = model_dir / "model_explainer.json"
    pkl_path = model_dir / "model_explainer.pkl"

    if json_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(json_path))
        return booster
    if pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"No explainer artifact in {model_dir}")


def random_features(n_samples: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Plausible-magnitude inputs (mostly small floats; some realistic spread)."""
    return rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(np.float32)


def _python_classifier_proba(model, X: np.ndarray) -> np.ndarray:
    out = model.predict(xgb.DMatrix(X), validate_features=False)
    return out.astype(np.float32)


def _python_regressor_predict(model, X: np.ndarray) -> np.ndarray:
    out = model.predict(xgb.DMatrix(X), validate_features=False)
    return out.astype(np.float32).reshape(-1, 1)


def verify_one(
    triton,
    model_dir: Path,
    n_samples: int,
    atol: float,
    rng: np.random.Generator,
) -> Tuple[bool, str]:
    """Compare Python vs Triton predictions for one model. Returns (ok, message)."""
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    name = meta["model_name"]
    n_features = int(meta["n_features"])
    is_classifier = meta.get("model_purpose") == "classifier"

    X = random_features(n_samples, n_features, rng)
    py_model = load_python_model(model_dir)

    if is_classifier:
        py_out = _python_classifier_proba(py_model, X)
        # Triton path
        equipment_type = meta["equipment_type"]
        _, triton_out = triton.predict_classifier(equipment_type, X)
    else:
        py_out = _python_regressor_predict(py_model, X)
        equipment_type = meta["equipment_type"]
        health_column = meta["model_purpose"]
        triton_out = triton.predict_health(equipment_type, health_column, X).reshape(-1, 1)

    if py_out.shape != triton_out.shape:
        return False, f"{name}: shape mismatch py={py_out.shape} triton={triton_out.shape}"

    if not np.allclose(py_out, triton_out, atol=atol):
        max_abs = float(np.abs(py_out - triton_out).max())
        return False, f"{name}: max |diff|={max_abs:.6f} exceeds atol={atol}"

    return True, f"{name}: OK ({n_samples}x{n_features}, max |diff|={float(np.abs(py_out - triton_out).max()):.2e})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "triton" / "model_repository",
    )
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    if not args.repo_root.exists():
        logger.error(f"Model repository {args.repo_root} not found. Run export first.")
        return 2

    triton = TritonInferenceClient(url=args.triton_url, model_repo_root=args.repo_root)
    if not triton.is_healthy():
        logger.error(f"Triton at {args.triton_url} is not ready")
        return 2

    rng = np.random.default_rng(42)
    passed: list[str] = []
    failed: list[str] = []
    for model_dir in sorted(args.repo_root.iterdir()):
        if not (model_dir / "metadata.json").exists():
            continue
        try:
            ok, msg = verify_one(triton, model_dir, args.n_samples, args.atol, rng)
        except Exception as e:
            ok = False
            msg = f"{model_dir.name}: exception {e!r}"
        (passed if ok else failed).append(msg)
        logger.info(("[PASS] " if ok else "[FAIL] ") + msg)

    logger.info(f"Summary: {len(passed)} passed, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
