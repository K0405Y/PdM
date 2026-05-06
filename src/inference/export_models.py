"""Export MLflow-tracked models to a Triton FIL model repository.

For each (equipment_type, model_purpose) the script:
  1. Resolves the latest MLflow model from the configured experiment
  2. Loads it via mlflow.sklearn.load_model()
  3. Saves the native XGBoost JSON (FIL backend reads this directly)
  4. Saves a copy as `model_explainer.json` for SHAP TreeExplainer at startup
  5. Writes a metadata.json sidecar with feature columns, class names,
     QuantileTransformer params, and (for classifiers) the augmented input layout
  6. Writes a K-means background sample for SHAP baselines
  7. Generates the corresponding config.pbtxt

RandomForest regressors are converted via treelite. XGBoost is the default
training output and the dominant case.

Usage:
    python -m src.inference.export_models --equipment-types turbine compressor pump
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import ast
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.ml.data_loader import (
    get_health_columns,
    get_sensor_columns,
    load_table_config,
)
from src.inference.config_generator import generate_config
from src.inference.model_registry import ModelRegistry

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2] / "triton" / "model_repository"
EXPERIMENT = "pdm_end_to_end"
BACKGROUND_SAMPLE_SIZE = 100


@dataclass
class ExportTarget:
    """One model to export.

    `purpose` is either "classifier" or a health column name. `model_name` is
    the Triton-facing model directory (e.g. `turbine_classifier`).
    """

    equipment_type: str
    purpose: str
    model_name: str
    health_input_columns: List[str] = field(default_factory=list)


def list_targets(equipment_types: List[str]) -> List[ExportTarget]:
    """Build the export list: 1 classifier + N health regressors per equipment type."""
    targets: List[ExportTarget] = []
    for eq in equipment_types:
        health_cols = get_health_columns(eq)
        targets.append(
            ExportTarget(
                equipment_type=eq,
                purpose="classifier",
                model_name=f"{eq}_classifier",
                health_input_columns=list(health_cols),
            )
        )
        for col in health_cols:
            targets.append(
                ExportTarget(
                    equipment_type=eq,
                    purpose=col,
                    model_name=f"{eq}_{col}",
                )
            )
    return targets


def setup_mlflow() -> None:
    """Configure the MLflow client from env vars."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI not set. Cannot resolve models from MLflow."
        )
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")


def find_model_id(
    experiment_name: str,
    equipment_type: str,
    purpose: str,
    selection: str = "best",
) -> Tuple[str, str]:
    """Resolve a logged model_id for this equipment_type/purpose.

    selection:
        "best"   - rank by the most informative metric for this target:
                     classifier -> max metrics.test_macro_f1
                     health     -> min metrics.<purpose>_rmse
                   Falls back to most-recent if the metric is missing on all runs.
        "latest" - rank by start_time DESC (most recent training run first).

    Returns:
        (mlflow_model_id, mlflow_run_id)
    """
    if selection not in ("best", "latest"):
        raise ValueError(f"selection must be 'best' or 'latest', got {selection!r}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found")

    # Run names look like: f"{model_type}_{mode}_{equipment_type}_{timestamp}"
    # For classifier: mode='classifier'; for health regressors: mode='health'
    if purpose == "classifier":
        mode = "classifier"
        target_model_name = f"{equipment_type}_classifier_model"
        metric_key = "metrics.test_macro_f1"
        metric_ascending = False
    else:
        mode = "health"
        target_model_name = f"{equipment_type}_{purpose}_model"
        metric_key = f"metrics.{purpose}_rmse"
        metric_ascending = True

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name LIKE '%_{mode}_{equipment_type}_%'",
        max_results=50,
    )
    if runs.empty:
        raise RuntimeError(
            f"No MLflow runs found for {equipment_type}/{purpose} "
            f"(experiment={experiment_name}, mode={mode})"
        )

    if selection == "best" and metric_key in runs.columns and runs[metric_key].notna().any():
        scored = runs.dropna(subset=[metric_key]).sort_values(metric_key, ascending=metric_ascending)
        unscored = runs[runs[metric_key].isna()].sort_values("start_time", ascending=False)
        runs = pd.concat([scored, unscored], ignore_index=True)
    else:
        if selection == "best":
            logger.warning(
                f"  metric {metric_key} not found on any run; falling back to most-recent"
            )
        runs = runs.sort_values("start_time", ascending=False).reset_index(drop=True)

    for _, run in runs.iterrows():
        run_id = run["run_id"]
        logged_models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id='{run_id}' AND name='{target_model_name}'",
            output_format="list",
        )
        if logged_models:
            mlflow_model = logged_models[0]
            metric_val = run.get(metric_key)
            metric_repr = f"{metric_val:.4f}" if isinstance(metric_val, float) and not pd.isna(metric_val) else "n/a"
            logger.info(
                f"  selected run {run_id} ({selection}; {metric_key}={metric_repr})"
            )
            return mlflow_model.model_id, run_id

    raise RuntimeError(
        f"No logged model with name '{target_model_name}' found in any matching run "
        f"of experiment '{experiment_name}' for {equipment_type}"
    )


def load_mlflow_model(model_id: str):
    """Load a model artifact via MLflow given its model_id (m-...)."""
    return mlflow.sklearn.load_model(f"models:/{model_id}")


def fetch_classifier_class_names(mlflow_run_id: str) -> Optional[List[str]]:
    """Recover human-readable class names from a classifier run's MLflow params.

    train_failure_classifier.py logs the per-class sample counts as a
    stringified dict. sklearn's LabelEncoder sorts labels alphabetically, so
    sorting the dict keys gives the exact index->name mapping the model uses.
    """
    try:
        run = mlflow.get_run(mlflow_run_id)
    except Exception:
        return None
    params = run.data.params
    raw = params.get("train_class_distribution_resampled") or params.get("train_class_distribution")
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            return None
        return [str(k) for k in parsed.keys()]
    except (ValueError, SyntaxError):
        return None


def detect_model_format(model: Any) -> str:
    """Return the FIL model_type string for the given fitted model."""
    if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        return "xgboost_json"
    return "treelite_checkpoint"


def save_xgboost_model(model: Any, dest: Path) -> None:
    """Write XGBoost model in native JSON format for FIL.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    booster.save_model(str(dest))

    with open(dest, "r", encoding="utf-8") as f:
        payload = json.load(f)

    gbm_model = payload.get("learner", {}).get("gradient_booster", {}).get("model", {})
    cats = gbm_model.get("cats")
    if cats is not None:
        non_empty = any(cats.get(k) for k in ("enc", "feature_segments", "sorted_idx"))
        if non_empty:
            raise RuntimeError(
                f"{dest}: booster uses categorical features; cannot strip 'cats' block "
                "without losing model fidelity. Re-train without categoricals or upgrade Triton."
            )
        gbm_model.pop("cats", None)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f)


def save_model_for_fil(model: Any, version_dir: Path) -> str:
    """Persist the model in the format FIL expects.

    Returns the model_type string for the config.pbtxt.
    """
    fmt = detect_model_format(model)
    if fmt == "xgboost_json":
        save_xgboost_model(model, version_dir / "model.json")
    else:
        raise NotImplementedError(f"Unsupported model type for FIL export: {type(model)}")
    return fmt


def extract_regressor_base_score(model_json_path: Path) -> Optional[float]:
    """Read the scalar base_score from a saved XGBoost JSON regressor.

    Returns None for classifiers (multi-element base_score) or when the field
    is absent. Treelite-in-FIL-24.08 does not apply base_score for regressors,
    so the inference client adds it back from metadata.

    XGBoost serializes base_score as a stringified array, e.g. "[1.32E0]" for
    a regressor or "[0.14,0.14,...]" for a softprob classifier.
    """
    with open(model_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("learner", {}).get("learner_model_param", {}).get("base_score")
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            parts = [p.strip() for p in s[1:-1].split(",") if p.strip()]
            if len(parts) != 1:
                return None  # multiclass
            s = parts[0]
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(raw, list):
        if len(raw) != 1:
            return None
        try:
            return float(raw[0])
        except (TypeError, ValueError):
            return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def save_explainer_artifacts(model: Any, model_dir: Path) -> None:
    """Save artifacts the FastAPI SHAP explainer manager needs at startup.
    For XGBoost: native JSON (load with xgb.XGBClassifier()/load_model()).
    """
    if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        booster.save_model(str(model_dir / "model_explainer.json"))
    else:
        with open(model_dir / "model_explainer.pkl", "wb") as f:
            pickle.dump(model, f)


def kmeans_background(X: np.ndarray, k: int = BACKGROUND_SAMPLE_SIZE) -> np.ndarray:
    """Build a K-means summary of training data for SHAP baselines.

    SHAP recommends ~100 representative samples rather than the full training
    set; K-means centroids capture the distribution at low cost.
    """
    n = X.shape[0]
    if n <= k:
        return X.astype(np.float32)
    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)


def feature_columns_from_signature(model_meta: Any) -> List[str]:
    """Extract the ordered feature column list from MLflow model signature."""
    # MLflow's `Schema` exposes inputs as a list of `ColSpec` with `.name`
    sig = model_meta.signature
    if sig is None or sig.inputs is None:
        raise RuntimeError("MLflow model has no input signature; cannot recover feature order")
    return [col.name for col in sig.inputs]


def load_signature_columns(model_id: str) -> List[str]:
    """Load the input feature column order saved alongside the MLflow model."""
    # Use the artifact path so we don't need a live tracking server beyond load
    info = mlflow.models.get_model_info(f"models:/{model_id}")
    if info.signature is None or info.signature.inputs is None:
        raise RuntimeError(f"No signature for MLflow model {model_id}")
    return [col.name for col in info.signature.inputs]


def export_regressor_target_transformer(
    mlflow_run_id: str,
    health_column: str,
    model_dir: Path,
    model_version: int,
) -> Optional[str]:
    """Download a regressor target-transformer artifact from MLflow, if present."""
    candidate_paths = [
        f"target_transformers/{health_column}.pkl",
        f"target_transformers/{health_column}.joblib",
        f"{health_column}_target_transformer.pkl",
        f"target_transformer_{health_column}.pkl",
    ]

    def _copy_downloaded(local_path: Path, source_path: str) -> str:
        suffix = local_path.suffix or ".pkl"
        out_name = f"target_transformer_v{model_version}{suffix}"
        shutil.copy2(local_path, model_dir / out_name)
        logger.info(f"  wrote target transformer to {out_name} (from {source_path})")
        return out_name

    for artifact_path in candidate_paths:
        try:
            local = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run_id,
                    artifact_path=artifact_path,
                )
            )
        except Exception:
            continue
        if local.is_file():
            return _copy_downloaded(local, artifact_path)

    # Fallback: search artifact tree for a likely per-column transformer file.
    try:
        queue: List[str] = [""]
        visited = set()
        while queue:
            prefix = queue.pop(0)
            if prefix in visited:
                continue
            visited.add(prefix)

            infos = (
                mlflow.artifacts.list_artifacts(run_id=mlflow_run_id)
                if prefix == ""
                else mlflow.artifacts.list_artifacts(
                    run_id=mlflow_run_id, artifact_path=prefix
                )
            )
            for info in infos:
                rel_path = info.path
                name_lower = Path(rel_path).name.lower()

                if info.is_dir:
                    if (
                        prefix == ""
                        or "transform" in name_lower
                        or "quantile" in name_lower
                    ):
                        queue.append(rel_path)
                    continue

                if not name_lower.endswith((".pkl", ".pickle", ".joblib")):
                    continue
                if "transform" not in name_lower and "quantile" not in name_lower:
                    continue
                if health_column.lower() not in rel_path.lower():
                    continue

                local = Path(
                    mlflow.artifacts.download_artifacts(
                        run_id=mlflow_run_id,
                        artifact_path=rel_path,
                    )
                )
                if local.is_file():
                    return _copy_downloaded(local, rel_path)
    except Exception:
        pass

    return None


def write_metadata(
    model_dir: Path,
    target: ExportTarget,
    feature_columns: List[str],
    mlflow_model_id: str,
    mlflow_run_id: str,
    model_format: str,
    n_classes: Optional[int],
    class_names: Optional[List[str]],
    target_transformer_path: Optional[str],
    background_path: Optional[str],
    raw_feature_columns: Optional[List[str]] = None,
    health_input_columns: Optional[List[str]] = None,
    base_score: Optional[float] = None,
) -> dict:
    """Write metadata.json describing the model. Returns the dict written."""
    metadata = {
        "equipment_type": target.equipment_type,
        "model_purpose": target.purpose,
        "model_name": target.model_name,
        "mlflow_model_id": mlflow_model_id,
        "mlflow_run_id": mlflow_run_id,
        "model_format": model_format,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "n_classes": n_classes,
        "class_names": class_names,
        "target_transformer_path": target_transformer_path,
        "background_data_path": background_path,
    }
    if raw_feature_columns is not None:
        metadata["raw_feature_columns"] = raw_feature_columns
    if health_input_columns is not None:
        metadata["health_input_columns"] = health_input_columns
    if base_score is not None:
        metadata["base_score"] = base_score

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def derive_classifier_layout(
    feature_columns: List[str],
    health_input_columns: List[str],
) -> Tuple[List[str], List[str]]:
    """Split classifier feature_columns into (raw_feature_columns, health_input_columns).

    The classifier was trained with `[raw_features..., health_cols...]`. We
    locate health columns by name and return the prefix before them as the
    raw-feature contract for the inference router.
    """
    health_set = set(health_input_columns)
    raw = [c for c in feature_columns if c not in health_set]
    # Order of health columns in the classifier input is the order found in feature_columns
    health = [c for c in feature_columns if c in health_set]
    return raw, health


def export_target(
    target: ExportTarget,
    repo_root: Path,
    registry: ModelRegistry,
    background_provider: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    selection: str = "best",
) -> Optional[dict]:
    """Export one model. Returns the metadata dict, or None if skipped."""
    logger.info(f"Resolving MLflow model for {target.model_name}")
    experiment = EXPERIMENT
    mlflow_model_id, mlflow_run_id = find_model_id(
        experiment_name=experiment,
        equipment_type=target.equipment_type,
        purpose=target.purpose,
        selection=selection,
    )
    if not registry.needs_new_version(target.model_name, mlflow_model_id):
        logger.info(
            f"  {target.model_name}: already at MLflow model {mlflow_model_id}, skipping"
        )
        return None

    logger.info(f"  loading model {mlflow_model_id}")
    model = load_mlflow_model(mlflow_model_id)

    feature_columns = load_signature_columns(mlflow_model_id)
    n_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    if isinstance(model, xgb.XGBClassifier):
        n_classes = int(model.n_classes_)
        # The training data was label-encoded, so model.classes_ is just [0..N-1].
        # Recover the real failure-mode names from the MLflow run params.
        class_names = fetch_classifier_class_names(mlflow_run_id)
        if class_names is None or len(class_names) != n_classes:
            try:
                class_names = [str(c) for c in model.classes_]
            except Exception:
                class_names = [str(i) for i in range(n_classes)]

    next_version = registry.latest_version(target.model_name) + 1
    model_dir = repo_root / target.model_name
    version_dir = model_dir / str(next_version)
    version_dir.mkdir(parents=True, exist_ok=True)

    # 1. FIL-format model
    model_format = save_model_for_fil(model, version_dir)
    logger.info(f"  wrote FIL model ({model_format}) to {version_dir}")

    # 2. SHAP explainer copy
    save_explainer_artifacts(model, model_dir)

    # 3. K-means background sample for SHAP
    background_path: Optional[str] = None
    if background_provider is not None:
        bg_key = (target.equipment_type, target.purpose)
        bg = background_provider.get(bg_key)
        if bg is not None:
            bg_filename = f"background_data_v{next_version}.npy"
            np.save(model_dir / bg_filename, bg)
            background_path = bg_filename
            logger.info(f"  wrote background data ({bg.shape}) to {bg_filename}")

    # 4. Classifier-specific augmented layout
    raw_feature_columns: Optional[List[str]] = None
    health_input_columns: Optional[List[str]] = None
    if target.purpose == "classifier":
        raw_feature_columns, health_input_columns = derive_classifier_layout(
            feature_columns=feature_columns,
            health_input_columns=target.health_input_columns,
        )

    target_transformer_path: Optional[str] = None
    if target.purpose != "classifier":
        target_transformer_path = export_regressor_target_transformer(
            mlflow_run_id=mlflow_run_id,
            health_column=target.purpose,
            model_dir=model_dir,
            model_version=next_version,
        )

    # 5. metadata.json
    base_score: Optional[float] = None
    if target.purpose != "classifier" and model_format == "xgboost_json":
        base_score = extract_regressor_base_score(version_dir / "model.json")
        if base_score is not None:
            logger.info(f"  captured base_score={base_score:.6f} for client-side offset")

    metadata = write_metadata(
        model_dir=model_dir,
        target=target,
        feature_columns=feature_columns,
        mlflow_model_id=mlflow_model_id,
        mlflow_run_id=mlflow_run_id,
        model_format=model_format,
        n_classes=n_classes,
        class_names=class_names,
        target_transformer_path=target_transformer_path,
        background_path=background_path,
        raw_feature_columns=raw_feature_columns,
        health_input_columns=health_input_columns,
        base_score=base_score,
    )

    # 6. config.pbtxt (regenerate every export so n_features/n_classes track the model)
    config_text = generate_config(
        name=target.model_name,
        purpose="classifier" if target.purpose == "classifier" else "regressor",
        n_features=len(feature_columns),
        n_classes=n_classes,
        model_type=model_format,
    )
    with open(model_dir / "config.pbtxt", "w", encoding="utf-8") as f:
        f.write(config_text)

    # 7. Record in the registry
    registry.add_version(
        model_name=target.model_name,
        mlflow_model_id=mlflow_model_id,
        mlflow_run_id=mlflow_run_id,
        extra={"model_format": model_format, "n_features": len(feature_columns)},
    )
    return metadata


def load_background_data(
    equipment_types: List[str],
    background_dir: Optional[Path],
) -> Dict[Tuple[str, str], np.ndarray]:
    """Optional: load training-data parquet files and build K-means backgrounds.

    Looks for files named `<equipment>_train_classifier.parquet` and
    `<equipment>_train_regressor.parquet` in `background_dir`. If absent,
    SHAP will fall back to a zero baseline.
    """
    if background_dir is None or not background_dir.exists():
        return {}

    out: Dict[Tuple[str, str], np.ndarray] = {}
    for eq in equipment_types:
        for purpose, suffix in (("classifier", "classifier"), ("regressor", "regressor")):
            path = background_dir / f"{eq}_train_{suffix}.parquet"
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                X = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                bg = kmeans_background(X)
                if purpose == "classifier":
                    out[(eq, "classifier")] = bg
                else:
                    # Apply same background to every health regressor under this equipment
                    for col in get_health_columns(eq):
                        out[(eq, col)] = bg
                logger.info(f"Built background ({bg.shape}) from {path.name}")
            except Exception as e:
                logger.warning(f"Failed to build background from {path}: {e}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equipment-types",
        nargs="+",
        default=["turbine", "compressor", "pump"],
        help="Equipment types to export",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Triton model repository root",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Optional directory holding *_train_classifier.parquet / *_train_regressor.parquet for SHAP backgrounds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export even if registry shows the same MLflow model_id",
    )
    parser.add_argument(
        "--selection",
        choices=["best", "latest"],
        default="best",
        help="How to pick the MLflow run per target: 'best' uses test_macro_f1 (classifier) "
             "or <purpose>_rmse (health); 'latest' uses the most recent training run.",
    )
    args = parser.parse_args(argv)

    setup_mlflow()
    repo_root: Path = args.repo_root
    repo_root.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(repo_root)
    if args.force:
        # Wipe the registry but keep on-disk versions; new exports will bump version numbers
        logger.info("--force: ignoring registry, exporting all targets")

    targets = list_targets(args.equipment_types)
    backgrounds = load_background_data(args.equipment_types, args.background_dir)
    if not backgrounds:
        logger.info(
            "No background data parquets found; SHAP will use zero baselines. "
            "Pass --background-dir to enable K-means backgrounds."
        )

    exported = 0
    failed = 0
    for target in targets:
        try:
            if args.force:
                # Force a new version by clearing the registry entry for this model
                registry._data.pop(target.model_name, None)
            result = export_target(
                target=target,
                repo_root=repo_root,
                registry=registry,
                background_provider=backgrounds,
                selection=args.selection,
            )
            if result is not None:
                exported += 1
        except Exception as e:
            failed += 1
            logger.error(f"Failed to export {target.model_name}: {e}", exc_info=True)

    registry.save()
    logger.info(f"Done: {exported} exported, {failed} failed, registry at {registry.path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
