"""
Unified training entry point.

    python train.py --task classification
    python train.py --task regression
    python train.py --task classification --force-recompute
    python train.py --task regression --config configs/turbine.yaml

Loads raw telemetry + failures, runs the shared label/split/feature-engineering
pipeline once, caches the engineered splits to disk, then dispatches to either
the failure classifier or the health regressor training scripts.

Cache invalidation is hash-based: the cache key encodes (equipment_type,
prediction_horizon, split fractions, source-data fingerprint, feature_prep
code hash). Anything that affects the engineered features changes the key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.ml.data_loader import (
    get_engine,
    get_health_columns,
    load_equipment_ids,
    load_failures,
    load_telemetry,
)
from src.ml.evaluate import (
    evaluate_classifier_model,
    evaluate_health_regressors,
)
from src.ml.feature_prep import (
    compute_cumulative_features,
    compute_derived_features,
    compute_regressor_indicators,
    label_telemetry,
    prepare_xy,
    select_features,
    temporal_train_test_split,
    temporal_validation_split,
)
from src.ml.train_failure_classifier import (
    setup_mlflow,
    train_fleet_model_with_tuning,
)
from src.ml.train_health_estimators import train_health_estimators

logger = logging.getLogger("train")

CACHE_ROOT = PROJECT_ROOT / "cache" / "features"
FEATURE_PREP_FILE = PROJECT_ROOT / "src" / "ml" / "feature_prep.py"
DATA_LOADER_FILE = PROJECT_ROOT / "src" / "ml" / "data_loader.py"


# Config
@dataclass
class PipelineConfig:
    """Knobs that affect the engineered-feature output."""
    equipment_type: str = "turbine"
    prediction_horizon_hours: float = 168.0
    test_fraction: float = 0.25
    val_fraction: float = 0.10

    # Training-only knobs
    experiment_name: str = "pdm_end_to_end"
    n_cv_folds: int = 5
    n_iter: int = 10
    model_type: str = "xgboost"
    log_to_mlflow: bool = True

    @classmethod
    def from_file(cls, path: Path) -> "PipelineConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# Cache key / fingerprinting
def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _data_fingerprint(telemetry: pd.DataFrame, failures: pd.DataFrame) -> Dict[str, str]:
    """source-data fingerprint.
    Captures size + temporal extent + failure event identity. If the upstream
    tables change in any way that matters for FE, at least one of these moves.
    """
    def _max_str(s: pd.Series) -> str:
        return str(s.max()) if len(s) else ""

    return {
        "telemetry_rows": str(len(telemetry)),
        "telemetry_max_time": _max_str(telemetry["sample_time"]),
        "telemetry_min_time": str(telemetry["sample_time"].min()) if len(telemetry) else "",
        "failure_rows": str(len(failures)),
        "failure_max_time": _max_str(failures["failure_time"]) if "failure_time" in failures.columns else "",
        "failure_modes": ",".join(sorted(failures["failure_mode_code"].unique())) if "failure_mode_code" in failures.columns else "",
    }


def compute_cache_key(cfg: PipelineConfig, fingerprint: Dict[str, str]) -> str:
    """Deterministic hash over everything that affects the engineered features."""
    payload = {
        "equipment_type": cfg.equipment_type,
        "prediction_horizon_hours": cfg.prediction_horizon_hours,
        "test_fraction": cfg.test_fraction,
        "val_fraction": cfg.val_fraction,
        "fingerprint": fingerprint,
        "code_hashes": {
            "feature_prep": _hash_file(FEATURE_PREP_FILE),
            "data_loader": _hash_file(DATA_LOADER_FILE),
        },
        "schema_version": 1,
    }
    serialized = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:16]


def cache_dir_for(cfg: PipelineConfig, key: str) -> Path:
    return CACHE_ROOT / cfg.equipment_type / key


# Shared pipeline: load → label → split → engineer → cache
def load_raw(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull telemetry + failures from Postgres, normalize equipment id column."""
    db_url = os.environ.get("POSTGRES_URL", "")
    if not db_url:
        raise RuntimeError("POSTGRES_URL not set; cannot load raw data")

    engine = get_engine(db_url)
    telemetry = load_telemetry(engine, cfg.equipment_type)
    failures = load_failures(engine, cfg.equipment_type)
    equipment_ids = load_equipment_ids(engine, cfg.equipment_type)
    logger.info(
        "Raw: telemetry=%s rows, failures=%s, units=%s",
        f"{len(telemetry):,}", f"{len(failures):,}", len(equipment_ids),
    )
    return telemetry, failures


def engineer_features(
    telemetry: pd.DataFrame, failures: pd.DataFrame, cfg: PipelineConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label → temporal split → derived + cumulative features."""
    labeled = label_telemetry(telemetry, failures, cfg.prediction_horizon_hours)
    train_df, test_df = temporal_train_test_split(labeled, cfg.test_fraction)
    train_df, val_df = temporal_validation_split(train_df, val_fraction=cfg.val_fraction)
    logger.info("Split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        logger.info("Engineering features on %s (%d rows)…", split_name, len(df))
        compute_derived_features(df, cfg.equipment_type)
        compute_cumulative_features(df, cfg.equipment_type)

    return train_df, val_df, test_df


def save_cache(
    cache_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: Dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(cache_dir / "train.parquet", index=False)
    val_df.to_parquet(cache_dir / "val.parquet", index=False)
    test_df.to_parquet(cache_dir / "test.parquet", index=False)
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Cached engineered splits to %s", cache_dir)


def load_cache(cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    train_df = pd.read_parquet(cache_dir / "train.parquet")
    val_df = pd.read_parquet(cache_dir / "val.parquet")
    test_df = pd.read_parquet(cache_dir / "test.parquet")
    with open(cache_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    logger.info(
        "Loaded cached splits from %s (train=%d, val=%d, test=%d)",
        cache_dir, len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df, meta


def cache_is_valid(cache_dir: Path) -> bool:
    return (
        cache_dir.is_dir()
        and (cache_dir / "train.parquet").exists()
        and (cache_dir / "val.parquet").exists()
        and (cache_dir / "test.parquet").exists()
        and (cache_dir / "meta.json").exists()
    )


def get_engineered_splits(
    cfg: PipelineConfig, use_cache: bool, force_recompute: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Return (train, val, test, cache_key), loading from cache when valid.

    Even when force_recompute=True we still load raw data once to compute the
    fingerprint, so the new cache key reflects current source state.
    """
    telemetry, failures = load_raw(cfg)
    fingerprint = _data_fingerprint(telemetry, failures)
    key = compute_cache_key(cfg, fingerprint)
    cache_dir = cache_dir_for(cfg, key)

    if use_cache and not force_recompute and cache_is_valid(cache_dir):
        train_df, val_df, test_df, _ = load_cache(cache_dir)
        return train_df, val_df, test_df, key

    if force_recompute and cache_dir.exists():
        logger.info("--force-recompute: recomputing despite cache at %s", cache_dir)

    t0 = time.time()
    train_df, val_df, test_df = engineer_features(telemetry, failures, cfg)
    logger.info("Feature engineering took %.1fs", time.time() - t0)

    meta = {
        "cache_key": key,
        "config": asdict(cfg),
        "fingerprint": fingerprint,
        "row_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "feature_columns": [c for c in train_df.columns],
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    save_cache(cache_dir, train_df, val_df, test_df, meta)
    return train_df, val_df, test_df, key


# Task: classification
def run_classification(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Dict:
    setup_mlflow(experiment_name=cfg.experiment_name)

    feature_cols = select_features(train_df, cfg.equipment_type, mode="classifier")
    logger.info("Classifier features (%d): %s", len(feature_cols), feature_cols)

    X_train, y_train, le, medians = prepare_xy(train_df, feature_cols)
    X_val, _, _, _ = prepare_xy(val_df, feature_cols, medians=medians)
    X_test, _, _, _ = prepare_xy(test_df, feature_cols, medians=medians)

    y_train = le.transform(train_df["label"].values)
    y_val = le.transform(val_df["label"].values)
    y_test = le.transform(test_df["label"].values)

    model, run_id, model_id = train_fleet_model_with_tuning(
        X_train, y_train, X_val, y_val,
        label_encoder=le,
        equipment_type=cfg.equipment_type,
        n_cv_folds=cfg.n_cv_folds,
        n_iter=cfg.n_iter,
        log_to_mlflow=cfg.log_to_mlflow,
        groups=train_df["equipment_id"].values,
    )

    metrics = evaluate_classifier_model(
        model, X_test, y_test, le,
        dataset_name="test",
        log_to_mlflow=cfg.log_to_mlflow,
        run_id=run_id,
        feature_names=feature_cols,
        log_test_data=True,
        model_id=model_id,
    )
    logger.info(
        "Classifier — acc=%.4f rocauc_macro=%.4f macro_f1=%.4f weighted_f1=%.4f",
        metrics["accuracy"], metrics.get("roc_auc_macro", 0.0),
        metrics["macro_f1"], metrics["weighted_f1"],
    )
    return metrics


# Task: regression
def _regressor_group_split(
    all_df: pd.DataFrame, regressor_cols, health_columns,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame]:
    """Group-aware 70/10/20 split by equipment_id."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(gss.split(all_df, groups=all_df["equipment_id"]))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(
        gss2.split(all_df.iloc[trainval_idx],
                   groups=all_df.iloc[trainval_idx]["equipment_id"])
    )

    train_reg = all_df.iloc[trainval_idx].iloc[train_idx]
    val_reg = all_df.iloc[trainval_idx].iloc[val_idx]
    test_reg = all_df.iloc[test_idx]

    return (
        train_reg[regressor_cols], val_reg[regressor_cols], test_reg[regressor_cols],
        train_reg[health_columns], val_reg[health_columns], test_reg[health_columns],
        train_reg,
    )


def run_regression(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> Dict:
    setup_mlflow(experiment_name=cfg.experiment_name)

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    health_columns = get_health_columns(cfg.equipment_type)
    compute_regressor_indicators(all_df, cfg.equipment_type)
    regressor_cols = select_features(all_df, cfg.equipment_type, mode="regressor")
    logger.info("Regressor features (%d): %s", len(regressor_cols), regressor_cols)

    (X_train, X_val, X_test,
     h_train, h_val, h_test,
     train_reg) = _regressor_group_split(all_df, regressor_cols, health_columns)

    logger.info("Regressor split: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    regressors, run_id, model_ids, transformers = train_health_estimators(
        X_train, h_train, X_val, h_val,
        equipment_type=cfg.equipment_type,
        model_type=cfg.model_type,
        health_columns=health_columns,
        n_cv_folds=cfg.n_cv_folds,
        n_iter=cfg.n_iter,
        log_to_mlflow=cfg.log_to_mlflow,
        groups=train_reg["equipment_id"].values,
        use_target_transform=False,
        train_labels=train_reg["label"].values,
    )

    metrics = evaluate_health_regressors(
        regressors, X_test, h_test,
        log_to_mlflow=cfg.log_to_mlflow,
        dataset_name="true",
        run_id=run_id,
        model_ids=model_ids,
        feature_names=regressor_cols,
        log_test_data=True,
        target_transformers=transformers,
    )
    for col, m in metrics.items():
        logger.info("  %s — R²=%.4f MAE=%.4f RMSE=%.4f", col, m["r2"], m["mae"], m["rmse"])
    return metrics


# CLI

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified PdM training entry point.")
    p.add_argument("--task", required=True, choices=["classification", "regression"])
    p.add_argument("--config", type=Path, default=None,
                   help="Optional YAML config file (overrides defaults).")
    p.add_argument("--equipment-type", choices=["turbine", "compressor", "pump"],
                   help="Override equipment_type from config.")
    p.add_argument("--use-cache", dest="use_cache", action="store_true", default=True,
                   help="Use cached engineered features when valid (default).")
    p.add_argument("--no-cache", dest="use_cache", action="store_false",
                   help="Skip cache lookup (still writes a new cache).")
    p.add_argument("--force-recompute", action="store_true",
                   help="Recompute engineered features even if cache exists.")
    p.add_argument("--no-mlflow", action="store_true",
                   help="Skip MLflow logging for this run.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    cfg = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()
    if args.equipment_type:
        cfg.equipment_type = args.equipment_type
    if args.no_mlflow:
        cfg.log_to_mlflow = False

    logger.info("Task=%s | %s", args.task, asdict(cfg))

    train_df, val_df, test_df, key = get_engineered_splits(
        cfg, use_cache=args.use_cache, force_recompute=args.force_recompute,
    )
    logger.info("Cache key: %s", key)

    if args.task == "classification":
        run_classification(train_df, val_df, test_df, cfg)
    else:
        run_regression(train_df, val_df, test_df, cfg)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())