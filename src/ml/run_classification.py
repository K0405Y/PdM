"""
End-to-End Failure Mode Classification Runner

Usage:
    python -m src.ml.run_classification --equipment-type turbine
    python -m src.ml.run_classification --equipment-type compressor --horizon 48
    python -m src.ml.run_classification --equipment-type pump --output-dir models/
"""

import os
import sys
import argparse
import logging
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dotenv import load_dotenv
load_dotenv()

from src.ml.data_loader import get_engine, load_telemetry, load_failures, load_equipment_ids
from src.ml.feature_prep import (
    label_telemetry,
    select_features,
    prepare_xy,
    equipment_train_test_split,
    temporal_validation_split,
)
from src.ml.train_xgb_classifier import (
    setup_mlflow,
    train_fleet_model,
    fine_tune_per_equipment,
    save_model,
)
from src.ml.evaluate import (
    evaluate_model,
    evaluate_all_vs_sensor_only,
    get_feature_importance,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_classification(
    equipment_type: str,
    prediction_horizon_hours: float = 72.0,
    output_dir: str = "models",
    test_fraction: float = 0.3,
    db_url: str = None,
):
    """
    Full pipeline: load → label → split → train → evaluate.

    Args:
        equipment_type: 'turbine', 'compressor', or 'pump'
        prediction_horizon_hours: Look-back window for failure labeling
        output_dir: Directory to save trained models
        test_fraction: Fraction of equipment for test set
        db_url: PostgreSQL URL (or uses POSTGRES_URL env var)
    """
    logger.info(f"{'='*60}")
    logger.info(f"FAILURE MODE CLASSIFICATION: {equipment_type.upper()}")
    logger.info(f"Prediction horizon: {prediction_horizon_hours}h")
    logger.info(f"{'='*60}")

    # --- 1. Load data ---
    engine = get_engine(db_url)
    telemetry = load_telemetry(engine, equipment_type)
    failures = load_failures(engine, equipment_type)
    equipment_ids = load_equipment_ids(engine, equipment_type)

    if telemetry.empty:
        logger.error(f"No telemetry data found for {equipment_type}")
        return

    # --- 2. Label telemetry ---
    labeled = label_telemetry(telemetry, failures, prediction_horizon_hours)

    # --- 3. Equipment-based train/test split ---
    train_df, test_df = equipment_train_test_split(
        labeled, equipment_ids, test_fraction=test_fraction
    )

    # --- 4. Temporal validation split within training set ---
    train_df, val_df = temporal_validation_split(train_df, val_fraction=0.2)

    # --- 5. Train SENSOR_ONLY model (primary / production model) ---
    logger.info("\n--- SENSOR_ONLY MODEL ---")
    sensor_features = select_features(train_df, equipment_type, mode="sensor_only")
    X_train_s, y_train_s, le = prepare_xy(train_df, sensor_features)
    X_val_s, y_val_s, _ = prepare_xy(val_df, sensor_features)
    # Re-use same label encoder
    y_val_s = le.transform(val_df['label'].values)

    try:
        setup_mlflow(db_url)
    except Exception as e:
        logger.warning(f"MLflow setup failed ({e}), training without tracking")

    model_sensor, sensor_run_id = train_fleet_model(
        X_train_s, y_train_s, X_val_s, y_val_s,
        label_encoder=le,
        equipment_type=equipment_type,
        mode="sensor_only",
        log_to_mlflow=True,
    )

    # --- 6. Evaluate on test set ---
    X_test_s = test_df[sensor_features].copy()
    for col in X_test_s.columns:
        if X_test_s[col].dtype in ('float64', 'float32', 'int64', 'int32'):
            X_test_s[col] = X_test_s[col].fillna(0)
    y_test = le.transform(test_df['label'].values)

    results_sensor = evaluate_model(
        model_sensor, X_test_s, y_test, le, "sensor_only_test"
    )

    # --- 7. Train FULL model (with health features) for gap analysis ---
    logger.info("\n--- FULL MODEL (for gap analysis) ---")
    full_features = select_features(train_df, equipment_type, mode="full")
    X_train_f, y_train_f, _ = prepare_xy(train_df, full_features)
    y_train_f = le.transform(train_df['label'].values)
    X_val_f = val_df[full_features].copy()
    for col in X_val_f.columns:
        if X_val_f[col].dtype in ('float64', 'float32', 'int64', 'int32'):
            X_val_f[col] = X_val_f[col].fillna(0)
    y_val_f = le.transform(val_df['label'].values)

    model_full, _ = train_fleet_model(
        X_train_f, y_train_f, X_val_f, y_val_f,
        label_encoder=le,
        equipment_type=equipment_type,
        mode="all",
        params={'n_estimators': 300},
        log_to_mlflow=False,
    )

    X_test_f = test_df[full_features].copy()
    for col in X_test_f.columns:
        if X_test_f[col].dtype in ('float64', 'float32', 'int64', 'int32'):
            X_test_f[col] = X_test_f[col].fillna(0)

    gap_results = evaluate_all_vs_sensor_only(
        model_full, model_sensor,
        X_test_f, X_test_s,
        y_test, le,
    )

    # --- 8. Feature importance ---
    fi = get_feature_importance(model_sensor, sensor_features)

    # --- 9. Save model ---
    model_path = os.path.join(output_dir, f"{equipment_type}_failure_classifier.json")
    save_model(model_sensor, model_path)

    # Save label encoder mapping
    le_path = os.path.join(output_dir, f"{equipment_type}_label_encoder.json")
    os.makedirs(os.path.dirname(le_path), exist_ok=True)
    with open(le_path, 'w') as f:
        json.dump({
            'classes': list(le.classes_),
            'equipment_type': equipment_type,
            'prediction_horizon_hours': prediction_horizon_hours,
            'sensor_features': sensor_features,
        }, f, indent=2)
    logger.info(f"Label encoder saved to {le_path}")

    # --- 10. Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Equipment type:     {equipment_type}")
    logger.info(f"Sensor-only F1:     {results_sensor['macro_f1']:.4f}")
    logger.info(f"Full-mode F1:       {gap_results['full']['macro_f1']:.4f}")
    logger.info(f"Generalization gap:  {gap_results['gap']['macro_f1_gap']:.4f}")
    logger.info(f"Model saved:        {model_path}")
    logger.info(f"Top feature:        {fi.iloc[0]['feature']}")

    return {
        'model': model_sensor,
        'results': results_sensor,
        'gap': gap_results['gap'],
        'feature_importance': fi,
        'label_encoder': le,
    }


def main():
    parser = argparse.ArgumentParser(description='PdM Failure Mode Classification')
    parser.add_argument('--equipment-type', '-e', required=True,
                        choices=['turbine', 'compressor', 'pump'],
                        help='Equipment type to classify')
    parser.add_argument('--prediction-horizon', '-p', type=float, default=72.0,
                        help='Hours before failure to label as pre-failure (default: 72)')
    parser.add_argument('--output-dir', '-o', default='models',
                        help='Directory for saved models (default: models/)')
    parser.add_argument('--test-fraction', '-t', type=float, default=0.3,
                        help='Fraction of equipment for test set (default: 0.3)')
    parser.add_argument('--db-url', default=None,
                        help='PostgreSQL URL (default: POSTGRES_URL env var)')

    args = parser.parse_args()

    run_classification(
        equipment_type=args.equipment_type,
        prediction_horizon_hours=args.prediction_horizon,
        output_dir=args.output_dir,
        test_fraction=args.test_fraction,
        db_url=args.db_url,
    )


if __name__ == '__main__':
    main()
