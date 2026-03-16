"""
Health Indicator Estimation via XGBoost Regressors

Trains per-component regressors to estimate health indicators (health_hgp,
health_blade, health_bearing, health_fuel) from sensor features + physics-based
indicators. Used to bridge the sensor-only → ground truth accuracy gap.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.preprocessing import QuantileTransformer
from src.ml.data_loader import load_table_config
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# Features where higher values indicate worse health (health decreases)
_MONOTONE_DECREASING = {
    'operating_hours', 'lifecycle_position',
    'cummax_vibration_rms', 'cummax_vibration_peak',
    'cum_efficiency_loss',
    'cummax_egt', 'cum_fuel_flow_increase',
    'vibration_rms_mm_s', 'vibration_peak_mm_s',
    'vibration_kurtosis', 'vibration_crest_factor',
}
# Note: crest_factor_deviation, kurtosis_excess, cummax_crest_deviation,
# cummax_kurtosis_excess are left unconstrained — they help blade/HGP
# discrimination but are not universally monotone across all health indicators

# Features where higher values indicate better health (health increases)
_MONOTONE_INCREASING = {
    'efficiency_fraction',
    'cummin_efficiency',
}

# Train only on degraded samples for these columns (above threshold, sensors carry no signal)
_DEGRADED_THRESHOLDS = {
    'health_hgp': 0.90,
    'health_blade': 0.85,
}


def _build_monotone_constraints(feature_names: List[str]) -> Tuple[int, ...]:
    """Build monotone_constraints tuple for XGBoost from feature names."""
    constraints = []
    for name in feature_names:
        if name in _MONOTONE_DECREASING:
            constraints.append(-1)
        elif name in _MONOTONE_INCREASING:
            constraints.append(1)
        else:
            constraints.append(0)
    return tuple(constraints)


def train_health_estimators( X_train: pd.DataFrame, health_train: pd.DataFrame, X_val: pd.DataFrame, health_val: pd.DataFrame, 
                            equipment_type: str, health_columns: List[str], n_cv_folds: int = 5, n_iter: int = 20, 
                            log_to_mlflow: bool = True
                            ) -> Tuple[Dict[str, xgb.XGBRegressor], Optional[str], Dict, Dict]:
    """
    Train one XGBoost regressor per health indicator column.

    Args:
        X_train: Training sensor features (already normalized)
        health_train: Ground truth health columns for training
        X_val: Validation sensor features
        health_val: Ground truth health columns for validation
        equipment_type: 'turbine', 'compressor', or 'pump'
        health_columns: List of health column names to estimate
        n_cv_folds: Number of CV folds
        n_iter: Number of random hyperparameter combinations per column
        log_to_mlflow: Whether to log to MLflow

    Returns:
        Tuple of (regressors dict, MLflow run_id, model_ids dict, target_transformers dict)
    """
    cfg = load_table_config().get('xgb_config', {})
    param_distributions = cfg.get('tuning_distributions')
    cfg_defaults = cfg.get('default_params', {})

    monotone_constraints = _build_monotone_constraints(list(X_train.columns))

    fixed_params = {
        'objective': 'reg:pseudohubererror',
        'eval_metric': 'mae',
        'tree_method': cfg_defaults.get('tree_method', 'hist'),
        'random_state': cfg_defaults.get('random_state', 42),
        'verbosity': 0,
        'early_stopping_rounds': cfg_defaults.get('early_stopping_rounds', 10),
        'monotone_constraints': monotone_constraints,
    }

    # GroupKFold on equipment_id
    groups = X_train['equipment_id'].values
    cv_splitter = GroupKFold(n_splits=n_cv_folds)

    regressors = {}
    model_ids = {}
    target_transformers = {}
    run_id = None

    if log_to_mlflow:
        run_name = f"health_estimator_{equipment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        parent_run = mlflow.start_run(run_name=run_name, tags={
            "mlflow.source.name": "src/ml/train_health_estimators.py",
            "mlflow.source.type": "LOCAL",
        })
        run_id = parent_run.info.run_id
        mlflow.log_params({
            'equipment_type': equipment_type,
            'model_scope': 'health_estimation',
            'n_cv_folds': n_cv_folds,
            'n_iter': n_iter,
            'health_columns': str(health_columns),
            'objective': 'reg:pseudohubererror',
            'monotone_constraints': str(monotone_constraints),
        })

    param_list = list(ParameterSampler(
        param_distributions, n_iter=n_iter, random_state=42
    ))

    for col in health_columns:
        logger.info(f"Training regressor for {col}...")
        y_train_raw = health_train[col].values
        y_val_raw = health_val[col].values

        # Use all training data (degraded filtering removed — hurts test-set R²
        # because model never sees healthy-range data during training)
        X_train_col = X_train
        y_train_raw_col = y_train_raw
        X_val_col = X_val
        y_val_raw_col = y_val_raw
        groups_col = groups

        # Target transformation to spread clustered-near-1.0 distribution
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(y_train_raw_col)))
        y_train_col = qt.fit_transform(y_train_raw_col.reshape(-1, 1)).ravel()
        y_val_col = qt.transform(y_val_raw_col.reshape(-1, 1)).ravel()
        target_transformers[col] = qt

        col_cv_splitter = GroupKFold(n_splits=n_cv_folds)

        best_score = float('inf')
        best_params = None

        for i, trial_params in enumerate(param_list):
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(
                col_cv_splitter.split(X_train_col, y_train_col, groups_col)
            ):
                X_fold_train = X_train_col.iloc[train_idx]
                y_fold_train = y_train_col[train_idx]
                X_fold_val = X_train_col.iloc[val_idx]
                y_fold_val = y_train_col[val_idx]

                model = xgb.XGBRegressor(**fixed_params, **trial_params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False,
                )
                fold_scores.append(model.best_score)

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            logger.info(f"  {col} Trial {i+1}/{n_iter}: mae={mean_score:.4f} +/- {std_score:.4f}")

            if log_to_mlflow:
                with mlflow.start_run(
                    run_name=f"trial_{col}_{i+1}",
                    nested=True,
                    tags={"mlflow.source.name": "src/ml/train_health_estimators.py", "mlflow.source.type": "LOCAL"}
                ):
                    mlflow.log_params(trial_params)
                    mlflow.set_tag("health_column", col)
                    mlflow.log_metric('mean_cv_mae', mean_score)
                    mlflow.log_metric('std_cv_mae', std_score)
                    for fold_idx, score in enumerate(fold_scores):
                        mlflow.log_metric(f'fold_{fold_idx}_mae', score)

            if mean_score < best_score:
                best_score = mean_score
                best_params = trial_params

        logger.info(f"  {col} best params (mae={best_score:.4f}): {best_params}")

        # Train final model on filtered training set with best params
        final_model = xgb.XGBRegressor(**fixed_params, **best_params)
        final_model.fit(
            X_train_col, y_train_col,
            eval_set=[(X_val_col, y_val_col)],
            verbose=False,
        )

        val_mae = final_model.best_score
        logger.info(f"  {col} final val mae: {val_mae:.4f}")

        if log_to_mlflow:
            mlflow.log_params({f'best_{k}_{col}': v for k, v in best_params.items()})

            X_train_float = X_train.astype({c: 'float64' for c in X_train.select_dtypes('integer').columns})
            signature = infer_signature(X_train_float, final_model.predict(X_train_float))

            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                name=f"{col}_model",
                signature = signature,
                input_example=X_train_float.iloc[:1]
            )
            model_ids[col] = model_info.model_id
            mlflow.log_metrics({
                f'best_cv_mae_{col}': best_score,
                f'final_val_mae_{col}': val_mae,
            }, model_id=model_info.model_id)

        regressors[col] = final_model

    if log_to_mlflow:
        mlflow.end_run()

    return regressors, run_id, model_ids, target_transformers


def estimate_health_features(
    X: pd.DataFrame,
    health_regressors: Dict[str, xgb.XGBRegressor],
    target_transformers: Optional[Dict[str, QuantileTransformer]] = None,
) -> pd.DataFrame:
    """
    Predict health indicators from sensor features using trained regressors.

    Args:
        X: Sensor feature DataFrame
        health_regressors: Dict mapping health column name to trained regressor
        target_transformers: Dict mapping health column name to fitted QuantileTransformer.
                            If provided, inverse-transforms predictions back to original scale.

    Returns:
        DataFrame with estimated health columns, same index as X
    """
    estimated = {}
    for col, regressor in health_regressors.items():
        preds = regressor.predict(X)
        if target_transformers and col in target_transformers:
            preds = target_transformers[col].inverse_transform(preds.reshape(-1, 1)).ravel()
        estimated[col] = np.clip(preds, 0.0, 1.0)

    return pd.DataFrame(estimated, index=X.index)
