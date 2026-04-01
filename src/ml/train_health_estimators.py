"""
Health Indicator Estimation via Multiple Regressor Types

Trains per-component regressors to estimate health indicators (health_hgp,
health_blade, health_bearing, health_fuel) from sensor features + physics-based
indicators. Supports XGBoost, CatBoost, and RandomForest.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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

# Features where higher values indicate better health (health increases)
_MONOTONE_INCREASING = {
    'efficiency_fraction',
    'cummin_efficiency',
}

VALID_MODEL_TYPES = ('xgboost', 'random_forest')

# Health columns where monotonic constraints hurt (entangled physics)
_UNCONSTRAINED_COLUMNS = {'health_hgp', 'health_blade', 'health_bearing', 'seal_health_primary', 'seal_health_secondary'}


def _build_monotone_constraints(feature_names: List[str], health_col: str = '') -> Tuple[int, ...]:
    """Build monotone_constraints tuple from feature names.

    Disables constraints for entangled health columns (HGP, blade) where
    the non-monotonic EGT/efficiency relationship makes constraints counterproductive.
    """
    if health_col in _UNCONSTRAINED_COLUMNS:
        return tuple(0 for _ in feature_names)
    constraints = []
    for name in feature_names:
        if name in _MONOTONE_DECREASING:
            constraints.append(-1)
        elif name in _MONOTONE_INCREASING:
            constraints.append(1)
        else:
            constraints.append(0)
    return tuple(constraints)


def _build_model_config(model_type: str, feature_names: List[str], cfg: Dict, health_col: str = '',
                        ) -> Tuple[Dict, List[Dict], type]:
    """
    Build fixed params, param search list, and model class for the given model type.

    Returns:
        (fixed_params, param_distributions, model_class)
    """
    monotone_constraints = _build_monotone_constraints(feature_names, health_col)
    cfg_defaults = cfg.get('xgb_config', {}).get('default_params', {})

    if model_type == 'xgboost':
        fixed_params = {
            'objective': 'reg:pseudohubererror',
            'eval_metric': 'mae',
            'tree_method': cfg_defaults.get('tree_method', 'hist'),
            'random_state': cfg_defaults.get('random_state', 42),
            'verbosity': 0,
            'early_stopping_rounds': cfg_defaults.get('early_stopping_rounds', 10),
            'monotone_constraints': monotone_constraints,
        }
        param_dists = cfg.get('xgb_config', {}).get('tuning_distributions', {})
        return fixed_params, param_dists, xgb.XGBRegressor

    elif model_type == 'random_forest':
        fixed_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'criterion': 'absolute_error',
        }
        param_dists = {}  # empty = skip tuning loop
        return fixed_params, param_dists, RandomForestRegressor

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Must be one of {VALID_MODEL_TYPES}")


def _fit_model(model, X_train, y_train, X_val, y_val, model_type: str):
    """Fit a model with appropriate arguments for each model type."""
    if model_type == 'xgboost':
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:  # random_forest
        model.fit(X_train, y_train)


def _get_val_score(model, X_val, y_val, model_type: str) -> float:
    """Get validation MAE for a fitted model."""
    if model_type == 'xgboost':
        return model.best_score if hasattr(model, 'best_score') and model.best_score else mean_absolute_error(y_val, model.predict(X_val))
    else:
        return mean_absolute_error(y_val, model.predict(X_val))


def train_health_estimators(X_train: pd.DataFrame, health_train: pd.DataFrame, X_val: pd.DataFrame,
                            health_val: pd.DataFrame, equipment_type: str, health_columns: List[str],
                            model_type: str = 'xgboost', n_cv_folds: int = 5, n_iter: int = 20,
                            log_to_mlflow: bool = True, mode = 'health',
                            use_target_transform: bool = True,
                            best_params_override: Optional[Dict[str, Dict]] = None,
                            groups: Optional[np.ndarray] = None,
                            ) -> Tuple[Dict[str, Any], Optional[str], Dict, Dict]:
    """
    Train one regressor per health indicator column.

    Args:
        model_type: 'xgboost' or 'random_forest'
        best_params_override: Optional dict mapping health column name to pre-tuned params dict.
                             Skips CV tuning for columns with overrides.

    Returns:
        Tuple of (regressors dict, MLflow run_id, model_ids dict, target_transformers dict)
    """
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {VALID_MODEL_TYPES}, got '{model_type}'")

    cfg = load_table_config()
    feature_names = list(X_train.columns)

    # GroupKFold on equipment_id
    if groups is None:
        groups = X_train['equipment_id'].values

    regressors = {}
    model_ids = {}
    target_transformers = {}
    deferred_metrics = {}  # {col: {metric_name: value}} — logged after all models registered
    run_id = None

    if log_to_mlflow:
        run_name = f"{model_type}_{mode}_{equipment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        parent_run = mlflow.start_run(run_name=run_name, tags={
            "mlflow.source.name": "src/ml/train_health_estimators.py",
            "mlflow.source.type": "LOCAL",
        })
        run_id = parent_run.info.run_id
        mlflow.log_params({
            'equipment_type': equipment_type,
            'model_type': model_type,
            'model_scope': 'health_estimation',
            'n_cv_folds': n_cv_folds,
            'n_iter': n_iter,
            'health_columns': str(health_columns),
        })

    # Build config once (shared param_list across all columns)
    fixed_params, param_distributions, model_class = _build_model_config(model_type, feature_names, cfg)
    skip_tuning = not param_distributions
    if not skip_tuning:
        param_list = list(ParameterSampler(
            param_distributions, n_iter=n_iter, random_state=42
        ))

    for col in health_columns:
        # Override monotone constraints per column (unconstrained for entangled targets)
        col_params = dict(fixed_params)
        if model_type == 'xgboost':
            col_params['monotone_constraints'] = _build_monotone_constraints(feature_names, health_col=col)
        constrained = col not in _UNCONSTRAINED_COLUMNS and model_type == 'xgboost'
        print(f"Training {model_type} regressor for {col} (monotone={'yes' if constrained else 'no'})...")
        logger.info(f"Training {model_type} regressor for {col} (monotone={'yes' if constrained else 'no'})...")
        y_train_raw = health_train[col].values
        y_val_raw = health_val[col].values

        X_train_col = X_train
        X_val_col = X_val
        groups_col = groups

        # Target transformation to spread clustered-near-1.0 distribution
        if use_target_transform:
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(y_train_raw)))
            y_train_col = qt.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
            y_val_col = qt.transform(y_val_raw.reshape(-1, 1)).ravel()
            target_transformers[col] = qt
        else:
            y_train_col = y_train_raw
            y_val_col = y_val_raw

        best_params = {}

        # Use override params if provided, skip CV
        if best_params_override and col in best_params_override:
            best_params = best_params_override[col]
            print(f"  {col} using override params (skipping CV): {best_params}")
        elif skip_tuning:
            # Direct training with fixed params (no CV search)
            print(f"  {col} skipping CV tuning (fixed params)")
        else:
            col_cv_splitter = GroupKFold(n_splits=n_cv_folds)
            best_score = float('inf')

            for i, trial_params in enumerate(param_list):
                fold_scores = []

                for fold_idx, (train_idx, val_idx) in enumerate(
                    col_cv_splitter.split(X_train_col, y_train_col, groups_col)
                ):
                    X_fold_train = X_train_col.iloc[train_idx]
                    y_fold_train = y_train_col[train_idx]
                    X_fold_val = X_train_col.iloc[val_idx]
                    y_fold_val = y_train_col[val_idx]

                    model = model_class(**col_params, **trial_params)
                    _fit_model(model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, model_type)
                    fold_scores.append(_get_val_score(model, X_fold_val, y_fold_val, model_type))

                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)

                print(f"  {col} Trial {i+1}/{n_iter}: mae={mean_score:.4f} +/- {std_score:.4f}")
                logger.info(f"  {col} Trial {i+1}/{n_iter}: mae={mean_score:.4f} +/- {std_score:.4f}")

                if log_to_mlflow:
                    with mlflow.start_run(
                        run_name=f"trial_{col}_{i+1}",
                        nested=True,
                        tags={"mlflow.source.name": "src/ml/train_health_estimators.py", "mlflow.source.type": "LOCAL"}
                    ):
                        mlflow.log_params(trial_params)
                        mlflow.set_tag("health_column", col)
                        mlflow.set_tag("model_type", model_type)
                        mlflow.log_metric('mean_cv_mae', mean_score)
                        mlflow.log_metric('std_cv_mae', std_score)
                        for fold_idx, score in enumerate(fold_scores):
                            mlflow.log_metric(f'fold_{fold_idx}_mae', score)

                if mean_score < best_score:
                    best_score = mean_score
                    best_params = trial_params

            logger.info(f"  {col} best params (mae={best_score:.4f}): {best_params}")

        # Train final model
        final_model = model_class(**col_params, **best_params)
        _fit_model(final_model, X_train_col, y_train_col, X_val_col, y_val_col, model_type)

        val_mae = _get_val_score(final_model, X_val_col, y_val_col, model_type)
        print(f"  {col} final val mae: {val_mae:.4f}")
        logger.info(f"  {col} final val mae: {val_mae:.4f}")

        if log_to_mlflow:
            if best_params:
                mlflow.log_params({f'best_{k}_{col}': v for k, v in best_params.items()})

            X_train_float = X_train.astype({c: 'float64' for c in X_train.select_dtypes('integer').columns})
            signature = infer_signature(X_train_float, final_model.predict(X_train_float))

            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                name=f"{equipment_type}_{col}_model",
                signature=signature,
                input_example=X_train_float.iloc[:1]
            )
            model_ids[col] = model_info.model_id
            col_metrics = {f'final_val_mae_{col}': val_mae}
            if not skip_tuning and not (best_params_override and col in best_params_override):
                col_metrics[f'best_cv_mae_{col}'] = best_score
            deferred_metrics[col] = col_metrics

        regressors[col] = final_model

    # Log cv/val metrics AFTER all models are registered to avoid
    # later models inheriting earlier models' metrics
    if log_to_mlflow:
        for col, metrics in deferred_metrics.items():
            mlflow.log_metrics(metrics, model_id=model_ids[col])
        mlflow.end_run()

    return regressors, run_id, model_ids, target_transformers


def estimate_health_features(
    X: pd.DataFrame,
    health_regressors: Dict[str, Any],
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
