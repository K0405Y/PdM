"""
XGBoost Failure Mode Classifier
Trains fleet-level XGBoost models per equipment type.
Includes hybrid fine-tuning for individual equipment.
Logs experiments to MLflow with PostgreSQL backend.
"""

import os
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder
from src.ml.feature_prep import compute_sample_weights
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


def setup_mlflow(experiment_name: str = "pdm_failure_classification"):
    """
    Configure MLflow client to connect to tracking server.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root,
        )
        logger.info(f"Created MLflow experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment: {experiment_name} (id={experiment_id})")

    return {
        "tracking_uri": tracking_uri,
        "experiment_id": experiment_id,
    }


def train_fleet_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
    equipment_type: str,
    mode: str = "sensor_only",
    params: Dict = None,
    log_to_mlflow: bool = True,
) -> Tuple[xgb.XGBClassifier, Optional[str]]:
    """
    Train a fleet-level XGBoost classifier for one equipment type.

    Args:
        X_train: Training features
        y_train: Training labels (encoded)
        X_val: Validation features
        y_val: Validation labels (encoded)
        label_encoder: Fitted LabelEncoder
        equipment_type: 'turbine', 'compressor', or 'pump'
        mode: Feature mode - 'sensor_only' or 'all'
        params: XGBoost hyperparameters (optional overrides)
        log_to_mlflow: Whether to log to MLflow

    Returns:
        Tuple of (trained XGBClassifier, MLflow run_id or None)
    """
    n_classes = len(label_encoder.classes_)

    default_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 1,
    }
    if params:
        default_params.update(params)

    # Extract n_estimators and early_stopping for fit()
    n_estimators = default_params.pop('n_estimators', 300)
    early_stopping_rounds = default_params.pop('early_stopping_rounds', 20)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **default_params,
    )

    sample_weights = compute_sample_weights(y_train)

    logger.info(f"Training fleet model for {equipment_type} "
                f"({len(X_train)} train, {len(X_val)} val, {n_classes} classes)")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=False,
    )

    run_id = None
    if log_to_mlflow:
        run_name = f"fleet_{equipment_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id

        mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
        mlflow.set_tag("mlflow.source.type", "LOCAL")

        mlflow.log_params({
            'equipment_type': equipment_type,
            'model_scope': 'fleet',
            'feature_mode': mode,
            'n_classes': n_classes,
            'classes': str(list(label_encoder.classes_)),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_features': X_train.shape[1],
            'feature_names': str(list(X_train.columns)),
            **{k: v for k, v in model.get_params().items()
            if k in ('max_depth', 'learning_rate', 'n_estimators',
                        'subsample', 'colsample_bytree', 'min_child_weight')},
        })

        # Class imbalance info
        classes, counts = np.unique(y_train, return_counts=True)
        class_dist = {label_encoder.inverse_transform([c])[0]: int(cnt)
                      for c, cnt in zip(classes, counts)}
        class_weights = {label_encoder.inverse_transform([c])[0]: round(
            float(len(y_train) / (len(classes) * cnt)), 4)
            for c, cnt in zip(classes, counts)}
        mlflow.log_param('train_class_distribution', str(class_dist))
        mlflow.log_param('class_weights', str(class_weights))
        mlflow.log_metric('imbalance_ratio', float(max(counts) / min(counts)))

        best_score = model.best_score
        mlflow.log_metric('best_val_mlogloss', best_score)
        mlflow.log_metric('best_iteration', model.best_iteration)

        X_train_float = X_train.astype({c: 'float64' for c in X_train.select_dtypes('integer').columns})
        signature = infer_signature(X_train_float, model.predict(X_train_float))

        mlflow.sklearn.log_model(
            sk_model=model,
            name=f"{mode}_model",
            signature=signature,
            input_example=X_train_float.iloc[:1],
        )

        logger.info(
            f"Logged fleet model to MLflow run '{run_name}' (best_val_mlogloss={best_score:.4f})"
        )
        mlflow.end_run()
    return model, run_id


def fine_tune_per_equipment(
    fleet_model: xgb.XGBClassifier,
    X_equip: pd.DataFrame,
    y_equip: np.ndarray,
    equipment_id: int,
    equipment_type: str,
    label_encoder: LabelEncoder,
    learning_rate: float = 0.01,
    n_estimators: int = 50,
    log_to_mlflow: bool = True,
) -> xgb.XGBClassifier:
    """
    Fine-tune a fleet model on a single equipment's data (hybrid approach).

    Takes the fleet-trained XGBoost model and continues boosting with
    a lower learning rate on one equipment's data.

    Args:
        fleet_model: Pre-trained fleet XGBClassifier
        X_equip: Features for this equipment
        y_equip: Labels for this equipment (encoded)
        equipment_id: Equipment ID being fine-tuned
        equipment_type: 'turbine', 'compressor', or 'pump'
        label_encoder: Fitted LabelEncoder
        learning_rate: Lower LR for fine-tuning
        n_estimators: Additional boosting rounds

    Returns:
        Fine-tuned XGBClassifier
    """
    if len(X_equip) < 10:
        logger.warning(f"Equipment {equipment_id}: only {len(X_equip)} samples, skipping fine-tune")
        return fleet_model

    # Create a new model with same params but lower LR
    params = fleet_model.get_params()
    params['learning_rate'] = learning_rate
    params['n_estimators'] = n_estimators

    ft_model = xgb.XGBClassifier(**params)

    sample_weights = compute_sample_weights(y_equip) if len(np.unique(y_equip)) > 1 else None

    logger.info(f"Fine-tuning for equipment {equipment_id} "
                f"({len(X_equip)} samples, lr={learning_rate}, rounds={n_estimators})")

    ft_model.fit(
        X_equip, y_equip,
        sample_weight=sample_weights,
        xgb_model=fleet_model.get_booster(),
        verbose=False,
    )

    if log_to_mlflow:
        run_name = f"finetune_{equipment_type}_eq{equipment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)

        mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
        mlflow.set_tag("mlflow.source.type", "LOCAL")

        mlflow.log_params({
            'equipment_type': equipment_type,
            'model_scope': 'fine_tuned',
            'equipment_id': equipment_id,
            'fine_tune_lr': learning_rate,
            'fine_tune_rounds': n_estimators,
            'n_samples': len(X_equip),
        })
        mlflow.sklearn.log_model(ft_model, name="finetune_model")
        logger.info(f"Logged fine-tuned model for equipment {equipment_id} (run '{run_name}')")
        mlflow.end_run()

    return ft_model


def train_fleet_model_with_tuning(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
    equipment_type: str,
    mode: str = "sensor_only",
    n_cv_folds: int = 5,
    n_iter: int = 20,
    param_distributions: Dict = None,
    log_to_mlflow: bool = True,
) -> Tuple[xgb.XGBClassifier, Optional[str]]:
    """
    Train a fleet-level XGBoost classifier with hyperparameter tuning via
    randomized search and grouped cross-validation. Each trial is logged
    as a nested MLflow run.

    Args:
        X_train: Training features
        y_train: Training labels (encoded)
        X_val: Validation features
        y_val: Validation labels (encoded)
        label_encoder: Fitted LabelEncoder
        equipment_type: 'turbine', 'compressor', or 'pump'
        mode: Feature mode - 'sensor_only' or 'all'
        n_cv_folds: Number of cross-validation folds
        n_iter: Number of random parameter combinations to try
        param_distributions: Search space (dict of param -> list of values)
        log_to_mlflow: Whether to log to MLflow

    Returns:
        Tuple of (best trained XGBClassifier, MLflow parent run_id or None)
    """
    from sklearn.model_selection import ParameterSampler
    from src.ml.feature_prep import get_grouped_cv_splitter

    n_classes = len(label_encoder.classes_)

    if param_distributions is None:
        param_distributions = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 500],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5, 10],
        }

    fixed_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
        'early_stopping_rounds': 20,
    }

    cv_splitter, groups = get_grouped_cv_splitter(X_train, y_train, n_splits=n_cv_folds)
    sample_weights = compute_sample_weights(y_train)

    run_id = None
    best_score = float('inf')
    best_params = None

    if log_to_mlflow:
        run_name = f"tuning_{equipment_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        parent_run = mlflow.start_run(run_name=run_name)
        run_id = parent_run.info.run_id
        mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
        mlflow.set_tag("mlflow.source.type", "LOCAL")
        mlflow.log_params({
            'equipment_type': equipment_type,
            'feature_mode': mode,
            'model_scope': 'tuning',
            'n_cv_folds': n_cv_folds,
            'n_iter': n_iter,
        })

    param_list = list(ParameterSampler(
        param_distributions, n_iter=n_iter, random_state=42
    ))

    logger.info(f"Starting hyperparameter tuning: {n_iter} trials x {n_cv_folds} folds")

    for i, trial_params in enumerate(param_list):
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv_splitter.split(X_train, y_train, groups)
        ):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]
            fold_weights = sample_weights[train_idx]

            model = xgb.XGBClassifier(**fixed_params, **trial_params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                sample_weight=fold_weights,
                verbose=False,
            )
            fold_scores.append(model.best_score)

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        logger.info(f"Trial {i+1}/{n_iter}: mlogloss={mean_score:.4f} +/- {std_score:.4f} | {trial_params}")

        if log_to_mlflow:
            with mlflow.start_run(
                run_name=f"trial_{i+1}",
                nested=True,
            ):
                mlflow.log_params(trial_params)
                mlflow.log_metric('mean_cv_mlogloss', mean_score)
                mlflow.log_metric('std_cv_mlogloss', std_score)
                for fold_idx, score in enumerate(fold_scores):
                    mlflow.log_metric(f'fold_{fold_idx}_mlogloss', score)

        if mean_score < best_score:
            best_score = mean_score
            best_params = trial_params

    logger.info(f"Best params (mlogloss={best_score:.4f}): {best_params}")

    # Train final model with best params on full training data
    final_model = xgb.XGBClassifier(**fixed_params, **best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=False,
    )

    if log_to_mlflow:
        mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
        mlflow.log_metric('best_cv_mlogloss', best_score)
        mlflow.log_metric('final_val_mlogloss', final_model.best_score)
        mlflow.log_metric('final_best_iteration', final_model.best_iteration)

        X_train_float = X_train.astype({c: 'float64' for c in X_train.select_dtypes('integer').columns})
        signature = infer_signature(X_train_float, final_model.predict(X_train_float))
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name=f"{mode}_tuned_model",
            signature=signature,
            input_example=X_train_float.iloc[:1],
        )

        # Class imbalance info
        classes, counts = np.unique(y_train, return_counts=True)
        class_dist = {label_encoder.inverse_transform([c])[0]: int(cnt)
                      for c, cnt in zip(classes, counts)}
        mlflow.log_param('train_class_distribution', str(class_dist))
        mlflow.log_metric('imbalance_ratio', float(max(counts) / min(counts)))

        mlflow.end_run()

    return final_model, run_id