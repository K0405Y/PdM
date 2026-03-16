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
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterSampler
from src.ml.feature_prep import get_grouped_cv_splitter
from src.ml.feature_prep import compute_sample_weights
from src.ml.data_loader import load_table_config
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

def _undersample_majority(X: pd.DataFrame, y: np.ndarray, le: LabelEncoder) -> Tuple[pd.DataFrame, np.ndarray]:
    """Undersample NORMAL class to 3x the largest failure class count."""
    classes, counts = np.unique(y, return_counts=True)
    normal_label = le.transform(['NORMAL'])[0]
    failure_counts = [c for cls, c in zip(classes, counts) if cls != normal_label]
    if not failure_counts:
        return X, y
    max_failure = max(failure_counts)
    normal_count = counts[classes == normal_label]
    if len(normal_count) == 0 or normal_count[0] <= max_failure * 3:
        return X, y
    sampling_strategy = {normal_label: max_failure * 3}
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    logger.info(f"Undersampled NORMAL: {normal_count[0]} -> {max_failure * 3} "
                f"(3x largest failure class: {max_failure})")
    return X_resampled, y_resampled

def _smote_oversample(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply SMOTE to oversample minority classes.

    Adapts k_neighbors to the smallest class count to avoid errors.
    """
    class_counts = Counter(y)
    min_count = min(class_counts.values())
    if min_count < 2:
        logger.warning(f"Smallest class has {min_count} sample(s), skipping SMOTE")
        return X, y
    k = min(5, min_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    new_counts = Counter(y_resampled)
    logger.info(f"SMOTE applied (k={k}): {dict(class_counts)} -> {dict(new_counts)}")
    return X_resampled, y_resampled

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

def train_fleet_model_with_tuning(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                                  label_encoder: LabelEncoder, equipment_type: str, mode: str = "classifier",
                                  n_cv_folds: int = 5, n_iter: int = 20, param_distributions: Dict = None, log_to_mlflow: bool = True
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
        mode: Feature mode - 'ground_truth'
        n_cv_folds: Number of cross-validation folds
        n_iter: Number of random parameter combinations to try
        param_distributions: Search space (dict of param -> list of values)
        log_to_mlflow: Whether to log to MLflow

    Returns:
        Tuple of (best trained XGBClassifier, MLflow parent run_id or None, model_id or None)
    """
    n_classes = len(label_encoder.classes_)

    cfg = load_table_config().get('xgb_config', {})

    if param_distributions is None:
        param_distributions = cfg.get('tuning_distributions')

    cfg_defaults = cfg.get('default_params', {})
    fixed_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'tree_method': cfg_defaults.get('tree_method', 'hist'),
        'random_state': cfg_defaults.get('random_state', 42),
        'verbosity': 0,
        'early_stopping_rounds': cfg_defaults.get('early_stopping_rounds', 20)
    }

    cv_splitter, groups = get_grouped_cv_splitter(X_train, y_train, n_splits=n_cv_folds)
    sample_weights = compute_sample_weights(y_train)

    run_id = None
    model_id = None
    best_score = float('inf')
    best_params = None

    if log_to_mlflow:
        run_name = f"tuning_{equipment_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        parent_run = mlflow.start_run(run_name=run_name, tags={
            "mlflow.source.name": "src/ml/train_xgb_classifier.py",
            "mlflow.source.type": "LOCAL",
        })
        run_id = parent_run.info.run_id
        mlflow.log_params({
            'equipment_type': equipment_type,
            'model_scope': 'tuning',
            'n_cv_folds': n_cv_folds,
            'n_iter': n_iter
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
                tags={"mlflow.source.name": "src/ml/train_xgb_classifier.py", "mlflow.source.type": "LOCAL"},
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

    # Undersample NORMAL class, then oversample minority classes
    X_train_resampled, y_train_resampled = _undersample_majority(X_train, y_train, label_encoder)
    X_train_resampled, y_train_resampled = _smote_oversample(X_train_resampled, y_train_resampled)

    final_weights = compute_sample_weights(y_train_resampled)
    final_model = xgb.XGBClassifier(**fixed_params, **best_params)
    final_model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val, y_val)],
        sample_weight=final_weights,
        verbose=False
    )

    if log_to_mlflow:
        mlflow.log_params({f'best_{k}': v for k, v in best_params.items()})
        mlflow.log_metric('best_cv_mlogloss', best_score)
        mlflow.log_metric('final_val_mlogloss', final_model.best_score)
        mlflow.log_metric('final_best_iteration', final_model.best_iteration)
        mlflow.log_metric('final_best_accuracy', final_model.score(X_val, y_val))

        X_train_float = X_train.astype({c: 'float64' for c in X_train.select_dtypes('integer').columns})
        signature = infer_signature(X_train_float, final_model.predict(X_train_float))
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name=f"health_failure_type_{mode}_model",
            signature=signature,
            input_example=X_train_float.iloc[:1]
        )
        model_id = model_info.model_id

        # Class imbalance info
        classes, counts = np.unique(y_train, return_counts=True)
        class_dist = {label_encoder.inverse_transform([c])[0]: int(cnt)
                      for c, cnt in zip(classes, counts)}
        
        #smote resampled class distribution
        classes_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
        class_dist_resampled = {label_encoder.inverse_transform([c])[0]: int(cnt)
                                for c, cnt in zip(classes_resampled, counts_resampled)}

        mlflow.log_param('train_class_distribution', str(class_dist))
        mlflow.log_metric('imbalance_ratio', float(max(counts) / min(counts)))
        mlflow.log_param('train_class_distribution_resampled', str(class_dist_resampled))
        mlflow.end_run()

    return final_model, run_id, model_id
