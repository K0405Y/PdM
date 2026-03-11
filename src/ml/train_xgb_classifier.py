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


def train_fleet_model(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, label_encoder: LabelEncoder,
                      equipment_type: str, mode: str = "sensor_only", params: Dict = None, log_to_mlflow: bool = True,
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
        mode: Feature mode - 'sensor_only' or 'ground_truth'
        params: XGBoost hyperparameters (optional overrides)
        log_to_mlflow: Whether to log to MLflow

    Returns:
        Tuple of (trained XGBClassifier, MLflow run_id or None)
    """
    n_classes = len(label_encoder.classes_)

    cfg = load_table_config().get('xgb_config', {})
    default_params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'eval_metric': 'mlogloss',
        'verbosity': 1,
        **cfg.get('default_params', {})
    }
    if params:
        default_params.update(params)

    # Extract n_estimators and early_stopping for fit()
    n_estimators = default_params.pop('n_estimators', 300)
    early_stopping_rounds = default_params.pop('early_stopping_rounds', 20)

    model = xgb.XGBClassifier(n_estimators=n_estimators,early_stopping_rounds=early_stopping_rounds, **default_params)

    # Undersample NORMAL class to reduce imbalance
    X_train_resampled, y_train_resampled = _undersample_majority(X_train, y_train, label_encoder)

    sample_weights = compute_sample_weights(y_train_resampled)

    logger.info(f"Training fleet model for {equipment_type} "
                f"({len(X_train_resampled)} train, {len(X_val)} val, {n_classes} classes)")

    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=False
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
                        'subsample', 'colsample_bytree', 'min_child_weight')}
        })

        # Class imbalance info
        classes, counts = np.unique(y_train, return_counts=True)
        class_dist = {label_encoder.inverse_transform([c])[0]: int(cnt)
                      for c, cnt in zip(classes, counts)}
        class_weights = {label_encoder.inverse_transform([c])[0]: round(
            float(len(y_train) / (len(classes) * cnt)), 4)
            for c, cnt in zip(classes, counts)}
        
        #smote resampled class distribution
        classes_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
        class_dist_resampled = {label_encoder.inverse_transform([c])[0]: int(cnt)
                                for c, cnt in zip(classes_resampled, counts_resampled)}
        
        mlflow.log_param('train_class_distribution', str(class_dist))
        mlflow.log_param('class_weights', str(class_weights))
        mlflow.log_param('train_class_distribution_resampled', str(class_dist_resampled))
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
            input_example=X_train_float.iloc[:1]
        )

        logger.info(
            f"Logged fleet model to MLflow run '{run_name}' (best_val_mlogloss={best_score:.4f})"
        )
        mlflow.end_run()
    return model, run_id


def fine_tune_per_equipment(fleet_model: xgb.XGBClassifier, X_equip: pd.DataFrame, y_equip: np.ndarray, equipment_id: int,
                            equipment_type: str, label_encoder: LabelEncoder, learning_rate: float = 0.01, n_estimators: int = 50,
                            log_to_mlflow: bool = True) -> xgb.XGBClassifier:
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
        verbose=False
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
            'n_samples': len(X_equip)
        })
        mlflow.sklearn.log_model(ft_model, name="finetune_model")
        logger.info(f"Logged fine-tuned model for equipment {equipment_id} (run '{run_name}')")
        mlflow.end_run()

    return ft_model


def train_fleet_model_with_tuning(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                                  label_encoder: LabelEncoder, equipment_type: str, mode: str = "sensor_only",
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
        mode: Feature mode - 'sensor_only' or 'ground_truth'
        n_cv_folds: Number of cross-validation folds
        n_iter: Number of random parameter combinations to try
        param_distributions: Search space (dict of param -> list of values)
        log_to_mlflow: Whether to log to MLflow

    Returns:
        Tuple of (best trained XGBClassifier, MLflow parent run_id or None)
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
            ):
                mlflow.log_params(trial_params)
                mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
                mlflow.set_tag("mlflow.source.type", "LOCAL")
                mlflow.log_metric('mean_cv_mlogloss', mean_score)
                mlflow.log_metric('std_cv_mlogloss', std_score)
                for fold_idx, score in enumerate(fold_scores):
                    mlflow.log_metric(f'fold_{fold_idx}_mlogloss', score)

        if mean_score < best_score:
            best_score = mean_score
            best_params = trial_params

    logger.info(f"Best params (mlogloss={best_score:.4f}): {best_params}")

    # Undersample NORMAL class, then SMOTE oversample minority classes
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
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name=f"{mode}_tuned_model",
            signature=signature,
            input_example=X_train_float.iloc[:1]
        )

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
        mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
        mlflow.set_tag("mlflow.source.type", "LOCAL")

        mlflow.end_run()

    return final_model, run_id


def train_two_stage_model(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
    label_encoder: LabelEncoder, equipment_type: str,
    mode: str = "sensor_only",
    n_cv_folds: int = 5, n_iter: int = 20,
    param_distributions: Dict = None,
    log_to_mlflow: bool = True,
    stage1_recall_target: float = 0.95,
) -> Tuple[xgb.XGBClassifier, xgb.XGBClassifier, LabelEncoder, float, Optional[str]]:
    """Train a two-stage model: Stage 1 (NORMAL vs ABNORMAL) then Stage 2 (failure mode classification).

    Stage 1: Binary classifier separating NORMAL from all failure modes.
    Stage 2: Multiclass classifier trained only on failure-labeled data.

    Args:
        X_train, y_train: Training data (encoded with original label_encoder)
        X_val, y_val: Validation data
        label_encoder: Original 5-class LabelEncoder
        equipment_type: Equipment type string
        mode: Feature mode
        n_cv_folds: CV folds for tuning
        n_iter: Tuning iterations per stage
        param_distributions: XGBoost tuning space
        log_to_mlflow: Whether to log to MLflow
        stage1_recall_target: Target recall for anomaly detection stage

    Returns:
        (stage1_model, stage2_model, stage2_label_encoder, stage1_threshold, mlflow_run_id)
    """
    cfg = load_table_config().get('xgb_config', {})
    if param_distributions is None:
        param_distributions = cfg.get('tuning_distributions')
    cfg_defaults = cfg.get('default_params', {})

    normal_label = label_encoder.transform(['NORMAL'])[0]

    # === STAGE 1: Binary (NORMAL=0, ABNORMAL=1) ===
    y_train_binary = (y_train != normal_label).astype(int)
    y_val_binary = (y_val != normal_label).astype(int)

    fixed_params_s1 = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': cfg_defaults.get('tree_method', 'hist'),
        'random_state': cfg_defaults.get('random_state', 42),
        'verbosity': 0,
        'early_stopping_rounds': cfg_defaults.get('early_stopping_rounds', 20),
    }

    run_id = None
    if log_to_mlflow:
        run_name = f"two_stage_{equipment_type}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        parent_run = mlflow.start_run(run_name=run_name)
        run_id = parent_run.info.run_id
        mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
        mlflow.set_tag("mlflow.source.type", "LOCAL")
        mlflow.log_params({
            'equipment_type': equipment_type,
            'feature_mode': mode,
            'model_scope': 'two_stage',
            'n_cv_folds': n_cv_folds,
            'n_iter': n_iter,
            'stage1_recall_target': stage1_recall_target,
        })

    # Stage 1 tuning
    logger.info("=== STAGE 1: Binary anomaly detection (NORMAL vs ABNORMAL) ===")
    cv_splitter_s1, groups_s1 = get_grouped_cv_splitter(X_train, y_train_binary, n_splits=n_cv_folds)
    sample_weights_s1 = compute_sample_weights(y_train_binary)

    best_score_s1 = float('inf')
    best_params_s1 = None
    param_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))

    for i, trial_params in enumerate(param_list):
        fold_scores = []
        for train_idx, val_idx in cv_splitter_s1.split(X_train, y_train_binary, groups_s1):
            model = xgb.XGBClassifier(**fixed_params_s1, **trial_params)
            model.fit(
                X_train.iloc[train_idx], y_train_binary[train_idx],
                eval_set=[(X_train.iloc[val_idx], y_train_binary[val_idx])],
                sample_weight=sample_weights_s1[train_idx],
                verbose=False,
            )
            fold_scores.append(model.best_score)

        mean_score = np.mean(fold_scores)
        logger.info(f"Stage1 Trial {i+1}/{n_iter}: logloss={mean_score:.4f}")

        if log_to_mlflow:
            with mlflow.start_run(run_name=f"trial_stage1_{i+1}", nested=True):
                mlflow.log_params(trial_params)
                mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
                mlflow.set_tag("mlflow.source.type", "LOCAL")
                mlflow.log_metric('mean_cv_logloss', mean_score)
                for fold_idx, score in enumerate(fold_scores):
                    mlflow.log_metric(f'fold_{fold_idx}_logloss', score)

        if mean_score < best_score_s1:
            best_score_s1 = mean_score
            best_params_s1 = trial_params

    logger.info(f"Stage1 best params (logloss={best_score_s1:.4f}): {best_params_s1}")

    # Train final Stage 1 model
    X_s1_resampled, y_s1_resampled = _undersample_majority_binary(X_train, y_train_binary)
    X_s1_resampled, y_s1_resampled = _smote_oversample(X_s1_resampled, y_s1_resampled)
    s1_weights = compute_sample_weights(y_s1_resampled)

    stage1_model = xgb.XGBClassifier(**fixed_params_s1, **best_params_s1)
    stage1_model.fit(
        X_s1_resampled, y_s1_resampled,
        eval_set=[(X_val, y_val_binary)],
        sample_weight=s1_weights,
        verbose=False,
    )

    # Optimize Stage 1 threshold for high recall
    s1_proba = stage1_model.predict_proba(X_val)[:, 1]
    stage1_threshold = _optimize_binary_threshold(s1_proba, y_val_binary, stage1_recall_target)
    s1_preds = (s1_proba >= stage1_threshold).astype(int)
    from sklearn.metrics import recall_score, precision_score
    s1_recall = recall_score(y_val_binary, s1_preds)
    s1_precision = precision_score(y_val_binary, s1_preds, zero_division=0)
    logger.info(f"Stage1 threshold={stage1_threshold:.3f}, recall={s1_recall:.4f}, precision={s1_precision:.4f}")

    if log_to_mlflow:
        mlflow.log_params({f'best_stage1_{k}': v for k, v in best_params_s1.items()})
        mlflow.log_metric('best_stage1_cv_logloss', best_score_s1)
        mlflow.log_metric('final_stage1_val_logloss', stage1_model.best_score)
        mlflow.log_metric('stage1_threshold', stage1_threshold)
        mlflow.log_metric('stage1_recall', s1_recall)
        mlflow.log_metric('stage1_precision', s1_precision)

        # Stage 1 class distributions
        s1_orig_counts = Counter(y_train_binary)
        s1_resampled_counts = Counter(y_s1_resampled)
        mlflow.log_param('stage1_train_class_distribution', str(dict(s1_orig_counts)))
        mlflow.log_param('stage1_train_class_distribution_resampled', str(dict(s1_resampled_counts)))

        # Log stage1 model
        X_float = X_train.astype({c: 'float64' for c in X_train.select_dtypes('integer').columns})
        sig1 = infer_signature(X_float, stage1_model.predict(X_float))
        mlflow.sklearn.log_model(
            sk_model=stage1_model, name="stage1_binary_model",
            signature=sig1, input_example=X_float.iloc[:1],
        )

    # === STAGE 2: Multiclass on failure-labeled data only ===
    logger.info("=== STAGE 2: Failure mode classification ===")
    failure_mask_train = y_train != normal_label
    failure_mask_val = y_val != normal_label

    X_train_s2 = X_train[failure_mask_train].reset_index(drop=True)
    y_train_s2_orig = y_train[failure_mask_train]
    X_val_s2 = X_val[failure_mask_val].reset_index(drop=True)
    y_val_s2_orig = y_val[failure_mask_val]

    # Create Stage 2 label encoder (failure classes only)
    failure_class_names = [c for c in label_encoder.classes_ if c != 'NORMAL']
    stage2_le = LabelEncoder()
    stage2_le.fit(failure_class_names)
    n_classes_s2 = len(failure_class_names)

    # Re-encode using original label_encoder inverse then stage2_le
    y_train_s2_names = label_encoder.inverse_transform(y_train_s2_orig)
    y_val_s2_names = label_encoder.inverse_transform(y_val_s2_orig)
    y_train_s2 = stage2_le.transform(y_train_s2_names)
    y_val_s2 = stage2_le.transform(y_val_s2_names)

    fixed_params_s2 = {
        'objective': 'multi:softprob',
        'num_class': n_classes_s2,
        'eval_metric': 'mlogloss',
        'tree_method': cfg_defaults.get('tree_method', 'hist'),
        'random_state': cfg_defaults.get('random_state', 42),
        'verbosity': 0,
        'early_stopping_rounds': cfg_defaults.get('early_stopping_rounds', 20),
    }

    # Stage 2 tuning
    cv_splitter_s2, groups_s2 = get_grouped_cv_splitter(X_train_s2, y_train_s2, n_splits=n_cv_folds)
    sample_weights_s2 = compute_sample_weights(y_train_s2)

    best_score_s2 = float('inf')
    best_params_s2 = None

    for i, trial_params in enumerate(param_list):
        fold_scores = []
        for train_idx, val_idx in cv_splitter_s2.split(X_train_s2, y_train_s2, groups_s2):
            model = xgb.XGBClassifier(**fixed_params_s2, **trial_params)
            model.fit(
                X_train_s2.iloc[train_idx], y_train_s2[train_idx],
                eval_set=[(X_train_s2.iloc[val_idx], y_train_s2[val_idx])],
                sample_weight=sample_weights_s2[train_idx],
                verbose=False,
            )
            fold_scores.append(model.best_score)

        mean_score = np.mean(fold_scores)
        logger.info(f"Stage2 Trial {i+1}/{n_iter}: mlogloss={mean_score:.4f}")

        if log_to_mlflow:
            with mlflow.start_run(run_name=f"trial_stage2_{i+1}", nested=True):
                mlflow.log_params(trial_params)
                mlflow.set_tag("mlflow.source.name", "src/ml/train_xgb_classifier.py")
                mlflow.set_tag("mlflow.source.type", "LOCAL")
                mlflow.log_metric('mean_cv_mlogloss', mean_score)
                for fold_idx, score in enumerate(fold_scores):
                    mlflow.log_metric(f'fold_{fold_idx}_mlogloss', score)

        if mean_score < best_score_s2:
            best_score_s2 = mean_score
            best_params_s2 = trial_params

    logger.info(f"Stage2 best params (mlogloss={best_score_s2:.4f}): {best_params_s2}")

    # Train final Stage 2 model with SMOTE
    X_s2_resampled, y_s2_resampled = _smote_oversample(X_train_s2, y_train_s2)
    s2_weights = compute_sample_weights(y_s2_resampled)

    stage2_model = xgb.XGBClassifier(**fixed_params_s2, **best_params_s2)
    stage2_model.fit(
        X_s2_resampled, y_s2_resampled,
        eval_set=[(X_val_s2, y_val_s2)],
        sample_weight=s2_weights,
        verbose=False,
    )

    if log_to_mlflow:
        mlflow.log_params({f'best_stage2_{k}': v for k, v in best_params_s2.items()})
        mlflow.log_metric('best_stage2_cv_mlogloss', best_score_s2)
        mlflow.log_metric('final_stage2_val_mlogloss', stage2_model.best_score)

        # Stage 2 class distributions
        s2_orig_counts = Counter(y_train_s2)
        s2_orig_named = {stage2_le.inverse_transform([c])[0]: int(cnt)
                         for c, cnt in s2_orig_counts.items()}
        s2_resampled_counts = Counter(y_s2_resampled)
        s2_resampled_named = {stage2_le.inverse_transform([c])[0]: int(cnt)
                              for c, cnt in s2_resampled_counts.items()}
        mlflow.log_param('stage2_train_class_distribution', str(s2_orig_named))
        mlflow.log_param('stage2_train_class_distribution_resampled', str(s2_resampled_named))

        # Log stage2 model
        sig2 = infer_signature(X_float, stage2_model.predict(X_float))
        mlflow.sklearn.log_model(
            sk_model=stage2_model, name="stage2_failure_model",
            signature=sig2, input_example=X_float.iloc[:1],
        )

        mlflow.end_run()

    return stage1_model, stage2_model, stage2_le, stage1_threshold, run_id


def predict_two_stage(
    stage1_model: xgb.XGBClassifier,
    stage2_model: xgb.XGBClassifier,
    X: pd.DataFrame,
    stage1_threshold: float,
    label_encoder: LabelEncoder,
    stage2_le: LabelEncoder,
) -> np.ndarray:
    """Run two-stage inference: anomaly detection then failure classification.

    Args:
        stage1_model: Binary NORMAL/ABNORMAL model
        stage2_model: Multiclass failure mode model
        X: Features
        stage1_threshold: Binary threshold for anomaly detection
        label_encoder: Original 5-class LabelEncoder
        stage2_le: Stage 2 failure-only LabelEncoder

    Returns:
        Encoded predictions in original label_encoder space
    """
    normal_label = label_encoder.transform(['NORMAL'])[0]
    n_samples = len(X)
    final_preds = np.full(n_samples, normal_label)

    # Stage 1: detect anomalies
    s1_proba = stage1_model.predict_proba(X)[:, 1]
    abnormal_mask = s1_proba >= stage1_threshold

    if abnormal_mask.any():
        # Stage 2: classify failure mode on abnormal samples
        X_abnormal = X[abnormal_mask]
        s2_preds = stage2_model.predict(X_abnormal)
        # Map stage2 labels back to original label space
        s2_names = stage2_le.inverse_transform(s2_preds)
        s2_original = label_encoder.transform(s2_names)
        final_preds[abnormal_mask] = s2_original

    return final_preds


def _undersample_majority_binary(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Undersample majority class in binary setting to 3x minority."""
    counts = Counter(y)
    if len(counts) < 2:
        return X, y
    majority_label = max(counts, key=counts.get)
    minority_count = min(counts.values())
    target = minority_count * 3
    if counts[majority_label] <= target:
        return X, y
    sampling_strategy = {majority_label: target}
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    logger.info(f"Binary undersample: {counts[majority_label]} -> {target}")
    return X_resampled, y_resampled


def _optimize_binary_threshold(proba: np.ndarray, y_true: np.ndarray,
                                recall_target: float = 0.95) -> float:
    """Find the lowest threshold that achieves the target recall for the positive class."""
    from sklearn.metrics import recall_score
    best_threshold = 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (proba >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        if rec >= recall_target:
            best_threshold = t  # keep the highest threshold that meets recall target
    return best_threshold