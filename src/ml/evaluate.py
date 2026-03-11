"""
Evaluation Module for Failure Mode Classification

Computes metrics, confusion matrices, feature importance,
and the GROUND_TRUTH vs SENSOR_ONLY generalization gap.
"""

import os
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import nullcontext
import tempfile
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: np.ndarray, label_encoder: LabelEncoder,
    dataset_name: str = "test", log_to_mlflow: bool = False, run_id: str = None, feature_names: List[str] = None) -> Dict:
    """
    Evaluate model and compute classification metrics.

    Args:
        model: Trained XGBClassifier
        X_test: Test features
        y_test: True labels (encoded)
        label_encoder: For decoding class names
        dataset_name: Name for logging (e.g., 'test', 'sensor_only')
        log_to_mlflow: Whether to log metrics
        run_id: MLflow run ID to log into (uses active run if None)
        feature_names: Feature column names for importance logging

    Returns:
        Dict with metrics: accuracy, macro_f1, weighted_f1, roc_auc_macro,
        per_class metrics, confusion_matrix, report
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    class_names = list(label_encoder.classes_)

    try:
        roc_auc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except ValueError:
        roc_auc_macro = None

    all_labels = list(range(len(class_names)))
    report = classification_report(
        y_test, y_pred,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)


    logger.info(f"EVALUATION: {dataset_name}")
    logger.info(f"Accuracy:     {acc:.4f}")
    logger.info(f"Macro F1:     {macro_f1:.4f}")
    logger.info(f"Weighted F1:  {weighted_f1:.4f}")
    if roc_auc_macro is not None:
        logger.info(f"ROC AUC:      {roc_auc_macro:.4f}")
    logger.info(f"\n{report_str}")
    logger.info(f"Confusion Matrix:\n{cm}")

    results = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'roc_auc_macro': roc_auc_macro,
        'report': report,
        'report_str': report_str,
        'confusion_matrix': cm,
        'class_names': class_names,
    }

    if log_to_mlflow:
        ctx = mlflow.start_run(run_id=run_id) if run_id else nullcontext()
        with ctx:
            mlflow.log_metric(f'{dataset_name}_accuracy', acc)
            mlflow.log_metric(f'{dataset_name}_macro_f1', macro_f1)
            mlflow.log_metric(f'{dataset_name}_weighted_f1', weighted_f1)
            if roc_auc_macro is not None:
                mlflow.log_metric(f'{dataset_name}_roc_auc_macro', roc_auc_macro)

            for cls_name in class_names:
                if cls_name in report:
                    mlflow.log_metric(
                        f'{dataset_name}_precision_{cls_name}',
                        report[cls_name]['precision']
                    )
                    mlflow.log_metric(
                        f'{dataset_name}_recall_{cls_name}',
                        report[cls_name]['recall']
                    )
                    mlflow.log_metric(
                        f'{dataset_name}_f1_{cls_name}',
                        report[cls_name]['f1-score']
                    )

            # Log artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Confusion matrix
                cm_fig = plot_confusion_matrix(
                    cm, class_names,
                    title=f"{dataset_name} Confusion Matrix"
                )
                cm_path = os.path.join(tmpdir, f"{dataset_name}_confusion_matrix.png")
                cm_fig.savefig(cm_path, dpi=150)
                plt.close(cm_fig)
                mlflow.log_artifact(cm_path)

                # Feature importance
                if feature_names is not None:
                    fi_df = get_feature_importance(model, feature_names)
                    fi_fig = plot_feature_importance(
                        fi_df,
                        title=f"{dataset_name} Feature Importance"
                    )
                    fi_path = os.path.join(tmpdir, f"{dataset_name}_feature_importance.png")
                    fi_fig.savefig(fi_path, dpi=150)
                    plt.close(fi_fig)
                    mlflow.log_artifact(fi_path)

                # Classification report text
                report_path = os.path.join(tmpdir, f"{dataset_name}_classification_report.txt")
                with open(report_path, 'w') as f:
                    f.write(report_str)
                mlflow.log_artifact(report_path)

    return results


def evaluate_ground_truth_vs_sensor_only(model_ground_truth: xgb.XGBClassifier, model_sensor: xgb.XGBClassifier,
                                         X_test_ground_truth: pd.DataFrame, X_test_sensor: pd.DataFrame, y_test: np.ndarray,
                                         label_encoder: LabelEncoder) -> Dict:
    """
    Compare Ground Truth features mode vs SENSOR_ONLY mode performance.

    Args:
        model_ground_truth: Model trained with ground truth features        
        model_sensor: Model trained with sensor-only features
        X_test_ground_truth: Test features including ground truth columns
        X_test_sensor: Test features without ground truth columns
        y_test: True labels
        label_encoder: For decoding

    Returns:
        Dict with both evaluation results and the gap metrics
    """
    results_ground_truth = evaluate_model(
        model_ground_truth, X_test_ground_truth, y_test, label_encoder, "ground_truth_mode"
    )
    results_sensor = evaluate_model(
        model_sensor, X_test_sensor, y_test, label_encoder, "sensor_only_mode"
    )

    gap = {
        'accuracy_gap': results_ground_truth['accuracy'] - results_sensor['accuracy'],
        'macro_f1_gap': results_ground_truth['macro_f1'] - results_sensor['macro_f1'],
    }


    logger.info("GROUND TRUTH vs SENSOR_ONLY GENERALIZATION GAP")
    logger.info(f"Accuracy gap:  {gap['accuracy_gap']:.4f} "
                f"({results_ground_truth['accuracy']:.4f} -> {results_sensor['accuracy']:.4f})")
    logger.info(f"Macro F1 gap:  {gap['macro_f1_gap']:.4f} "
                f"({results_ground_truth['macro_f1']:.4f} -> {results_sensor['macro_f1']:.4f})")

    if gap['macro_f1_gap'] > 0.15:
        logger.warning("Large generalization gap (>15%) — model may rely on all features "
                       "not available in production")

    return {
        'ground_truth': results_ground_truth,
        'sensor_only': results_sensor,
        'gap': gap,
    }


def optimize_thresholds(model: xgb.XGBClassifier, X_val: pd.DataFrame,
                        y_val: np.ndarray, label_encoder: LabelEncoder,
                        metric: str = 'macro_f1') -> Dict[str, float]:
    """Find per-class probability thresholds that maximize macro F1 on validation set.

    For each class, searches thresholds from 0.05 to 0.90 in steps of 0.05.
    Uses a margin-based prediction: predict the class with the highest
    (probability - threshold) among those exceeding their threshold.

    Args:
        model: Trained XGBClassifier
        X_val: Validation features
        y_val: Validation labels (encoded)
        label_encoder: For decoding class names
        metric: Optimization target ('macro_f1')

    Returns:
        Dict mapping class name to optimal threshold
    """
    y_proba = model.predict_proba(X_val)
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    # Start with equal thresholds
    best_thresholds = np.full(n_classes, 0.5)
    best_score = -1.0

    # Grid search: optimize one class at a time, repeat until stable
    for _ in range(3):  # 3 passes for convergence
        for cls_idx in range(n_classes):
            best_t = best_thresholds[cls_idx]
            for t in np.arange(0.05, 0.91, 0.05):
                trial = best_thresholds.copy()
                trial[cls_idx] = t
                y_pred = _predict_with_threshold_array(y_proba, trial)
                score = f1_score(y_val, y_pred, average='macro', zero_division=0)
                if score > best_score:
                    best_score = score
                    best_t = t
            best_thresholds[cls_idx] = best_t

    thresholds = {name: float(best_thresholds[i]) for i, name in enumerate(class_names)}
    logger.info(f"Optimized thresholds (macro_f1={best_score:.4f}): {thresholds}")
    return thresholds


def predict_with_thresholds(model: xgb.XGBClassifier, X: pd.DataFrame,
                            thresholds: Dict[str, float],
                            label_encoder: LabelEncoder) -> np.ndarray:
    """Predict using per-class thresholds instead of argmax.

    For each sample, selects the class with the highest margin
    (probability - threshold) among classes exceeding their threshold.
    Falls back to argmax if no class exceeds its threshold.

    Args:
        model: Trained XGBClassifier
        X: Features
        thresholds: Dict mapping class name to threshold
        label_encoder: For class ordering

    Returns:
        Encoded predictions array
    """
    y_proba = model.predict_proba(X)
    threshold_array = np.array([thresholds[name] for name in label_encoder.classes_])
    return _predict_with_threshold_array(y_proba, threshold_array)


def _predict_with_threshold_array(y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Internal: predict using threshold array."""
    margins = y_proba - thresholds
    # Where any class exceeds threshold, pick highest margin
    any_above = np.any(margins > 0, axis=1)
    preds = np.argmax(y_proba, axis=1)  # fallback: argmax
    preds[any_above] = np.argmax(margins[any_above], axis=1)
    return preds


def get_feature_importance(model: xgb.XGBClassifier, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Get feature importance from XGBoost model.

    Args:
        model: Trained XGBClassifier
        feature_names: Feature column names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature names and importance scores, sorted descending
    """
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop {top_n} Features:")
    for _, row in fi_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:.4f}")

    return fi_df.head(top_n)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str = "Confusion Matrix") -> plt.Figure:
    """Generate a confusion matrix heatmap figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def evaluate_two_stage_model(
    stage1_model: xgb.XGBClassifier,
    stage2_model: xgb.XGBClassifier,
    X_test: pd.DataFrame, y_test: np.ndarray,
    label_encoder: LabelEncoder,
    stage2_le: LabelEncoder,
    stage1_threshold: float,
    dataset_name: str = "two_stage_test",
    log_to_mlflow: bool = False,
    run_id: str = None,
    feature_names: List[str] = None,
) -> Dict:
    """Evaluate two-stage model (anomaly detection + failure classification).

    Runs the two-stage inference pipeline, then computes the same metrics
    as evaluate_model() in the original 5-class label space.

    Args:
        stage1_model: Binary NORMAL/ABNORMAL model
        stage2_model: Multiclass failure mode model
        X_test: Test features
        y_test: True labels (encoded with original label_encoder)
        label_encoder: Original 5-class LabelEncoder
        stage2_le: Stage 2 failure-only LabelEncoder
        stage1_threshold: Binary threshold for anomaly detection
        dataset_name: Name for logging
        log_to_mlflow: Whether to log metrics
        run_id: MLflow run ID
        feature_names: Feature column names for importance logging

    Returns:
        Dict with metrics (same format as evaluate_model)
    """
    from src.ml.train_xgb_classifier import predict_two_stage

    y_pred = predict_two_stage(
        stage1_model, stage2_model, X_test,
        stage1_threshold, label_encoder, stage2_le,
    )

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    class_names = list(label_encoder.classes_)

    # ROC AUC requires probabilities — not straightforward for two-stage,
    # so we skip it here
    roc_auc_macro = None

    all_labels = list(range(len(class_names)))
    report = classification_report(
        y_test, y_pred, labels=all_labels, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred, labels=all_labels, target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"EVALUATION: {dataset_name}")
    logger.info(f"Accuracy:     {acc:.4f}")
    logger.info(f"Macro F1:     {macro_f1:.4f}")
    logger.info(f"Weighted F1:  {weighted_f1:.4f}")
    logger.info(f"\n{report_str}")
    logger.info(f"Confusion Matrix:\n{cm}")

    results = {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'roc_auc_macro': roc_auc_macro,
        'report': report,
        'report_str': report_str,
        'confusion_matrix': cm,
        'class_names': class_names,
    }

    if log_to_mlflow:
        ctx = mlflow.start_run(run_id=run_id) if run_id else nullcontext()
        with ctx:
            mlflow.log_metric(f'{dataset_name}_accuracy', acc)
            mlflow.log_metric(f'{dataset_name}_macro_f1', macro_f1)
            mlflow.log_metric(f'{dataset_name}_weighted_f1', weighted_f1)

            for cls_name in class_names:
                if cls_name in report:
                    mlflow.log_metric(f'{dataset_name}_precision_{cls_name}', report[cls_name]['precision'])
                    mlflow.log_metric(f'{dataset_name}_recall_{cls_name}', report[cls_name]['recall'])
                    mlflow.log_metric(f'{dataset_name}_f1_{cls_name}', report[cls_name]['f1-score'])

            with tempfile.TemporaryDirectory() as tmpdir:
                # Confusion matrix
                cm_fig = plot_confusion_matrix(cm, class_names, title=f"{dataset_name} Confusion Matrix")
                cm_path = os.path.join(tmpdir, f"{dataset_name}_confusion_matrix.png")
                cm_fig.savefig(cm_path, dpi=150)
                plt.close(cm_fig)
                mlflow.log_artifact(cm_path)

                # Stage 1 confusion matrix (binary)
                normal_label = label_encoder.transform(['NORMAL'])[0]
                y_test_binary = (y_test != normal_label).astype(int)
                y_pred_binary = (y_pred != normal_label).astype(int)
                cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
                cm_bin_fig = plot_confusion_matrix(cm_binary, ['NORMAL', 'ABNORMAL'],
                                                   title=f"{dataset_name} Stage1 Confusion Matrix")
                cm_bin_path = os.path.join(tmpdir, f"{dataset_name}_stage1_confusion_matrix.png")
                cm_bin_fig.savefig(cm_bin_path, dpi=150)
                plt.close(cm_bin_fig)
                mlflow.log_artifact(cm_bin_path)

                # Feature importance for both stages
                if feature_names is not None:
                    fi_s1 = get_feature_importance(stage1_model, feature_names)
                    fi_s1_fig = plot_feature_importance(fi_s1, title=f"{dataset_name} Stage1 Feature Importance")
                    fi_s1_path = os.path.join(tmpdir, f"{dataset_name}_feature_importance_stage1.png")
                    fi_s1_fig.savefig(fi_s1_path, dpi=150)
                    plt.close(fi_s1_fig)
                    mlflow.log_artifact(fi_s1_path)

                    fi_s2 = get_feature_importance(stage2_model, feature_names)
                    fi_s2_fig = plot_feature_importance(fi_s2, title=f"{dataset_name} Stage2 Feature Importance")
                    fi_s2_path = os.path.join(tmpdir, f"{dataset_name}_feature_importance_stage2.png")
                    fi_s2_fig.savefig(fi_s2_path, dpi=150)
                    plt.close(fi_s2_fig)
                    mlflow.log_artifact(fi_s2_path)

                # Classification report text
                report_path = os.path.join(tmpdir, f"{dataset_name}_classification_report.txt")
                with open(report_path, 'w') as f:
                    f.write(report_str)
                mlflow.log_artifact(report_path)

    return results


def plot_feature_importance(fi_df: pd.DataFrame, title: str = "Feature Importance") -> plt.Figure:
    """Generate a horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = fi_df.sort_values("importance", ascending=True)
    ax.barh(data["feature"], data["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig
