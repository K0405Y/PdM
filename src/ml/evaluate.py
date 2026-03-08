"""
Evaluation Module for Failure Mode Classification

Computes metrics, confusion matrices, feature importance,
and the FULL vs SENSOR_ONLY generalization gap.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    dataset_name: str = "test",
    log_to_mlflow: bool = False,
    run_id: str = None,
    feature_names: List[str] = None,
) -> Dict:
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
        roc_auc_macro = roc_auc_score(
            y_test, y_proba, multi_class='ovr', average='macro'
        )
    except ValueError:
        roc_auc_macro = None

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION: {dataset_name}")
    logger.info(f"{'='*60}")
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


def evaluate_all_vs_sensor_only(
    model_all: xgb.XGBClassifier,
    model_sensor: xgb.XGBClassifier,
    X_test_all: pd.DataFrame,
    X_test_sensor: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict:
    """
    Compare All features mode vs SENSOR_ONLY mode performance.

    Args:
        model_all: Model trained with health features and derived features
        model_sensor: Model trained with sensor-only features
        X_test_all: Test features including health columns and derived features
        X_test_sensor: Test features without health columns
        y_test: True labels
        label_encoder: For decoding

    Returns:
        Dict with both evaluation results and the gap metrics
    """
    results_all = evaluate_model(
        model_all, X_test_all, y_test, label_encoder, "all_mode"
    )
    results_sensor = evaluate_model(
        model_sensor, X_test_sensor, y_test, label_encoder, "sensor_only_mode"
    )

    gap = {
        'accuracy_gap': results_all['accuracy'] - results_sensor['accuracy'],
        'macro_f1_gap': results_all['macro_f1'] - results_sensor['macro_f1'],
    }


    logger.info("ALL vs SENSOR_ONLY GENERALIZATION GAP")
    logger.info(f"Accuracy gap:  {gap['accuracy_gap']:.4f} "
                f"({results_all['accuracy']:.4f} -> {results_sensor['accuracy']:.4f})")
    logger.info(f"Macro F1 gap:  {gap['macro_f1_gap']:.4f} "
                f"({results_all['macro_f1']:.4f} -> {results_sensor['macro_f1']:.4f})")

    if gap['macro_f1_gap'] > 0.15:
        logger.warning("Large generalization gap (>15%) — model may rely on all features "
                       "not available in production")

    return {
        'all': results_all,
        'sensor_only': results_sensor,
        'gap': gap,
    }


def get_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
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


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
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


def plot_feature_importance(
    fi_df: pd.DataFrame,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Generate a horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = fi_df.sort_values("importance", ascending=True)
    ax.barh(data["feature"], data["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig
