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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def evaluate_classifier_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: np.ndarray, label_encoder: LabelEncoder,
    dataset_name: str = "test", log_to_mlflow: bool = False, run_id: str = None, feature_names: List[str] = None,
    log_test_data: bool = False, model_id: str = None) -> Dict:
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

                # Test data
                if log_test_data:
                    x_path = os.path.join(tmpdir, f"{dataset_name}_X.parquet")
                    y_path = os.path.join(tmpdir, f"{dataset_name}_y.parquet")
                    X_test.to_parquet(x_path)
                    pd.Series(y_test, name="label").to_frame().to_parquet(y_path)
                    mlflow.log_artifact(x_path)
                    mlflow.log_artifact(y_path)

            # Log metrics to the model (visible on model page)
            if model_id is not None:
                model_metrics = {
                    f'{dataset_name}_accuracy': acc,
                    f'{dataset_name}_macro_f1': macro_f1,
                    f'{dataset_name}_weighted_f1': weighted_f1,
                }
                if roc_auc_macro is not None:
                    model_metrics[f'{dataset_name}_roc_auc_macro'] = roc_auc_macro
                mlflow.log_metrics(metrics=model_metrics, model_id=model_id)

    return results

def evaluate_health_regressors(health_regressors: Dict, X_test: pd.DataFrame, health_test: pd.DataFrame,
                                log_to_mlflow: bool = False, dataset_name: str = "test",run_id: str = None,
                                model_ids: Dict[str, str] = None, feature_names: List[str] = None,
                                log_test_data: bool = False, target_transformers: Dict = None,
                               ) -> Dict[str, Dict[str, float]]:
    """
    Evaluate health indicator regressors on test data.

    Args:
        health_regressors: Dict mapping health column name to trained XGBRegressor
        X_test: Test sensor features
        health_test: Ground truth health columns for test set
        log_to_mlflow: Whether to log metrics and artifacts to MLflow
        run_id: MLflow run ID to log into (uses active run if None)
        model_ids: Dict mapping health column name to MLflow model IDs for logging metrics to models
        feature_names: Feature column names for importance logging
        log_test_data: Whether to log test data as artifacts
        target_transformers: Dict mapping health column name to fitted QuantileTransformer
    Returns:
        Dict mapping health column name to {'r2', 'mae', 'rmse'}
    """

    results = {}

    for col, regressor in health_regressors.items():
        y_true = health_test[col].values
        y_pred = regressor.predict(X_test)
        if target_transformers and col in target_transformers:
            y_pred = target_transformers[col].inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_pred = np.clip(y_pred, 0.0, 1.0)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        results[col] = {'r2': r2, 'mae': mae, 'rmse': rmse}
        logger.info(f"Health regressor {col}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    if log_to_mlflow:
        ctx = mlflow.start_run(run_id=run_id) if run_id else nullcontext()
        with ctx:
            with tempfile.TemporaryDirectory() as tmpdir:
                for col, metrics in results.items():
                    mid = model_ids.get(col) if model_ids else None
                    col_regressor = health_regressors[col]
                    mlflow.log_metric(f'{col}_r2', metrics['r2'], model_id=mid)
                    mlflow.log_metric(f'{col}_mae', metrics['mae'], model_id=mid)
                    mlflow.log_metric(f'{col}_rmse', metrics['rmse'], model_id=mid)

                    col_y_true = health_test[col].values
                    col_y_pred = col_regressor.predict(X_test)
                    if target_transformers and col in target_transformers:
                        col_y_pred = target_transformers[col].inverse_transform(col_y_pred.reshape(-1, 1)).ravel()
                    col_y_pred = np.clip(col_y_pred, 0.0, 1.0)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(col_y_true, col_y_pred, alpha=0.1, s=1)
                    ax.plot([0, 1], [0, 1], 'r--', label='Perfect')
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title(f'{col}: R²={metrics["r2"]:.4f}')
                    ax.legend()
                    plt.tight_layout()

                    path = os.path.join(tmpdir, f'{col}_scatter.png')
                    fig.savefig(path, dpi=150)
                    plt.close(fig)
                    mlflow.log_artifact(path)

                    #feature importance
                    if feature_names is not None:
                        fi_df = get_feature_importance(col_regressor, feature_names)
                        fi_fig = plot_feature_importance(
                            fi_df,
                            title=f"{col} Feature Importance"
                        )
                        fi_path = os.path.join(tmpdir, f"{col}_feature_importance.png")
                        fi_fig.savefig(fi_path, dpi=150)
                        plt.close(fi_fig)
                        mlflow.log_artifact(fi_path)

                # Log test data 
                if log_test_data:
                    x_path = os.path.join(tmpdir, f'health_X_{dataset_name}.parquet')
                    y_path = os.path.join(tmpdir, f'health_y_{dataset_name}.parquet')
                    X_test.to_parquet(x_path)
                    health_test.to_parquet(y_path)
                    mlflow.log_artifact(x_path)
                    mlflow.log_artifact(y_path)
            
              
    return results

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

def get_feature_importance(model, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Get feature importance from XGBoost model.

    Args:
        model: Trained XGBoost model
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

def plot_feature_importance(fi_df: pd.DataFrame, title: str = "Feature Importance") -> plt.Figure:
    """Generate a horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = fi_df.sort_values("importance", ascending=True)
    ax.barh(data["feature"], data["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig
