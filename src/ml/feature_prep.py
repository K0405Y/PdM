"""
Feature Preparation for Failure Mode Classification

Handles:
- Labeling telemetry rows with failure mode codes (look-back window)
- Feature column selection (SENSOR_ONLY vs FULL)
- Class weight computation for imbalanced data
- Train/test splitting via equipment-based strategy
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.ml.data_loader import get_sensor_columns, get_health_columns, get_derived_columns

logger = logging.getLogger(__name__)

NORMAL_LABEL = "NORMAL"


def label_telemetry(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    prediction_horizon_hours: float = 72.0,
) -> pd.DataFrame:
    """
    Label each telemetry row with the upcoming failure mode or NORMAL.

    For each failure event, all telemetry rows for that equipment within
    `prediction_horizon_hours` before the failure are labeled with the
    failure_mode_code. Everything else is NORMAL.

    Args:
        telemetry: Telemetry DataFrame with columns [equipment_id, sample_time, ...]
        failures: Failure DataFrame with columns [equipment_id, failure_time, failure_mode_code]
        prediction_horizon_hours: Hours before failure to label as pre-failure

    Returns:
        Telemetry DataFrame with added 'label' column
    """
    df = telemetry.copy()
    df['label'] = NORMAL_LABEL

    if failures.empty:
        logger.warning("No failure events — all rows labeled NORMAL")
        return df

    horizon = pd.Timedelta(hours=prediction_horizon_hours)

    for _, failure in failures.iterrows():
        eq_id = failure['equipment_id']
        f_time = pd.Timestamp(failure['failure_time'])
        mode = failure['failure_mode_code']
        window_start = f_time - horizon

        mask = (
            (df['equipment_id'] == eq_id) &
            (df['sample_time'] >= window_start) &
            (df['sample_time'] <= f_time)
        )
        df.loc[mask, 'label'] = mode

    label_counts = df['label'].value_counts()
    logger.info(f"Label distribution:\n{label_counts.to_string()}")
    return df


def select_features(
    df: pd.DataFrame,
    equipment_type: str,
    mode: str = "sensor_only",
) -> List[str]:
    """
    Select feature columns based on output mode.

    Args:
        df: Labeled telemetry DataFrame
        equipment_type: 'turbine', 'compressor', or 'pump'
        mode: 'sensor_only' or 'all'

    Returns:
        List of feature column names
    """
    sensor_cols = get_sensor_columns(equipment_type)

    if mode == "all":
        health_cols = get_health_columns(equipment_type)
        derived_cols = get_derived_columns(equipment_type)
        feature_cols = sensor_cols + health_cols + derived_cols
    else:
        feature_cols = sensor_cols

    # Add equipment_id for fleet-level model
    feature_cols = ['equipment_id', 'operating_hours'] + [
        c for c in feature_cols if c not in ('equipment_id', 'operating_hours')
    ]

    # Only keep columns that exist in the DataFrame
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        logger.warning(f"Missing columns (skipped): {missing}")

    return available


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Prepare X (features) and y (encoded labels) for XGBoost.

    Args:
        df: Labeled telemetry DataFrame
        feature_cols: Columns to use as features

    Returns:
        (X, y_encoded, label_encoder)
    """
    X = df[feature_cols].copy()

    # Fill NaN with 0 for numeric columns (sensor gaps)
    for col in X.columns:
        if X[col].dtype in ('float64', 'float32', 'int64', 'int32'):
            X[col] = X[col].fillna(0)

    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    logger.info(f"Classes: {list(le.classes_)}")
    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y, le


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute inverse-frequency sample weights for class imbalance.

    Args:
        y: Encoded label array

    Returns:
        Weight array (same length as y)
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    class_weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    weights = np.array([class_weights[yi] for yi in y])
    logger.info(f"Class weights: { {int(c): round(w, 2) for c, w in class_weights.items()} }")
    return weights


def equipment_train_test_split(
    df: pd.DataFrame,
    equipment_ids: List[int],
    test_fraction: float = 0.3,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by equipment ID — no data leakage between train/test.

    Args:
        df: Labeled telemetry DataFrame
        equipment_ids: All equipment IDs
        test_fraction: Fraction of equipment for test set
        seed: Random seed

    Returns:
        (train_df, test_df)
    """
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(equipment_ids))
    n_test = max(1, int(len(ids) * test_fraction))
    test_ids = set(rng.choice(ids, size=n_test, replace=False))
    train_ids = set(ids) - test_ids

    train_df = df[df['equipment_id'].isin(train_ids)].copy()
    test_df = df[df['equipment_id'].isin(test_ids)].copy()

    logger.info(f"Train equipment: {sorted(train_ids)}, Test equipment: {sorted(test_ids)}")
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    return train_df, test_df


def temporal_validation_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within-equipment temporal split for validation.

    For each equipment, the last `val_fraction` of rows become validation.

    Args:
        df: Training DataFrame
        val_fraction: Fraction of each equipment's timeline for validation

    Returns:
        (train_df, val_df)
    """
    train_parts = []
    val_parts = []

    for eq_id, group in df.groupby('equipment_id'):
        group = group.sort_values('sample_time')
        split_idx = int(len(group) * (1 - val_fraction))
        train_parts.append(group.iloc[:split_idx])
        val_parts.append(group.iloc[split_idx:])

    return pd.concat(train_parts), pd.concat(val_parts)


def get_grouped_cv_splitter(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5):
    """
    Create a StratifiedGroupKFold splitter using equipment_id as groups.
    Prevents data from the same equipment appearing in both train and validation folds.

    Returns:
        Tuple of (splitter, groups array)
    """
    from sklearn.model_selection import StratifiedGroupKFold
    groups = X['equipment_id'].values
    return StratifiedGroupKFold(n_splits=n_splits), groups
