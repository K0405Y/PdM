"""
Feature Preparation for Failure Mode Classification

Handles:
- Labeling telemetry rows with failure mode codes (look-back window)
- Feature column selection (sensor_only vs ground_truth)
- Class weight computation for imbalanced data
- Train/test splitting via equipment-based strategy
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.ml.data_loader import get_sensor_columns, get_health_columns, load_table_config
from src.data_simulation.ml_utils import FeatureEngineer

logger = logging.getLogger(__name__)

NORMAL_LABEL = "NORMAL"


def label_telemetry(telemetry: pd.DataFrame, failures: pd.DataFrame, prediction_horizon_hours: float = 72.0) -> pd.DataFrame:
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


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for a DataFrame using FeatureEngineer.

    Records must be sorted by (equipment_id, sample_time) for correct
    rolling-window calculations. Resets buffers per equipment unit.

    Returns:
        DataFrame with derived feature columns added in-place.
    """
    fe = FeatureEngineer()
    derived_rows = []

    for _, group in df.groupby('equipment_id', sort=False):
        group = group.sort_values('sample_time')
        fe.reset()
        for _, row in group.iterrows():
            derived_rows.append(fe.compute(row.to_dict()))

    derived_df = pd.DataFrame(derived_rows, index=df.index)
    for col in derived_df.columns:
        df[col] = derived_df[col]

    return df


def select_features(df: pd.DataFrame, equipment_type: str, mode: str = "sensor_only") -> List[str]:
    """
    Select feature columns based on output mode.

    Both modes include derived features (computed via FeatureEngineer).
    Derived features are production-available since they're computed from
    sensor data only, no ground truth needed.

    Args:
        df: Labeled telemetry DataFrame
        equipment_type: 'turbine', 'compressor', or 'pump'
        mode: 'sensor_only' or 'ground_truth'

    Returns:
        List of feature column names
    """
    sensor_cols = get_sensor_columns(equipment_type)

    # Derived features are available in both modes (computed from sensors)
    cfg = load_table_config()
    derived_col_names = [
        c for c in cfg.get('derived_columns', [])
        if c != 'operating_state'  # operating_state is computed separately
    ]
    if derived_col_names and not all(c in df.columns for c in derived_col_names):
        logger.info("Computing derived features via FeatureEngineer...")
        compute_derived_features(df)

    if mode == "ground_truth":
        health_cols = get_health_columns(equipment_type)
        feature_cols = sensor_cols + health_cols + derived_col_names
    else:
        feature_cols = sensor_cols + derived_col_names

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


def impute_features(X: pd.DataFrame, medians: Dict[str, float] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Impute missing values: forward-fill, then median, then 0 fallback.

    Args:
        X: Feature DataFrame
        medians: Pre-computed medians from training set (use for test/val).
                 If None, computes medians from X (training mode).

    Returns:
        (imputed X, median dict for reuse on test/val sets)
    """
    computed_medians = {}
    for col in X.columns:
        if X[col].dtype in ('float64', 'float32', 'int64', 'int32'):
            X[col] = X[col].ffill()
            if medians and col in medians:
                X[col] = X[col].fillna(medians[col])
            else:
                col_median = X[col].median()
                computed_medians[col] = col_median if not pd.isna(col_median) else 0.0
                X[col] = X[col].fillna(computed_medians[col])
            X[col] = X[col].fillna(0)

    return X, medians if medians else computed_medians


def prepare_xy(df: pd.DataFrame,feature_cols: List[str], medians: Dict[str, float] = None) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, Dict[str, float]]:
    """
    Prepare X (features) and y (encoded labels) for XGBoost.

    Args:
        df: Labeled telemetry DataFrame
        feature_cols: Columns to use as features
        medians: Pre-computed medians for imputation (None = compute from data)

    Returns:
        (X, y_encoded, label_encoder, medians)
    """
    X = df[feature_cols].copy()
    X, medians_out = impute_features(X, medians)

    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    logger.info(f"Classes: {list(le.classes_)}")
    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y, le, medians_out


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


def temporal_train_test_split(df: pd.DataFrame, test_fraction: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time — train on earlier data, test on later data.
    Adjusts split point earlier if needed to ensure every failure mode
    present in training also appears in the test set.

    Args:
        df: Labeled telemetry DataFrame with 'sample_time' column
        test_fraction: Fraction of data (by time) for test set

    Returns:
        (train_df, test_df)
    """
    df_sorted = df.sort_values('sample_time').reset_index(drop=True)
    all_failure_modes = set(df_sorted.loc[df_sorted['label'] != NORMAL_LABEL, 'label'].unique())

    # Start with the default split point
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    min_split_idx = int(len(df_sorted) * 0.5)  # never give test more than 50%

    # Walk split point earlier until all failure modes are in the test set
    while split_idx > min_split_idx:
        test_modes = set(df_sorted.iloc[split_idx:].loc[
            df_sorted.iloc[split_idx:]['label'] != NORMAL_LABEL, 'label'
        ].unique())
        missing = all_failure_modes - test_modes
        if not missing:
            break
        # Move split earlier by 1% of data
        split_idx -= max(1, int(len(df_sorted) * 0.01))

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    test_modes = set(test_df.loc[test_df['label'] != NORMAL_LABEL, 'label'].unique())
    missing = all_failure_modes - test_modes
    if missing:
        logger.warning(f"Failure modes missing from test set (insufficient data): {missing}")

    logger.info(f"Temporal split: train up to {train_df['sample_time'].max()}, "
                f"test from {test_df['sample_time'].min()}")
    logger.info(f"Train: {len(train_df)} rows ({train_df['label'].value_counts().to_dict()})")
    logger.info(f"Test:  {len(test_df)} rows ({test_df['label'].value_counts().to_dict()})")
    return train_df, test_df


def temporal_validation_split(df: pd.DataFrame, val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
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