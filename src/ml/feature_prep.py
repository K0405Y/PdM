"""
Feature Preparation for Failure Mode Classification

Handles:
- Labeling telemetry rows with failure mode codes (look-back window)
- Feature column selection (sensor_only vs ground_truth)
- Class weight computation for imbalanced data
- Train/test splitting via equipment-based strategy
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from src.ml.data_loader import get_sensor_columns, get_health_columns, load_table_config
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

NORMAL_LABEL = "NORMAL"


class FeatureEngineer:
    """Compute derived features from raw sensor readings.

    Usable standalone for inference pipelines or internally by DataOutputFormatter.
    Maintains rolling-window buffers for trend/stability features.
    Key mappings are loaded from table_config.yaml.

    Usage:
        # Streaming (one record at a time, maintains state):
        fe = FeatureEngineer()
        features = fe.compute(sensor_record)

        # Batch (list of time-ordered records, resets buffers first):
        fe = FeatureEngineer()
        all_features = fe.compute_batch(records)
    """

    def __init__(self, sample_interval_min: int = 5):
        cfg = load_table_config()
        fe_cfg = cfg['feature_engineering']

        # Build reverse lookup: DB column name → state key
        # so FeatureEngineer works with both simulator records and DB records
        self._db_to_state: Dict[str, str] = {}
        for eq_cfg in cfg['equipment_types'].values():
            mappings = eq_cfg.get('telemetry', {}).get('column_mappings', {})
            for state_key, db_col in mappings.items():
                if db_col != state_key:
                    self._db_to_state[db_col] = state_key

        self._vibration_keys = tuple(fe_cfg['vibration_keys'])
        self._temperature_keys = tuple(fe_cfg['temperature_keys'])
        self._speed_key = fe_cfg['speed_key']
        self._efficiency_key = fe_cfg['efficiency_key']
        self._pressure_keys = fe_cfg['pressure_keys']
        self._load_keys = fe_cfg['load_keys']

        self._fuel_flow_key = fe_cfg.get('fuel_flow_key', 'fuel_flow')

        self._samples_per_day = int(24 * 60 / sample_interval_min)
        self._vibration_history = deque(maxlen=self._samples_per_day * 30)
        self._temperature_history = deque(maxlen=self._samples_per_day * 30)
        self._speed_history = deque(maxlen=int(60 / sample_interval_min))
        self._efficiency_history = deque(maxlen=self._samples_per_day * 30)

        # Delta tracking (previous values)
        self._prev_vibration: Optional[float] = None
        self._prev_temperature: Optional[float] = None
        self._prev_efficiency: Optional[float] = None

        # 2nd derivative tracking (previous deltas)
        self._prev_vibration_delta: Optional[float] = None
        self._prev_temp_delta: Optional[float] = None
        self._prev_efficiency_delta: Optional[float] = None

        # Exponentially weighted moving average state
        self._vibration_ewma: Optional[float] = None
        self._temp_ewma: Optional[float] = None
        self._ewma_alpha = 0.1

    def reset(self):
        """Clear all rolling buffers and state."""
        self._vibration_history.clear()
        self._temperature_history.clear()
        self._speed_history.clear()
        self._efficiency_history.clear()
        self._prev_vibration = None
        self._prev_temperature = None
        self._prev_efficiency = None
        self._prev_vibration_delta = None
        self._prev_temp_delta = None
        self._prev_efficiency_delta = None
        self._vibration_ewma = None
        self._temp_ewma = None

    def compute(self, record: Dict) -> Dict:
        """Compute derived features for a single record.

        Updates internal rolling buffers, so call in time order for
        correct trend/stability calculations.
        """
        values = self._extract_telemetry_values(record)

        if 'vibration' in values and np.isfinite(values['vibration']):
            self._vibration_history.append(values['vibration'])
        if 'temperature' in values and np.isfinite(values['temperature']):
            self._temperature_history.append(values['temperature'])
        if 'speed' in values and np.isfinite(values['speed']):
            self._speed_history.append(values['speed'])
        if 'efficiency' in values and np.isfinite(values['efficiency']):
            self._efficiency_history.append(values['efficiency'])

        features = {}
        spd = self._samples_per_day

        # Vibration trend features (7d, 14d, 30d)
        vib_list = list(self._vibration_history)
        if len(vib_list) >= 2:
            # 7d trend (use last 7 days of buffer)
            vib_7d = vib_list[-(spd * 7):] if len(vib_list) > spd * 7 else vib_list
            x = np.arange(len(vib_7d))
            features['vibration_trend_7d'] = float(np.polyfit(x, vib_7d, 1)[0])
        else:
            features['vibration_trend_7d'] = 0.0

        if len(vib_list) >= spd * 14:
            vib_14d = vib_list[-(spd * 14):]
            x = np.arange(len(vib_14d))
            features['vibration_trend_14d'] = float(np.polyfit(x, vib_14d, 1)[0])
        else:
            features['vibration_trend_14d'] = features['vibration_trend_7d']

        if len(vib_list) >= spd * 21:
            x = np.arange(len(vib_list))
            features['vibration_trend_30d'] = float(np.polyfit(x, vib_list, 1)[0])
        else:
            features['vibration_trend_30d'] = features['vibration_trend_14d']

        # Temperature variation features (24h, 7d, 14d)
        temp_list = list(self._temperature_history)
        if len(temp_list) >= 2:
            temp_24h = temp_list[-spd:] if len(temp_list) > spd else temp_list
            features['temp_variation_24h'] = float(np.std(temp_24h))
        else:
            features['temp_variation_24h'] = 0.0

        if len(temp_list) >= spd * 7:
            features['temp_variation_7d'] = float(np.std(temp_list[-(spd * 7):]))
        else:
            features['temp_variation_7d'] = features['temp_variation_24h']

        if len(temp_list) >= spd * 14:
            features['temp_variation_14d'] = float(np.std(temp_list[-(spd * 14):]))
        else:
            features['temp_variation_14d'] = features['temp_variation_7d']

        # Speed stability
        if len(self._speed_history) >= 2:
            arr = np.array(self._speed_history)
            mean_speed = np.mean(arr)
            if mean_speed > 0:
                features['speed_stability'] = float(np.std(arr) / mean_speed)
            else:
                features['speed_stability'] = 0.0
        else:
            features['speed_stability'] = 0.0

        # Efficiency degradation rate (7d, 14d, 30d)
        eff_list = list(self._efficiency_history)
        if len(eff_list) >= 2:
            eff_7d = eff_list[-(spd * 7):] if len(eff_list) > spd * 7 else eff_list
            x = np.arange(len(eff_7d))
            features['efficiency_degradation_rate'] = float(np.polyfit(x, eff_7d, 1)[0])
        else:
            features['efficiency_degradation_rate'] = 0.0

        if len(eff_list) >= spd * 14:
            eff_14d = eff_list[-(spd * 14):]
            x = np.arange(len(eff_14d))
            features['efficiency_degradation_rate_14d'] = float(np.polyfit(x, eff_14d, 1)[0])
        else:
            features['efficiency_degradation_rate_14d'] = features['efficiency_degradation_rate']

        if len(eff_list) >= spd * 21:
            x = np.arange(len(eff_list))
            features['efficiency_degradation_rate_30d'] = float(np.polyfit(x, eff_list, 1)[0])
        else:
            features['efficiency_degradation_rate_30d'] = features['efficiency_degradation_rate_14d']

        # Pressure ratio
        discharge = self._resolve(record, self._pressure_keys['discharge'])
        suction = self._resolve(record, self._pressure_keys['suction'])
        if discharge is not None and suction is not None and suction > 0:
            features['pressure_ratio'] = discharge / suction

        # Load factor
        speed_val = self._resolve(record, self._load_keys['speed'])
        target_val = self._resolve(record, self._load_keys['speed_target'])
        if speed_val is not None and target_val is not None and target_val > 0:
            features['load_factor'] = speed_val / target_val

        # Delta (1st derivative) features
        vib = values.get('vibration')
        temp = values.get('temperature')
        eff = values.get('efficiency')

        if vib is not None and self._prev_vibration is not None:
            vib_delta = vib - self._prev_vibration
        else:
            vib_delta = 0.0
        features['vibration_delta'] = vib_delta
        if vib is not None:
            self._prev_vibration = vib

        if temp is not None and self._prev_temperature is not None:
            temp_delta = temp - self._prev_temperature
        else:
            temp_delta = 0.0
        features['temp_delta'] = temp_delta
        if temp is not None:
            self._prev_temperature = temp

        if eff is not None and self._prev_efficiency is not None:
            eff_delta = eff - self._prev_efficiency
        else:
            eff_delta = 0.0
        features['efficiency_delta'] = eff_delta
        if eff is not None:
            self._prev_efficiency = eff

        # 2nd derivative (acceleration) features
        if self._prev_vibration_delta is not None:
            features['vibration_acceleration'] = vib_delta - self._prev_vibration_delta
        else:
            features['vibration_acceleration'] = 0.0
        self._prev_vibration_delta = vib_delta

        if self._prev_temp_delta is not None:
            features['temp_acceleration'] = temp_delta - self._prev_temp_delta
        else:
            features['temp_acceleration'] = 0.0
        self._prev_temp_delta = temp_delta

        if self._prev_efficiency_delta is not None:
            features['efficiency_acceleration'] = eff_delta - self._prev_efficiency_delta
        else:
            features['efficiency_acceleration'] = 0.0
        self._prev_efficiency_delta = eff_delta

        # Trend acceleration: difference between short and long trend slopes
        features['vibration_trend_acceleration'] = (
            features['vibration_trend_7d'] - features['vibration_trend_14d']
        )
        features['efficiency_trend_acceleration'] = (
            features['efficiency_degradation_rate'] - features['efficiency_degradation_rate_14d']
        )

        # Rolling min/max features (24h, 7d, 14d)
        if len(vib_list) >= 2:
            features['vibration_max_24h'] = float(np.max(vib_list[-spd:]))
            features['vibration_max_7d'] = float(np.max(vib_list[-(spd * 7):]))
            features['vibration_max_14d'] = float(np.max(vib_list[-(spd * 14):]))
        else:
            features['vibration_max_24h'] = 0.0
            features['vibration_max_7d'] = 0.0
            features['vibration_max_14d'] = 0.0

        if len(temp_list) >= 2:
            features['temp_max_24h'] = float(np.max(temp_list[-spd:]))
            features['temp_max_7d'] = float(np.max(temp_list[-(spd * 7):]))
            features['temp_max_14d'] = float(np.max(temp_list[-(spd * 14):]))
        else:
            features['temp_max_24h'] = 0.0
            features['temp_max_7d'] = 0.0
            features['temp_max_14d'] = 0.0

        if len(eff_list) >= 2:
            features['efficiency_min_7d'] = float(np.min(eff_list[-(spd * 7):]))
            features['efficiency_min_14d'] = float(np.min(eff_list[-(spd * 14):]))
            features['efficiency_min_30d'] = float(np.min(eff_list))
        else:
            features['efficiency_min_7d'] = 0.0
            features['efficiency_min_14d'] = 0.0
            features['efficiency_min_30d'] = 0.0

        # EWMA features
        if vib is not None:
            if self._vibration_ewma is None:
                self._vibration_ewma = vib
            else:
                self._vibration_ewma = self._ewma_alpha * vib + (1 - self._ewma_alpha) * self._vibration_ewma
            features['vibration_ewma'] = self._vibration_ewma
        else:
            features['vibration_ewma'] = self._vibration_ewma if self._vibration_ewma is not None else 0.0

        if temp is not None:
            if self._temp_ewma is None:
                self._temp_ewma = temp
            else:
                self._temp_ewma = self._ewma_alpha * temp + (1 - self._ewma_alpha) * self._temp_ewma
            features['temp_ewma'] = self._temp_ewma
        else:
            features['temp_ewma'] = self._temp_ewma if self._temp_ewma is not None else 0.0

        # Cross-sensor interaction features
        if temp is not None and vib is not None and vib > 0:
            features['temp_vibration_ratio'] = temp / vib
        else:
            features['temp_vibration_ratio'] = 0.0

        if eff is not None and 'load_factor' in features and features['load_factor'] > 0:
            features['efficiency_load_ratio'] = eff / features['load_factor']
        else:
            features['efficiency_load_ratio'] = 0.0

        # New cross-sensor interactions
        fuel_flow = self._resolve(record, self._fuel_flow_key)

        if vib is not None and eff is not None and eff > 0:
            features['vibration_efficiency_ratio'] = vib / eff
        else:
            features['vibration_efficiency_ratio'] = 0.0

        if temp is not None and eff is not None and eff > 0:
            features['temp_efficiency_ratio'] = temp / eff
        else:
            features['temp_efficiency_ratio'] = 0.0

        if vib is not None and temp is not None:
            features['vibration_temp_product'] = vib * temp
        else:
            features['vibration_temp_product'] = 0.0

        if fuel_flow is not None and eff is not None and eff > 0:
            features['fuel_efficiency_ratio'] = fuel_flow / eff
        else:
            features['fuel_efficiency_ratio'] = 0.0

        speed = values.get('speed')
        if vib is not None and speed is not None and speed > 0:
            features['vibration_speed_ratio'] = vib / speed
        else:
            features['vibration_speed_ratio'] = 0.0

        if temp is not None and fuel_flow is not None and fuel_flow > 0:
            features['temp_fuel_ratio'] = temp / fuel_flow
        else:
            features['temp_fuel_ratio'] = 0.0

        # Spectral features (24h window)
        features.update(self._compute_spectral_features(vib_list, 'vibration', spd))
        features.update(self._compute_spectral_features(temp_list, 'temp', spd))

        # Failure-mode-specific indicator features
        oil_temp = self._resolve(record, 'oil_temp')
        vib_peak = self._resolve(record, 'vibration_peak')

        if vib is not None and oil_temp is not None:
            features['bearing_indicator'] = vib * oil_temp
        else:
            features['bearing_indicator'] = 0.0

        if temp is not None and eff is not None:
            features['hgp_indicator'] = temp * (1.0 - eff)
        else:
            features['hgp_indicator'] = 0.0

        if vib is not None and vib_peak is not None and eff is not None:
            features['blade_indicator'] = vib * vib_peak / max(eff, 0.01)
        else:
            features['blade_indicator'] = 0.0

        if fuel_flow is not None and eff is not None:
            features['fuel_indicator'] = fuel_flow * (1.0 - eff)
        else:
            features['fuel_indicator'] = 0.0

        return features

    def _compute_spectral_features(self, signal_list: list, prefix: str,
                                   samples_per_day: int) -> Dict:
        """Compute FFT-based spectral features from a signal window.

        Args:
            signal_list: Time-ordered signal values (from deque).
            prefix: Feature name prefix ('vibration' or 'temp').
            samples_per_day: Number of samples in one day.

        Returns:
            Dict with spectral entropy, dominant frequency, and spectral centroid.
        """
        result = {
            f'{prefix}_spectral_entropy': 0.0,
            f'{prefix}_dominant_freq': 0.0,
            f'{prefix}_spectral_centroid': 0.0,
        }

        if len(signal_list) < samples_per_day:
            return result

        window = np.array(signal_list[-samples_per_day:])
        # Remove DC component
        window = window - np.mean(window)

        fft_vals = np.fft.rfft(window)
        power = np.abs(fft_vals) ** 2
        total_power = np.sum(power)

        if total_power <= 0:
            return result

        # Normalized power spectrum (probability distribution)
        p = power / total_power
        # Avoid log(0)
        p_safe = p[p > 0]
        result[f'{prefix}_spectral_entropy'] = float(-np.sum(p_safe * np.log(p_safe)))

        freqs = np.fft.rfftfreq(len(window))
        result[f'{prefix}_dominant_freq'] = float(freqs[np.argmax(power)])
        result[f'{prefix}_spectral_centroid'] = float(np.sum(freqs * p))

        return result

    def compute_batch(self, records: List[Dict]) -> List[Dict]:
        """Compute derived features for a batch of time-ordered records.

        Resets buffers before processing so results are self-contained.
        Returns a new list; each dict is the original record augmented
        with derived feature keys.
        """
        self.reset()
        results = []
        for record in records:
            features = self.compute(record)
            augmented = record.copy()
            augmented.update(features)
            results.append(augmented)
        return results

    def _resolve(self, record: Dict, state_key: str) -> Optional[float]:
        """Look up a value by state key, falling back to DB column name."""
        if state_key in record:
            return record[state_key]
        # Check all DB column names that map to this state key
        for db_col, sk in self._db_to_state.items():
            if sk == state_key and db_col in record:
                return record[db_col]
        return None

    def _extract_telemetry_values(self, record: Dict) -> Dict:
        """Extract canonical telemetry values from equipment-specific keys."""
        values = {}

        for key in self._vibration_keys:
            val = self._resolve(record, key)
            if val is not None:
                values['vibration'] = val
                break

        temps = []
        for key in self._temperature_keys:
            val = self._resolve(record, key)
            if val is not None and isinstance(val, (int, float)):
                temps.append(val)
        if temps:
            values['temperature'] = max(temps)

        speed = self._resolve(record, self._speed_key)
        if speed is not None:
            values['speed'] = speed
        efficiency = self._resolve(record, self._efficiency_key)
        if efficiency is not None:
            values['efficiency'] = efficiency

        return values


def label_telemetry(telemetry: pd.DataFrame, failures: pd.DataFrame, 
                        prediction_horizon_hours: float = 168.0) -> pd.DataFrame:
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


def compute_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-equipment cumulative degradation features in-place.

    These are monotonic proxies for health degradation, useful for
    health indicator regression where raw sensors lack signal above
    the simulator's coupling thresholds.

    Column names are read from ``cumulative_columns`` in table_config.yaml.
    """
    cfg = load_table_config()
    cumulative_columns = cfg.get('cumulative_columns', [])
    if not cumulative_columns:
        logger.warning("No cumulative_columns defined in table_config.yaml — skipping")
        return df

    # Detect sensor column names dynamically
    vib_rms_col = next((c for c in df.columns if 'vibration_rms' in c), None)
    vib_peak_col = next((c for c in df.columns if 'vibration_peak' in c and 'crest' not in c), None)
    eff_col = next((c for c in df.columns if 'efficiency' in c and 'min' not in c and 'degradation' not in c and 'ratio' not in c and 'load' not in c and 'delta' not in c and 'acceleration' not in c), None)
    egt_col = next((c for c in df.columns if 'egt' in c.lower() or 'exhaust' in c.lower()), None)
    fuel_col = next((c for c in df.columns if 'fuel_flow' in c), None)

    for col_name in cumulative_columns:
        df[col_name] = 0.0

    for _, group in df.groupby('equipment_id', sort=False):
        group = group.sort_values('sample_time')
        sorted_idx = group.index

        if 'cummax_vibration_rms' in cumulative_columns and vib_rms_col:
            df.loc[sorted_idx, 'cummax_vibration_rms'] = group[vib_rms_col].cummax().values

        if 'cummax_vibration_peak' in cumulative_columns and vib_peak_col:
            df.loc[sorted_idx, 'cummax_vibration_peak'] = group[vib_peak_col].cummax().values

        if 'cum_efficiency_loss' in cumulative_columns and eff_col:
            first_eff = group[eff_col].iloc[0]
            df.loc[sorted_idx, 'cum_efficiency_loss'] = (first_eff - group[eff_col]).values

        if 'lifecycle_position' in cumulative_columns and 'operating_hours' in df.columns:
            max_hours = group['operating_hours'].max()
            if max_hours > 0:
                df.loc[sorted_idx, 'lifecycle_position'] = (group['operating_hours'] / max_hours).values

        if 'cummax_egt' in cumulative_columns and egt_col:
            df.loc[sorted_idx, 'cummax_egt'] = group[egt_col].cummax().values

        if 'cum_fuel_flow_increase' in cumulative_columns and fuel_col:
            first_fuel = group[fuel_col].iloc[0]
            df.loc[sorted_idx, 'cum_fuel_flow_increase'] = (group[fuel_col] - first_fuel).values

        if 'cummin_efficiency' in cumulative_columns and eff_col:
            df.loc[sorted_idx, 'cummin_efficiency'] = group[eff_col].cummin().values

    logger.info(f"Computed cumulative features: {cumulative_columns}")
    return df


def select_features(df: pd.DataFrame, equipment_type: str, mode: str = "regressor") -> List[str]:
    """
    Select feature columns based on output mode.

    Derived features are production-available since they're computed from
    sensor data only, no ground truth needed.

    Args:
        df: Labeled telemetry DataFrame
        equipment_type: 'turbine', 'compressor', or 'pump'
        mode: 'classifier', or 'regressor'

    Returns:
        List of feature column names
    """
    sensor_cols = get_sensor_columns(equipment_type)

    cfg = load_table_config()

    if mode == "regressor":
        # Raw sensors + cumulative features only (no FeatureEngineer derived features)
        cumulative = [c for c in cfg.get('cumulative_columns', []) if c in df.columns]
        feature_cols = sensor_cols + cumulative
    else:
        # Classifier mode: sensors + health + derived features
        derived_col_names = [
            c for c in cfg.get('derived_columns', [])
            if c != 'operating_state'
        ]
        if derived_col_names and not all(c in df.columns for c in derived_col_names):
            logger.info("Computing derived features via FeatureEngineer...")
            compute_derived_features(df)
        health_cols = get_health_columns(equipment_type)
        feature_cols = sensor_cols + health_cols + derived_col_names
        
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


def normalize_per_equipment(df: pd.DataFrame, feature_cols: List[str],
                            label_col: str = 'label',
                            stats: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Z-score normalize feature columns per equipment using NORMAL-labeled data as baseline.

    Works with both sensor_only and ground_truth feature sets.

    Args:
        df: DataFrame with equipment_id, label, and feature columns
        feature_cols: Columns to normalize (sensor + derived + health as applicable)
        label_col: Label column name
        stats: Pre-computed per-equipment stats (for test/inference). If None, computes from data.

    Returns:
        (normalized DataFrame, per-equipment stats dict)
    """
    computed_stats = {} if stats is None else None
    cols_to_norm = [c for c in feature_cols if c in df.columns and c != 'equipment_id'
                    and c != 'operating_hours']

    for eq_id in df['equipment_id'].unique():
        eq_mask = df['equipment_id'] == eq_id

        if stats is None:
            normal_mask = eq_mask & (df[label_col] == NORMAL_LABEL)
            eq_stats = {}
            for col in cols_to_norm:
                mean = df.loc[normal_mask, col].mean()
                std = df.loc[normal_mask, col].std()
                eq_stats[col] = (float(mean) if not pd.isna(mean) else 0.0,
                                 float(std) if not pd.isna(std) and std > 0 else 1.0)
                df.loc[eq_mask, col] = (df.loc[eq_mask, col] - eq_stats[col][0]) / eq_stats[col][1]
            computed_stats[eq_id] = eq_stats
        else:
            eq_stats = stats.get(eq_id, {})
            for col in cols_to_norm:
                if col in eq_stats:
                    mean, std = eq_stats[col]
                    df.loc[eq_mask, col] = (df.loc[eq_mask, col] - mean) / std

    logger.info(f"Per-equipment normalization applied to {len(cols_to_norm)} columns")
    return df, stats if stats is not None else computed_stats


def select_features_by_importance(model, X_val: np.ndarray, y_val: np.ndarray,
                                   feature_cols: List[str], threshold: float = 0.0) -> List[str]:
    """Drop features with negative or zero permutation importance.

    Args:
        model: Trained model with predict method
        X_val: Validation feature array
        y_val: Validation labels (encoded)
        feature_cols: List of feature column names
        threshold: Minimum importance to keep a feature

    Returns:
        List of feature names to keep
    """
    result = permutation_importance(model, X_val, y_val, n_repeats=10,
                                     random_state=42, scoring='f1_macro')
    keep = [f for f, imp in zip(feature_cols, result.importances_mean) if imp > threshold]
    dropped = [f for f in feature_cols if f not in keep]
    logger.info(f"Feature selection: keeping {len(keep)}/{len(feature_cols)}, dropped: {dropped}")
    return keep


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

    # Ensure every failure mode has samples in the test set
    test_modes = set(test_df.loc[test_df['label'] != NORMAL_LABEL, 'label'].unique())
    missing = all_failure_modes - test_modes
    for mode in missing:
        mode_in_train = train_df[train_df['label'] == mode]
        if len(mode_in_train) == 0:
            continue
        n_move = max(1, int(len(mode_in_train) * test_fraction))
        move_idx = mode_in_train.tail(n_move).index
        test_df = pd.concat([test_df, train_df.loc[move_idx]])
        train_df = train_df.drop(move_idx)
        logger.warning(f"Moved {n_move} {mode} samples from train to test (was 0 in test)")

    # Re-check after redistribution
    still_missing = all_failure_modes - set(test_df.loc[test_df['label'] != NORMAL_LABEL, 'label'].unique())
    if still_missing:
        logger.warning(f"Failure modes still missing from test set: {still_missing}")

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
    groups = X['equipment_id'].values
    return StratifiedGroupKFold(n_splits=n_splits), groups