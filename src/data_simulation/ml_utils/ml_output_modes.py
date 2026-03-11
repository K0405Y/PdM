"""
ML Output Modes for Realistic Model Evaluation

Provides different output modes for training vs evaluation to simulate
realistic predictive maintenance scenarios where ground-truth health
indicators are not available in production.

Key Features:
- ground_truth: All telemetry including health indicators (training)
- sensor_only: Only measurable sensors (realistic evaluation)

Reference: Industrial SCADA systems, real-world labeling practices
"""

import os
from typing import Dict, List, Optional, Set
from collections import deque
from enum import Enum
from datetime import datetime
import yaml
import numpy as np


def _load_config() -> Dict:
    """Load table_config.yaml from project root."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'table_config.yaml'
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _collect_health_columns(cfg: Dict) -> Set[str]:
    """Collect all health columns across equipment types."""
    health = set()
    for eq_cfg in cfg['equipment_types'].values():
        health.update(eq_cfg.get('telemetry', {}).get('health_columns', []))
    return health


class OutputMode(Enum):
    """Data output modes for different use cases."""
    GROUND_TRUTH = "ground_truth"      # All data including ground-truth health
    SENSOR_ONLY = "sensor_only"        # Only measurable sensors


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
        cfg = _load_config()
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


class DataOutputFormatter:
    """Formats simulation output based on selected mode.

    Ground-truth health fields and internal state fields are loaded from
    table_config.yaml rather than hardcoded.
    """

    def __init__(self,
                 output_mode: OutputMode = OutputMode.GROUND_TRUTH,
                 sample_interval_min: int = 5):
        cfg = _load_config()
        self.ground_truth_fields = _collect_health_columns(cfg)
        self.internal_state_fields = set(cfg.get('internal_state_fields', []))

        self.output_mode = output_mode
        self._feature_engineer = FeatureEngineer(sample_interval_min=sample_interval_min)

    def format_record(self,
                     telemetry_record: Dict,
                     timestamp: datetime) -> Optional[Dict]:
        """Format a single telemetry record according to output mode."""
        if self.output_mode == OutputMode.GROUND_TRUTH:
            return telemetry_record

        elif self.output_mode == OutputMode.SENSOR_ONLY:
            return self._format_sensor_only(telemetry_record)

        return telemetry_record

    def _format_sensor_only(self, record: Dict) -> Dict:
        """Remove ground truth health and internal state fields, add noise, compute derived features."""
        filtered = {
            k: v for k, v in record.items()
            if k not in self.ground_truth_fields and k not in self.internal_state_fields
        }
        filtered = self._add_realistic_noise(filtered)
        # Compute derived features from the original (unfiltered, un-noised) record
        derived = self._feature_engineer.compute(record)
        filtered.update(derived)
        return filtered

    def _add_realistic_noise(self, record: Dict) -> Dict:
        """Add realistic sensor measurement noise."""
        noisy = record.copy()
        noise_config = {
            'temperature': 0.5,
            'pressure': 5.0,
            'speed': 10.0,
            'vibration': 0.05,
            'flow': 2.0,
            'current': 0.5,
        }

        for key, value in noisy.items():
            if not isinstance(value, (int, float)):
                continue

            sensor_type = None
            if 'temp' in key.lower():
                sensor_type = 'temperature'
            elif 'pressure' in key.lower():
                sensor_type = 'pressure'
            elif 'speed' in key.lower() or 'rpm' in key.lower():
                sensor_type = 'speed'
            elif 'vib' in key.lower():
                sensor_type = 'vibration'
            elif 'flow' in key.lower():
                sensor_type = 'flow'
            elif 'current' in key.lower():
                sensor_type = 'current'

            if sensor_type and sensor_type in noise_config:
                noise_std = noise_config[sensor_type]
                noise = np.random.normal(0, noise_std)
                if np.random.random() < 0.01:
                    noise *= 3.0
                noisy[key] = value + noise

        return noisy


class TrainTestSplitter:
    """
    Utilities for creating train/test splits for ML model development.
    """

    @staticmethod
    def temporal_split(telemetry_records: List[Dict],
                      train_fraction: float = 0.7) -> tuple:
        """
        Temporal train/test split.

        Args:
            telemetry_records: List of telemetry records (assumed time-ordered)
            train_fraction: Fraction for training set

        Returns:
            (train_records, test_records)
        """
        split_idx = int(len(telemetry_records) * train_fraction)
        return telemetry_records[:split_idx], telemetry_records[split_idx:]

    @staticmethod
    def equipment_based_split(telemetry_records: List[Dict],
                             equipment_ids: List[int],
                             test_equipment: List[int]) -> tuple:
        """
        Split by equipment (avoid data leakage).

        Args:
            telemetry_records: All telemetry records
            equipment_ids: All equipment IDs
            test_equipment: Equipment IDs to hold out for testing

        Returns:
            (train_records, test_records)
        """
        train = [r for r in telemetry_records
                if r['equipment_id'] not in test_equipment]
        test = [r for r in telemetry_records
               if r['equipment_id'] in test_equipment]

        return train, test

    @staticmethod
    def stratified_failure_split(telemetry_records: List[Dict],
                                 failure_records: List[Dict],
                                 test_fraction: float = 0.3) -> tuple:
        """
        Ensure both train and test sets have failure examples.

        Args:
            telemetry_records: All telemetry
            failure_records: Failure events
            test_fraction: Fraction for test set

        Returns:
            (train_telemetry, test_telemetry, train_failures, test_failures)
        """
        # Group telemetry by equipment
        equipment_groups = {}
        for record in telemetry_records:
            eq_id = record['equipment_id']
            if eq_id not in equipment_groups:
                equipment_groups[eq_id] = []
            equipment_groups[eq_id].append(record)

        # Identify which equipment had failures
        failed_equipment = set(f['equipment_id'] for f in failure_records)
        healthy_equipment = set(equipment_groups.keys()) - failed_equipment

        # Split failed equipment to ensure both sets have failures
        failed_list = list(failed_equipment)
        n_test_failures = max(1, int(len(failed_list) * test_fraction))

        import random
        test_failed = set(random.sample(failed_list, n_test_failures))
        train_failed = failed_equipment - test_failed

        # Split healthy equipment
        healthy_list = list(healthy_equipment)
        n_test_healthy = int(len(healthy_list) * test_fraction)
        test_healthy = set(random.sample(healthy_list, n_test_healthy))
        train_healthy = healthy_equipment - test_healthy

        # Combine equipment IDs
        test_equipment = test_failed | test_healthy
        train_equipment = train_failed | train_healthy

        # Split telemetry
        train_telem = [r for r in telemetry_records if r['equipment_id'] in train_equipment]
        test_telem = [r for r in telemetry_records if r['equipment_id'] in test_equipment]

        # Split failures
        train_fail = [f for f in failure_records if f['equipment_id'] in train_equipment]
        test_fail = [f for f in failure_records if f['equipment_id'] in test_equipment]

        return train_telem, test_telem, train_fail, test_fail


if __name__ == '__main__':
    """Demonstration of output modes."""
    print("ML Output Modes - Demonstration")

    # Mock telemetry record
    mock_record = {
        'timestamp': datetime.now(),
        'equipment_id': 1,
        'speed_rpm': 3000,
        'bearing_temp_de_celsius': 75.5,
        'vibration_rms_mm_s': 2.3,
        'health_bearing': 0.82,  # Ground truth
        'health_impeller': 0.90,  # Ground truth
        'operating_mode': 'steady_state',  # Internal state
        'differential_temp_C': 45.0  # Internal state
    }

    # Test each output mode
    for mode in OutputMode:
        print(f"\n--- {mode.value.upper()} MODE")
        formatter = DataOutputFormatter(output_mode=mode, sample_interval_min=5)
        formatted = formatter.format_record(mock_record, mock_record['timestamp'])

        if formatted:
            print(f"Keys in output: {list(formatted.keys())}")
            print(f"Sample values:")
            for key in list(formatted.keys())[:5]:
                print(f"  {key}: {formatted[key]}")

    # Demonstrate train/test splitting
    print("\nTRAIN/TEST SPLITTING")

    # Mock dataset
    mock_telemetry = [
        {'equipment_id': 1, 'sample': i} for i in range(1000)
    ] + [
        {'equipment_id': 2, 'sample': i} for i in range(800)
    ]

    mock_failures = [
        {'equipment_id': 1, 'failure_time': datetime.now()}
    ]

    # Temporal split
    train, test = TrainTestSplitter.temporal_split(mock_telemetry, train_fraction=0.7)
    print(f"Temporal split: {len(train)} train, {len(test)} test")

    # Stratified split
    train_t, test_t, train_f, test_f = TrainTestSplitter.stratified_failure_split(
        mock_telemetry, mock_failures, test_fraction=0.3
    )
    print(f"Stratified split: {len(train_t)} train telem, {len(test_t)} test telem")
    print(f"                  {len(train_f)} train failures, {len(test_f)} test failures")