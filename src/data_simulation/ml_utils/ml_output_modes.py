"""
ML Output Modes for Realistic Model Evaluation

Provides different output modes for training vs evaluation to simulate
realistic predictive maintenance scenarios where ground-truth health
indicators are not available in production.

Key Features:
- sensor_only: Only measurable sensors (realistic evaluation)
- full: All telemetry including health (training)
- delayed_labels: Health labels with configurable delay (labeling lag)
- derived_features: Pre-computed features for ML pipelines

Reference: Industrial SCADA systems, real-world labeling practices
"""

import os
from typing import Dict, List, Optional, Set
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
import json
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
    FULL = "full"                      # All data including ground-truth health
    SENSOR_ONLY = "sensor_only"        # Only measurable sensors
    DELAYED_LABELS = "delayed_labels"  # Labels available after delay
    DERIVED_FEATURES = "derived_features"  # Include engineered features


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

        self._vibration_keys = tuple(fe_cfg['vibration_keys'])
        self._temperature_keys = tuple(fe_cfg['temperature_keys'])
        self._speed_key = fe_cfg['speed_key']
        self._efficiency_key = fe_cfg['efficiency_key']
        self._pressure_keys = fe_cfg['pressure_keys']
        self._load_keys = fe_cfg['load_keys']

        samples_per_day = int(24 * 60 / sample_interval_min)
        self._vibration_history = deque(maxlen=samples_per_day * 7)
        self._temperature_history = deque(maxlen=samples_per_day)
        self._speed_history = deque(maxlen=int(60 / sample_interval_min))
        self._efficiency_history = deque(maxlen=samples_per_day * 7)

    def reset(self):
        """Clear all rolling buffers."""
        self._vibration_history.clear()
        self._temperature_history.clear()
        self._speed_history.clear()
        self._efficiency_history.clear()

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

        if len(self._vibration_history) >= 2:
            x = np.arange(len(self._vibration_history))
            coeffs = np.polyfit(x, list(self._vibration_history), 1)
            features['vibration_trend_7d'] = float(coeffs[0])
        else:
            features['vibration_trend_7d'] = 0.0

        if len(self._temperature_history) >= 2:
            features['temp_variation_24h'] = float(np.std(list(self._temperature_history)))
        else:
            features['temp_variation_24h'] = 0.0

        if len(self._speed_history) >= 2:
            arr = np.array(self._speed_history)
            mean_speed = np.mean(arr)
            if mean_speed > 0:
                features['speed_stability'] = float(np.std(arr) / mean_speed)
            else:
                features['speed_stability'] = 0.0
        else:
            features['speed_stability'] = 0.0

        if len(self._efficiency_history) >= 2:
            x = np.arange(len(self._efficiency_history))
            coeffs = np.polyfit(x, list(self._efficiency_history), 1)
            features['efficiency_degradation_rate'] = float(coeffs[0])
        else:
            features['efficiency_degradation_rate'] = 0.0

        discharge_key = self._pressure_keys['discharge']
        suction_key = self._pressure_keys['suction']
        if discharge_key in record and suction_key in record:
            suction = record.get(suction_key, 1)
            if suction > 0:
                features['pressure_ratio'] = record[discharge_key] / suction

        speed_key = self._load_keys['speed']
        target_key = self._load_keys['speed_target']
        if speed_key in record and target_key in record:
            if record[target_key] > 0:
                features['load_factor'] = record[speed_key] / record[target_key]

        return features

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

    def _extract_telemetry_values(self, record: Dict) -> Dict:
        """Extract canonical telemetry values from equipment-specific keys."""
        values = {}

        for key in self._vibration_keys:
            if key in record:
                values['vibration'] = record[key]
                break

        temps = [record[k] for k in self._temperature_keys
                 if k in record and isinstance(record[k], (int, float))]
        if temps:
            values['temperature'] = max(temps)

        if self._speed_key in record:
            values['speed'] = record[self._speed_key]
        if self._efficiency_key in record:
            values['efficiency'] = record[self._efficiency_key]

        return values


class DataOutputFormatter:
    """Formats simulation output based on selected mode.

    Ground-truth health fields and internal state fields are loaded from
    table_config.yaml rather than hardcoded.
    """

    def __init__(self,
                 output_mode: OutputMode = OutputMode.FULL,
                 label_delay_hours: int = 168,
                 sample_interval_min: int = 5):
        cfg = _load_config()
        self.ground_truth_fields = _collect_health_columns(cfg)
        self.internal_state_fields = set(cfg.get('internal_state_fields', []))

        self.output_mode = output_mode
        self.label_delay_hours = label_delay_hours
        self.label_buffer = []
        self._feature_engineer = FeatureEngineer(sample_interval_min=sample_interval_min)

    def format_record(self,
                     telemetry_record: Dict,
                     timestamp: datetime) -> Optional[Dict]:
        """Format a single telemetry record according to output mode."""
        if self.output_mode == OutputMode.FULL:
            return telemetry_record

        elif self.output_mode == OutputMode.SENSOR_ONLY:
            return self._format_sensor_only(telemetry_record)

        elif self.output_mode == OutputMode.DELAYED_LABELS:
            return self._format_delayed_labels(telemetry_record, timestamp)

        elif self.output_mode == OutputMode.DERIVED_FEATURES:
            formatted = telemetry_record.copy()
            formatted['features'] = json.dumps(self._feature_engineer.compute(telemetry_record))
            return formatted

        return telemetry_record

    def _format_sensor_only(self, record: Dict) -> Dict:
        """Remove ground truth health and internal state fields, add noise."""
        filtered = {
            k: v for k, v in record.items()
            if k not in self.ground_truth_fields and k not in self.internal_state_fields
        }
        return self._add_realistic_noise(filtered)

    def _format_delayed_labels(self, record: Dict, timestamp: datetime) -> Optional[Dict]:
        """Health indicators available after configurable delay."""
        self.label_buffer.append({'timestamp': timestamp, 'record': record.copy()})
        cutoff_time = timestamp - timedelta(hours=self.label_delay_hours)
        ready_records = [r for r in self.label_buffer if r['timestamp'] <= cutoff_time]

        if not ready_records:
            return self._format_sensor_only(record)

        ready = ready_records[0]
        self.label_buffer.remove(ready)
        return ready['record']

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
        print(f"\n--- {mode.value.upper()} MODE ---")
        formatter = DataOutputFormatter(output_mode=mode)
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