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

from typing import Dict, List, Optional
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np


class OutputMode(Enum):
    """Data output modes for different use cases."""
    FULL = "full"                      # All data including ground-truth health
    SENSOR_ONLY = "sensor_only"        # Only measurable sensors
    DELAYED_LABELS = "delayed_labels"  # Labels available after delay
    DERIVED_FEATURES = "derived_features"  # Include engineered features


class DataOutputFormatter:
    """
    Formats simulation output based on selected mode.
    """

    # Define which fields are "ground truth" vs "measurable"
    GROUND_TRUTH_FIELDS = {
        'health_hgp', 'health_blade', 'health_bearing', 'health_fuel',
        'health_impeller', 'health_seal', 'health_seal_primary',
        'health_seal_secondary', 'health_bearing_de', 'health_bearing_nde'
    }

    # Fields that would never be measured in reality (internal states)
    INTERNAL_STATE_FIELDS = {
        'operating_mode', 'thermal_stress', 'degradation_multiplier',
        'differential_temp_C'  
    }

    def __init__(self,
                 output_mode: OutputMode = OutputMode.FULL,
                 label_delay_hours: int = 168):  # 1 week default
        """
        Initialize output formatter.

        Args:
            output_mode: Desired output mode
            label_delay_hours: Hours of delay for DELAYED_LABELS mode
        """
        self.output_mode = output_mode
        self.label_delay_hours = label_delay_hours
        self.label_buffer = []  # For delayed labels mode

        # Rolling history buffers for derived feature computation
        # Window sizes assume 5-min sample intervals
        self._vibration_history = deque(maxlen=2016)    # 7 days
        self._temperature_history = deque(maxlen=288)   # 24 hours
        self._speed_history = deque(maxlen=12)          # 1 hour
        self._efficiency_history = deque(maxlen=2016)   # 7 days

    def format_record(self,
                     telemetry_record: Dict,
                     timestamp: datetime) -> Optional[Dict]:
        """
        Format a single telemetry record according to output mode.

        Args:
            telemetry_record: Raw telemetry from simulator
            timestamp: Record timestamp

        Returns:
            Formatted record or None (if delayed labels not ready)
        """
        if self.output_mode == OutputMode.FULL:
            return self._format_full(telemetry_record, timestamp)

        elif self.output_mode == OutputMode.SENSOR_ONLY:
            return self._format_sensor_only(telemetry_record, timestamp)

        elif self.output_mode == OutputMode.DELAYED_LABELS:
            return self._format_delayed_labels(telemetry_record, timestamp)

        elif self.output_mode == OutputMode.DERIVED_FEATURES:
            return self._format_with_features(telemetry_record, timestamp)

        return telemetry_record  # Fallback

    def _format_full(self, record: Dict, timestamp: datetime) -> Dict:
        """Full mode: all data including ground truth."""
        return record  # No filtering

    def _format_sensor_only(self, record: Dict, timestamp: datetime) -> Dict:
        """Sensor-only mode: remove ground truth health indicators."""
        filtered = {}

        for key, value in record.items():
            # Skip ground truth fields
            if key in self.GROUND_TRUTH_FIELDS:
                continue
            # Skip internal state fields
            if key in self.INTERNAL_STATE_FIELDS:
                continue
            filtered[key] = value

        # Add realistic sensor noise
        filtered = self._add_realistic_noise(filtered)

        return filtered

    def _format_delayed_labels(self,
                               record: Dict,
                               timestamp: datetime) -> Optional[Dict]:
        """
        Delayed labels mode: health indicators available after delay.

        Simulates real-world scenario where labels come from inspection
        reports that take time to process.
        """
        # Add current record to buffer with timestamp
        self.label_buffer.append({
            'timestamp': timestamp,
            'record': record.copy()
        })

        # Check if we have a record old enough to release
        cutoff_time = timestamp - timedelta(hours=self.label_delay_hours)

        # Find records older than cutoff
        ready_records = [r for r in self.label_buffer if r['timestamp'] <= cutoff_time]

        if not ready_records:
            # Return sensor-only version while waiting for labels
            return self._format_sensor_only(record, timestamp)

        # Release oldest ready record (FIFO)
        ready = ready_records[0]
        self.label_buffer.remove(ready)

        # Return with full labels (they're now "available")
        return ready['record']

    def _format_with_features(self, record: Dict, timestamp: datetime) -> Dict:
        """Add derived features commonly used in ML pipelines."""
        formatted = record.copy()
        # Serialize to JSON for database compatibility
        formatted['features'] = json.dumps(self._compute_derived_features(record))
        return formatted

    def _add_realistic_noise(self, record: Dict) -> Dict:
        """
        Add realistic sensor measurement noise.

        Real sensors have noise, drift, and occasional outliers.
        """
        noisy = record.copy()

        # Noise levels for different sensor types (typical industrial sensors)
        noise_config = {
            'temperature': 0.5,    # °C
            'pressure': 5.0,       # kPa
            'speed': 10.0,         # RPM
            'vibration': 0.05,     # mm/s
            'flow': 2.0,           # m³/hr
            'current': 0.5         # A
        }

        for key, value in noisy.items():
            if not isinstance(value, (int, float)):
                continue

            # Determine sensor type from key name
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

                # Occasional outliers (1% chance)
                if np.random.random() < 0.01:
                    noise *= 3.0

                noisy[key] = value + noise

        return noisy

    def _extract_telemetry_values(self, record: Dict) -> Dict:
        """Extract canonical telemetry values from equipment-specific keys."""
        values = {}

        # Vibration: prefer vibration_rms, fall back to orbit_amplitude (compressor)
        if 'vibration_rms' in record:
            values['vibration'] = record['vibration_rms']
        elif 'orbit_amplitude' in record:
            values['vibration'] = record['orbit_amplitude']

        # Temperature: max of available bearing/process temps
        temp_keys = ['bearing_temp_de', 'bearing_temp_nde', 'fluid_temp',
                     'exhaust_gas_temp', 'oil_temp', 'discharge_temp']
        temps = [record[k] for k in temp_keys if k in record and isinstance(record[k], (int, float))]
        if temps:
            values['temperature'] = max(temps)

        # Speed and efficiency: universal keys
        if 'speed' in record:
            values['speed'] = record['speed']
        if 'efficiency' in record:
            values['efficiency'] = record['efficiency']

        return values

    def _compute_derived_features(self, record: Dict) -> Dict:
        """
        Compute derived features from rolling history buffers.

        Features are computed from deque-based sliding windows that fill
        progressively. Returns 0.0 during cold-start (< 2 samples).
        """
        values = self._extract_telemetry_values(record)

        # Append to history buffers (guard against NaN)
        if 'vibration' in values and np.isfinite(values['vibration']):
            self._vibration_history.append(values['vibration'])
        if 'temperature' in values and np.isfinite(values['temperature']):
            self._temperature_history.append(values['temperature'])
        if 'speed' in values and np.isfinite(values['speed']):
            self._speed_history.append(values['speed'])
        if 'efficiency' in values and np.isfinite(values['efficiency']):
            self._efficiency_history.append(values['efficiency'])

        features = {}

        # vibration_trend_7d: slope of linear fit over vibration history
        if len(self._vibration_history) >= 2:
            x = np.arange(len(self._vibration_history))
            coeffs = np.polyfit(x, list(self._vibration_history), 1)
            features['vibration_trend_7d'] = float(coeffs[0])
        else:
            features['vibration_trend_7d'] = 0.0

        # temp_variation_24h: std deviation over temperature history
        if len(self._temperature_history) >= 2:
            features['temp_variation_24h'] = float(np.std(list(self._temperature_history)))
        else:
            features['temp_variation_24h'] = 0.0

        # speed_stability: coefficient of variation over speed history
        if len(self._speed_history) >= 2:
            arr = np.array(self._speed_history)
            mean_speed = np.mean(arr)
            if mean_speed > 0:
                features['speed_stability'] = float(np.std(arr) / mean_speed)
            else:
                features['speed_stability'] = 0.0
        else:
            features['speed_stability'] = 0.0

        # efficiency_degradation_rate: slope over efficiency history
        if len(self._efficiency_history) >= 2:
            x = np.arange(len(self._efficiency_history))
            coeffs = np.polyfit(x, list(self._efficiency_history), 1)
            features['efficiency_degradation_rate'] = float(coeffs[0])
        else:
            features['efficiency_degradation_rate'] = 0.0

        # Ratio features (from current record)
        if 'discharge_pressure' in record and 'suction_pressure' in record:
            suction = record.get('suction_pressure', 1)
            if suction > 0:
                features['pressure_ratio'] = record['discharge_pressure'] / suction

        # Operating regime
        if 'speed' in record and 'speed_target' in record:
            if record['speed_target'] > 0:
                features['load_factor'] = record['speed'] / record['speed_target']

        return features


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