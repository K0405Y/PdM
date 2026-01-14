# ML Output Modes Module

## Overview

The `ml_output_modes.py` module provides flexible data output formatting for realistic machine learning model training and evaluation. It addresses the critical distinction between training scenarios (where ground-truth health labels are available) and production deployment (where only sensor measurements exist), enabling more accurate ML model validation.

## Purpose

A common pitfall in predictive maintenance ML is training models with access to ground-truth health indicators that would never be available in production. This creates unrealistic performance expectations and models that fail when deployed. This module enables:

- **Realistic Evaluation**: Test models with only production-available data (sensors)
- **Training Data Preparation**: Include ground-truth labels for supervised learning
- **Labeling Lag Simulation**: Model real-world inspection delays (weeks to months)
- **Feature Engineering**: Add derived features for ML pipelines
- **Sensor Noise Injection**: Realistic measurement uncertainty
- **Proper Train/Test Splits**: Temporal and equipment-based splitting to avoid data leakage

**Key Principle**: Models should be trained on features available in production and evaluated under production constraints.

## Key Features

- **4 Output Modes**: Full, sensor-only, delayed labels, derived features
- **Ground-Truth Filtering**: Automatic removal of unmeasurable fields
- **Realistic Sensor Noise**: Add measurement uncertainty typical of industrial sensors
- **Label Delay Simulation**: Configurable inspection/labeling lag
- **Train/Test Splitting**: Temporal, equipment-based, and stratified failure splits
- **Feature Engineering Framework**: Template for derived feature calculation

## Module Components

### OutputMode Enum

Defines four data output modes for different use cases:

```python
class OutputMode(Enum):
    FULL = "full"                      # All data including ground-truth health
    SENSOR_ONLY = "sensor_only"        # Only measurable sensors
    DELAYED_LABELS = "delayed_labels"  # Labels available after delay
    DERIVED_FEATURES = "derived_features"  # Include engineered features
```

### DataOutputFormatter Class

Formats simulation output according to selected mode.

### TrainTestSplitter Class

Utilities for creating proper ML train/test splits.

## Output Modes

### 1. FULL Mode

**Description**: Complete simulation output including all ground-truth health indicators.

**Use Case**: Model training and development

**Includes**:
- All sensor measurements
- Ground-truth health values (`health_bearing`, `health_impeller`, etc.)
- Internal state variables (`operating_mode`, `thermal_stress`, etc.)
- No filtering applied

**Example Output**:
```python
{
    'equipment_id': 1,
    'timestamp': '2024-01-15 10:30:00',
    'speed_rpm': 3000,
    'bearing_temp_de_celsius': 75.5,
    'vibration_rms_mm_s': 2.3,
    'health_bearing': 0.82,          # Ground truth
    'health_impeller': 0.90,          # Ground truth
    'operating_mode': 'steady_state', # Internal state
    'thermal_stress': 0.35            # Internal state
}
```

**When to Use**:
- Model training (supervised learning)
- Feature importance analysis
- Debugging simulation issues
- Algorithm development

**Important**: Never use FULL mode for model evaluation - creates unrealistic performance metrics.

### 2. SENSOR_ONLY Mode

**Description**: Only production-available sensor measurements, with realistic noise.

**Use Case**: Model evaluation, production deployment testing

**Excludes**:
- Ground-truth health indicators
- Internal simulation states
- Derived maintenance labels

**Includes**:
- Temperature sensors (with ±0.5°C noise)
- Pressure sensors (with ±5 kPa noise)
- Speed sensors (with ±10 RPM noise)
- Vibration sensors (with ±0.05 mm/s noise)
- Flow sensors (with ±2.0 m³/hr noise)
- Current sensors (with ±0.5 A noise)

**Example Output**:
```python
{
    'equipment_id': 1,
    'timestamp': '2024-01-15 10:30:00',
    'speed_rpm': 3008.2,              # With noise (true: 3000)
    'bearing_temp_de_celsius': 76.1,  # With noise (true: 75.5)
    'vibration_rms_mm_s': 2.35,       # With noise (true: 2.3)
    # No health fields
    # No internal state fields
}
```

**Sensor Noise Model**:
- Gaussian noise: `measurement = true_value + N(0, σ)`
- Occasional outliers: 1% chance of 3σ noise spike
- Sensor-type specific noise levels (calibrated to industrial sensors)

**When to Use**:
- Model evaluation (final performance metrics)
- Production deployment simulation
- Robustness testing
- Algorithm validation

**Critical**: Use this mode to generate realistic performance expectations for production.

### 3. DELAYED_LABELS Mode

**Description**: Health labels available after configurable time delay, simulating inspection lag.

**Use Case**: Semi-supervised learning, online learning scenarios

**Real-World Scenario**:
- Equipment operates continuously (sensor data collected)
- Periodic inspections occur (weekly, monthly, quarterly)
- Inspection reports processed and labeled (weeks to months delay)
- Health labels become available long after measurement

**Example Timeline**:
```
Day 0: Sensor data collected → available immediately
Day 7: Inspection performed
Day 14: Inspection report filed
Day 21: Health label assigned → available 21 days after measurement
```

**Configuration**:
- `label_delay_hours`: Time delay for label availability (default: 168 hours = 1 week)

**Buffer Management**:
- Records stored in FIFO buffer
- Sensor-only output returned for recent records (labels not yet available)
- Full record (with labels) returned once delay period passes

**Example Usage**:
```python
formatter = DataOutputFormatter(
    output_mode=OutputMode.DELAYED_LABELS,
    label_delay_hours=168  # 1 week delay
)

# Day 0: Record generated
record_day0 = formatter.format_record(raw_record, timestamp_day0)
# Returns sensor-only (labels not ready)

# Day 7: One week later
record_day7 = formatter.format_record(raw_record_day7, timestamp_day7)
# Returns day 0 record with full labels (delay passed)
# Returns day 7 record as sensor-only
```

**When to Use**:
- Online learning algorithm development
- Semi-supervised learning (large unlabeled + small labeled dataset)
- Realistic model update scenarios
- Active learning strategies

**Applications**:
- Continuous model retraining with new inspections
- Label efficiency studies (how many inspections needed?)
- Cost optimization (inspection frequency vs model performance)

### 4. DERIVED_FEATURES Mode

**Description**: Include pre-computed engineered features commonly used in ML pipelines.

**Use Case**: Feature engineering research, AutoML baselines

**Includes**:
- All standard sensor measurements
- Derived features (ratios, trends, statistics)
- Optional ground-truth (configurable)

**Feature Categories**:

**Ratio Features**:
- Pressure ratio: `discharge_pressure / suction_pressure`
- Load factor: `actual_speed / target_speed`
- Efficiency: `actual_power / theoretical_power`

**Trend Features** (require historical window):
- 7-day vibration trend
- 24-hour temperature variation
- 30-day efficiency degradation rate

**Statistical Features** (require historical window):
- Speed stability (coefficient of variation)
- Temperature variability (standard deviation)
- Vibration peak-to-mean ratio

**Operating Regime Features**:
- Load classification: idle/part-load/full-load
- Operating mode: startup/steady/shutdown
- Duty cycle: hours operated / total hours

**Example Output**:
```python
{
    'equipment_id': 1,
    'timestamp': '2024-01-15 10:30:00',
    'speed_rpm': 3000,
    'bearing_temp_de_celsius': 75.5,
    # ... standard sensors ...
    'features': {
        'pressure_ratio': 5.2,
        'load_factor': 0.87,
        'vibration_trend_7d': 0.015,      # mm/s per day
        'temp_variation_24h': 3.2,        # °C std dev
        'speed_stability': 0.02,          # CV
        'efficiency_degradation_rate': -0.001  # per day
    }
}
```

**When to Use**:
- Feature importance studies
- AutoML baseline comparison
- Domain-knowledge feature validation
- Rapid prototyping

**Note**: Full implementation requires historical data tracking. Module provides framework and placeholders.

## DataOutputFormatter Class

### Initialization

```python
def __init__(self,
             output_mode: OutputMode = OutputMode.FULL,
             label_delay_hours: int = 168)
```

**Parameters**:
- `output_mode`: Desired output mode (default: FULL)
- `label_delay_hours`: Delay for DELAYED_LABELS mode (default: 168 hours = 1 week)

### format_record()

Format a single telemetry record according to output mode.

```python
def format_record(self,
                 telemetry_record: Dict,
                 timestamp: datetime) -> Optional[Dict]
```

**Parameters**:
- `telemetry_record`: Raw telemetry from simulator
- `timestamp`: Record timestamp

**Returns**: Formatted record or `None` (if delayed labels not ready and no buffer output)

**Logic**: Dispatches to mode-specific formatter based on `output_mode`

### Field Classification

**Ground Truth Fields** (removed in SENSOR_ONLY):
```python
GROUND_TRUTH_FIELDS = {
    'health_hgp', 'health_blade', 'health_bearing', 'health_fuel',
    'health_impeller', 'health_seal', 'health_seal_primary',
    'health_seal_secondary', 'health_bearing_de', 'health_bearing_nde'
}
```

**Internal State Fields** (removed in SENSOR_ONLY):
```python
INTERNAL_STATE_FIELDS = {
    'operating_mode', 'thermal_stress', 'degradation_multiplier',
    'differential_temp_C'
}
```

### Sensor Noise Model

Realistic measurement noise based on industrial sensor specifications:

| Sensor Type | Noise Std Dev | Typical Range | Relative Accuracy |
|-------------|---------------|---------------|-------------------|
| Temperature | 0.5°C | 0-500°C | 0.1-0.5% |
| Pressure | 5.0 kPa | 0-1000 kPa | 0.5% |
| Speed | 10.0 RPM | 0-10000 RPM | 0.1% |
| Vibration | 0.05 mm/s | 0-20 mm/s | 2-5% |
| Flow | 2.0 m³/hr | 0-1000 m³/hr | 0.2-1% |
| Current | 0.5 A | 0-500 A | 0.1-0.5% |

**Outlier Model**: 1% probability of 3σ spike (simulates transient electrical noise, EMI)

**Implementation**:
```python
noise = np.random.normal(0, noise_std)
if np.random.random() < 0.01:  # 1% chance
    noise *= 3.0  # Outlier
noisy_value = true_value + noise
```

## TrainTestSplitter Class

### Purpose

Proper train/test splitting is critical to avoid data leakage and ensure realistic model evaluation.

**Common Mistakes**:
- Random split across time (future data leaks to training)
- Random split across equipment (same equipment in both train and test)
- Unbalanced failure distribution (all failures in train or test)

### temporal_split()

Split dataset by time (chronological).

```python
@staticmethod
def temporal_split(telemetry_records: List[Dict],
                  train_fraction: float = 0.7) -> tuple
```

**Parameters**:
- `telemetry_records`: Time-ordered telemetry records
- `train_fraction`: Fraction for training (default: 0.7)

**Returns**: `(train_records, test_records)`

**Logic**:
- Split at index `N * train_fraction`
- Training: first 70% chronologically
- Testing: last 30% chronologically

**Use Case**: Time-series forecasting, sequence models

**Example**:
```python
# 10,000 records from Jan 1 to Dec 31
records = [...]  # Chronologically ordered

train, test = TrainTestSplitter.temporal_split(records, train_fraction=0.7)
# train: Jan 1 - Sep 15 (7,000 records)
# test:  Sep 16 - Dec 31 (3,000 records)
```

**Advantage**: Realistic evaluation (predict future from past)

**Disadvantage**: Test set may have different distribution (seasonal effects, long-term trends)

### equipment_based_split()

Split by equipment to avoid data leakage.

```python
@staticmethod
def equipment_based_split(telemetry_records: List[Dict],
                         equipment_ids: List[int],
                         test_equipment: List[int]) -> tuple
```

**Parameters**:
- `telemetry_records`: All telemetry records
- `equipment_ids`: All equipment IDs
- `test_equipment`: Equipment IDs held out for testing

**Returns**: `(train_records, test_records)`

**Logic**:
- Training: records from equipment NOT in `test_equipment`
- Testing: records from equipment in `test_equipment`

**Use Case**: Generalization to new equipment instances

**Example**:
```python
# 100 equipment, hold out 20 for testing
all_equipment = list(range(1, 101))
test_equipment = [81, 82, 83, ..., 100]  # Last 20

train, test = TrainTestSplitter.equipment_based_split(
    records,
    all_equipment,
    test_equipment
)
# train: 80 equipment (all time periods)
# test:  20 equipment (all time periods)
```

**Advantage**: Tests generalization to new equipment

**Disadvantage**: Requires sufficient number of equipment instances

### stratified_failure_split()

Ensure both train and test sets contain failure examples.

```python
@staticmethod
def stratified_failure_split(telemetry_records: List[Dict],
                             failure_records: List[Dict],
                             test_fraction: float = 0.3) -> tuple
```

**Parameters**:
- `telemetry_records`: All telemetry
- `failure_records`: Failure events
- `test_fraction`: Fraction for test set (default: 0.3)

**Returns**: `(train_telemetry, test_telemetry, train_failures, test_failures)`

**Logic**:
1. Identify equipment with failures vs healthy equipment
2. Split failed equipment to ensure both sets have failures
3. Split healthy equipment proportionally
4. Distribute telemetry according to equipment assignments

**Use Case**: Imbalanced datasets, rare failure events

**Example**:
```python
# 100 equipment: 10 failed, 90 healthy
# Ensure both train and test have failure examples

train_t, test_t, train_f, test_f = TrainTestSplitter.stratified_failure_split(
    telemetry,
    failures,
    test_fraction=0.3
)

# Result:
# train: 70 equipment (7 failed, 63 healthy)
# test:  30 equipment (3 failed, 27 healthy)
# Both sets have failure examples for model training/evaluation
```

**Advantage**: Balanced failure representation in both sets

**Disadvantage**: Requires multiple equipment with failures

## Usage Examples

### Training vs Evaluation

```python
from ml_output_modes import DataOutputFormatter, OutputMode
from gas_turbine import GasTurbine

turbine = GasTurbine(name='GT-001')

# Generate training data (with labels)
formatter_train = DataOutputFormatter(output_mode=OutputMode.FULL)
training_data = []

for i in range(10000):
    raw = turbine.next_state()
    formatted = formatter_train.format_record(raw, datetime.now())
    training_data.append(formatted)

# Training data includes health labels - use for model training
# train_model(training_data)

# Generate evaluation data (realistic - no labels)
formatter_eval = DataOutputFormatter(output_mode=OutputMode.SENSOR_ONLY)
evaluation_data = []

turbine2 = GasTurbine(name='GT-002')
for i in range(5000):
    raw = turbine2.next_state()
    formatted = formatter_eval.format_record(raw, datetime.now())
    evaluation_data.append(formatted)

# Evaluation data has no health labels - realistic production scenario
# predictions = model.predict(evaluation_data)
```

### Delayed Label Simulation

```python
from ml_output_modes import DataOutputFormatter, OutputMode
from datetime import datetime, timedelta

# Simulate online learning with weekly inspections
formatter = DataOutputFormatter(
    output_mode=OutputMode.DELAYED_LABELS,
    label_delay_hours=168  # 1 week
)

current_time = datetime.now()
labeled_count = 0
unlabeled_count = 0

for day in range(365):  # 1 year simulation
    current_time += timedelta(days=1)

    # Generate daily telemetry
    raw_record = turbine.next_state()
    formatted = formatter.format_record(raw_record, current_time)

    if 'health_bearing' in formatted:
        # Label available (inspection completed)
        labeled_count += 1
        # Use for model update
    else:
        # Label not yet available
        unlabeled_count += 1
        # Use for prediction only

print(f"Labeled: {labeled_count}, Unlabeled: {unlabeled_count}")
# Result: ~52 labeled (weekly inspections), ~313 unlabeled
```

### Feature Engineering Pipeline

```python
from ml_output_modes import DataOutputFormatter, OutputMode

formatter = DataOutputFormatter(output_mode=OutputMode.DERIVED_FEATURES)

# Generate data with derived features
feature_data = []
for i in range(10000):
    raw = turbine.next_state()
    formatted = formatter.format_record(raw, datetime.now())

    # Extract features for ML
    features = formatted['features']
    X = [
        formatted['speed_rpm'],
        formatted['bearing_temp_de_celsius'],
        features['pressure_ratio'],
        features['load_factor'],
        # ... more features
    ]

    y = formatted.get('health_bearing', None)  # Label if available

    feature_data.append((X, y))

# Use for model training
```

### Proper Train/Test Split

```python
from ml_output_modes import TrainTestSplitter

# Scenario: 50 equipment, 365 days each, 5 failures

# Temporal split (within-equipment prediction)
train, test = TrainTestSplitter.temporal_split(
    all_telemetry,
    train_fraction=0.7
)
# Train on first 255 days, test on last 110 days

# Equipment split (generalization to new equipment)
test_equipment_ids = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
train, test = TrainTestSplitter.equipment_based_split(
    all_telemetry,
    equipment_ids=list(range(1, 51)),
    test_equipment=test_equipment_ids
)
# Train on 40 equipment, test on 10 equipment

# Stratified split (ensure failures in both sets)
train_t, test_t, train_f, test_f = TrainTestSplitter.stratified_failure_split(
    all_telemetry,
    failure_records,
    test_fraction=0.3
)
# Ensure both train and test have failure examples
print(f"Train failures: {len(train_f)}, Test failures: {len(test_f)}")
```

## Best Practices

### Output Mode Selection

**Model Development**:
1. Use FULL mode for training
2. Use SENSOR_ONLY mode for evaluation
3. Report both performances (shows generalization gap)

**Production Deployment**:
1. Use SENSOR_ONLY exclusively
2. Never use health indicators as features
3. Validate with delayed labels when available

**Research**:
1. Use DERIVED_FEATURES to test feature importance
2. Compare domain features vs learned features
3. Use DELAYED_LABELS to study label efficiency

### Train/Test Splitting Strategy

**For Time-Series Models**:
- Use temporal split
- Respect temporal order
- Consider seasonality

**For Equipment Generalization**:
- Use equipment-based split
- Test on unseen equipment
- Validate across different operating conditions

**For Rare Failures**:
- Use stratified split
- Ensure failure representation
- Consider class imbalance methods

**General Rule**: Combine strategies
- Split by equipment (avoid leakage)
- Then split temporally within equipment (respect time)
- Stratify failures across equipment groups

### Sensor Noise Handling

**Training**:
- Include noise in training data (improves robustness)
- Use moderate noise levels (match production sensors)
- Consider data augmentation with varied noise

**Evaluation**:
- Test with realistic noise levels
- Include outlier scenarios (1% spike rate)
- Validate robustness to sensor failures (missing values)

### Feature Engineering

**Domain Features** (DERIVED_FEATURES mode):
- Start with physics-based features (pressure ratio, efficiency)
- Add temporal features (trends, variability)
- Include operating regime indicators

**Learned Features** (deep learning):
- Use raw sensors as input
- Let model learn features
- Compare to domain features

**Hybrid Approach**:
- Combine domain and learned features
- Use domain features for interpretability
- Use learned features for performance

## Validation Guidelines

### Model Performance Reporting

Always report performance on SENSOR_ONLY mode:

```python
# Training performance (optimistic)
train_performance = evaluate_model(model, training_data_FULL)

# Realistic performance (production expectation)
eval_performance = evaluate_model(model, evaluation_data_SENSOR_ONLY)

print(f"Training accuracy: {train_performance['accuracy']:.3f}")
print(f"Production accuracy: {eval_performance['accuracy']:.3f}")
print(f"Generalization gap: {train_performance['accuracy'] - eval_performance['accuracy']:.3f}")
```

**Key Metrics**:
- Accuracy on SENSOR_ONLY mode (realistic)
- Precision/Recall for failure prediction
- False positive rate (cost of unnecessary maintenance)
- Early detection time (hours before failure)

### Cross-Validation Strategy

**Time-Series Cross-Validation**:
```python
# Rolling window approach
for fold in range(5):
    train_end = start + timedelta(days=250)
    test_start = train_end
    test_end = test_start + timedelta(days=50)

    train = records[start:train_end]
    test = records[test_start:test_end]

    model.fit(train)
    score = model.evaluate(test)
```

**Equipment-Based Cross-Validation**:
```python
# K-fold across equipment
from sklearn.model_selection import GroupKFold

splitter = GroupKFold(n_splits=5)
for train_idx, test_idx in splitter.split(X, y, groups=equipment_ids):
    train_data = data[train_idx]
    test_data = data[test_idx]

    model.fit(train_data)
    score = model.evaluate(test_data)
```

## Limitations and Extensions

### Current Limitations

1. **Static Noise Model**: Fixed noise levels, no sensor degradation
2. **No Missing Data**: Real sensors fail, producing NaN values
3. **Feature Placeholders**: Derived features need historical window implementation
4. **No Sensor Drift**: Real sensors drift over time, requiring recalibration
5. **Fixed Delay**: DELAYED_LABELS uses constant delay, not variable inspection schedules

### Potential Enhancements

1. **Dynamic Sensor Degradation**:
   - Increase noise over time (sensor aging)
   - Model sensor failures (missing data)
   - Calibration drift

2. **Advanced Noise Models**:
   - Correlated noise (electromagnetic interference)
   - Temperature-dependent noise
   - Systematic bias

3. **Historical Feature Calculation**:
   - Implement rolling window statistics
   - Trend calculation from historical data
   - Frequency-domain features (FFT)

4. **Variable Label Delays**:
   - Random inspection intervals
   - Condition-triggered inspections
   - Resource-constrained labeling

5. **Data Quality Indicators**:
   - Sensor health status
   - Calibration timestamps
   - Data quality flags

6. **Multi-Modal Data**:
   - Images from inspections
   - Acoustic signatures
   - Oil analysis results

## References

1. Schlachter, G., et al. (2020). "On the Importance of Time Series Splitting for Machine Learning." IEEE Access.

2. Saxena, A., et al. (2008). "Metrics for Offline Evaluation of Prognostic Performance." International Journal of Prognostics and Health Management.

3. Settles, B. (2012). "Active Learning." Synthesis Lectures on Artificial Intelligence and Machine Learning.

4. Emmert-Streib, F., et al. (2019). "An Introductory Review of Deep Learning for Prediction Models With Big Data." Frontiers in Artificial Intelligence.

5. ISO 13374-2:2007 - Condition monitoring and diagnostics of machines -- Data processing, communication and presentation -- Part 2: Data processing

## See Also

- [pipeline_enhanced.md](pipeline_enhanced.md) - High-performance data generation pipeline
- [data_pipeline.md](data_pipeline.md) - Standard data ingestion and storage
- Feature engineering modules (time_domain_features.py, frequency_domain_features.py)
