# Gas Turbine Simulator Module

## Overview

The `gas_turbine.py` module simulates an industrial gas turbine typical of offshore platforms and LNG facilities, providing realistic telemetry data for predictive maintenance machine learning systems. It implements physics-based degradation models, thermodynamic performance simulation, and integrates with all enhancement modules for comprehensive equipment behavior modeling.

## Purpose

Gas turbines are critical power generation assets in oil & gas operations. This simulator enables:

- **ML Training Data Generation**: Run-to-failure datasets with realistic degradation signatures
- **Algorithm Development**: Test predictive maintenance algorithms before production deployment
- **Edge Case Coverage**: Process upsets, incipient faults, and maintenance events
- **Multi-Mode Degradation**: Hot gas path, blade erosion, bearing wear, fuel system fouling
- **Performance Degradation**: Thermodynamic efficiency loss with component health

## Architecture Overview

The gas turbine simulator follows a **modular enhancement architecture**:

1. **Core Simulator** (`GasTurbine` class): Base equipment physics and degradation
2. **Enhancement Modules** (optional): Physics, simulation, and ML utilities
3. **Integration Layer**: Conditional activation of enhancements based on initialization flags

## Key Features

### Core Features (Always Available)

- **Multi-Mode Degradation**: 4 independent failure modes with exponential wear curves
- **Thermodynamic Modeling**: EGT, compressor discharge, fuel flow with efficiency loss
- **Vibration Signatures**: Fault-specific harmonic content based on component health
- **Operating Envelope**: Speed (3K-15K RPM), EGT (400-620°C), vibration limits (API 670)
- **Time-to-Failure Tracking**: Operating hours and health progression

### Enhancement Features (Optional)

- **Advanced Vibration**: Bearing defect frequencies with envelope modulation
- **Thermal Transients**: Startup/shutdown stress with mode-specific degradation multipliers
- **Environmental Variability**: Temperature, pressure, humidity, salt/dust exposure
- **Maintenance Events**: Routine, minor, major overhauls with imperfect restoration
- **Incipient Faults**: Discrete fault initiation (cracks, spalls, contamination)
- **Process Upsets**: Abnormal operating conditions (overload, trips, thermal shocks)
- **Output Formatting**: Training vs evaluation data modes

## Module Components

### GasTurbineHealthModel Class

Manages four independent degradation pathways using generalized wear equation.

**Degradation Model**:
```
h(t) = 1 - d - exp(a * t^b)
```

Where:
- `h(t)`: Health at time t (0.0 = failed, 1.0 = new)
- `d`: Asymptotic wear offset
- `a`: Degradation rate coefficient
- `b`: Degradation acceleration exponent

**Health Range Constraint**:

The degradation formula has a mathematical constraint: maximum possible health is `1 - d`. For example, with `d = 0.04`, the maximum health is 0.96. If initial health exceeds this limit, the standard time-to-failure (ttf) calculation fails because `log(1 - d - h)` produces a negative argument.

**Hybrid Health Generator**:

To handle cases where initial health exceeds the formula's maximum (e.g., initial fuel health of 0.98 when d=0.04 limits max to 0.96), the health model uses a hybrid approach:

1. **Phase 1 (Linear Degradation)**: When `initial_health > 1 - d`, health degrades linearly at a slow rate until it reaches the formula's valid range
2. **Phase 2 (Exponential Degradation)**: Once health is within the valid range, the standard exponential formula takes over

```python
def _hybrid_health_generator(initial_health, max_formula_health, d, a, b, threshold):
    linear_rate = 0.00002  # Per timestep
    current_health = initial_health

    # Phase 1: Linear degradation until within formula range
    while current_health > max_formula_health:
        if current_health < threshold:
            return
        yield t_virtual, current_health
        current_health -= linear_rate * (1.0 + random.gauss(0, 0.05))

    # Phase 2: Switch to exponential formula
    transition_health = max_formula_health - 0.001
    ttf = math.pow(math.log(1 - d - transition_health) / a, 1 / b)
    for t in range(int(ttf), -1, -1):
        h = 1 - d - math.exp(a * t**b)
        if h < threshold:
            break
        yield t, h
```

**Failure Modes**:

| Mode | Description | Parameters (d, a, b) | Threshold | Typical Lifespan |
|------|-------------|----------------------|-----------|------------------|
| HGP (Hot Gas Path) | Combustion liner cracking | (0.05, -0.25, 0.22) | 0.45 | 8,000-12,000 hrs |
| Blade | Leading edge erosion | (0.03, -0.30, 0.20) | 0.40 | 10,000-15,000 hrs |
| Bearing | Lubrication/mechanical | (0.08, -0.35, 0.25) | 0.35 | 5,000-10,000 hrs |
| Fuel | Nozzle fouling | (0.04, -0.20, 0.18) | 0.50 | 12,000-18,000 hrs |

**Operating Severity Multiplier**:
```python
severity = 1.0

# Speed penalty (above rated)
if speed_factor > 1.0:
    severity *= (1.0 + 0.5 * (speed_factor - 1.0)**2)

# Temperature penalty (above rated)
if temp_factor > 1.0:
    severity *= (1.0 + 0.3 * (temp_factor - 1.0)**2)
```

### VibrationSignalGenerator Class

Generates realistic vibration signals with fault-specific harmonic content.

**Base Harmonics (Healthy)**:
- 1X (shaft frequency): 0.3 mm/s
- 2X: 0.15 mm/s
- 3X: 0.05 mm/s

**Fault Signatures**:

| Fault Type | Harmonics | Amplitude Multiplier | Trigger Condition |
|------------|-----------|----------------------|-------------------|
| Unbalance | 1X | 3.0x | HGP health < 0.8 |
| Misalignment | 2X | 2.5x | - |
| Blade Rub | 1X, 3X, 5X, 7X | 2.0x | Blade health < 0.75 |
| Bearing Defect | 3.5X, 7X, 10.5X | 1.5x | Bearing health < 0.7 |

**Noise Floor**: Gaussian noise, σ = 0.05 mm/s (instrumentation noise)

### GasTurbine Class

Main simulator class integrating all components.

**Operating Limits** (API 670, Industry Standards):

| Parameter | Min | Max | Rated/Nominal | Alarm | Trip |
|-----------|-----|-----|---------------|-------|------|
| Speed (RPM) | 3,000 | 15,000 | 9,500 | - | - |
| EGT (°C) | 400 | 620 | 520 | 600 | 620 |
| Vibration (mm/s) | - | - | 0.5-1.0 | 2.2 | 3.0 |
| Oil Temperature (°C) | 70 | 130 | 95 | 120 | 130 |
| Fuel Flow (kg/s) | 0.5 | 3.8 | 2.5 | - | - |

## Integration with Enhancement Modules

### 1. Physics Module Integration

#### Enhanced Vibration (physics.vibration_enhanced)

**When Enabled** (`enable_enhanced_vibration=True`):
- Replaces base vibration generator
- Uses `EnhancedVibrationGenerator` with bearing geometry
- Provides: RMS, peak, crest factor, kurtosis
- Includes envelope modulation for bearing defects

**Integration Point** (line 620-647):
```python
if self.use_enhanced_vibration:
    vib_signal, vib_metrics = self.vib_generator_enhanced.generate_bearing_vibration(
        rpm=self.speed,
        bearing_health=health_state.get('bearing', 1.0),
        duration=1.0
    )
    vib_rms = vib_metrics.get('rms', 0)
    vib_crest = vib_metrics.get('crest_factor', 0)
    vib_kurtosis = vib_metrics.get('kurtosis', 0)
```

**Benefit**: Realistic bearing fault progression with ISO 10816 severity alignment

#### Thermal Transients (physics.thermal_transient)

**When Enabled** (`enable_thermal_transients=True`):
- Tracks operating mode (startup, loading, steady, shutdown)
- Applies mode-specific degradation multipliers
- Startup: 2.5x degradation
- Loading: 1.4x degradation

**Integration Point** (line 558-570):
```python
if self.use_thermal_model:
    thermal_state = self.thermal_model.step(
        target_speed=self.speed_target,
        rated_speed=self.LIMITS['speed_rated'],
        timestep_minutes=1/60
    )
    thermal_multiplier = thermal_state.get('degradation_multiplier', 1.0)
    severity *= thermal_multiplier  # Increases degradation during transients
```

**Benefit**: Captures accelerated wear during startups/shutdowns (30-40% of total damage)

#### Environmental Conditions (physics.environmental_conditions)

**When Enabled** (`enable_environmental=True`):
- Modifies ambient temperature and pressure
- Seasonal and diurnal variations
- Eight location-specific profiles available
- Supports both synthetic and real weather data

**Location Profile Selection:**

| Location Type | Region | Key Characteristics | Use Cases |
|--------------|--------|---------------------|-----------|
| OFFSHORE | Marine platforms | High salt (0.9), moderate temp (15°C) | North Sea (UK/Norway), Gulf of Mexico (US), West Africa offshore |
| DESERT | Arid regions | Extreme dust (0.95), high temp swings (30°C ±15°C) | Saudi Arabia, UAE, Kuwait, Libya, Algeria |
| ARCTIC | Polar regions | Extreme cold (-15°C), high ice risk (0.95) | Russia (Yamal), Alaska, Northern Canada |
| TROPICAL | Equatorial | High humidity (85%), minimal seasons (28°C) | Indonesia, Malaysia, Nigeria (coastal), Gabon |
| TEMPERATE | Mid-latitudes | 4-season pattern (12°C), balanced | USA, UK, Germany, China |
| SAHEL | West Africa | High dust (0.80), Harmattan season (30°C) | Nigeria (north), Chad, Niger, Sudan |
| SAVANNA | Semi-arid Africa | Moderate dust (0.5), Southern Hemisphere (25°C) | South Africa, Angola, Mozambique |

**Integration Point** (line 379-398):
```python
if self.use_environmental:
    env_cond = self.env_model.get_conditions(self.elapsed_hours)
    self.ambient_temp = env_cond.get('ambient_temp_C', self.ambient_temp)
    self.ambient_pressure = env_cond.get('pressure_kPa', self.ambient_pressure)
```

**Benefit**: Realistic environmental variability affects turbine performance and degradation

**Two Integration Methods:**

**Method 1: Synthetic Location Profile** (pass `location_type`):
```python
from gas_turbine import GasTurbine
from physics.environmental_conditions import LocationType

gt = GasTurbine(
    name='GT-001',
    location_type=LocationType.SAHEL,  # Synthetic Sahel climate
    enable_environmental=True
)
```

**Method 2: Real Weather API** (pass `env_model`):
```python
from gas_turbine import GasTurbine
from physics.weather_api_client import create_hybrid_environment

# Create real weather source
env_source = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_key",
    location_name="Lagos",
    country="Nigeria",
    cache_enabled=True
)

# Pass directly to constructor - seamless integration!
gt = GasTurbine(
    name='GT-001',
    env_model=env_source,  # Real weather from Lagos, Nigeria
    enable_environmental=True
)
```

See [weather_api_client.md](weather_api_client.md) and [environmental_conditions.md](environmental_conditions.md) for detailed documentation.

### 2. Simulation Module Integration

#### Maintenance Scheduler (simulation.maintenance_events)

**When Enabled** (`enable_maintenance=True`):
- Checks maintenance requirements (time-based, condition-based, opportunistic)
- Performs maintenance with imperfect restoration
- Tracks maintenance history

**Integration Point** (line 654-673):
```python
if self.use_maintenance:
    maint_type = self.maint_scheduler.check_maintenance_required(
        operating_hours=self.operating_hours,
        health_state=health_state,
        is_planned_shutdown=(self.speed == 0)
    )

    if maint_type:
        maint_action = self.maint_scheduler.perform_maintenance(
            maint_type,
            current_health=health_state,
            operating_hours=self.operating_hours,
            timestamp=self.current_timestamp
        )
        # Restore health (imperfect - quality factor dependent)
        self.health_model.health = maint_action.health_after
        health_state = maint_action.health_after
```

**Benefit**: Realistic maintenance cycles with infant mortality and restoration variability

#### Incipient Fault Simulator (simulation.incipient_faults)

**When Enabled** (`enable_incipient_faults=True`):
- Checks for discrete fault initiation (Poisson process)
- Propagates active faults (Paris law for cracks, debris feedback for spalls)
- Adjusts component health based on fault size

**Integration Point** (line 573-605):
```python
if self.use_faults:
    # Check initiation (stress-dependent Poisson)
    fault_event = self.fault_sim.check_fault_initiation(
        operating_hours_increment=1/3600,
        stress_factor=severity,
        timestamp=self.current_timestamp,
        operating_hours=self.operating_hours,
        component_list=['hgp', 'blade', 'bearing', 'fuel']
    )

    # Propagate existing faults
    self.fault_sim.propagate_faults(1/3600, severity)

    # Adjust health for fault impact
    health_state = self.fault_sim.adjust_health_for_faults(health_state)
```

**Benefit**: Captures localized defects (70-80% of real failures) vs uniform degradation

#### Process Upset Simulator (simulation.process_upsets)

**When Enabled** (`enable_process_upsets=True`):
- Checks for upset initiation (Poisson process)
- Applies upset effects to sensor readings
- Calculates upset damage to components

**Integration Point** (line 588-614):
```python
if self.use_upsets:
    # Check for upset initiation
    upset_event = self.upset_sim.check_upset_initiation(
        timestep_seconds=1,
        timestamp=self.current_timestamp,
        operating_state={'speed': self.speed, 'egt': self.egt}
    )

    # Apply upset damage
    if self.upset_sim.active_upset:
        health_state = self.upset_sim.calculate_upset_damage(health_state)
```

**Benefit**: Edge case scenarios (30-40% of unplanned downtime originates from upsets)

### 3. ML Utilities Integration

#### Data Output Formatter (ml_utils.ml_output_modes)

**When Enabled** (`output_mode` parameter set):
- Formats telemetry based on mode (FULL, SENSOR_ONLY, DELAYED_LABELS, DERIVED_FEATURES)
- Removes ground-truth for evaluation
- Adds realistic sensor noise

**Integration Point** (line 736-740):
```python
if self.use_output_formatter:
    state = self.output_formatter.format_record(state, self.current_timestamp)
```

**Benefit**: Proper train/test separation, realistic evaluation conditions

## Thermodynamic Model

### Efficiency Calculation

Equipment efficiency degrades with blade and HGP health:

```python
efficiency = 0.85 + 0.15 * (blade_health * hgp_health)
```

Range: 0.85 (new) → 0.70 (heavily degraded, threshold ~0.47)

### Exhaust Gas Temperature (EGT)

EGT increases with load and degradation:

```python
# Base EGT from load
load_fraction = speed / speed_rated
base_egt = egt_min + (egt_nominal - egt_min) * load_fraction

# Degradation penalty (higher firing temps for same output)
egt_penalty = (2.0 - hgp_health - blade_health) * 30  # Up to 60°C

target_egt = base_egt + egt_penalty
```

### Fuel Flow

Fuel flow increases with load and inversely with efficiency:

```python
base_fuel = fuel_min + (fuel_max - fuel_min) * load_fraction
actual_fuel = base_fuel / (efficiency * fuel_health)
```

### Oil Temperature

Oil temperature correlates with load and bearing health (friction heating):

```python
target_oil = oil_min + (oil_nominal - oil_min) * load_fraction
target_oil += (1.0 - bearing_health) * 25  # Friction penalty
```

## Usage Examples

### Basic Simulation

```python
from gas_turbine import GasTurbine

# Create turbine with default settings
turbine = GasTurbine(
    name='GT-001',
    initial_health={
        'hgp': 0.92,
        'blade': 0.95,
        'bearing': 0.90,
        'fuel': 0.93
    }
)

# Set operating speed
turbine.set_speed(9500)  # Rated speed

# Run simulation
for i in range(3600):  # 1 hour at 1-second intervals
    try:
        state = turbine.next_state()
        print(f"Speed: {state['speed']} RPM, EGT: {state['exhaust_gas_temp']}°C")
    except Exception as e:
        print(f"Failure: {e}")
        break
```

### Simulation with All Enhancements (Synthetic Weather)

```python
from gas_turbine import GasTurbine
from physics.environmental_conditions import LocationType
from ml_utils.ml_output_modes import OutputMode

turbine = GasTurbine(
    name='GT-001',
    initial_health={'hgp': 0.85, 'blade': 0.90, 'bearing': 0.80, 'fuel': 0.88},
    ambient_temp=15.0,
    ambient_pressure=101.3,

    # Enable enhancements with synthetic location
    location_type=LocationType.SAHEL,  # West African Harmattan conditions
    enable_enhanced_vibration=True,
    enable_thermal_transients=True,
    enable_environmental=True,
    enable_maintenance=True,
    enable_incipient_faults=True,
    enable_process_upsets=True,
    output_mode=OutputMode.FULL  # Training data
)

turbine.set_speed(9500)

for i in range(3600):
    state = turbine.next_state()

    # State includes enhancement data
    if 'vibration_crest_factor' in state:
        print(f"Enhanced vibration: CF={state['vibration_crest_factor']:.2f}")

    if 'upset_active' in state and state['upset_active']:
        print(f"Upset: {state['upset_type']}, severity={state['upset_severity']:.2f}")

    if 'num_active_faults' in state and state['num_active_faults'] > 0:
        print(f"Active faults: {state['num_active_faults']}")
```

### Simulation with Real Weather API

```python
from gas_turbine import GasTurbine
from physics.weather_api_client import create_hybrid_environment
from physics.environmental_conditions import EnvironmentalConditions, LocationType
from ml_utils.ml_output_modes import OutputMode
from datetime import datetime, timedelta

# Create hybrid environment with real weather and synthetic fallback
fallback = EnvironmentalConditions(LocationType.TROPICAL)
env_source = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key_here",
    location_name="Port Harcourt",  # Nigerian coastal installation
    country="Nigeria",
    fallback_source=fallback,  # Falls back to synthetic Tropical on API failure
    cache_enabled=True
)

turbine = GasTurbine(
    name='GT-PH-001',
    initial_health={'hgp': 0.85, 'blade': 0.90, 'bearing': 0.80, 'fuel': 0.88},

    # Pass real weather directly via env_model parameter
    env_model=env_source,  # Real weather from Port Harcourt, Nigeria
    enable_enhanced_vibration=True,
    enable_thermal_transients=True,
    enable_environmental=True,
    enable_maintenance=True,
    enable_incipient_faults=True,
    enable_process_upsets=True,
    output_mode=OutputMode.FULL
)

turbine.set_speed(9500)

# Simulate 30 days with real weather
start_time = datetime(2025, 1, 1)
for hour in range(30 * 24):  # 30 days, hourly
    timestamp = start_time + timedelta(hours=hour)

    # Real weather is automatically fetched and cached
    state = turbine.next_state()

    # Log real weather conditions
    if hour % 24 == 0:  # Daily log
        day = hour // 24
        print(f"Day {day}: Temp={state.get('ambient_temp', 'N/A')}°C, "
              f"EGT={state['exhaust_gas_temp']:.1f}°C, "
              f"Efficiency={state['efficiency']:.3f}")
```

### Run-to-Failure Dataset Generation

```python
from gas_turbine import generate_turbine_dataset

telemetry, failures = generate_turbine_dataset(
    n_machines=10,              # 10 turbines
    n_cycles_per_machine=100,   # 100 operating cycles each
    cycle_duration_range=(60, 300),  # 1-5 minutes per cycle
    random_seed=42
)

print(f"Generated {len(telemetry)} telemetry records")
print(f"Captured {len(failures)} failures")

# Failure distribution
for failure in failures:
    print(f"{failure['machineID']}: {failure['code']} - {failure['message']}")
```

### Training vs Evaluation Data

```python
from gas_turbine import GasTurbine
from ml_utils.ml_output_modes import OutputMode

# Training data (with ground-truth health)
turbine_train = GasTurbine('GT-Train', output_mode=OutputMode.FULL)
train_data = []

for i in range(10000):
    state = turbine_train.next_state()
    # state includes 'health_hgp', 'health_blade', etc.
    train_data.append(state)

# Evaluation data (sensor-only, realistic)
turbine_eval = GasTurbine('GT-Eval', output_mode=OutputMode.SENSOR_ONLY)
eval_data = []

for i in range(5000):
    state = turbine_eval.next_state()
    # state excludes health indicators, includes sensor noise
    eval_data.append(state)

# Train model on train_data (has labels)
# Evaluate model on eval_data (no labels - realistic production scenario)
```

## Telemetry Output

### Standard Fields (Always Present)

| Field | Type | Range | Units | Description |
|-------|------|-------|-------|-------------|
| speed | float | 0-15000 | RPM | Current rotor speed |
| speed_target | float | 0-15000 | RPM | Target speed setpoint |
| exhaust_gas_temp | float | 400-620 | °C | Turbine exhaust temperature |
| oil_temp | float | 70-130 | °C | Lube oil temperature |
| fuel_flow | float | 0.5-3.8 | kg/s | Fuel mass flow rate |
| vibration_rms | float | 0-3.0 | mm/s | RMS vibration velocity |
| vibration_peak | float | 0-10 | mm/s | Peak vibration velocity |
| compressor_discharge_temp | float | 100-450 | °C | Compressor outlet temp |
| compressor_discharge_pressure | float | 1000-1500 | kPa | Compressor outlet pressure |
| ambient_temp | float | -30-50 | °C | Ambient air temperature |
| ambient_pressure | float | 90-105 | kPa | Ambient air pressure |
| efficiency | float | 0.70-0.85 | - | Thermal efficiency |
| operating_hours | float | 0+ | hrs | Cumulative operating hours |
| health_hgp | float | 0-1 | - | Hot gas path health |
| health_blade | float | 0-1 | - | Blade health |
| health_bearing | float | 0-1 | - | Bearing health |
| health_fuel | float | 0-1 | - | Fuel system health |

### Enhancement Fields (Conditional)

| Field | Condition | Type | Description |
|-------|-----------|------|-------------|
| vibration_crest_factor | Enhanced vibration | float | Crest factor (peak/RMS) |
| vibration_kurtosis | Enhanced vibration | float | Signal kurtosis (bearing defect indicator) |
| num_active_faults | Incipient faults | int | Count of active faults |
| total_faults_initiated | Incipient faults | int | Total faults initiated |
| upset_active | Process upsets | bool | Upset currently active |
| upset_type | Process upsets | str | Type of upset (if active) |
| upset_severity | Process upsets | float | Upset severity 0-1 (if active) |

## Failure Modes

### Failure Detection

Equipment raises exception when component health drops below threshold:

```python
try:
    state = turbine.next_state()
except Exception as e:
    failure_code = str(e)
    # F_HGP, F_BLADE, F_BEARING, F_FUEL, F_VIB_TRIP
```

### Failure Mode Catalog

| Code | Description | Typical Cause | Threshold | Detection |
|------|-------------|---------------|-----------|-----------|
| F_HGP | Hot gas path degradation | Combustion liner cracking, thermal fatigue | 0.45 | Health monitoring |
| F_BLADE | Blade erosion | Leading edge degradation, FOD | 0.40 | Health monitoring |
| F_BEARING | Bearing failure | Lubrication failure, mechanical wear | 0.35 | Health monitoring |
| F_FUEL | Fuel system fouling | Nozzle blockage, contamination | 0.50 | Health monitoring |
| F_VIB_TRIP | Vibration trip | Excessive vibration > 3.0 mm/s | - | Trip condition |

## Performance Considerations

### Computational Cost

- **Base simulation**: ~0.1 ms per timestep
- **With all enhancements**: ~0.5 ms per timestep
- **Memory**: ~5 KB per turbine instance

### Timestep Recommendations

- **High fidelity**: 1 second (captures transients, vibration evolution)
- **Standard**: 10 seconds (good balance)
- **Coarse**: 60 seconds (long-term trends only)

### Dataset Generation

**Example**: 10 turbines, 180 days, 10-second sampling
- Records: ~1.5M per turbine = 15M total
- Generation time: ~2 hours (sequential) or ~20 minutes (parallel with 8 cores)
- Storage: ~5 GB uncompressed

## Validation and Calibration

### Operating Envelope Validation

Operating limits based on:
- API 670: Vibration monitoring standards
- OEM specifications: GE LM2500, Siemens SGT-750
- Industry operational data: Offshore platform installations

### Degradation Rate Calibration

Degradation parameters calibrated from:
- OEM maintenance intervals (HGP inspection every 8,000-12,000 hours)
- Failure mode distributions from OREDA database
- Hot gas path component life studies (Boyce, 2011)

### Vibration Signature Validation

Vibration patterns validated against:
- Bently Nevada case studies
- ISO 10816 severity standards
- Turbomachinery vibration handbook (Eisenmann & Eisenmann)

## Best Practices

### Initial Health Selection

**Health Range Constraints**:

Each component has a maximum initial health based on the degradation formula's asymptotic offset `d`:
- `max_health = 1 - d`

| Component | d value | Max Health | Recommended Range |
|-----------|---------|------------|-------------------|
| HGP | 0.05 | 0.95 | 0.50 - 0.95 |
| Blade | 0.03 | 0.97 | 0.45 - 0.97 |
| Bearing | 0.08 | 0.92 | 0.40 - 0.92 |
| Fuel | 0.04 | 0.96 | 0.55 - 0.96 |

**Note**: If initial health exceeds these limits, the hybrid health generator automatically uses linear degradation until health enters the valid formula range.

**New Equipment** (commissioned < 1 year):
```python
initial_health = {'hgp': 0.95, 'blade': 0.97, 'bearing': 0.92, 'fuel': 0.96}
```

**Mid-Life** (2-4 years operation):
```python
initial_health = {'hgp': 0.80, 'blade': 0.85, 'bearing': 0.75, 'fuel': 0.82}
```

**Approaching Overhaul** (4-6 years):
```python
initial_health = {'hgp': 0.60, 'blade': 0.70, 'bearing': 0.55, 'fuel': 0.68}
```

### Enhancement Selection

**For Training Data**:
- Enable all enhancements
- Use `OutputMode.FULL` for labels

**For Algorithm Testing**:
- Enable enhancements of interest
- Use `OutputMode.SENSOR_ONLY` for evaluation

**For Baseline Comparison**:
- Disable all enhancements
- Compare performance with/without physics models

### Location Type Selection

| Location | Typical Application | Key Characteristics |
|----------|---------------------|---------------------|
| OFFSHORE | Platform, FPSO | Salt corrosion, humidity, moderate temps |
| DESERT | Middle East onshore | Extreme temps, dust, low humidity |
| ARCTIC | Alaska, Northern Canada | Extreme cold, icing risk |
| TROPICAL | SE Asia, equatorial | High humidity, monsoons |
| TEMPERATE | North America, Europe | Seasonal variation, moderate |

## Limitations and Extensions

### Current Limitations

1. **Single-Spool Model**: Does not model multi-spool turbines (HP/LP spools)
2. **Simplified Thermodynamics**: No detailed cycle analysis (compressor maps, turbine maps)
3. **No Control System**: Speed control is simplified exponential approach
4. **Fixed Degradation Paths**: Cannot model component replacement between failures
5. **No Auxiliary Systems**: Generator, gearbox, fuel system components not modeled

### Potential Enhancements

1. **Multi-Spool Modeling**:
   - Separate HP and LP spool dynamics
   - Inter-spool coupling and surge

2. **Advanced Thermodynamics**:
   - Detailed compressor/turbine performance maps
   - Off-design performance prediction
   - Heat rate calculations

3. **Control System**:
   - PID speed controller
   - Acceleration/deceleration limits
   - Load controller

4. **Component Replacement**:
   - Partial overhauls (HGP only, blade only)
   - Component-specific restoration

5. **Auxiliary Systems**:
   - Generator vibration and electrical signatures
   - Gearbox dynamics
   - Fuel system pressure/flow control

## References

1. Boyce, M. P. (2011). "Gas Turbine Engineering Handbook" (4th ed.). Butterworth-Heinemann.

2. API 670:2014 - "Machinery Protection Systems"

3. Bently Nevada. (2002). "Fundamentals of Vibration." Technical Manual.

4. ISO 10816 - Mechanical vibration -- Evaluation of machine vibration by measurements on non-rotating parts

5. OREDA (Offshore Reliability Data). (2015). "Equipment Failure Rates and Reliability Data."

6. Eisenmann, R. C., & Eisenmann, R. C. (1997). "Machinery Malfunction Diagnosis and Correction." Prentice Hall.

7. Kurz, R., & Brun, K. (2001). "Degradation in Gas Turbine Systems." ASME Journal of Engineering for Gas Turbines and Power.

8. GE Energy. (2010). "LM2500 Gas Turbine Technical Manual."

## See Also

- [compressor.md](compressor.md) - Compressor simulation with surge protection
- [pump.md](pump.md) - Pump simulation with cavitation modeling
- [environmental_conditions.md](environmental_conditions.md) - Synthetic environmental modeling with African location types
- [weather_api_client.md](weather_api_client.md) - Real weather API integration and caching
- [vibration_enhanced.md](vibration_enhanced.md) - Advanced vibration generation
- [thermal_transient.md](thermal_transient.md) - Startup/shutdown thermal stress
- [maintenance_events.md](maintenance_events.md) - Maintenance scheduling and restoration
- [incipient_faults.md](incipient_faults.md) - Discrete fault initiation and growth
- [process_upsets.md](process_upsets.md) - Abnormal operating conditions
