# Pump Simulator Module

## Overview

The `pump.py` module simulates industrial pumps typical of offshore platforms, refineries, and process facilities. As the most numerous rotating equipment in oil & gas operations, pumps are critical for production continuity and safety. This simulator provides comprehensive cavitation modeling, hydraulic performance tracking, and mechanical seal health monitoring.

## Purpose

Centrifugal pumps represent the largest population of rotating equipment (50-70% of all rotating assets) with typical costs of $50K-500K and high criticality. This simulator enables:

- **Cavitation Detection**: NPSH monitoring and acoustic signature generation
- **Mechanical Seal Health**: Seal degradation and leakage tracking
- **Hydraulic Performance**: BEP deviation, efficiency curves, head-flow characteristics
- **Motor Current Signature**: Electrical anomaly detection
- **Multi-Mode Degradation**: Impeller erosion, seal wear, bearing damage

## Architecture Overview

The pump follows the **modular enhancement architecture**:

```
┌──────────────────────────────────────────────────────────────────┐
│                 Pump Class                             │
│                  (Core Simulator)                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │ Hydraulic Model   │  │ Cavitation Model  │                   │
│  │ - Head-flow curve │  │ - NPSH calc       │                   │
│  │ - Efficiency      │  │ - Severity levels │                   │
│  │ - BEP tracking    │  │ - Margin          │                   │
│  └───────────────────┘  └───────────────────┘                   │
│                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │ Seal Model        │  │ Bearing Model     │                   │
│  │ - Health/leakage  │  │ - DE & NDE        │                   │
│  │ - Temp factors    │  │ - Temps           │                   │
│  └───────────────────┘  │ - Vibration gen   │                   │
│                         └───────────────────┘                   │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           Enhancement Integration Layer                     │ │
│  │       (Physics, Simulation, ML Utilities)                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  next_state() → Telemetry Dictionary                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Key Features

### Core Features

- **Cavitation Modeling**: NPSHr calculation, margin tracking, 4-level severity classification
- **Mechanical Seal Tracking**: Health degradation, leakage progression, failure detection
- **Bearing Health**: Independent DE/NDE bearings with temperature and vibration correlation
- **Hydraulic Performance**: Head-flow characteristics, efficiency curves, BEP deviation
- **Motor Current**: Properly sized motor with current signature based on hydraulic load
- **Impeller Degradation**: Erosion/corrosion modeling affecting performance

### Enhancement Features (Optional)

- **Advanced Vibration**: Bearing defect frequencies (BPFO, BPFI, BSF, FTF)
- **Thermal Transients**: Startup/shutdown stress and thermal shock
- **Environmental Effects**: Corrosion factors, temperature effects
- **Maintenance Events**: Seal replacement, bearing service, impeller refurbishment
- **Incipient Faults**: Discrete fault initiation (bearing spalls, seal damage, cavitation erosion)
- **Process Upsets**: Cavitation events, pump runout, thermal shocks

## Module Components

### CavitationModel Class

Models cavitation phenomena including NPSH monitoring and severity classification.

**NPSH Required Calculation**:
```python
npsh_r = base_npsh * flow_factor * speed_factor
```

Where:
- `flow_factor = 1.0 + 0.5 * (flow_ratio - 1.0)^2` (minimum at BEP)
- `speed_factor = (speed_ratio)^2` (affinity law)

**NPSH Margin**:
```
margin = NPSHa - NPSHr  (meters)
```

**Severity Classification**:

| Severity | Margin Range | Symptoms | Action |
|----------|--------------|----------|--------|
| NONE (0) | > alarm threshold | Normal operation | Monitor |
| INCIPIENT (1) | alarm to trip | Slight noise, minor bubbles | Increase NPSH |
| MODERATE (2) | trip to 0 | Audible noise, performance loss | Reduce flow |
| SEVERE (3) | < 0 | Loud cavitation, rapid damage | Emergency shutdown |

**Default Thresholds**:
- Base NPSHr: 3.5 m
- Alarm margin: 1.0 m
- Trip margin: 0.3 m

### MechanicalSealModel Class

Models mechanical seal health with temperature and contamination effects.

**Degradation**:
```python
effective_rate = base_rate * operating_severity * temp_factor * contamination_factor
health -= effective_rate * dt
```

- Base rate: 0.00008 per hour
- Temperature factor: Increases with fluid temp and seal face temp
- Contamination factor: Particulates accelerate wear

**Leakage Calculation**:
```python
leakage_factor = exp(2 * (1 - health))
leakage = base_leakage * leakage_factor
```

Base leakage: 0.5 L/hr (healthy)

**Leakage Progression Example**:

| Health | Leakage (L/hr) | Status |
|--------|----------------|--------|
| 1.0 (new) | 0.5 | Normal |
| 0.8 | 1.5 | Slight increase |
| 0.6 | 3.7 | Moderate |
| 0.4 | 9.1 | High - plan replacement |
| 0.3 (threshold) | 14.8 | FAILURE |

### PumpBearingModel Class

Models drive end (DE) and non-drive end (NDE) bearings with independent health tracking.

**Degradation Rates**:
- Drive end: 0.00006 per hour (higher load)
- Non-drive end: 0.00004 per hour

**Temperature Calculation**:
```python
base_temp = ambient + 20 * sqrt(speed_factor)
friction_heat = (1.0 - health) * 50
load_heat = (load_factor - 1.0) * 10  # If overload

bearing_temp = base_temp + friction_heat + load_heat
```

**Vibration Generation**:
Includes bearing defect frequencies:
- BPFO (Outer race): 3.58 × shaft freq
- BPFI (Inner race): 5.42 × shaft freq
- BSF (Ball spin): 2.37 × shaft freq
- FTF (Cage): 0.42 × shaft freq

### HydraulicPerformanceModel Class

Models pump head-flow characteristics and efficiency.

**Head Calculation** (Parabolic Curve):
```python
# Affinity law scaling
flow_at_design_speed = flow / speed_ratio

# Parabolic head curve
head = a * flow^2 + b * flow + c

# Scale with speed squared
head *= speed_ratio^2

# Apply impeller health degradation
head *= (0.85 + 0.15 * impeller_health)
```

**Efficiency Calculation**:
```python
# Best efficiency at design flow
flow_ratio = flow / (design_flow * speed_ratio)
efficiency = design_eff * (1 - 0.5 * (flow_ratio - 1.0)^2)

# Degradation from impeller condition
efficiency *= (0.9 + 0.1 * impeller_health)
```

**BEP Deviation**:
```python
bep_flow = design_flow * speed_ratio
deviation(%) = abs((flow - bep_flow) / bep_flow) * 100
```

### Pump Class

**Operating Limits** (API 610):

| Parameter | Min | Max | Rated | Alarm | Trip |
|-----------|-----|-----|-------|-------|------|
| Speed (RPM) | 1,000 | 6,000 | 3,000 | - | - |
| Flow (m³/hr) | 10 | 500 | 150 | - | - |
| Head (m) | 20 | 200 | 80 | - | - |
| Vibration (mm/s) | - | - | 2-3 | 4.5 | 7.0 |
| Bearing Temp (°C) | - | 95 | 60 | 85 | 95 |
| Seal Leakage (L/hr) | - | - | 0.5-2 | 10 | - |
| Motor Current Ratio | - | 1.15 | 1.0 | 1.10 | 1.15 |

## Motor Current Calculation

Properly sized motor based on hydraulic power:

**Motor Sizing**:
```python
# Design point hydraulic power
design_power = (density * g * design_flow * design_head) / (efficiency * 3600)

# Motor rated power (25% margin)
motor_rated_power = design_power * 1.25

# Motor rated current (480V, 3-phase, PF=0.85, eff=0.92)
motor_rated_current = (motor_power * 1000) / (sqrt(3) * 480 * 0.85 * 0.92)
```

**Operating Current**:
```python
# Current proportional to actual power
motor_current = (actual_power * 1000) / (sqrt(3) * 480 * 0.85 * 0.92)

current_ratio = motor_current / motor_rated_current
```

**Overload Detection**: Trip at current_ratio > 1.15 (115% rated)

## Integration with Enhancement Modules

### Physics Module Integration

#### Enhanced Vibration

**Integration Point** (line 733-752):
```python
if self.use_enhanced_vibration:
    vib_metrics = self.vibration_generator.generate(
        speed=self.speed,
        load=self.flow / self.design_flow,
        degradation=1.0 - impeller_health
    )
    vib_rms = vib_metrics.get('rms', 0.0)
    vib_peak = vib_metrics.get('peak', 0.0)
else:
    vib_signal = self.bearing_model.generate_vibration(self.speed)
    vib_rms = sqrt(mean(vib_signal^2))
    vib_peak = max(abs(vib_signal))
```

**Benefit**: Realistic bearing fault progression with defect frequency signatures

#### Thermal Transients

**Integration Point** (line 658-670):
```python
if self.use_thermal_model:
    thermal_state = self.thermal_model.step(
        ambient_temp=ambient_temp,
        operating_temp=self.fluid_temp,
        severity=severity
    )
    thermal_multiplier = thermal_state.get('degradation_multiplier', 1.0)
    severity *= thermal_multiplier
```

**Benefit**: Captures thermal shock effects on seals and bearings

#### Environmental Conditions

**Integration Point** (line 636-649):
```python
if self.use_environmental:
    env_state = self.environmental_conditions.get_state()
    ambient_temp = env_state.get('ambient_temp', 40.0)
    corrosion_factor = env_state.get('corrosion_factor', 1.0)
```

**Benefit**: Realistic environmental effects on corrosion and performance

#### Location Type Selection

Eight pre-configured location profiles model diverse environmental conditions affecting pump performance:

| Location Type | Region | Key Characteristics | Use Cases |
|--------------|--------|---------------------|-----------|
| **OFFSHORE** | Marine platforms | High salt (0.9), corrosive environment (15°C), 75% humidity | North Sea (UK/Norway), Gulf of Mexico (US), West Africa offshore |
| **DESERT** | Arid regions | Extreme heat (30°C), high dust (0.95), low humidity (20%) | Saudi Arabia, UAE, Kuwait, Libya, Algeria |
| **ARCTIC** | Polar regions | Extreme cold (-15°C), ice risk (0.95), viscosity challenges | Russia (Yamal), Alaska, Northern Canada |
| **TROPICAL** | Equatorial zones | High humidity (85%), warm (28°C), corrosion acceleration | Indonesia, Malaysia, Nigeria (coastal), Gabon, Venezuela |
| **TEMPERATE** | Mid-latitudes | 4-season pattern (12°C mean), moderate humidity (65%) | USA, UK, Germany, Netherlands, China |
| **SAHEL** | West Africa | High dust (0.80), Harmattan season, semi-arid (30°C) | Nigeria (north), Chad, Niger, Sudan |
| **SAVANNA** | Semi-arid Africa | Southern Hemisphere pattern, moderate dust (0.5), seasonal (25°C) | South Africa, Angola, Mozambique |

**Environmental Impact on Pump Performance**:
- **Temperature**: Affects fluid viscosity, vapor pressure (NPSHr), and seal performance
- **Humidity**: Influences corrosion rates on casing and shaft
- **Pressure**: Alters NPSHa calculations and cavitation risk
- **Salt/Dust**: Accelerates seal wear, impeller erosion, and bearing contamination

#### Integration Methods

**Method 1: Synthetic Location Profile**
```python
from physics.environmental_conditions import LocationType

pump = Pump(
    name='CP-001',
    location_type=LocationType.SAVANNA,  # South African installation
    enable_environmental=True
)
```

**Method 2: Real Weather API Integration**
```python
from physics.weather_api_client import create_hybrid_environment

# Create real weather data source
env_source = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key_here",
    location_name="Port Harcourt",
    country="Nigeria",
    cache_enabled=True
)

pump = Pump(
    name='CP-001',
    env_model=env_source,  # Use real weather data directly
    enable_environmental=True
)
```

For detailed environmental modeling and weather API integration, see:
- [environmental_conditions.md](environmental_conditions.md) - Synthetic location profiles
- [weather_api_client.md](weather_api_client.md) - Real-time weather integration

### Simulation Module Integration

#### Maintenance Scheduler

**Integration Point** (line 772-797):
```python
if self.use_maintenance:
    maint_state = self.maintenance_scheduler.step(
        operating_hours=self.operating_hours,
        component_health={
            'impeller': impeller_health,
            'seal': seal_health,
            'bearing_de': bearing_de_health,
            'bearing_nde': bearing_nde_health
        }
    )

    if maint_state.get('performed', False):
        # Restore component health
        for component in maint_state.get('maintained_components', []):
            if component == 'impeller':
                self.impeller_health = min(1.0, self.impeller_health + 0.05)
            # ... seal and bearing restoration
```

**Pump-Specific Maintenance**:
- Routine (2000 hrs): Seal inspection, alignment check
- Minor (8000 hrs): Seal replacement, bearing inspection
- Major (20000 hrs): Impeller refurbishment, shaft inspection

#### Incipient Fault Simulator

**Integration Point** (line 673-691):
```python
if self.use_faults:
    fault_state = self.fault_simulator.step(
        operating_hours=self.operating_hours,
        current_severity=severity
    )

    for component, active in fault_state.get('active_faults', {}).items():
        if active:
            if component == 'impeller':
                self.impeller_health -= propagation_rate
            elif component == 'seal':
                self.seal_model.health -= propagation_rate
            # ... bearing fault propagation
```

**Pump-Specific Faults**:
- Impeller: Erosion initiation, fatigue cracks
- Seal: Face damage, contamination
- Bearings: Spalls, wear patterns
- Cavitation erosion: Material removal from cavitation bubble collapse

#### Process Upset Simulator

**Integration Point** (line 694-702):
```python
if self.use_upsets:
    upset_state = self.upset_simulator.step()

    if upset_state.get('active', False):
        upset_damage = upset_state.get('damage_multiplier', 1.0)
        severity *= upset_damage
```

**Pump-Specific Upsets**:
- **Cavitation event**: NPSHa drops critically low (8% damage potential)
- **Pump runout**: Operation far from BEP, high radial loads (2% damage)
- **Thermal shock**: Rapid temperature change (3% damage)

## Hydraulic Performance Model Details

### Head-Flow Curve Shape

Typical pump curve:

```
  Head
   ^
   |     \
   |      \___
   |          \___
   |              \___
   |                  \____
   +-----------------------> Flow
  Shutoff                  Runout
   Head    BEP             Flow
```

**Parabolic Approximation**:
```
head = -a * flow^2 + c
```

Where:
- `a = design_head / (1.5 * design_flow)^2`
- `c = design_head * 1.1` (shutoff head)

### Efficiency Curve

Bell-shaped efficiency curve peaking at BEP:

```
Efficiency
   ^
   |       /--\
   |      /    \
   |     /      \
   |    /        \
   +-------------> Flow
       BEP
```

### Best Efficiency Point (BEP)

**Importance**:
- Operating away from BEP causes:
  - Increased hydraulic losses
  - Radial thrust on impeller
  - Higher bearing loads
  - Accelerated seal wear

**Recommended Operating Range**: BEP ± 20%

**BEP Deviation Example**:

| Flow | BEP Deviation | Status | Impact |
|------|---------------|--------|--------|
| 150 m³/hr | 0% | Optimal | Normal wear |
| 120 m³/hr | 20% | Acceptable | Slightly increased wear |
| 90 m³/hr | 40% | Poor | 2x degradation rate |
| 60 m³/hr | 60% | Very poor | 3x degradation rate |

## Cavitation Phenomena

### NPSH Concept

**Net Positive Suction Head Available (NPSHa)**:
```
NPSHa = (P_suction / (ρ * g)) + (v^2 / (2 * g)) - (P_vapor / (ρ * g)) - Z_pump
```

Typically provided as system parameter in simulator.

**NPSH Required (NPSHr)**:
Pump characteristic that increases with:
- Flow rate (parabolic increase away from BEP)
- Speed (squared with affinity laws)
- Impeller wear (3-5% increase)

### Cavitation Severity Progression

**Incipient Cavitation** (Margin: 0.3-1.0 m):
- Slight noise (15-100 kHz acoustic)
- Small vapor bubbles at impeller eye
- Minor vibration increase (+10%)
- No immediate damage

**Moderate Cavitation** (Margin: 0-0.3 m):
- Audible "gravel" noise
- Performance loss (5-10% head reduction)
- Vibration increase (+50%)
- Slow erosion rate (0.1 mm/1000 hrs)

**Severe Cavitation** (Margin < 0):
- Very loud cavitation noise
- Significant performance loss (15%+ head reduction)
- High vibration (+200%)
- Rapid erosion (1-5 mm/1000 hrs)
- Emergency shutdown required

## Usage Examples

### Basic Pump Simulation

```python
from pump import Pump

pump = Pump(
    name='CP-001',
    initial_health={
        'impeller': 0.90,
        'seal': 0.85,
        'bearing_de': 0.82,
        'bearing_nde': 0.86
    },
    design_flow=150,
    design_head=80,
    design_speed=3000,
    fluid_density=850,  # Crude oil
    npsh_available=8.0
)

pump.set_speed(3000)

for i in range(3600):
    state = pump.next_state()

    # Monitor cavitation
    if state['cavitation_severity'] > 0:
        print(f"CAVITATION: Severity={state['cavitation_severity']}, "
              f"Margin={state['npsh_margin']:.2f} m")

    # Monitor seal leakage
    if state['seal_leakage'] > 10:
        print(f"HIGH SEAL LEAKAGE: {state['seal_leakage']:.1f} L/hr")

    # Monitor motor current
    if state['motor_current_ratio'] > 1.10:
        print(f"HIGH MOTOR CURRENT: {state['motor_current_ratio']:.2f}")
```

### Pump with All Enhancements

```python
from pump import Pump
from physics.environmental_conditions import LocationType
from ml_utils.ml_output_modes import OutputMode

pump = Pump(
    name='CP-001',
    initial_health={
        'impeller': 0.88,
        'seal': 0.80,
        'bearing_de': 0.75,
        'bearing_nde': 0.78
    },
    design_flow=200,
    design_head=100,
    design_speed=3000,
    fluid_density=1025,  # Seawater
    npsh_available=6.0,

    # Enhancements
    location_type=LocationType.OFFSHORE,
    enable_enhanced_vibration=True,
    enable_thermal_transients=True,
    enable_environmental=True,
    enable_maintenance=True,
    enable_incipient_faults=True,
    enable_process_upsets=True,
    output_mode=OutputMode.FULL
)

pump.set_speed(3000)

for i in range(3600):
    state = pump.next_state()

    # Enhanced monitoring
    if state.get('upset_active'):
        print(f"Upset: {state['upset_type']}")

    if state.get('num_active_faults', 0) > 0:
        fault_info = state.get('active_fault_details', {})
        for comp, fault in fault_info.items():
            print(f"Fault in {comp}: type={fault['type']}, size={fault['size']:.2f}")
```

### Pump with Real Weather API (Nigerian Coastal Installation)

```python
from pump import Pump
from physics.weather_api_client import create_hybrid_environment
from physics.environmental_conditions import EnvironmentalConditions, LocationType
from ml_utils.ml_output_modes import OutputMode

# Create fallback for when API unavailable
fallback = EnvironmentalConditions(LocationType.TROPICAL)

# Configure real weather integration for Port Harcourt crude oil pumping station
env_source = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key_here",
    location_name="Port Harcourt",
    country="Nigeria",
    fallback_source=fallback,
    cache_enabled=True,
    cache_ttl_hours=24
)

pump = Pump(
    name='CP-PH-001',
    initial_health={
        'impeller': 0.92,
        'seal': 0.88,
        'bearing_de': 0.85,
        'bearing_nde': 0.87
    },
    design_flow=250,
    design_head=120,
    design_speed=3000,
    fluid_density=850,  # Crude oil
    npsh_available=9.0,

    # Use real weather data from Port Harcourt, Nigeria
    env_model=env_source,
    enable_environmental=True,
    enable_enhanced_vibration=True,
    enable_thermal_transients=True,
    enable_maintenance=True,
    enable_incipient_faults=True,
    enable_process_upsets=True,
    output_mode=OutputMode.FULL
)

pump.set_speed(3000)

# Simulate 30 days of crude oil pumping with tropical climate effects
for hour in range(720):
    state = pump.next_state()

    # Monitor environmental impact
    if hour % 24 == 0:  # Daily summary
        day = hour // 24
        print(f"Day {day}: Temp={state.get('ambient_temp_C', 'N/A'):.1f}°C, "
              f"Pressure={state.get('pressure_kPa', 'N/A'):.1f} kPa, "
              f"Humidity={state.get('humidity_percent', 'N/A'):.0f}%")

    # Cavitation monitoring
    if state['cavitation_severity'] > 0:
        print(f"Hour {hour}: CAVITATION - Severity={state['cavitation_severity']}, "
              f"NPSH margin={state['npsh_margin']:.2f} m")

    # Seal health tracking (critical in high humidity)
    if state['seal_leakage'] > 10:
        print(f"Hour {hour}: HIGH SEAL LEAKAGE - {state['seal_leakage']:.1f} L/hr")

    # Enhanced fault monitoring
    if state.get('num_active_faults', 0) > 0:
        print(f"Hour {hour}: Active faults detected - {state['num_active_faults']}")
```

### Multiple Pump Services

```python
from pump import generate_pump_dataset

# Generate dataset with varied pump services
telemetry, failures = generate_pump_dataset(
    n_machines=10,
    n_cycles_per_machine=200,
    cycle_duration_range=(60, 300),
    random_seed=42
)

# Services: Crude Booster, Seawater Injection, Process Water, Methanol, Fire Water

print(f"Generated {len(telemetry)} records")
print(f"Failures: {len(failures)}")

# Analyze by service type
from collections import defaultdict
by_service = defaultdict(list)

for record in telemetry:
    by_service[record['service']].append(record)

for service, records in by_service.items():
    print(f"{service}: {len(records)} records")
```

### Cavitation Monitoring

```python
from pump import Pump

pump = Pump(
    name='CP-001',
    design_flow=150,
    design_head=80,
    npsh_available=5.0  # Marginal NPSH
)

pump.set_speed(3000)

cavitation_history = []

for i in range(1000):
    state = pump.next_state()

    cavitation_history.append({
        'time': i,
        'npsh_margin': state['npsh_margin'],
        'cavitation_severity': state['cavitation_severity'],
        'vibration_rms': state['vibration_rms'],
        'impeller_health': state['health_impeller']
    })

    if state['cavitation_severity'] == 3:  # Severe
        print(f"Time {i}: SEVERE CAVITATION - Emergency shutdown")
        print(f"  NPSH margin: {state['npsh_margin']:.2f} m")
        print(f"  Vibration: {state['vibration_rms']:.2f} mm/s")
        print(f"  Impeller health: {state['health_impeller']:.3f}")
        break
```

## Telemetry Output

### Standard Fields

| Field | Type | Range | Units | Description |
|-------|------|-------|-------|-------------|
| speed | float | 0-6000 | RPM | Pump speed |
| speed_target | float | 0-6000 | RPM | Target speed |
| flow | float | 0-500 | m³/hr | Volumetric flow rate |
| head | float | 0-200 | m | Developed head |
| efficiency | float | 0.1-0.95 | - | Hydraulic efficiency |
| power | float | 0-500 | kW | Hydraulic power |
| suction_pressure | float | 50-500 | kPa | Inlet pressure |
| discharge_pressure | float | 100-2000 | kPa | Outlet pressure |
| fluid_temp | float | 0-150 | °C | Process fluid temperature |
| motor_current | float | 0-200 | A | Motor current draw |
| motor_current_ratio | float | 0-1.5 | - | Current / rated current |
| vibration_rms | float | 0-10 | mm/s | RMS vibration velocity |
| vibration_peak | float | 0-30 | mm/s | Peak vibration velocity |
| bearing_temp_de | float | 30-95 | °C | Drive end bearing temp |
| bearing_temp_nde | float | 30-95 | °C | Non-drive end bearing temp |
| npsh_available | float | 0-20 | m | Available NPSH |
| npsh_required | float | 2-15 | m | Required NPSH |
| npsh_margin | float | -10-15 | m | NPSHa - NPSHr |
| cavitation_severity | int | 0-3 | - | Cavitation level (0=none, 3=severe) |
| seal_leakage | float | 0.5-50 | L/hr | Mechanical seal leakage |
| bep_deviation | float | 0-100 | % | Deviation from BEP |
| operating_hours | float | 0+ | hrs | Cumulative hours |
| health_impeller | float | 0-1 | - | Impeller health |
| health_seal | float | 0-1 | - | Seal health |
| health_bearing_de | float | 0-1 | - | DE bearing health |
| health_bearing_nde | float | 0-1 | - | NDE bearing health |

## Failure Modes

### Failure Catalog

| Code | Description | Threshold/Condition | Detection |
|------|-------------|---------------------|-----------|
| F_IMPELLER | Impeller degradation | Health < 0.35 | Health monitoring |
| F_SEAL | Mechanical seal failure | Health < 0.30 | Health monitoring |
| F_BEARING_DRIVE_END | DE bearing failure | Health < 0.28 | Health monitoring |
| F_BEARING_NON_DRIVE_END | NDE bearing failure | Health < 0.28 | Health monitoring |
| F_BEARING_OVERTEMP | Bearing overtemperature | Temp > 95°C | Temperature monitoring |
| F_HIGH_VIBRATION | Excessive vibration | RMS > 7.0 mm/s | Vibration monitoring |
| F_CAVITATION | Severe cavitation | Margin < 0.3 m | NPSH monitoring |
| F_MOTOR_OVERLOAD | Motor overload | Current ratio > 1.15 | Current monitoring |

## Performance Considerations

- **Base simulation**: ~0.12 ms per timestep
- **With enhancements**: ~0.55 ms per timestep
- **Memory**: ~6 KB per instance

## Validation and Calibration

### Operating Envelope

Based on:
- API 610: Pumps for Petroleum Industry
- Hydraulic Institute Standards
- Flowserve and Sulzer pump specifications

### Hydraulic Performance

Head-flow curves calibrated from:
- OEM pump curves (various sizes)
- Affinity law validation
- Field performance test data

### Cavitation Thresholds

NPSH values from:
- API 610 recommendations
- Hydraulic Institute cavitation studies
- Field experience (offshore platforms)

### Bearing Temperature Limits

Limits from:
- API 610 bearing temperature requirements
- SKF bearing handbook
- Field alarm/trip setpoints

## Best Practices

### Service Selection

**Crude Oil Transfer**:
```python
pump = Pump(
    design_flow=200,
    design_head=100,
    fluid_density=850,  # Crude
    npsh_available=8.0
)
```

**Seawater Injection**:
```python
pump = Pump(
    design_flow=300,
    design_head=150,
    fluid_density=1025,  # Seawater
    npsh_available=10.0
)
```

**Fire Water Service**:
```python
pump = Pump(
    design_flow=400,
    design_head=120,
    fluid_density=1000,  # Water
    npsh_available=12.0
)
```

### NPSH Management

**Adequate NPSH**: NPSHa > NPSHr + 3 m
**Marginal NPSH**: NPSHa > NPSHr + 1 m (monitor closely)
**Insufficient NPSH**: NPSHa < NPSHr + 1 m (increase suction pressure)

### BEP Operation

**Optimal**: BEP ± 10%
**Acceptable**: BEP ± 20%
**Avoid**: BEP ± 40%+ (accelerated wear)

## Limitations and Extensions

### Current Limitations

1. **Single-Speed Motor**: Variable speed drives not modeled
2. **Newtonian Fluids**: Non-Newtonian rheology not included
3. **Single Impeller**: Multi-stage pumps not modeled
4. **Fixed System Curve**: No pipeline resistance modeling
5. **No Priming**: Assumes pump always primed

### Potential Enhancements

1. **VFD Modeling**:
   - Variable speed operation
   - Harmonic content in motor current

2. **Non-Newtonian Fluids**:
   - Viscosity effects
   - Reynolds number corrections

3. **Multi-Stage Pumps**:
   - Stage-by-stage performance
   - Inter-stage balancing

4. **System Integration**:
   - Pipeline pressure drop
   - System curve interaction

5. **Startup Sequence**:
   - Priming detection
   - Minimum flow recirculation

## References

1. API 610:2010 - "Pumps for Petroleum, Petrochemical and Natural Gas Industries"

2. Hydraulic Institute. (2016). "Effects of Liquid Viscosity on Rotodynamic Pump Performance."

3. Gülich, J. F. (2020). "Pumps" (4th ed.). Springer.

4. Karassik, I. J., et al. (2008). "Pump Handbook" (4th ed.). McGraw-Hill.

5. ISO 10816 - Mechanical vibration -- Evaluation of machine vibration

6. SKF Group. (2014). "Bearing damage and failure analysis."

## See Also

- [gas_turbine.md](gas_turbine.md) - Gas turbine simulation
- [compressor.md](compressor.md) - Compressor with surge modeling
- [environmental_conditions.md](environmental_conditions.md) - Synthetic environmental modeling
- [weather_api_client.md](weather_api_client.md) - Real weather API integration
- [vibration_enhanced.md](vibration_enhanced.md) - Advanced bearing vibration
- [process_upsets.md](process_upsets.md) - Cavitation events and pump runout
- [incipient_faults.md](incipient_faults.md) - Bearing spall and seal damage progression
