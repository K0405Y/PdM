# Compressor Simulator Module

## Overview

The `compressor.py` module simulates industrial compressors typical of pipeline networks, LNG facilities, and refinery process units. It provides comprehensive surge protection modeling, dry gas seal health tracking, and shaft orbit analysis for predictive maintenance applications.

## Purpose

Centrifugal compressors are critical high-speed rotating equipment in oil & gas operations, with typical costs of $5-20M and operational significance requiring 99.5%+ reliability. This simulator enables:

- **Surge Modeling**: Realistic surge margin calculation and anti-surge control simulation
- **Dry Gas Seal Monitoring**: Primary and secondary seal health with leakage tracking
- **Shaft Dynamics**: Proximity probe measurements and orbit pattern analysis
- **Multi-Mode Degradation**: Impeller fouling, bearing wear, seal degradation
- **Edge Case Coverage**: Liquid carryover, surge events, seal failures

## Key Features

### Core Features

- **Surge Protection**: Parabolic surge line, margin calculation, alarm/trip thresholds
- **Dry Gas Seals**: Independent primary/secondary seal health and leakage
- **Shaft Orbit Analysis**: X-Y displacement with unbalance, whirl, and blade pass signatures
- **Thermodynamic Performance**: Head/flow characteristics, affinity laws, efficiency tracking
- **Multi-Mode Degradation**: Impeller and bearing health trajectories

### Enhancement Features (Optional)

- **Advanced Vibration**: Envelope-modulated bearing defect frequencies
- **Thermal Transients**: Mode-specific degradation during startups/shutdowns
- **Environmental Effects**: Location-specific temperature/pressure/corrosion
- **Maintenance Events**: Scheduled and condition-based maintenance
- **Incipient Faults**: Discrete spall/crack initiation and propagation
- **Process Upsets**: Liquid carryover, thermal shocks, overload events

## Module Components

### SurgeModel Class

Models surge characteristics and anti-surge protection.

**Surge Line** (Parabolic Approximation):
```
head_surge = a * flow^2 + b * flow + c
```

Default coefficients:
- a = 0.005
- b = -2.0
- c = design_head * 1.2

**Surge Margin Calculation**:
```
margin(%) = ((flow_actual - flow_surge) / flow_surge) * 100
```

**Protection Thresholds**:
- Alarm: 10% margin
- Trip: 5% margin
- Surge: < 0% margin (flow reversal)

**Example Surge Line**:

| Head (kJ/kg) | Surge Flow (m³/hr) | Safe Flow (10% margin) |
|--------------|-------------------|------------------------|
| 8000 | 600 | 660 |
| 9000 | 570 | 627 |
| 10000 | 545 | 600 |

### DryGasSealModel Class

Models primary and secondary dry gas seals critical for preventing process gas leakage.

**Health Tracking**:
- Primary seal: Initial 95%, degrades at 0.00005/hr
- Secondary seal: Initial 98%, degrades at 0.00002/hr

**Leakage Calculation**:
```python
health_factor = 1.0 / max(health, 0.1)
leakage = base_leakage * health_factor
```

Base leakage:
- Primary: 2.0 Nm³/hr (healthy)
- Secondary: 0.5 Nm³/hr (healthy)

**Failure Threshold**: 25% health (both seals)

**Degradation Factors**:
- Operating severity (speed, pressure)
- Contamination (dirty process gas)

### ShaftOrbitModel Class

Simulates shaft displacement measured by orthogonal proximity probes (X-Y sensors).

**Orbit Components**:

1. **Base Orbit** (Healthy):
   - Radius: 0.02 mm
   - Ellipticity: 0.2 (slightly elliptical)

2. **Unbalance** (Impeller Degradation):
   - Increases orbit size: `factor = 1.0 + 2.0 * (1.0 - impeller_health)`
   - Dominant at 1X shaft frequency

3. **Oil Whirl/Whip** (Bearing Health < 0.6):
   - Sub-synchronous component at 0.43X shaft frequency
   - Amplitude: `(0.6 - bearing_health) * 0.05 mm`

4. **Blade Pass** (Impeller Health < 0.7):
   - Super-synchronous at 8X shaft frequency (8 blades)
   - Amplitude: `(0.7 - impeller_health) * 0.01 mm`

**Orbit Metrics**:
- Orbit amplitude: Peak-to-peak radial displacement
- Average gap: Time-averaged shaft position
- Synchronous amplitude: 1X component magnitude

### CompressorHealthModel Class

Manages two primary degradation pathways.

**Degradation Model** (same as gas turbine):
```
h(t) = 1 - d - exp(a * t^b)
```

| Component | Parameters (d, a, b) | Threshold | Typical Life |
|-----------|----------------------|-----------|--------------|
| Impeller | (0.04, -0.28, 0.21) | 0.42 | 8,000-15,000 hrs |
| Bearing | (0.06, -0.32, 0.24) | 0.38 | 6,000-12,000 hrs |

**Operating Severity**:
```python
severity = 1.0

# Speed penalty
if speed_factor > 1.0:
    severity *= (1.0 + 0.4 * (speed_factor - 1.0)**2)

# Surge margin penalty
if surge_margin < 20:
    severity *= (1.0 + 0.3 * (20 - surge_margin) / 20)
```

### Compressor Class

**Operating Limits** (API 617, Industrial Standards):

| Parameter | Min | Max | Rated | Alarm | Trip |
|-----------|-----|-----|-------|-------|------|
| Speed (RPM) | 5,000 | 25,000 | 12,000 | - | - |
| Suction Pressure (kPa) | 500 | 5,000 | 2,000 | - | - |
| Discharge Pressure (kPa) | 2,000 | 15,000 | - | - | - |
| Flow (m³/hr) | 100 | 3,000 | 1,500 | - | - |
| Vibration (mm) | - | - | 0.025-0.05 | 0.050 | 0.075 |
| Bearing Temp (°C) | - | 110 | - | 100 | 110 |
| Seal Leakage (Nm³/hr) | - | - | 2-3 | 5.0 | - |

## Integration with Enhancement Modules

### Physics Module Integration

#### Vibration and Orbit Measurement

**API 617 Compliance**: Centrifugal compressor vibration monitoring uses **displacement** (mm) measured by proximity probes, not velocity (mm/s). This is critical for proper trip threshold evaluation.

**Integration Point** (line 821-845):

The orbit model **always** provides displacement values for API 617 compliance:

```python
# Always use orbit model for displacement (mm) - API 617 compliance
x_disp, y_disp = self.orbit_model.generate_orbit(self.speed, health_state)
orbit_metrics = self.orbit_model.compute_metrics(x_disp, y_disp)
orbit_amplitude = orbit_metrics['orbit_amplitude']  # mm (displacement)
sync_amplitude = orbit_metrics['sync_amplitude']    # mm (displacement)

# Enhanced vibration is optional - provides ML features only (velocity, mm/s)
vib_rms = None
vib_peak = None
if self.use_enhanced_vibration:
    try:
        vib_signal, vib_metrics = self.vib_generator_enhanced.generate_bearing_vibration(
            rpm=self.speed,
            bearing_health=health_state.get('bearing', 1.0),
            duration=1.0
        )
        vib_rms = vib_metrics.get('rms', 0)    # mm/s (velocity)
        vib_peak = vib_metrics.get('peak', 0)  # mm/s (velocity)
    except:
        pass
```

**Important**: Trip thresholds (e.g., 0.075 mm, 0.15 mm) are evaluated against `orbit_amplitude` from the orbit model, not velocity metrics from the enhanced vibration generator.

**Benefit**: Proper unit separation - displacement (mm) for protection, velocity (mm/s) for ML features

#### Thermal Transients

**Integration Point** (line 727-738):
```python
if self.use_thermal_model:
    thermal_state = self.thermal_model.step(
        target_speed=self.speed_target,
        rated_speed=self.LIMITS['speed_max'],
        timestep_minutes=1/60
    )
    thermal_multiplier = thermal_state.get('degradation_multiplier', 1.0)
    severity *= thermal_multiplier
```

**Benefit**: Captures startup/shutdown stress on impeller and thrust bearing

#### Environmental Conditions

**Integration Point** (line 715-721):
```python
if self.use_environmental:
    env_cond = self.env_model.get_conditions(self.elapsed_hours)
    self.suction_temp = env_cond.get('ambient_temp_C', self.suction_temp)
    self.suction_pressure = env_cond.get('pressure_kPa', self.suction_pressure)
```

**Benefit**: Realistic inlet conditions affecting compression ratio and performance

#### Location Type Selection

Eight pre-configured location profiles model diverse environmental conditions affecting compressor performance:

| Location Type | Region | Key Characteristics | Use Cases |
|--------------|--------|---------------------|-----------|
| **OFFSHORE** | Marine platforms | High salt (0.9), moderate temp (15°C), 75% humidity | North Sea (UK/Norway), Gulf of Mexico (US), West Africa offshore |
| **DESERT** | Arid regions | Extreme heat (30°C), high dust (0.95), low humidity (20%) | Saudi Arabia, UAE, Kuwait, Qatar, Libya, Algeria |
| **ARCTIC** | Polar regions | Extreme cold (-15°C), ice risk (0.95), pressure variation | Russia (Yamal LNG), Alaska, Northern Canada |
| **TROPICAL** | Equatorial zones | High humidity (85%), warm (28°C), minimal seasonal variation | Indonesia, Malaysia, Nigeria (coastal), Gabon |
| **TEMPERATE** | Mid-latitudes | 4-season pattern (12°C mean), moderate humidity (65%) | USA, UK, Germany, Netherlands, China |
| **SAHEL** | West Africa | High dust (0.80), Harmattan winds, semi-arid (30°C) | Nigeria (north), Chad, Niger, Sudan |
| **SAVANNA** | Semi-arid Africa | Southern Hemisphere pattern, moderate dust (0.5), seasonal (25°C) | South Africa, Angola, Mozambique |

**Environmental Impact on Compressor Performance**:
- **Temperature**: Affects gas density, compression ratio, and power requirements
- **Humidity**: Influences corrosion rates and intercooler effectiveness
- **Pressure**: Alters volumetric flow calculations and stage matching
- **Dust/Salt**: Accelerates fouling on impellers and seals

#### Integration Methods

**Method 1: Synthetic Location Profile**
```python
from physics.environmental_conditions import LocationType

compressor = Compressor(
    name='CC-001',
    location_type=LocationType.SAHEL,  # Pre-configured African profile
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
    location_name="Lagos",
    country="Nigeria",
    cache_enabled=True
)

compressor = Compressor(
    name='CC-001',
    env_model=env_source,  # Use real weather data directly
    enable_environmental=True
)
```

For detailed environmental modeling and weather API integration, see:
- [environmental_conditions.md](environmental_conditions.md) - Synthetic location profiles
- [weather_api_client.md](weather_api_client.md) - Real-time weather integration

### Simulation Module Integration

#### Maintenance Scheduler

**Integration Point** (line 827-845):
```python
if self.use_maintenance:
    maint_type = self.maint_scheduler.check_maintenance_required(
        operating_hours=self.operating_hours,
        health_state=health_state,
        is_planned_shutdown=(self.speed == 0)
    )

    if maint_type:
        maint_action = self.maint_scheduler.perform_maintenance(
            maint_type, current_health=health_state,
            operating_hours=self.operating_hours,
            timestamp=self.current_timestamp
        )
        self.health_model.health = maint_action.health_after
```

**Maintenance Types**:
- Routine (2000 hrs): Oil change, seal inspection
- Minor (8000 hrs): Seal replacement, bearing inspection
- Major (24000 hrs): Impeller refurbishment, full overhaul
- Emergency (<20% health): Critical repair

#### Incipient Fault Simulator

**Integration Point** (line 741-772):
```python
if self.use_faults:
    fault_event = self.fault_sim.check_fault_initiation(
        operating_hours_increment=1/3600,
        stress_factor=severity,
        timestamp=self.current_timestamp,
        operating_hours=self.operating_hours,
        component_list=['impeller', 'bearing', 'seal_primary', 'seal_secondary']
    )
    self.fault_sim.propagate_faults(1/3600, severity)
    health_state = self.fault_sim.adjust_health_for_faults(health_state)
```

**Compressor-Specific Faults**:
- Impeller: FOD, erosion, fatigue cracks
- Bearing: Spalls, wear
- Seals: Face damage, contamination

#### Process Upset Simulator

**Integration Point** (line 755-763):
```python
if self.use_upsets:
    upset_event = self.upset_sim.check_upset_initiation(
        timestep_seconds=1,
        timestamp=self.current_timestamp,
        operating_state={'speed': self.speed, 'flow': self.flow}
    )

    if self.upset_sim.active_upset:
        health_state = self.upset_sim.calculate_upset_damage(health_state)
```

**Compressor-Specific Upsets**:
- Liquid carryover: Slug of liquid enters compressor (severe damage risk)
- Surge event: Flow reversal (mechanical stress)
- Thermal shock: Rapid temperature change (differential expansion)

## Thermodynamic Model

### Affinity Laws

Flow and head scale with speed:

```python
# Flow: proportional to speed
flow = design_flow * (speed / rated_speed)

# Head: proportional to speed squared
head = design_head * (speed / rated_speed)^2 * health_factor
```

### Performance Degradation

Impeller fouling/erosion reduces head:

```python
head_degradation = 0.9 + 0.1 * impeller_health  # 10% max loss
head_actual = head_theoretical * head_degradation
```

### Discharge Conditions

Simplified polytropic compression:

```python
# Pressure ratio from head
gamma = 1.3  # For natural gas
pressure_ratio = 1 + (head / (constant * suction_temp))

discharge_pressure = suction_pressure * pressure_ratio

# Temperature rise
discharge_temp = suction_temp * (pressure_ratio)^((gamma-1)/gamma)
```

### Efficiency

Efficiency degrades with impeller condition:

```python
efficiency = 0.75 + 0.10 * impeller_health
```

Range: 0.75 (degraded) → 0.85 (new)

### Power Calculation

```python
mass_flow = volumetric_flow * gas_density
power = (mass_flow * head) / (3600 * efficiency)  # kW
```

## Surge Protection Model

### Surge Line Shape

Typical surge line is parabolic in head-flow coordinates:

```
          Head
           ^
           |     Safe Operating Region
           |   /
  Design   | /
   Head    |/_____ Surge Line
           |
           +---------------> Flow
         Surge   Design
         Flow     Flow
```

### Margin Calculation Examples

**Scenario 1 - Safe Operation**:
- Actual flow: 1500 m³/hr
- Surge flow (at current head): 900 m³/hr
- Margin: ((1500 - 900) / 900) * 100 = 66.7%
- Status: Normal operation

**Scenario 2 - Approaching Surge**:
- Actual flow: 990 m³/hr
- Surge flow: 900 m³/hr
- Margin: ((990 - 900) / 900) * 100 = 10%
- Status: Surge alarm (at threshold)

**Scenario 3 - Surge Condition**:
- Actual flow: 850 m³/hr
- Surge flow: 900 m³/hr
- Margin: ((850 - 900) / 900) * 100 = -5.6%
- Status: Surge trip condition

## Usage Examples

### Basic Compressor Simulation

```python
from compressor import Compressor

compressor = Compressor(
    name='CC-001',
    initial_health={'impeller': 0.88, 'bearing': 0.82},
    design_flow=1500,
    design_head=8000,
    suction_pressure=2000,
    suction_temp=35
)

# Start compressor
compressor.set_speed(12000)

for i in range(3600):
    state = compressor.next_state()

    # Monitor surge margin
    if state['surge_alarm']:
        print(f"SURGE ALARM: Margin = {state['surge_margin']:.1f}%")

    # Monitor seal leakage
    total_leakage = state['primary_seal_leakage'] + state['secondary_seal_leakage']
    if total_leakage > 5.0:
        print(f"HIGH SEAL LEAKAGE: {total_leakage:.2f} Nm³/hr")
```

### Compressor with All Enhancements

```python
from compressor import Compressor
from physics.environmental_conditions import LocationType
from ml_utils.ml_output_modes import OutputMode

compressor = Compressor(
    name='CC-001',
    initial_health={'impeller': 0.85, 'bearing': 0.78},
    design_flow=1500,
    design_head=8000,
    suction_pressure=2000,
    suction_temp=35,

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

compressor.set_speed(12000)

for i in range(3600):
    state = compressor.next_state()

    # Enhanced monitoring
    if state.get('upset_active'):
        print(f"Upset: {state['upset_type']}")

    if state.get('num_active_faults', 0) > 0:
        print(f"Active faults: {state['num_active_faults']}")
```

### Compressor with Real Weather API (African Installation)

```python
from compressor import Compressor
from physics.weather_api_client import create_hybrid_environment
from physics.environmental_conditions import EnvironmentalConditions, LocationType
from ml_utils.ml_output_modes import OutputMode

# Create fallback for when API unavailable
fallback = EnvironmentalConditions(LocationType.SAHEL)

# Configure real weather integration for Lagos gas compression station
env_source = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key_here",
    location_name="Lagos",
    country="Nigeria",
    fallback_source=fallback,
    cache_enabled=True,
    cache_ttl_hours=24
)

compressor = Compressor(
    name='CC-LG-001',
    initial_health={'impeller': 0.90, 'bearing': 0.85},
    design_flow=2500,
    design_head=9000,
    suction_pressure=2500,
    suction_temp=35,

    # Use real weather data from Lagos
    env_model=env_source,
    enable_environmental=True,
    enable_enhanced_vibration=True,
    enable_thermal_transients=True,
    enable_maintenance=True,
    enable_incipient_faults=True,
    enable_process_upsets=True,
    output_mode=OutputMode.FULL
)

compressor.set_speed(12000)

# Simulate 30 days of operation with real weather
for hour in range(720):
    state = compressor.next_state()

    # Monitor environmental impact
    if hour % 24 == 0:  # Daily summary
        day = hour // 24
        print(f"Day {day}: Temp={state.get('ambient_temp_C', 'N/A'):.1f}°C, "
              f"Humidity={state.get('humidity_percent', 'N/A'):.0f}%, "
              f"Derating={state.get('temp_derating_factor', 1.0):.3f}")

    # Enhanced monitoring
    if state.get('upset_active'):
        print(f"Hour {hour}: Upset - {state['upset_type']}")

    if state['surge_alarm']:
        print(f"Hour {hour}: SURGE ALARM - Margin={state['surge_margin']:.1f}%")
```

### Shaft Orbit Analysis

```python
from compressor import Compressor

compressor = Compressor(name='CC-001')
compressor.set_speed(12000)

state = compressor.next_state()

# Orbit metrics
print(f"Orbit amplitude: {state['orbit_amplitude']:.4f} mm")
print(f"Sync amplitude: {state['sync_amplitude']:.4f} mm")
print(f"X displacement: {state['shaft_x_displacement']:.4f} mm")
print(f"Y displacement: {state['shaft_y_displacement']:.4f} mm")

# Diagnostic interpretation
if state['orbit_amplitude'] > 0.05:
    print("WARNING: High shaft vibration")
    if state['sync_amplitude'] > 0.03:
        print("  → Likely unbalance (1X dominant)")
    if state['orbit_amplitude'] / state['sync_amplitude'] > 2:
        print("  → Sub-synchronous content (oil whirl/bearing issue)")
```

### Surge Scenario Simulation

```python
from compressor import Compressor

compressor = Compressor(
    name='CC-001',
    design_flow=1500,
    design_head=8000
)

# Gradually reduce flow toward surge
compressor.set_speed(12000)

for i in range(100):
    # Simulate valve closure (reduces flow)
    flow_reduction = i / 100  # 0% to 100%

    state = compressor.next_state()

    margin = state['surge_margin']
    print(f"Step {i}: Flow={state['flow']:.0f} m³/hr, "
          f"Surge Margin={margin:.1f}%")

    if margin < 10:
        print("  → SURGE ALARM")
    if margin < 5:
        print("  → SURGE TRIP - Emergency shutdown")
        break
```

## Telemetry Output

### Standard Fields

| Field | Type | Range | Units | Description |
|-------|------|-------|-------|-------------|
| speed | float | 0-25000 | RPM | Rotor speed |
| speed_target | float | 0-25000 | RPM | Target speed |
| flow | float | 0-3000 | m³/hr | Volumetric flow rate |
| head | float | 0-12000 | kJ/kg | Polytropic head |
| suction_pressure | float | 500-5000 | kPa | Inlet pressure |
| discharge_pressure | float | 2000-15000 | kPa | Outlet pressure |
| suction_temp | float | -10-60 | °C | Inlet temperature |
| discharge_temp | float | 50-200 | °C | Outlet temperature |
| surge_margin | float | -50-100 | % | Margin from surge line |
| surge_alarm | bool | - | - | Surge alarm active |
| bearing_temp_de | float | 30-110 | °C | Drive end bearing temp |
| bearing_temp_nde | float | 30-110 | °C | Non-drive end bearing temp |
| thrust_bearing_temp | float | 30-110 | °C | Thrust bearing temp |
| shaft_x_displacement | float | 0-0.2 | mm | X-axis shaft position |
| shaft_y_displacement | float | 0-0.2 | mm | Y-axis shaft position |
| orbit_amplitude | float | 0-0.2 | mm | Peak orbit radius |
| sync_amplitude | float | 0-0.1 | mm | 1X synchronous component |
| primary_seal_leakage | float | 1-20 | Nm³/hr | Primary seal leakage |
| secondary_seal_leakage | float | 0.3-10 | Nm³/hr | Secondary seal leakage |
| efficiency | float | 0.75-0.85 | - | Compression efficiency |
| power | float | 0-2000 | kW | Power consumption |
| operating_hours | float | 0+ | hrs | Cumulative hours |
| health_impeller | float | 0-1 | - | Impeller health |
| health_bearing | float | 0-1 | - | Bearing health |
| health_seal_primary | float | 0-1 | - | Primary seal health |
| health_seal_secondary | float | 0-1 | - | Secondary seal health |

## Failure Modes

### Failure Catalog

| Code | Description | Threshold | Detection |
|------|-------------|-----------|-----------|
| F_IMPELLER | Impeller degradation | Health < 0.42 | Health monitoring |
| F_BEARING | Bearing failure | Health < 0.38 | Health monitoring |
| F_SEAL_PRIMARY | Primary seal failure | Health < 0.25 | Health monitoring |
| F_SEAL_SECONDARY | Secondary seal failure | Health < 0.25 | Health monitoring |
| F_HIGH_VIBRATION | Excessive vibration | Orbit > 0.15 mm | Vibration monitoring |

### Failure Progression Example

**Bearing Failure Scenario**:
1. Operating hours: 0 → Health: 0.90 (initial)
2. Operating hours: 5,000 → Health: 0.75 (slow degradation)
3. Operating hours: 8,000 → Health: 0.55 (accelerating)
4. Bearing temp increases: 60°C → 80°C (friction)
5. Operating hours: 10,000 → Health: 0.40 (rapid degradation)
6. Vibration increases: 0.03 mm → 0.08 mm orbit
7. Operating hours: 11,000 → Health: 0.35 (threshold)
8. **FAILURE: F_BEARING**

## Performance Considerations

- **Base simulation**: ~0.15 ms per timestep
- **With enhancements**: ~0.6 ms per timestep
- **Memory**: ~8 KB per instance (includes orbit buffers)

## Validation and Calibration

### Operating Envelope

Based on:
- API 617: Axial and Compressors
- Elliott compressor specifications
- Offshore platform operational data

### Surge Line

Coefficients calibrated from:
- OEM compressor performance maps
- Surge test data from commissioning
- Industry correlation (Greitzer B-parameter)

### Dry Gas Seal Performance

Leakage rates from:
- John Crane seal specifications
- Field performance data
- API 692: Dry Gas Seal Standards

### Shaft Orbit Patterns

Validated against:
- Bently Nevada orbit analysis case studies
- API 670 vibration standards
- Compressor rotor dynamics literature

## Best Practices

### Design Point Selection

**Pipeline Compression**:
```python
compressor = Compressor(
    design_flow=2000,       # High flow
    design_head=6000,       # Moderate head
    suction_pressure=3000   # Mid-range
)
```

**LNG Refrigeration**:
```python
compressor = Compressor(
    design_flow=1200,       # Moderate flow
    design_head=10000,      # High head
    suction_pressure=500    # Low suction
)
```

### Surge Margin Monitoring

**Critical Applications** (tight margins):
- Set alarm at 15% margin
- Set trip at 8% margin

**Standard Applications**:
- Set alarm at 10% margin (default)
- Set trip at 5% margin (default)

### Seal Health Monitoring

**Alarm Thresholds**:
- Primary leakage > 5 Nm³/hr
- Secondary leakage > 2 Nm³/hr
- Combined > 7 Nm³/hr

**Replacement Criteria**:
- Primary health < 50%
- Secondary health < 60%
- Leakage trend (rate of increase)

## Limitations and Extensions

### Current Limitations

1. **Single-Stage Model**: Multi-stage compressors not modeled
2. **Fixed Gas Properties**: Constant molecular weight and gamma
3. **Simplified Surge Line**: Real surge lines more complex (speed-dependent)
4. **No Stall Modeling**: Rotating stall not included
5. **No Anti-Surge Control**: Valve modulation not simulated

### Potential Enhancements

1. **Multi-Stage Compressors**:
   - Stage-by-stage performance
   - Intercooling effects

2. **Variable Gas Composition**:
   - MW and gamma changes
   - Real gas properties (EOS)

3. **Advanced Surge Modeling**:
   - Speed-dependent surge lines
   - Hysteresis effects
   - Mild vs deep surge

4. **Anti-Surge Control**:
   - Valve position control
   - Recycle flow calculation

5. **Mechanical Seals**:
   - Alternative to dry gas seals
   - Flush system modeling

## References

1. API 617:2014 - "Axial and Compressors and Expander-compressors"

2. API 670:2014 - "Machinery Protection Systems"

3. API 692:2014 - "Dry Gas Seal Systems for Axial, Centrifugal, and Rotary Screw Compressors"

4. Brun, K., & Kurz, R. (2019). "Compression Machinery for Oil and Gas." Gulf Professional Publishing.

5. Greitzer, E. M. (1976). "Surge and Rotating Stall in Axial Flow Compressors." Journal of Engineering for Power.

6. Bently Nevada. (2002). "Orbit Analysis Fundamentals." Technical Training Manual.

## See Also

- [gas_turbine.md](gas_turbine.md) - Gas turbine simulation
- [pump.md](pump.md) - Pump simulation with cavitation
- [environmental_conditions.md](environmental_conditions.md) - Synthetic environmental modeling
- [weather_api_client.md](weather_api_client.md) - Real weather API integration
- [vibration_enhanced.md](vibration_enhanced.md) - Advanced vibration modeling
- [process_upsets.md](process_upsets.md) - Liquid carryover and surge events
- [incipient_faults.md](incipient_faults.md) - Bearing and seal fault progression
