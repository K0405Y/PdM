# Thermal Transient Module

## Overview

The `thermal_transient.py` module models temperature dynamics during equipment startup, shutdown, and load changes. It simulates differential thermal expansion between components with different thermal masses, capturing the thermal stress and accelerated degradation that occurs during transient operations.

## Purpose

Research shows that 60-70% of thermal fatigue in rotating equipment occurs during transients rather than steady-state operation. Thermal stresses during rapid startups can cause 2-3x normal degradation rates due to:

- Differential expansion between rotor and casing
- Thermal gradients causing mechanical stress
- Repeated thermal cycling (low-cycle fatigue)
- Clearance variations affecting vibration and rubbing

This module provides physics-based thermal modeling to:
- Generate realistic degradation patterns for ML training
- Identify high-stress operating periods
- Support maintenance optimization studies
- Enable condition monitoring during transients

## Key Features

- **Operating Mode State Machine**: Seven distinct operating states
- **Multi-Component Thermal Modeling**: Separate time constants for bearings, casing, and rotor
- **Differential Expansion Tracking**: Monitors rotor-casing temperature difference
- **Degradation Multipliers**: Mode-dependent accelerated wear factors
- **Startup Cycle Counting**: Tracks cumulative thermal cycles
- **Configurable Thermal Properties**: Customizable time constants and stress limits

## Module Components

### OperatingMode Enum

Defines seven equipment operating states:

```python
class OperatingMode(Enum):
    COLD_STANDBY = "cold_standby"      # Stopped, at ambient temperature
    STARTUP = "startup"                 # Accelerating from standby
    LOADING = "loading"                 # Increasing load at constant speed
    STEADY_STATE = "steady_state"       # Normal operation at rated conditions
    UNLOADING = "unloading"            # Decreasing load
    SHUTDOWN = "shutdown"               # Decelerating to stop
    HOT_STANDBY = "hot_standby"        # Stopped but still warm
```

### ThermalMassProperties Dataclass

Encapsulates thermal characteristics of equipment components:

```python
@dataclass
class ThermalMassProperties:
    tau_bearing: float = 8.0           # Bearing time constant (minutes)
    tau_casing: float = 25.0           # Casing time constant (minutes)
    tau_rotor: float = 45.0            # Rotor time constant (minutes)
    max_differential: float = 80.0     # Maximum safe ΔT (°C)
    stress_per_deg: float = 0.015      # Stress factor per °C differential
```

**Time Constants (τ):**
- Time for component to reach 63.2% of temperature change
- Smaller τ = faster heating/cooling response
- Bearings heat quickly (small thermal mass)
- Rotors heat slowly (large thermal mass)

**Physical Basis:**
- Bearings: Lightweight, good conduction to oil
- Casing: Medium mass, large surface area for convection
- Rotor: Heavy, insulated by air gap, slow conduction path

### ThermalTransientModel Class

Main class for simulating thermal dynamics.

## Operating Mode State Machine

### State Transitions

The model automatically transitions between operating modes based on speed and temperature:

```
COLD_STANDBY → STARTUP → LOADING → STEADY_STATE → UNLOADING → SHUTDOWN → HOT_STANDBY
      ↑                                                                        ↓
      └─────────────────────────── (cool down) ──────────────────────────────┘
```

### Transition Logic

**Speed Ratio = Current Speed / Rated Speed**

| Speed Ratio | Previous Mode | New Mode | Condition |
|-------------|---------------|----------|-----------|
| < 0.05 | Any | COLD_STANDBY | Stopped and rotor temp < ambient + 20°C |
| < 0.05 | Any | HOT_STANDBY | Stopped but rotor still warm |
| 0.05 - 0.30 | COLD/HOT_STANDBY | STARTUP | Accelerating from stop |
| 0.05 - 0.30 | STEADY_STATE | SHUTDOWN | Decelerating |
| 0.30 - 0.60 | STARTUP | LOADING | Increasing load |
| 0.30 - 0.60 | STEADY_STATE | UNLOADING | Decreasing load |
| > 0.60 | Any | STEADY_STATE | Normal operation |

### Mode Duration Tracking

Duration counter resets on each mode change. This enables time-dependent behavior:
- High degradation during first 30 minutes of startup
- Mode-specific stress calculations
- Startup cycle counting

## Thermal Modeling

### Component Temperature Dynamics

Each component (bearing, casing, rotor) follows first-order exponential thermal response:

```
dT/dt = (T_target - T_current) / τ
```

**Discrete Solution:**
```
T_new = T_current + (1 - e^(-Δt/τ)) * (T_target - T_current)
```

Where:
- `T_current`: Current component temperature (°C)
- `T_target`: Target temperature based on load (°C)
- `τ`: Component time constant (minutes)
- `Δt`: Timestep duration (minutes)

### Target Temperature Calculation

Target temperatures are load-dependent:

**Stopped (load < 5%):**
- Bearing: ambient + 5°C
- Casing: ambient + 10°C
- Rotor: ambient + 15°C

**Operating (load ≥ 5%):**
- Bearing: ambient + 30 + 60 * load_fraction (°C)
- Casing: ambient + 25 + 80 * load_fraction (°C)
- Rotor: ambient + 35 + 120 * load_fraction (°C)

**Example at full load (100%):**
- Bearing: ambient + 90°C
- Casing: ambient + 105°C
- Rotor: ambient + 155°C

**Physical Basis:**
- Bearings heated by friction and oil temperature
- Casing heated by gas path and external convection
- Rotor heated internally, limited cooling

### Differential Expansion

Temperature difference between rotor and casing causes mechanical stress:

```
ΔT = |T_rotor - T_casing|
```

**Normalized Thermal Stress:**
```
thermal_stress = min(1.0, ΔT / ΔT_max)
```

Where `ΔT_max = 80°C` (typical maximum allowable differential)

**Physical Impact:**
- Large ΔT → clearance variations → potential rubbing
- Thermal gradients → mechanical stress in shaft/casing
- Rapid ΔT changes → thermal shock and fatigue

## Degradation Multipliers

Degradation rate multipliers quantify accelerated wear during transients.

### Base Multiplier: 1.0 (Steady State)

Normal degradation during steady-state operation serves as baseline.

### Mode-Specific Multipliers

#### Startup: 2.5x to 1.2x

Most critical mode. Multiplier decreases with time in mode:

```python
if mode_duration < 30 minutes:
    startup_factor = 2.5 - (mode_duration / 30) * 1.5  # 2.5 → 1.0
else:
    startup_factor = 1.2
```

**First 30 minutes:** 2.5x → 1.0x (linear decrease)
**After 30 minutes:** 1.2x (elevated but stable)

**Physical Basis:**
- Initial thermal shock highest
- Clearances changing rapidly
- Thermal gradients steepest
- Unsteady oil temperatures

#### Loading: 1.4x

Elevated stress while increasing power output:
- Thermal gradients increasing
- Dynamic response requirements
- Control system activity

#### Unloading: 1.2x

Moderate stress during power reduction:
- Cooling gradients
- Reverse thermal transients
- Lower absolute stress than loading

#### Shutdown: 1.3x

Moderate to high stress during coastdown:
- Rapid cooling
- Loss of forced convection
- Natural convection settling
- Temperature redistribution

#### Steady State: 1.0x

Baseline normal operation:
- Thermal equilibrium
- Stable clearances
- Predictable conditions

#### Standby States: 1.0x

No additional stress when stopped:
- Components at equilibrium
- No mechanical stress
- Only aging/environmental degradation

### Thermal Stress Contribution

Thermal stress amplifies degradation exponentially:

```python
thermal_factor = 1.0 + 2.0 * (thermal_stress ** 2)
```

**Example Values:**

| ΔT | Normalized Stress | Thermal Factor |
|----|-------------------|----------------|
| 0°C | 0.0 | 1.0x |
| 20°C | 0.25 | 1.125x |
| 40°C | 0.5 | 1.5x |
| 60°C | 0.75 | 2.125x |
| 80°C | 1.0 | 3.0x |

### Combined Degradation Multiplier

```
degradation_multiplier = mode_multiplier * thermal_factor
```

Capped at 5.0x maximum (prevents unrealistic acceleration)

**Example Scenario: Rapid Cold Start**
- Mode: STARTUP (5 minutes in) → 2.25x
- ΔT: 50°C → thermal_stress = 0.625 → thermal_factor = 1.78x
- Combined: 2.25 * 1.78 = 4.0x degradation

## Class Methods

### Initialization

```python
def __init__(self,
             ambient_temp: float = 25.0,
             thermal_properties: ThermalMassProperties = None)
```

**Parameters:**
- `ambient_temp`: Ambient temperature (°C), default 25.0
- `thermal_properties`: Custom thermal properties (optional, uses defaults if None)

**Initial State:**
- All components at ambient temperature
- Operating mode: COLD_STANDBY
- Mode duration: 0 minutes
- Startup count: 0

### step(target_speed, rated_speed, timestep_minutes)

Advance thermal model by one timestep.

**Parameters:**
- `target_speed`: Target rotor speed (RPM)
- `rated_speed`: Rated speed for normalization (RPM)
- `timestep_minutes`: Simulation timestep (minutes), default 1/60 (1 second)

**Returns:** Dictionary containing:
- `operating_mode`: Current mode (string)
- `mode_duration_min`: Time in current mode (minutes)
- `temp_bearing`: Bearing temperature (°C)
- `temp_casing`: Casing temperature (°C)
- `temp_rotor`: Rotor temperature (°C)
- `differential_temp`: |T_rotor - T_casing| (°C)
- `thermal_stress`: Normalized stress (0.0 to 1.0)
- `degradation_multiplier`: Degradation rate multiplier
- `startup_cycles`: Cumulative startup count

### get_thermal_state()

Get current thermal state for diagnostics.

**Returns:** Dictionary with current thermal state (similar to step() output)

## Usage Examples

### Basic Startup Simulation

```python
from thermal_transient import ThermalTransientModel

# Initialize model
model = ThermalTransientModel(ambient_temp=25.0)

# Simulate startup sequence
rated_speed = 10000  # RPM

# Start from cold
for minute in range(60):
    state = model.step(
        target_speed=10000,
        rated_speed=rated_speed,
        timestep_minutes=1.0
    )

    if minute % 10 == 0:
        print(f"Minute {minute}: Mode={state['operating_mode']}, "
              f"ΔT={state['differential_temp']:.1f}°C, "
              f"Degradation={state['degradation_multiplier']:.2f}x")
```

### Integration with Gas Turbine Simulator

```python
from gas_turbine import GasTurbine
from thermal_transient import ThermalTransientModel

turbine = GasTurbine(name='GT-001')
thermal_model = ThermalTransientModel(ambient_temp=25.0)

# Simulation loop
for hour in range(1000):
    # Get thermal state
    thermal_state = thermal_model.step(
        target_speed=turbine.speed,
        rated_speed=turbine.rated_speed,
        timestep_minutes=1/60  # 1 second timesteps
    )

    # Apply thermal degradation to turbine health
    degradation_rate = base_degradation * thermal_state['degradation_multiplier']
    turbine.health_hgp -= degradation_rate * dt

    # Add thermal state to telemetry
    telemetry = turbine.next_state()
    telemetry.update(thermal_state)
```

### Complete Start-Stop Cycle

```python
model = ThermalTransientModel(ambient_temp=25.0)
rated_speed = 10000

# Idle (10 minutes)
for _ in range(10):
    state = model.step(0, rated_speed, 1.0)

# Startup to 50% load (30 minutes)
for _ in range(30):
    state = model.step(5000, rated_speed, 1.0)

# Load to 100% (20 minutes)
for _ in range(20):
    state = model.step(10000, rated_speed, 1.0)

# Steady operation (60 minutes)
for _ in range(60):
    state = model.step(10000, rated_speed, 1.0)

# Shutdown (30 minutes)
for _ in range(30):
    state = model.step(0, rated_speed, 1.0)

print(f"Total startup cycles: {state['startup_cycles']}")
```

### Custom Thermal Properties

```python
from thermal_transient import ThermalTransientModel, ThermalMassProperties

# Heavy-duty industrial turbine (slower thermal response)
heavy_duty_props = ThermalMassProperties(
    tau_bearing=12.0,    # Slower than default (8 min)
    tau_casing=40.0,     # Slower than default (25 min)
    tau_rotor=75.0,      # Slower than default (45 min)
    max_differential=100.0,  # Higher tolerance
    stress_per_deg=0.012     # Lower stress per degree
)

model = ThermalTransientModel(
    ambient_temp=30.0,
    thermal_properties=heavy_duty_props
)
```

## Validation and Calibration

### Time Constants

Time constants are based on literature and manufacturer data:

**Bearings (τ = 8 min):**
- Small thermal mass (1-10 kg)
- Good thermal conduction to oil system
- Typical response: 5-15 minutes to equilibrium

**Casing (τ = 25 min):**
- Medium thermal mass (100-1000 kg)
- Surface convection dominates
- Typical response: 15-45 minutes to equilibrium

**Rotor (τ = 45 min):**
- Large thermal mass (500-5000 kg)
- Poor thermal conduction (air gap insulation)
- Typical response: 30-90 minutes to equilibrium

### Degradation Multipliers

Based on research literature:

**Startup Multiplier (2.5x):**
- Kurz & Brun (2012): "60-70% of thermal fatigue during transients"
- Implies 2-3x higher stress during startup vs. steady operation
- Matches observed bearing failure rates post-startup

**Differential Temperature Limits:**
- API 670 guidelines: Typical maximum ΔT = 50-100°C
- Conservative limit: 80°C used in model
- Manufacturer startup procedures limit ΔT rate to 2-5°C/min

### Operating Mode Transitions

Speed thresholds calibrated to typical turbine startup procedures:
- 0-5% speed: Turning gear / standby
- 5-30% speed: Accelerating through critical speeds
- 30-60% speed: Loading / synchronization
- 60-100% speed: Normal operation

## Performance Considerations

### Computational Cost

- **Per-step cost**: O(1) - constant time
- **Memory**: ~200 bytes (state variables)
- **CPU**: Negligible (simple exponential calculations)

### Timestep Selection

**Recommended timesteps:**
- **1 second (1/60 min)**: High fidelity, smooth transients
- **10 seconds (1/6 min)**: Good balance for most applications
- **1 minute**: Acceptable for long simulations, faster processing

**Stability:**
Model is numerically stable for any positive timestep (first-order exponential is unconditionally stable).

**Accuracy:**
Smaller timesteps capture faster dynamics better. For startup sequences, use Δt ≤ 1 minute.

## Limitations and Future Enhancements

### Current Limitations

1. **Simplified Heat Transfer**: First-order exponential (no spatial gradients)
2. **Uniform Component Temperatures**: Single temperature per component
3. **Fixed Time Constants**: Does not vary with operating conditions
4. **No Cooling System Modeling**: Oil/air cooling implicitly included
5. **Symmetric Heating/Cooling**: Same τ for heating and cooling

### Potential Enhancements

1. **Spatial Temperature Distribution**: FEA-style multi-node thermal network
2. **Dynamic Time Constants**: Vary τ with flow rates and heat transfer coefficients
3. **Cooling System Modeling**: Explicit oil temperature and flow simulation
4. **Asymmetric Response**: Different τ for heating vs. cooling
5. **Thermal Bowing**: Rotor deflection due to thermal gradients
6. **Clearance Calculation**: Radial clearance based on thermal expansion
7. **Material Properties**: Temperature-dependent material properties

## References

1. API 670:2014 - Machinery Protection Systems
2. Kurz, R., & Brun, K. (2012). "Degradation in Gas Turbine Systems." Journal of Engineering for Gas Turbines and Power, 134(3).
3. Boyce, M. P. (2011). "Gas Turbine Engineering Handbook" (4th ed.). Butterworth-Heinemann.
4. Diakunchak, I. S. (1992). "Performance Degradation in Industrial Gas Turbines." Journal of Engineering for Gas Turbines and Power, 114(2), 161-168.
5. Gülen, S. C. (2019). "Gas Turbines for Electric Power Generation." Cambridge University Press.

## See Also

- `environmental_conditions.py` - Environmental impact on equipment performance
- `vibration_enhanced.py` - Vibration increases during thermal transients
- `maintenance_events.py` - Thermal cycling tracked for maintenance scheduling
- `gas_turbine.py` - Main turbine simulator integrating thermal model
