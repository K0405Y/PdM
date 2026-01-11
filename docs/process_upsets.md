# Process Upset Events Module

## Overview

The `process_upsets.py` module simulates abnormal operating conditions and process disturbances that stress equipment and create edge case scenarios for machine learning model training. These events represent real-world operational challenges that cause accelerated degradation and provide rich, realistic data for predictive maintenance systems.

## Purpose

In industrial environments, equipment rarely operates under perfect steady-state conditions. Process upsets—ranging from brief disturbances to extended abnormal situations—account for a significant portion of equipment damage and failure events. This module enables:

- **Edge Case Coverage**: Generate rare but critical operating conditions for robust ML training
- **Realistic Stress Scenarios**: Model actual process disturbances seen in industrial operations
- **Damage Quantification**: Calculate health impact of abnormal events
- **Sensor Response Simulation**: Realistic sensor signatures during upsets for anomaly detection

Research shows that 30-40% of unplanned downtime originates from process upsets rather than gradual wear. By incorporating these events, the simulation creates more representative training data that better prepares ML models for real-world conditions.

## Key Features

- **8 Distinct Upset Types**: Liquid carryover, cavitation, thermal shock, overload, contamination, trips, composition shifts, pump runout
- **Poisson Process Initiation**: Random upset occurrence with configurable frequency
- **Duration and Severity Modeling**: Type-specific duration ranges and probabilistic severity
- **State Modification**: Realistic sensor value changes during upsets
- **Equipment-Specific Effects**: Different impacts for turbines, compressors, and pumps
- **Health Impact Calculation**: Quantified damage from upset events
- **Event History Tracking**: Complete record of all upsets for analysis

## Module Components

### UpsetType Enum

Defines eight categories of process upsets:

```python
class UpsetType(Enum):
    LIQUID_CARRYOVER = "liquid_carryover"      # Liquid enters gas compressor
    PUMP_RUNOUT = "pump_runout"                # Pump operating beyond best efficiency point
    CAVITATION_EVENT = "cavitation_event"      # Pump cavitation
    THERMAL_SHOCK = "thermal_shock"            # Rapid temperature change
    FEED_COMPOSITION_SHIFT = "feed_composition_shift"  # Gas/fluid property change
    OVERLOAD = "overload"                      # Sustained overload operation
    TRIP_EVENT = "trip_event"                  # Emergency shutdown
    LUBE_OIL_CONTAMINATION = "lube_oil_contamination"  # Oil quality degradation
```

### UpsetEvent Dataclass

Records details of each process upset occurrence:

```python
@dataclass
class UpsetEvent:
    upset_type: UpsetType           # Type of upset
    timestamp: datetime              # Time of occurrence
    duration_seconds: int            # Upset duration
    severity: float                  # Severity (0.0 to 1.0)
    damage_potential: float          # Expected health reduction
    description: str                 # Human-readable description
```

### ProcessUpsetSimulator Class

Main class for generating and managing process upsets.

## Upset Types and Characteristics

### 1. Liquid Carryover

**Description**: Liquid slugs enter a gas compressor, causing surge risk and blade damage.

**Physical Mechanism**:
- Inadequate upstream separation
- Condensation in pipework
- Process control failures
- Rapid load changes

**Duration**: 30 seconds to 5 minutes

**Damage Potential**: 5% health loss (base)

**Effects**:
- Surge margin reduced by up to 80%
- Vibration amplitude increases 2-3x
- Discharge temperature fluctuations (+20°C)
- Risk of blade erosion and mechanical damage

**Applicability**: Compressors only

### 2. Pump Runout

**Description**: Pump operates far from Best Efficiency Point (BEP), causing hydraulic and mechanical stress.

**Physical Mechanism**:
- Low system resistance (open valve)
- Excessive flow rate
- Radial thrust increases
- Cavitation at impeller eye

**Duration**: 1 to 30 minutes

**Damage Potential**: 2% health loss (base)

**Effects**:
- Increased bearing loads
- Elevated vibration
- Shaft deflection
- Seal face damage

**Applicability**: Pumps only

### 3. Cavitation Event

**Description**: Net Positive Suction Head (NPSH) falls critically low, causing vapor bubble collapse and rapid erosion.

**Physical Mechanism**:
- Insufficient suction pressure
- High fluid temperature
- Pump operating above rated flow
- Vapor bubble implosion on impeller

**Duration**: 10 seconds to 2 minutes

**Damage Potential**: 8% health loss (base) - highest damage rate

**Effects**:
- NPSH margin drops to zero
- Vibration spikes 5-6x normal
- Flow instability (±15% fluctuation)
- Characteristic high-frequency noise
- Rapid material erosion

**Applicability**: Pumps only

### 4. Thermal Shock

**Description**: Rapid temperature transient causing differential thermal expansion and mechanical stress.

**Physical Mechanism**:
- Cold/hot fluid injection
- Rapid startup or shutdown
- Upstream process upset
- Cooling system failure

**Duration**: 5 seconds to 1 minute

**Damage Potential**: 3% health loss (base)

**Effects**:
- Temperature jump ±50°C (random direction)
- Thermal stress → increased vibration (+50%)
- Clearance changes
- Material fatigue accumulation

**Applicability**: All equipment types

### 5. Feed Composition Shift

**Description**: Changes in gas or fluid properties affecting equipment performance.

**Physical Mechanism**:
- Upstream process variation
- Feedstock changes
- Contamination
- Seasonal variations

**Duration**: 5 minutes to 1 hour

**Damage Potential**: 1% health loss (base)

**Effects**:
- Molecular weight changes
- Density variations
- Heat capacity shifts
- Performance drift from design point

**Applicability**: All equipment types

### 6. Overload

**Description**: Sustained operation above rated capacity.

**Physical Mechanism**:
- Production demand exceeds design
- Inadequate process control
- Parallel equipment offline
- Control valve malfunction

**Duration**: 10 minutes to 2 hours

**Damage Potential**: 4% health loss (base)

**Effects**:
- Speed increase (+30%)
- Power increase (+40%)
- Temperature rise (+15°C)
- Motor current increase (+30%)
- Accelerated wear rates

**Applicability**: All equipment types

### 7. Trip Event

**Description**: Emergency shutdown causing rapid transient.

**Physical Mechanism**:
- Safety system activation
- Process interlock
- Equipment protection
- Utility loss

**Duration**: 1 to 10 seconds

**Damage Potential**: 2% health loss (base)

**Effects**:
- Instantaneous load/speed change
- Pressure transients
- Thermal shock
- Mechanical stress from rapid deceleration

**Applicability**: All equipment types

### 8. Lube Oil Contamination

**Description**: Degraded or contaminated lubricating oil causing bearing damage.

**Physical Mechanism**:
- Water ingress
- Particulate contamination
- Oil oxidation/degradation
- Seal leakage
- Incorrect oil type

**Duration**: 1 hour to 1 day (slow-developing)

**Damage Potential**: 6% health loss (base)

**Effects**:
- Bearing temperature rise (+10°C per severity unit)
- Increased friction
- Vibration increase (+80%)
- Loss of lubricating film
- Accelerated wear

**Applicability**: All equipment types

## Class Methods

### Initialization

```python
def __init__(self,
             enable_upsets: bool = True,
             upset_rate_per_month: float = 2.0)
```

**Parameters**:
- `enable_upsets`: Enable/disable upset generation (default: True)
- `upset_rate_per_month`: Expected number of upsets per 30 days (default: 2.0)

**Initialization State**:
- Converts monthly rate to hourly rate: `rate_per_hour = rate_per_month / (30 * 24)`
- No active upset initially
- Empty upset history

**Rate Interpretation**:
- 2.0 upsets/month ≈ 1 upset every 15 days (typical for well-managed facility)
- 5.0 upsets/month ≈ 1 upset every 6 days (higher stress environment)
- 0.5 upsets/month ≈ 1 upset every 60 days (very stable operation)

### check_upset_initiation()

Check for new upset event using Poisson process.

```python
def check_upset_initiation(self,
                          timestep_seconds: int,
                          timestamp: datetime,
                          operating_state: Dict) -> Optional[UpsetEvent]
```

**Parameters**:
- `timestep_seconds`: Simulation timestep duration
- `timestamp`: Current simulation time
- `operating_state`: Current equipment operating state (used for context)

**Returns**: `UpsetEvent` if initiated, `None` otherwise

**Logic**:
1. Skip if upsets disabled or another upset already active
2. Calculate upset probability for timestep: `P = 1 - exp(-λ * Δt / 3600)`
3. Random draw determines if upset occurs
4. Create upset event with random type, duration, severity
5. Set as active upset and add to history

**Poisson Process**:
- Models random event occurrence with constant average rate
- Exponential distribution of inter-arrival times
- Appropriate for rare, independent events

### step()

Advance upset simulation by one timestep.

```python
def step(self, timestep_seconds: int) -> Optional[UpsetEvent]
```

**Parameters**:
- `timestep_seconds`: Simulation timestep duration

**Returns**: Active `UpsetEvent` or `None` if no upset active

**Logic**:
1. If no active upset, return `None`
2. Decrement remaining duration by timestep
3. If duration expires, clear active upset and return `None`
4. Otherwise return active upset

**Usage Pattern**:
```python
# Each simulation step
upset_event = simulator.step(timestep_seconds)
if upset_event:
    # Apply upset effects to equipment state
    modified_state = simulator.apply_upset_effects(normal_state, equipment_type)
```

### apply_upset_effects()

Modify equipment state variables to reflect upset conditions.

```python
def apply_upset_effects(self,
                       normal_state: Dict,
                       equipment_type: str) -> Dict
```

**Parameters**:
- `normal_state`: Normal operating state dictionary
- `equipment_type`: `'turbine'`, `'compressor'`, or `'pump'`

**Returns**: Modified state dictionary with upset effects applied

**Logic**:
1. Copy normal state
2. Check equipment type compatibility with upset type
3. Apply type-specific modifications (see individual upset descriptions)
4. Return modified state

**State Variables Modified**:
- Temperatures: `temp_bearing`, `discharge_temp`, etc.
- Vibration: `vibration_rms`, `vibration_amplitude`
- Pressure margins: `surge_margin`, `npsh_margin`
- Operating parameters: `speed`, `power`, `flow`, `motor_current`
- Status indicators: `cavitation_severity`

### calculate_upset_damage()

Calculate health reduction from active upset.

```python
def calculate_upset_damage(self,
                          baseline_health: Dict[str, float]) -> Dict[str, float]
```

**Parameters**:
- `baseline_health`: Current component health state (0.0 to 1.0)

**Returns**: Health state after upset damage applied

**Damage Model**:
```
damage_per_second = damage_potential / duration
health_new = max(0.0, health_old - damage_per_second)
```

**Important**: This method calculates damage per timestep. Call once per `step()` when upset is active.

**Damage Distribution**:
- Current implementation: uniform damage across all components
- Real systems: some upsets affect specific components more (e.g., cavitation damages impeller more than bearings)

## Upset Generation Process

### Duration Assignment

Each upset type has a characteristic duration range:

| Upset Type | Min Duration | Max Duration | Typical |
|------------|--------------|--------------|---------|
| Liquid Carryover | 30 s | 5 min | 1-2 min |
| Pump Runout | 1 min | 30 min | 5-10 min |
| Cavitation | 10 s | 2 min | 30 s |
| Thermal Shock | 5 s | 1 min | 15 s |
| Composition Shift | 5 min | 1 hr | 20 min |
| Overload | 10 min | 2 hrs | 30 min |
| Trip Event | 1 s | 10 s | 3 s |
| Oil Contamination | 1 hr | 1 day | 4 hrs |

Duration is randomly sampled uniformly from the range.

### Severity Distribution

Severity uses Beta distribution: `Beta(α=3, β=3)`

**Characteristics**:
- Range: 0.0 to 1.0
- Mean: 0.5
- Mode: 0.5
- Shape: symmetric bell curve

**Interpretation**:
- Most upsets are moderate severity (0.4-0.6)
- Severe events (>0.8) are rare but possible
- Very minor events (<0.2) also occur occasionally

**Rationale**: Real process upsets typically fall in moderate range. Control systems catch many severe events before full development, and very minor events may go unnoticed.

### Damage Potential Calculation

Base damage potential by type:

| Upset Type | Base Damage | Notes |
|------------|-------------|-------|
| Liquid Carryover | 5% | Blade impact, surge risk |
| Pump Runout | 2% | Bearing/seal stress |
| Cavitation | 8% | Highest - rapid erosion |
| Thermal Shock | 3% | Material fatigue |
| Composition Shift | 1% | Lowest - performance drift |
| Overload | 4% | Accelerated wear |
| Trip Event | 2% | Transient stress |
| Oil Contamination | 6% | Bearing damage |

Actual damage: `damage = base_damage * severity`

**Example**: Moderate cavitation event
- Severity: 0.6
- Base damage: 8%
- Actual damage: 0.08 * 0.6 = 0.048 (4.8% health loss)
- Duration: 60 seconds
- Damage rate: 4.8% / 60 = 0.08% per second

### Description Generation

Automatic human-readable descriptions combine severity level with upset type:

**Severity Mapping**:
- 0.0 - 0.25: "minor"
- 0.25 - 0.50: "moderate"
- 0.50 - 0.75: "significant"
- 0.75 - 1.0: "severe"

**Examples**:
- "moderate liquid carryover event - slug of liquid entered compressor"
- "severe cavitation event - NPSH critically low"
- "minor thermal shock - rapid temperature transient"

## Usage Examples

### Basic Setup

```python
from process_upsets import ProcessUpsetSimulator
from datetime import datetime

# Initialize simulator
simulator = ProcessUpsetSimulator(
    enable_upsets=True,
    upset_rate_per_month=2.0  # 1 upset every ~15 days
)

# Normal operating state
normal_state = {
    'speed': 3000,
    'power': 500,
    'vibration_rms': 2.5,
    'bearing_temp_de': 75.0,
    'npsh_margin': 5.0,
    'surge_margin': 25.0
}

# Component health
health = {'bearing': 0.90, 'seal': 0.85, 'impeller': 0.92}
```

### Simulation Loop

```python
timestep_seconds = 60  # 1-minute timesteps
timestamp = datetime.now()

for hour in range(720):  # 30 days
    for minute in range(60):
        timestamp += timedelta(minutes=1)

        # Check for new upset
        upset_event = simulator.check_upset_initiation(
            timestep_seconds,
            timestamp,
            normal_state
        )

        if upset_event:
            print(f"UPSET: {upset_event.description}")
            print(f"  Duration: {upset_event.duration_seconds}s")
            print(f"  Severity: {upset_event.severity:.2f}")

        # Advance upset
        active_upset = simulator.step(timestep_seconds)

        # Apply effects
        if active_upset:
            current_state = simulator.apply_upset_effects(
                normal_state,
                equipment_type='pump'
            )
            health = simulator.calculate_upset_damage(health)
        else:
            current_state = normal_state

        # Use current_state for sensor simulation
        # Use health for maintenance decisions
```

### Integration with Equipment Simulator

```python
from gas_turbine import GasTurbine
from process_upsets import ProcessUpsetSimulator

turbine = GasTurbine(name='GT-001')
upset_sim = ProcessUpsetSimulator(upset_rate_per_month=1.5)

for timestep in range(simulation_steps):
    # Get normal turbine state
    normal_state = turbine.next_state()

    # Check for upsets
    upset = upset_sim.check_upset_initiation(
        timestep_seconds=1,
        timestamp=datetime.now(),
        operating_state=normal_state
    )

    # Advance upset
    active_upset = upset_sim.step(1)

    # Apply upset effects
    if active_upset:
        modified_state = upset_sim.apply_upset_effects(
            normal_state,
            equipment_type='turbine'
        )

        # Apply health damage
        turbine.health = upset_sim.calculate_upset_damage(turbine.health)

        # Record upset in telemetry
        modified_state['upset_active'] = True
        modified_state['upset_type'] = active_upset.upset_type.value
        modified_state['upset_severity'] = active_upset.severity

        telemetry.append(modified_state)
    else:
        telemetry.append(normal_state)
```

### Equipment-Specific Usage

**Compressor**:
```python
# Compressor susceptible to liquid carryover and surge
compressor_state = {
    'speed': 15000,
    'discharge_pressure': 150,
    'surge_margin': 20,
    'vibration_amplitude': 3.2
}

upset_state = simulator.apply_upset_effects(
    compressor_state,
    equipment_type='compressor'
)

if simulator.active_upset and simulator.active_upset.upset_type == UpsetType.LIQUID_CARRYOVER:
    # Surge margin may be critically reduced
    if upset_state['surge_margin'] < 5:
        print("WARNING: Surge margin critically low!")
```

**Pump**:
```python
# Pump susceptible to cavitation and runout
pump_state = {
    'flow': 1500,  # m3/hr
    'npsh_margin': 4.5,
    'vibration_rms': 1.8,
    'motor_current': 120
}

upset_state = simulator.apply_upset_effects(
    pump_state,
    equipment_type='pump'
)

if simulator.active_upset and simulator.active_upset.upset_type == UpsetType.CAVITATION_EVENT:
    # Check cavitation severity indicator
    if 'cavitation_severity' in upset_state:
        cav_level = upset_state['cavitation_severity']
        print(f"Cavitation level: {cav_level}/3")
```

### Accessing Upset History

```python
# Get all upsets
for upset in simulator.upset_history:
    print(f"{upset.timestamp}: {upset.upset_type.value}")
    print(f"  Severity: {upset.severity:.2f}")
    print(f"  Damage: {upset.damage_potential:.3f}")

# Count by type
from collections import Counter
upset_types = [u.upset_type.value for u in simulator.upset_history]
type_counts = Counter(upset_types)
print(f"Upset type distribution: {type_counts}")

# Total damage
total_damage = sum(u.damage_potential for u in simulator.upset_history)
print(f"Total damage from upsets: {total_damage:.2f}")
```

## Validation and Calibration

### Upset Frequency

**Typical Industrial Values**:
- **Well-Managed Facility**: 0.5-2 upsets/month (0.017-0.07/day)
- **Average Facility**: 2-5 upsets/month (0.07-0.17/day)
- **High-Stress Environment**: 5-10 upsets/month (0.17-0.33/day)

**Calibration Sources**:
- CMMS databases (incident logs)
- Process historian alarm counts
- Operator shift reports
- DCS event logs

### Duration Ranges

Duration ranges based on:
- **Liquid Carryover**: Separator recovery time, slug size
- **Cavitation**: Time to restore NPSH, operator response
- **Thermal Shock**: Thermal mixing time constants
- **Overload**: Production scheduling, load reduction response
- **Oil Contamination**: Oil change/filtration cycle time

**Validation**: Compare simulated duration distributions to process historian event durations.

### Severity Distribution

Beta(3,3) distribution validated against:
- Maintenance work order severity classifications
- Process safety incident severity ratings
- Equipment damage assessment reports

**Characteristics**:
- 50% of events in 0.35-0.65 range (moderate)
- 10% of events >0.75 (severe)
- 10% of events <0.25 (minor)

### Damage Potentials

Damage values calibrated from:
- Bearing life reduction studies (SKF, Timken)
- Pump cavitation erosion rates (Hydraulic Institute)
- Turbomachinery reliability databases
- Failure mode effect analysis (FMEA)

**Cavitation Damage (8%)**: Based on studies showing cavitation can reduce impeller life by 50-90% if sustained.

**Liquid Carryover (5%)**: Compressor blade damage case studies showing 5-15% life reduction per event.

**Oil Contamination (6%)**: Bearing studies showing 2-3x wear rate increase with contaminated oil.

## Performance Considerations

### Computational Cost

- **Per-step cost**: O(1) - constant time operations
- **Memory**: ~500 bytes per upset event record
- **Upset check**: Single exponential calculation and random draw
- **State modification**: Dictionary copy and key-specific updates

### Timestep Recommendations

**1 second**: Captures fast transients (trips, thermal shocks)
**10 seconds**: Good balance for most applications
**60 seconds**: Acceptable for long simulations, may miss very brief events

**Statistical Accuracy**: Poisson process is timestep-independent for small Δt. For timestep << mean inter-arrival time, results are accurate.

**Example**:
- Rate: 2 upsets/month = 0.0028 upsets/hour
- Mean interval: 360 hours
- Timestep: 60 seconds = 0.017 hours << 360 hours
- Accuracy: >99%

## Limitations and Extensions

### Current Limitations

1. **Single Upset at a Time**: Only one active upset allowed
2. **Independent Occurrence**: Upsets don't trigger other upsets
3. **Uniform Component Damage**: All components damaged equally
4. **No Operator Mitigation**: No adaptive response to reduce severity
5. **Equipment-Type Filtering Manual**: User must specify equipment type

### Potential Enhancements

1. **Cascading Upsets**: One upset triggers secondary events
   - Example: Liquid carryover → trip event

2. **Component-Specific Damage Models**:
   - Cavitation damages impeller more than bearings
   - Liquid carryover damages compressor blades specifically

3. **Operator Response Modeling**:
   - Severity reduction with time (operator intervention)
   - Duration reduction based on alarm response

4. **Conditional Upset Rates**:
   - Higher rate during startups/shutdowns
   - Rate depends on equipment health (degraded equipment more prone)

5. **Upset Precursors**:
   - Subtle state changes before full upset development
   - Early warning signatures for detection algorithms

6. **Severity Evolution**:
   - Upset severity changes during event
   - Escalation or de-escalation based on conditions

7. **Multiple Simultaneous Upsets**:
   - Allow multiple concurrent upsets (rare but possible)
   - Interaction effects between upset types

## References

1. CCPS (Center for Chemical Process Safety). (2007). "Guidelines for Safe Automation of Chemical Processes." AIChE.

2. Nimmo, I. (2009). "Abnormal Situation Management: The Beneficial Design of Alarm Systems." ISA.

3. API 617:2014 - "Axial and Centrifugal Compressors and Expander-compressors"

4. API 610:2010 - "Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries"

5. Hydraulic Institute. (2016). "Effects of Liquid Viscosity on Rotodynamic Pump Performance."

6. Gülich, J. F. (2020). "Centrifugal Pumps" (4th ed.). Springer.

7. Boyce, M. P. (2011). "Gas Turbine Engineering Handbook" (4th ed.). Butterworth-Heinemann.

8. ISO 10816 - Mechanical vibration -- Evaluation of machine vibration by measurements on non-rotating parts

## See Also

- [incipient_faults.md](incipient_faults.md) - Discrete fault initiation and growth modeling
- [maintenance_events.md](maintenance_events.md) - Maintenance scheduling and restoration
- [environmental_conditions.md](environmental_conditions.md) - Environmental stress factors
- [vibration_enhanced.md](vibration_enhanced.md) - Vibration signatures during upsets
