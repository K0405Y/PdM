# Maintenance Events Module

## Overview

The `maintenance_events.py` module simulates maintenance interventions including routine service, component replacement, major overhauls, and emergency repairs. It models realistic maintenance scheduling, imperfect maintenance restoration, and post-maintenance infant mortality to create comprehensive training data for predictive maintenance systems.

## Purpose

Maintenance is not a perfect restoration process. Real-world interventions have varying effectiveness, workmanship quality affects outcomes, and equipment sometimes fails shortly after maintenance (infant mortality). This module enables:

- **Realistic Maintenance Strategies**: Time-based, condition-based, and opportunistic scheduling
- **Imperfect Restoration**: Probabilistic health improvement based on maintenance quality
- **Infant Mortality**: Post-maintenance failure risk from reassembly errors or contamination
- **Maintenance Cost Tracking**: Economic optimization of maintenance strategies
- **Multiple Maintenance Types**: Routine, minor, major, and emergency interventions
- **Trigger Modeling**: Threshold-based and schedule-based maintenance initiation

Research shows that only 60-80% of preventive maintenance restores equipment to "like-new" condition, with significant variability based on workmanship. Additionally, 5-15% of failures occur within the first 100 operating hours after maintenance. Modeling these effects creates realistic training scenarios for ML systems.

## Key Features

- **4 Maintenance Types**: Routine, minor overhaul, major overhaul, emergency repair
- **3 Scheduling Strategies**: Time-based, condition-based, opportunistic
- **Probabilistic Restoration**: Health improvement with quality factor and random variation
- **Infant Mortality Modeling**: Early failure risk after maintenance
- **Component-Specific Actions**: Different components affected by each maintenance type
- **Cost and Downtime Tracking**: Economic analysis of maintenance strategies
- **Maintenance History**: Complete record of all interventions

## Module Components

### MaintenanceType Enum

Defines four categories of maintenance interventions:

```python
class MaintenanceType(Enum):
    ROUTINE = "routine"              # Oil/filter change
    MINOR_OVERHAUL = "minor_overhaul"  # Seal/bearing replacement
    MAJOR_OVERHAUL = "major_overhaul"  # Full refurbishment
    EMERGENCY = "emergency"            # Unplanned repair
```

### MaintenanceAction Dataclass

Records details of a maintenance intervention:

```python
@dataclass
class MaintenanceAction:
    maintenance_type: MaintenanceType  # Type of maintenance
    timestamp: datetime                 # Time of intervention
    components_affected: List[str]      # Which components were serviced
    health_before: Dict[str, float]     # Health before maintenance
    health_after: Dict[str, float]      # Health after maintenance
    cost: float                         # Cost in USD
    duration_hours: float               # Downtime duration
    quality_factor: float               # Workmanship quality (0.0 to 1.0)
```

### MaintenanceScheduler Class

Main class for managing maintenance scheduling and execution.

## Maintenance Types and Characteristics

### 1. Routine Maintenance

**Description**: Scheduled servicing including oil changes, filter replacement, and minor adjustments.

**Typical Activities**:
- Lubricating oil change
- Oil filter replacement
- Air filter replacement
- Visual inspection
- Alignment checks
- Vibration monitoring

**Schedule**: Every 2000 operating hours (default)

**Cost**: $2,000 ± 20% (base)

**Downtime**: 4 hours ± 20%

**Health Restoration**: 5% improvement (base)

**Components Affected**: All components (minor improvement across the board)

**Condition Threshold**: Health < 85%

**Physical Basis**:
- Fresh oil reduces friction and wear rate
- Clean filters improve lubrication effectiveness
- Adjustments correct minor misalignments
- Limited invasiveness, minimal restoration

### 2. Minor Overhaul

**Description**: Component-level replacement or refurbishment targeting degraded parts.

**Typical Activities**:
- Bearing replacement
- Seal replacement
- Gasket renewal
- Coupling alignment
- Balance check
- Limited disassembly

**Schedule**: Every 8000 operating hours (default)

**Cost**: $25,000 ± 20%

**Downtime**: 48 hours (2 days) ± 20%

**Health Restoration**: 25% improvement (base)

**Components Affected**: Worst 1-2 components (targeted replacement)

**Condition Threshold**: Health < 70%

**Physical Basis**:
- Replacing degraded bearings/seals restores local health
- Rebalancing reduces vibration
- Alignment improves load distribution
- Moderate restoration for replaced components

**Component Selection Logic**:
```python
# Target worst components
sorted_components = sorted(health.items(), key=lambda x: x[1])
components_to_service = [comp for comp, _ in sorted_components[:2]]
```

### 3. Major Overhaul

**Description**: Comprehensive refurbishment with full disassembly and restoration.

**Typical Activities**:
- Complete disassembly
- All bearings replaced
- All seals replaced
- Shaft inspection and repair
- Impeller/rotor refurbishment
- Casing inspection
- Alignment and balancing
- Performance testing

**Schedule**: Every 24,000 operating hours (default)

**Cost**: $150,000 ± 20%

**Downtime**: 240 hours (10 days) ± 20%

**Health Restoration**: 85% improvement (base)

**Components Affected**: All components (comprehensive restoration)

**Condition Threshold**: Health < 55%

**Physical Basis**:
- Near-complete restoration to like-new condition
- All wear surfaces renewed
- Systematic refurbishment of all subsystems
- High restoration but not perfect (residual fatigue, design limitations)

### 4. Emergency Maintenance

**Description**: Unplanned repair in response to critical health threshold or failure.

**Typical Activities**:
- Rapid diagnosis
- Minimal disassembly to access failure
- Quick fix or temporary repair
- Restore to operational state
- Limited testing

**Trigger**: Health < 20% (critical threshold)

**Cost**: $15,000 ± 20%

**Downtime**: 24 hours ± 20%

**Health Restoration**: 15% improvement (base)

**Components Affected**: Worst component only (targeted quick fix)

**Physical Basis**:
- Time-constrained repair (production pressure)
- Address immediate failure mode
- Limited scope, focus on operability not longevity
- Often a temporary fix until planned outage

**Important**: Emergency maintenance does NOT reset planned maintenance schedules (tracked separately from routine/minor/major cycles).

**Component Selection Logic**:
```python
# Only fix the worst component
worst_component = min(health.items(), key=lambda x: x[1])[0]
components_to_service = [worst_component]
```

## Maintenance Scheduling Strategies

### 1. Time-Based Scheduling

**Description**: Maintenance triggered by operating hours since last service.

**Parameters**:
- Routine interval: 2000 hours (default)
- Minor interval: 8000 hours (default)
- Major interval: 24000 hours (default)

**Logic**:
```python
hours_since_last = current_hours - last_maintenance_hours

if hours_since_major >= major_interval:
    return MaintenanceType.MAJOR_OVERHAUL
elif hours_since_minor >= minor_interval:
    return MaintenanceType.MINOR_OVERHAUL
elif hours_since_routine >= routine_interval:
    return MaintenanceType.ROUTINE
```

**Advantages**:
- Predictable scheduling (production planning)
- Simple to implement
- Regulatory compliance (mandated service intervals)

**Disadvantages**:
- May perform unnecessary maintenance (equipment still healthy)
- May miss deterioration between intervals (under-maintained)

**Typical Applications**:
- Safety-critical equipment (mandated intervals)
- Equipment without condition monitoring
- New equipment (following OEM recommendations)

### 2. Condition-Based Scheduling

**Description**: Maintenance triggered when component health falls below thresholds.

**Thresholds**:
- Routine: Health < 85%
- Minor overhaul: Health < 70%
- Major overhaul: Health < 55%
- Emergency: Health < 20% (highest priority)

**Logic**:
```python
min_health = min(component_health.values())

if min_health < 0.20:
    return MaintenanceType.EMERGENCY
elif min_health < 0.55:
    return MaintenanceType.MAJOR_OVERHAUL
elif min_health < 0.70:
    return MaintenanceType.MINOR_OVERHAUL
elif min_health < 0.85:
    return MaintenanceType.ROUTINE
```

**Priority Order**: Emergency > Major > Minor > Routine

**Advantages**:
- Optimizes maintenance timing (only when needed)
- Prevents over-maintenance
- Reduces unnecessary costs
- Extends equipment life

**Disadvantages**:
- Requires reliable condition monitoring
- Less predictable scheduling
- Risk of missing sudden failures

**Typical Applications**:
- Equipment with continuous monitoring (sensors)
- High-value equipment (cost of downtime justifies monitoring)
- Mature condition-based maintenance programs

### 3. Opportunistic Maintenance

**Description**: Maintenance during planned shutdowns even if not strictly required.

**Trigger**: Planned shutdown AND approaching maintenance due date

**Logic**:
```python
if is_planned_shutdown:
    hours_since_routine = current_hours - last_routine
    if hours_since_routine > routine_interval * 0.8:  # Within 80%
        return MaintenanceType.ROUTINE
```

**Threshold**: 80% of scheduled interval

**Example**: Routine maintenance due at 2000 hours. If planned shutdown occurs at 1600 hours (80%), perform maintenance opportunistically.

**Advantages**:
- Leverages existing downtime (no additional production loss)
- Prevents maintenance-induced downtime later
- Improves reliability during critical production periods

**Disadvantages**:
- May perform slightly premature maintenance
- Requires coordination with production scheduling

**Typical Applications**:
- Batch production (scheduled campaigns)
- Seasonal operations (annual turnarounds)
- Multi-unit facilities (staggered outages)

### Combined Strategy

The `MaintenanceScheduler` can use all three strategies simultaneously:

**Priority Order**:
1. **Condition-based** (highest priority): Immediate needs based on health
2. **Time-based**: Scheduled intervals
3. **Opportunistic**: Leverage shutdown windows

**Example Scenario**:
- Operating hours: 1800
- Component health: 87% (above all thresholds)
- Last routine: 0 hours
- Scheduled interval: 2000 hours
- Planned shutdown: Yes

**Decision**:
- Condition-based: No trigger (health > 85%)
- Time-based: No trigger (1800 < 2000)
- Opportunistic: **TRIGGER** (1800 > 0.8 * 2000 = 1600, and shutdown window available)
- **Result**: Perform routine maintenance

## Maintenance Restoration Model

### Health Improvement Calculation

Restoration is probabilistic, accounting for workmanship quality:

```python
# Quality factor (workmanship variability)
quality_factor = np.random.normal(0.9, 0.1)  # Mean 0.9, std 0.1
quality_factor = np.clip(quality_factor, 0.6, 1.0)  # Range: 0.6 to 1.0

# Base restoration for maintenance type
restoration = restoration_factors[maintenance_type]

# Quality affects restoration
restoration *= quality_factor

# Add random variation
improvement = restoration + np.random.normal(0, 0.02)

# Calculate new health
new_health = min(1.0, old_health + improvement)
```

### Restoration Factors

Base restoration amounts by maintenance type:

| Type | Base Restoration | Quality-Adjusted Range | Notes |
|------|------------------|------------------------|-------|
| Routine | 5% | 3% - 5% | Minor improvement |
| Minor Overhaul | 25% | 15% - 25% | Moderate restoration |
| Major Overhaul | 85% | 51% - 85% | Near-complete restoration |
| Emergency | 15% | 9% - 15% | Quick fix, limited restoration |

### Quality Factor Distribution

Quality factor follows normal distribution: `N(μ=0.9, σ=0.1)`, clipped to [0.6, 1.0]

**Distribution**:
- Mean: 0.9 (typical good workmanship)
- Standard deviation: 0.1
- Range: 0.6 to 1.0 (poor to perfect)

**Interpretation**:
- 0.6: Poor workmanship (rushed, inexperienced, inadequate tools)
- 0.8: Below average (some issues, training needed)
- 0.9: Good workmanship (typical skilled technician)
- 1.0: Perfect execution (ideal conditions, expert craftsman)

**Example: Minor Overhaul**
- Component health before: 60%
- Base restoration: 25%
- Quality factor: 0.85 (slightly below average)
- Actual restoration: 25% * 0.85 = 21.25%
- Random variation: +1.5% (example)
- Final improvement: 22.75%
- Health after: 60% + 22.75% = 82.75%

### Restoration Limits

Health cannot exceed 100%:
```python
new_health = min(1.0, old_health + improvement)
```

**Physical Basis**: Even perfect maintenance cannot exceed original design condition. Factors limiting full restoration:
- Residual fatigue damage (irreversible)
- Material aging
- Design limitations
- Dimensional changes from wear
- Metallurgical changes from thermal cycles

### Component-Specific Restoration

Different maintenance types affect different components:

**Routine**: All components (uniform minor improvement)
```python
all_components = list(health.keys())
components_affected = all_components
```

**Minor Overhaul**: Worst 1-2 components (targeted replacement)
```python
sorted_components = sorted(health.items(), key=lambda x: x[1])
components_affected = [comp for comp, _ in sorted_components[:2]]
```

**Major Overhaul**: All components (comprehensive restoration)
```python
all_components = list(health.keys())
components_affected = all_components
```

**Emergency**: Only worst component (quick fix)
```python
worst_component = min(health.items(), key=lambda x: x[1])[0]
components_affected = [worst_component]
```

**Example: Minor Overhaul**
- Components: bearing (50%), seal (65%), impeller (75%)
- Worst 2: bearing and seal
- Restoration: 25% to bearing and seal
- Result: bearing (75%), seal (90%), impeller (75% unchanged)

## Infant Mortality Model

### Post-Maintenance Failure Risk

Equipment has elevated failure risk immediately after maintenance due to:
- Improper reassembly
- Contamination during service
- Wrong parts or materials
- Insufficient tightening or over-torquing
- Misalignment
- Residual stresses from machining

### Infant Mortality Probability

```python
def check_infant_mortality(self, hours_since_maintenance, quality_factor):
    # Only applies in first 100 hours
    if hours_since_maintenance > 100:
        return False

    # Base failure rate (per hour)
    base_rate = 0.02  # 2% over 100 hours

    # Quality multiplier (poor quality = higher risk)
    quality_multiplier = 2.0 - quality_factor  # Range: 1.0 to 1.4

    # Time-dependent hazard (highest immediately after maintenance)
    time_factor = np.exp(-hours_since_maintenance / 20)  # Decays with τ=20 hrs

    # Effective failure rate
    effective_rate = base_rate * quality_multiplier * time_factor

    # Probabilistic check (per-hour probability)
    return random.random() < (effective_rate / 100)
```

### Failure Rate Characteristics

**Base Rate**: 2% cumulative risk over first 100 hours (if quality = 1.0)

**Quality Effect**:
- Quality 1.0 (perfect): 0.2% per hour at t=0, decaying to ~0% by 100 hours
- Quality 0.9 (good): 0.22% per hour at t=0
- Quality 0.7 (poor): 0.26% per hour at t=0
- Quality 0.6 (very poor): 0.28% per hour at t=0

**Time Decay**: Exponential with time constant τ = 20 hours

| Hours Since Maintenance | Relative Risk | Time Factor |
|------------------------|---------------|-------------|
| 0 | 100% | 1.0 |
| 10 | 61% | 0.61 |
| 20 | 37% | 0.37 |
| 40 | 14% | 0.14 |
| 60 | 5% | 0.05 |
| 100 | 0.7% | 0.007 |

**Cumulative Risk Examples**:
- Perfect quality (1.0): ~0.2% cumulative failure risk over 100 hours
- Good quality (0.9): ~0.4% cumulative risk
- Poor quality (0.7): ~1.2% cumulative risk
- Very poor quality (0.6): ~2.0% cumulative risk

### Detection and Response

**Usage in Simulation**:
```python
# After maintenance
maintenance_quality = action.quality_factor
hours_since_maintenance = 0

# Each timestep
hours_since_maintenance += timestep_hours

if simulator.check_infant_mortality(hours_since_maintenance, maintenance_quality):
    # Infant mortality failure occurred
    print("WARNING: Post-maintenance failure detected!")

    # Apply sudden health reduction
    for component in health:
        health[component] *= 0.7  # 30% health loss

    # Trigger emergency maintenance
    emergency_action = scheduler.perform_maintenance(
        MaintenanceType.EMERGENCY,
        health,
        operating_hours,
        timestamp
    )
```

## Cost and Downtime Tracking

### Cost Model

Base costs with ±20% variability:

| Maintenance Type | Base Cost | Typical Range | Notes |
|-----------------|-----------|---------------|-------|
| Routine | $2,000 | $1,600 - $2,400 | Consumables, labor |
| Minor Overhaul | $25,000 | $20,000 - $30,000 | Parts, skilled labor |
| Major Overhaul | $150,000 | $120,000 - $180,000 | Complete refurbishment |
| Emergency | $15,000 | $12,000 - $18,000 | Rush charges, overtime |

**Variability Sources**:
- Labor rate fluctuations
- Parts availability and pricing
- Contractor vs. in-house labor
- Overnight/weekend premiums
- Site accessibility

**Calculation**:
```python
base_cost = base_costs[maintenance_type]
cost = base_cost * np.random.uniform(0.8, 1.2)
```

### Downtime Model

Base downtime with ±20% variability:

| Maintenance Type | Base Downtime | Typical Range | Notes |
|-----------------|---------------|---------------|-------|
| Routine | 4 hours | 3.2 - 4.8 hours | Quick service |
| Minor Overhaul | 48 hours | 38 - 58 hours | 2-day outage |
| Major Overhaul | 240 hours | 192 - 288 hours | 10-day outage |
| Emergency | 24 hours | 19 - 29 hours | 1-day rush repair |

**Variability Sources**:
- Technician experience
- Parts lead time
- Unexpected findings during service
- Weather/access delays
- Test and startup time

**Calculation**:
```python
base_duration = base_durations[maintenance_type]
duration = base_duration * np.random.uniform(0.8, 1.2)
```

### Maintenance Summary

Track cumulative statistics:

```python
summary = scheduler.get_maintenance_summary()

{
    'total_events': 15,
    'total_cost': 387450.23,
    'total_downtime': 684.5,  # hours
    'by_type': {
        'routine': {'count': 8, 'cost': 16234.56, 'downtime': 32.1},
        'minor_overhaul': {'count': 5, 'cost': 121215.67, 'downtime': 240.4},
        'major_overhaul': {'count': 2, 'cost': 250000.00, 'downtime': 480.0},
        'emergency': {'count': 0, 'cost': 0, 'downtime': 0}
    }
}
```

## Usage Examples

### Basic Setup

```python
from maintenance_events import MaintenanceScheduler, MaintenanceType
from datetime import datetime, timedelta

# Initialize scheduler with all strategies enabled
scheduler = MaintenanceScheduler(
    enable_time_based=True,
    enable_condition_based=True,
    enable_opportunistic=True
)

# Equipment health state
health = {
    'bearing': 0.92,
    'seal': 0.88,
    'impeller': 0.90
}

operating_hours = 0.0
timestamp = datetime.now()
```

### Simulation Loop

```python
for step in range(10000):  # 10,000 hours
    operating_hours += 1.0
    timestamp += timedelta(hours=1)

    # Simulate degradation
    for component in health:
        health[component] -= np.random.uniform(0.00005, 0.00015)

    # Check for maintenance
    maintenance_needed = scheduler.check_maintenance_required(
        operating_hours,
        health,
        is_planned_shutdown=False  # Set True during planned outages
    )

    if maintenance_needed:
        # Perform maintenance
        action = scheduler.perform_maintenance(
            maintenance_needed,
            health,
            operating_hours,
            timestamp
        )

        # Update health with restored values
        health = action.health_after

        # Log maintenance event
        print(f"[{operating_hours:.0f} hrs] {maintenance_needed.value.upper()}")
        print(f"  Components: {action.components_affected}")
        print(f"  Cost: ${action.cost:,.0f}")
        print(f"  Downtime: {action.duration_hours:.1f} hrs")
        print(f"  Quality: {action.quality_factor:.2f}")
        print(f"  Health: {action.health_before} -> {action.health_after}")

    # Check infant mortality (if recent maintenance)
    if scheduler.maintenance_history:
        last_action = scheduler.maintenance_history[-1]
        hours_since = operating_hours - (operating_hours - 1)  # Simplification

        if scheduler.check_infant_mortality(hours_since, last_action.quality_factor):
            print(f"[{operating_hours:.0f} hrs] INFANT MORTALITY FAILURE!")
            # Apply sudden health reduction
            for comp in health:
                health[comp] *= 0.7
```

### Integration with Equipment Simulator

```python
from gas_turbine import GasTurbine
from maintenance_events import MaintenanceScheduler

turbine = GasTurbine(name='GT-001')
scheduler = MaintenanceScheduler()

operating_hours = 0.0

for timestep in range(simulation_steps):
    dt_hours = timestep_seconds / 3600
    operating_hours += dt_hours

    # Get turbine health
    health = turbine.get_component_health()

    # Check maintenance
    maint_needed = scheduler.check_maintenance_required(
        operating_hours,
        health,
        is_planned_shutdown=turbine.is_shutdown
    )

    if maint_needed:
        action = scheduler.perform_maintenance(
            maint_needed,
            health,
            operating_hours,
            datetime.now()
        )

        # Update turbine health
        turbine.set_component_health(action.health_after)

        # Stop turbine for maintenance duration
        turbine.shutdown(duration_hours=action.duration_hours)

        # Log to telemetry
        telemetry.append({
            'timestamp': datetime.now(),
            'event': 'maintenance',
            'type': maint_needed.value,
            'cost': action.cost,
            'downtime': action.duration_hours,
            'health_improvement': {
                comp: action.health_after[comp] - action.health_before[comp]
                for comp in action.health_after
            }
        })
```

### Maintenance Optimization Study

```python
def compare_strategies(simulation_hours=50000):
    """Compare different maintenance strategies."""

    strategies = [
        {'time': True, 'condition': False, 'opportunistic': False},  # Time-based only
        {'time': False, 'condition': True, 'opportunistic': False},  # Condition-based only
        {'time': True, 'condition': True, 'opportunistic': True}     # Combined
    ]

    results = []

    for strategy in strategies:
        scheduler = MaintenanceScheduler(
            enable_time_based=strategy['time'],
            enable_condition_based=strategy['condition'],
            enable_opportunistic=strategy['opportunistic']
        )

        health = {'bearing': 1.0, 'seal': 1.0, 'impeller': 1.0}
        operating_hours = 0.0

        for hour in range(simulation_hours):
            operating_hours += 1.0

            # Degradation
            for comp in health:
                health[comp] -= 0.0001

            # Maintenance
            maint = scheduler.check_maintenance_required(operating_hours, health)
            if maint:
                action = scheduler.perform_maintenance(
                    maint, health, operating_hours, datetime.now()
                )
                health = action.health_after

        summary = scheduler.get_maintenance_summary()
        summary['strategy'] = strategy
        summary['final_health'] = health
        results.append(summary)

    return results

# Run comparison
results = compare_strategies()

for result in results:
    print(f"\nStrategy: {result['strategy']}")
    print(f"Total cost: ${result['total_cost']:,.0f}")
    print(f"Total downtime: {result['total_downtime']:.1f} hrs")
    print(f"Maintenance events: {result['total_events']}")
    print(f"Final health: {result['final_health']}")
```

### Custom Maintenance Intervals

```python
# Heavy-duty operation (shorter intervals)
scheduler_heavy = MaintenanceScheduler()
scheduler_heavy.routine_interval = 1500    # 25% shorter
scheduler_heavy.minor_interval = 6000
scheduler_heavy.major_interval = 18000

# Light-duty operation (longer intervals)
scheduler_light = MaintenanceScheduler()
scheduler_light.routine_interval = 3000    # 50% longer
scheduler_light.minor_interval = 12000
scheduler_light.major_interval = 36000

# Aggressive condition-based (lower thresholds)
scheduler_cbm = MaintenanceScheduler()
scheduler_cbm.condition_thresholds = {
    'routine': 0.90,    # More frequent routine (was 0.85)
    'minor': 0.75,      # Earlier minor overhaul (was 0.70)
    'major': 0.60,      # Earlier major overhaul (was 0.55)
    'emergency': 0.25   # Higher emergency threshold (was 0.20)
}
```

## Validation and Calibration

### Maintenance Intervals

**OEM Recommendations**:
- Gas turbines: 8,000 hrs (hot gas path), 25,000-50,000 hrs (major overhaul)
- Centrifugal compressors: 3,000 hrs (routine), 15,000 hrs (overhaul)
- Pumps: 2,000 hrs (routine), 10,000 hrs (overhaul)

**Industry Standards**:
- API 610 (pumps): Maintenance intervals based on service severity
- API 617 (compressors): Inspection intervals 3-5 years typical
- ISO 14224: Reliability data for maintenance planning

**Model Default (2000/8000/24000)**: Representative of moderate-duty rotating equipment.

### Restoration Effectiveness

**Literature Values**:
- Preventive maintenance restores 60-80% of degradation (source: Reliability Engineering)
- Emergency repairs provide 10-20% improvement (quick fixes)
- Major overhauls achieve 80-95% restoration (comprehensive refurbishment)

**Model Values (5%/25%/85%/15%)**: Calibrated to typical improvement from maintenance activities, accounting for imperfect restoration.

### Quality Factor Distribution

Based on maintenance quality studies:
- Mean 0.9: Most maintenance performed by skilled technicians
- Std 0.1: Variability from workmanship, conditions, time pressure
- Range 0.6-1.0: Realistic bounds (very poor to perfect)

**Validation**: Compare simulated maintenance effectiveness distribution to CMMS post-maintenance performance metrics.

### Infant Mortality Rate

**Literature**:
- 5-15% of failures occur within first 100 hours after maintenance
- Higher rates for complex equipment and rush jobs
- Lower rates with experienced technicians and rigorous QC

**Model (0.2-2% depending on quality)**: Conservative estimate, primarily capturing reassembly errors rather than all post-maintenance issues.

## Performance Considerations

### Computational Cost

- **Maintenance check**: O(N) where N = number of components (typically 3-5)
- **Maintenance execution**: O(N) for health updates
- **Infant mortality check**: O(1) per timestep
- **Memory**: ~500 bytes per maintenance event record

### Timestep Recommendations

Works with any timestep. For infant mortality accuracy, use timestep ≤ 1 hour during first 100 hours post-maintenance.

## Limitations and Extensions

### Current Limitations

1. **Fixed Cost/Duration**: No variation based on equipment size or location
2. **Independent Components**: Components maintained independently (no system-level effects)
3. **Single Quality Factor**: Quality applies uniformly to all restoration work
4. **No Learning**: Maintenance effectiveness doesn't improve with technician experience
5. **Instant Restoration**: Health improves immediately (no burn-in period)

### Potential Enhancements

1. **Dynamic Costing**:
   - Scale costs with equipment size
   - Geographic cost multipliers
   - Inflation adjustments over simulation time

2. **System-Level Maintenance**:
   - Interdependent component restoration
   - Alignment affects multiple components
   - Cleaning benefits entire system

3. **Component-Specific Quality**:
   - Different quality factors per component
   - Complexity-dependent quality (bearings easier than seals)

4. **Learning Curves**:
   - Maintenance effectiveness improves with repetition
   - Technician skill tracking
   - Learning from failures

5. **Graduated Restoration**:
   - Health improves over initial break-in period
   - Run-in wear phase after maintenance
   - Settling time for adjustments

6. **Maintenance Planning**:
   - Optimize maintenance grouping
   - Parts availability constraints
   - Weather/access windows

## References

1. Moubray, J. (1997). "Reliability-Centered Maintenance" (2nd ed.). Industrial Press.

2. ISO 14224:2016 - Petroleum, petrochemical and natural gas industries -- Collection and exchange of reliability and maintenance data for equipment

3. API 610:2010 - Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries

4. API 617:2014 - Axial and Centrifugal Compressors and Expander-compressors

5. Smith, A. M., & Hinchcliffe, G. R. (2003). "RCM--Gateway to World Class Maintenance." Butterworth-Heinemann.

6. Nakagawa, T. (2005). "Maintenance Theory of Reliability." Springer.

7. Ebeling, C. E. (1997). "An Introduction to Reliability and Maintainability Engineering." McGraw-Hill.

8. Jardine, A. K., & Tsang, A. H. (2013). "Maintenance, Replacement, and Reliability: Theory and Applications" (2nd ed.). CRC Press.

9. Vachtsevanos, G., et al. (2006). "Intelligent Fault Diagnosis and Prognosis for Engineering Systems." Wiley.

10. Blanchard, B. S., et al. (1995). "Maintainability: A Key to Effective Serviceability and Maintenance Management." Wiley.

## See Also

- [incipient_faults.md](incipient_faults.md) - Fault initiation and growth (what maintenance repairs)
- [process_upsets.md](process_upsets.md) - Process disturbances (may trigger emergency maintenance)
- [thermal_transient.md](thermal_transient.md) - Startup/shutdown stress (accelerates maintenance needs)
- [environmental_conditions.md](environmental_conditions.md) - Environmental factors affecting maintenance intervals
