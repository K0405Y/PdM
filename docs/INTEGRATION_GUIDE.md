# Integration Guide: Enhanced PdM Simulation Modules

This guide explains how to integrate the new enhanced simulation modules into your existing PdM data generation system.

## Overview of New Modules

### Physics Realism Enhancements
1. **vibration_enhanced.py** - Envelope-modulated vibration with bearing defect signatures
2. **thermal_transient.py** - Startup/shutdown thermal stress modeling
3. **maintenance_events.py** - Realistic maintenance interventions
4. **environmental_conditions.py** - Daily/seasonal environmental variability

### Performance Improvements
5. **pipeline_enhanced.py** - Generator-based output, parallel simulation, bulk DB insertion

### ML Readiness
6. **ml_output_modes.py** - Sensor-only mode, train/test splitting
7. **incipient_faults.py** - Discrete fault initiation and propagation
8. **process_upsets.py** - Process upset events for edge cases

---

## Quick Start: Step-by-Step Integration

### 1. Enhanced Vibration Signatures

**Objective**: Replace simple sinusoidal vibration with realistic envelope-modulated signals.

**Original Code** (gas_turbine.py):
```python
class VibrationSignalGenerator:
    def generate(self, rpm, health_state, duration=1.0):
        signal = np.zeros(n_samples)
        signal += 0.5 * np.sin(2 * np.pi * f_shaft * t)
        # ... more sinusoids
        return signal
```

**Enhanced Integration**:
```python
# At module level
from src.data_generation.vibration_enhanced import EnhancedVibrationGenerator, BearingGeometry

# In GasTurbine.__init__():
bearing_geometry = BearingGeometry(n_balls=9, ball_diameter=12.0, pitch_diameter=60.0)
self.vib_generator = EnhancedVibrationGenerator(
    sample_rate=10240,
    resonance_freq=3000,
    bearing_geometry=bearing_geometry
)

# In next_state():
vib_signal, vib_metrics = self.vib_generator.generate_bearing_vibration(
    rpm=self.speed,
    bearing_health=health_state.get('bearing', 1.0),
    duration=1.0
)

# Add metrics to telemetry state
state.update({
    'vibration_rms': vib_metrics['rms'],
    'vibration_peak': vib_metrics['peak'],
    'vibration_crest_factor': vib_metrics['crest_factor'],
    'vibration_kurtosis': vib_metrics['kurtosis']
})
```

**Benefits**:
- Realistic amplitude modulation patterns
- Detectable with envelope analysis
- Progresses from incipient to severe defects
- Additional diagnostic metrics (crest factor, kurtosis)

---

### 2. Thermal Transient Modeling

**Objective**: Model startup/shutdown thermal stress and operating modes.

**Integration in Equipment Simulators**:

```python
# In equipment __init__():
from src.data_generation.thermal_transient import ThermalTransientModel, ThermalMassProperties

self.thermal_model = ThermalTransientModel(
    ambient_temp=25.0,
    thermal_properties=ThermalMassProperties(
        tau_bearing=8.0,
        tau_casing=25.0,
        tau_rotor=45.0
    )
)

# In next_state() method:
# Update thermal model
thermal_state = self.thermal_model.step(
    target_speed=self.speed_target,
    rated_speed=self.LIMITS['speed_rated'],
    timestep_minutes=1/60  # Assuming 1-second timesteps
)

# Use degradation multiplier
severity = self._calculate_operating_severity()
severity *= thermal_state['degradation_multiplier']  # Apply thermal stress

# Add thermal data to telemetry
state.update({
    'operating_mode': thermal_state['operating_mode'],
    'temp_rotor_C': thermal_state['temp_rotor'],
    'temp_casing_C': thermal_state['temp_casing'],
    'differential_temp_C': thermal_state['differential_temp'],
    'thermal_stress': thermal_state['thermal_stress'],
    'startup_cycles': thermal_state['startup_cycles']
})
```

**Benefits**:
- 2-3x degradation during startup (realistic)
- Operating mode tracking
- Differential expansion stress
- Startup cycle counting

---

### 3. Maintenance Event Modeling

**Objective**: Periodic health restoration and imperfect maintenance simulation.

**Integration in Simulation Loop**:

```python
# In SimulationEngine class:
from src.data_generation.maintenance_events import MaintenanceScheduler, MaintenanceType

def _simulate_equipment(self, equipment, equipment_id, equipment_type):
    # Initialize maintenance scheduler
    maint_scheduler = MaintenanceScheduler(
        enable_time_based=True,
        enable_condition_based=True
    )

    for i in range(num_samples):
        # ... existing simulation code ...

        # Check maintenance required
        maint_type = maint_scheduler.check_maintenance_required(
            operating_hours=equipment.operating_hours,
            health_state=health_state,
            is_planned_shutdown=(duty_cycle[i] == False and duty_cycle[i+1] == False)
        )

        if maint_type:
            # Perform maintenance
            maint_action = maint_scheduler.perform_maintenance(
                maint_type,
                current_health=equipment.health_model.health,
                operating_hours=equipment.operating_hours,
                timestamp=sample_time
            )

            # Restore health
            equipment.health_model.health = maint_action.health_after

            # Log maintenance event
            logger.info(f"Maintenance performed on {equipment_id}: {maint_type.value}")

        # Check infant mortality
        if maint_scheduler.maintenance_history:
            last_maint = maint_scheduler.maintenance_history[-1]
            hours_since = equipment.operating_hours - last_maint.operating_hours_at_failure

            if maint_scheduler.check_infant_mortality(hours_since, last_maint.quality_factor):
                raise Exception("F_INFANT_MORTALITY")
```

**Benefits**:
- Realistic health recovery patterns
- Maintenance cost/downtime tracking
- Post-maintenance failures
- Multiple maintenance strategies

---

### 4. Environmental Variability

**Objective**: Add daily/seasonal cycles and location-specific characteristics.

**Integration**:

```python
# In SimulationEngine.__init__() or equipment __init__():
from src.data_generation.environmental_conditions import (
    EnvironmentalConditions, LocationType
)

self.env_model = EnvironmentalConditions(
    location_type=LocationType.OFFSHORE,  # or DESERT, ARCTIC, TROPICAL
    start_day_of_year=180  # Mid-summer
)

# In simulation loop:
env_conditions = self.env_model.get_conditions(elapsed_hours=i)

# Apply environmental impacts
self.ambient_temp = env_conditions['ambient_temp_C']
self.ambient_pressure = env_conditions['pressure_kPa']

# Update degradation based on environment
severity *= env_conditions['corrosion_factor']
severity *= env_conditions['fouling_factor']

# Add to telemetry
state.update(env_conditions)
```

**Benefits**:
- Realistic performance variations
- Location-specific degradation (salt, dust, ice)
- Temperature derating for turbines
- Weather event simulation

---

### 5. Memory-Efficient Generator-Based Pipeline

**Objective**: Replace memory-intensive batch processing with streaming generators.

**Old Approach** (data_pipeline.py):
```python
telemetry = []  # Accumulates all records in memory
for i in range(num_samples):
    state = equipment.next_state()
    telemetry.append(state)
return telemetry  # Returns huge list
```

**New Approach**:
```python
from src.data_generation.pipeline_enhanced import (
    GeneratorBasedSimulation,
    StreamingDataPipeline
)

# Create generator
gen_sim = GeneratorBasedSimulation(
    simulation_duration_days=180,
    sample_interval_minutes=10
)

# Stream telemetry directly to database
pipeline = StreamingDataPipeline(
    db_connection=db,
    batch_size=5000,
    use_bulk_insert=True  # Use PostgreSQL COPY
)

# Process stream
for equipment in equipment_list:
    stream = gen_sim.simulate_equipment_stream(
        equipment,
        equipment_id,
        equipment_type
    )

    pipeline.process_stream(
        stream,
        table_name='telemetry.gas_turbine_telemetry',
        column_mapping=TURBINE_TELEMETRY_MAPPING
    )
```

**Benefits**:
- O(batch_size) memory instead of O(total_records)
- 10-100x faster database insertion with COPY
- Progress tracking
- Can process datasets larger than RAM

---

### 6. Parallel Simulation

**Objective**: Utilize multiple CPU cores for independent equipment simulation.

**Implementation**:

```python
from src.data_generation.pipeline_enhanced import ParallelSimulator

# Prepare equipment configurations
equipment_configs = []
for turbine_id in turbine_ids:
    config = {
        'name': f'GT-{turbine_id}',
        'initial_health': {'hgp': 0.85, 'blade': 0.90, ...},
        'ambient_temp': 25.0
    }

    equipment_configs.append((
        GasTurbine,  # Equipment class
        turbine_id,  # ID
        config,      # Config dict
        {'duration_days': 180, 'sample_interval': 10, 'equipment_type': 'turbine'}
    ))

# Run parallel simulation
telemetry, failures = ParallelSimulator.simulate_equipment_parallel(
    equipment_configs,
    num_processes=8  # Use 8 CPU cores
)
```

**Benefits**:
- Near-linear speedup with CPU count
- Automatic load balancing
- Independent equipment instances

---

### 7. ML Output Modes

**Objective**: Create realistic evaluation datasets without ground-truth labels.

**Usage**:

```python
from src.data_generation.ml_output_modes import (
    DataOutputFormatter,
    OutputMode,
    TrainTestSplitter
)

# Training data: full labels
train_formatter = DataOutputFormatter(output_mode=OutputMode.FULL)

# Evaluation data: sensor-only (realistic)
eval_formatter = DataOutputFormatter(output_mode=OutputMode.SENSOR_ONLY)

# Production simulation: delayed labels (1 week lag)
prod_formatter = DataOutputFormatter(
    output_mode=OutputMode.DELAYED_LABELS,
    label_delay_hours=168
)

# Format records
for record in telemetry_stream:
    if is_training:
        formatted = train_formatter.format_record(record, timestamp)
    else:
        formatted = eval_formatter.format_record(record, timestamp)

    # Save formatted record

# Create proper train/test splits
train, test, train_fail, test_fail = TrainTestSplitter.stratified_failure_split(
    telemetry_records,
    failure_records,
    test_fraction=0.3
)
```

**Benefits**:
- Realistic model evaluation (no label leakage)
- Proper train/test splits
- Simulates production environment
- Adds realistic sensor noise

---

### 8. Incipient Fault Modeling

**Objective**: Discrete fault initiation events with gradual propagation.

**Integration**:

```python
from src.data_generation.incipient_faults import IncipientFaultSimulator

# In equipment __init__():
self.fault_sim = IncipientFaultSimulator(
    enable_incipient_faults=True,
    fault_rate_per_1000hrs=0.5
)

# In next_state():
# Check for new fault
fault_event = self.fault_sim.check_fault_initiation(
    operating_hours_increment=1/3600,  # 1 second
    stress_factor=severity,
    timestamp=current_time,
    operating_hours=self.operating_hours,
    component_list=['bearing', 'seal', 'impeller']
)

if fault_event:
    logger.warning(f"Fault initiated: {fault_event.fault_type.value} "
                  f"on {fault_event.affected_component}")

# Propagate existing faults
fault_sizes = self.fault_sim.propagate_faults(
    operating_hours_increment=1/3600,
    stress_factor=severity
)

# Adjust health for faults
health_state = self.health_model.step(severity)
health_state = self.fault_sim.adjust_health_for_faults(health_state)

# Add fault info to telemetry
state['active_faults'] = self.fault_sim.get_active_fault_summary()
```

**Benefits**:
- Realistic precursor signatures
- Discrete initiation events (crack, spall, contamination)
- Physics-based growth (Paris law for cracks)
- Multiple concurrent faults

---

### 9. Process Upset Events

**Objective**: Abnormal operating conditions and edge cases.

**Integration**:

```python
from src.data_generation.process_upsets import ProcessUpsetSimulator

# Initialize
upset_sim = ProcessUpsetSimulator(
    enable_upsets=True,
    upset_rate_per_month=2.0
)

# In simulation loop:
# Check for upset
upset_event = upset_sim.check_upset_initiation(
    timestep_seconds=60,
    timestamp=current_time,
    operating_state=state
)

if upset_event:
    logger.warning(f"Process upset: {upset_event.description}")

# Apply upset effects
state = upset_sim.apply_upset_effects(
    normal_state=state,
    equipment_type='pump'
)

# Apply upset damage
health = upset_sim.calculate_upset_damage(health)

# Add upset info to telemetry
if upset_sim.active_upset:
    state['upset_active'] = True
    state['upset_type'] = upset_sim.active_upset.upset_type.value
    state['upset_severity'] = upset_sim.active_upset.severity
```

**Benefits**:
- Edge case coverage (cavitation, surge, thermal shock)
- Rapid damage events
- Abnormal sensor patterns
- Enriched training data

---

## Complete Integration Example

Here's a complete example integrating all enhancements into a gas turbine simulator:

```python
from datetime import datetime, timedelta
import numpy as np

from src.data_generation.gas_turbine import GasTurbine
from src.data_generation.vibration_enhanced import EnhancedVibrationGenerator, BearingGeometry
from src.data_generation.thermal_transient import ThermalTransientModel
from src.data_generation.maintenance_events import MaintenanceScheduler
from src.data_generation.environmental_conditions import EnvironmentalConditions, LocationType
from src.data_generation.incipient_faults import IncipientFaultSimulator
from src.data_generation.process_upsets import ProcessUpsetSimulator
from src.data_generation.ml_output_modes import DataOutputFormatter, OutputMode


class EnhancedGasTurbine(GasTurbine):
    """Enhanced gas turbine with all new features."""

    def __init__(self, name, initial_health=None, location_type=LocationType.OFFSHORE):
        super().__init__(name, initial_health)

        # Enhanced vibration
        bearing_geom = BearingGeometry(n_balls=12, ball_diameter=15.0, pitch_diameter=75.0)
        self.vib_generator = EnhancedVibrationGenerator(
            sample_rate=10240,
            resonance_freq=2500,
            bearing_geometry=bearing_geom
        )

        # Thermal transient
        self.thermal_model = ThermalTransientModel(ambient_temp=25.0)

        # Maintenance
        self.maint_scheduler = MaintenanceScheduler()

        # Environment
        self.env_model = EnvironmentalConditions(location_type=location_type)

        # Incipient faults
        self.fault_sim = IncipientFaultSimulator(fault_rate_per_1000hrs=0.3)

        # Process upsets
        self.upset_sim = ProcessUpsetSimulator(upset_rate_per_month=1.5)

        # Output formatter
        self.output_formatter = DataOutputFormatter(OutputMode.FULL)

        self.elapsed_hours = 0.0

    def next_state_enhanced(self):
        """Enhanced next_state with all features."""

        # Get environmental conditions
        env_cond = self.env_model.get_conditions(self.elapsed_hours)
        self.ambient_temp = env_cond['ambient_temp_C']
        self.ambient_pressure = env_cond['pressure_kPa']

        # Update thermal model
        thermal_state = self.thermal_model.step(
            self.speed_target,
            self.LIMITS['speed_rated'],
            timestep_minutes=1/60
        )

        # Calculate operating severity
        severity = self._calculate_operating_severity()
        severity *= thermal_state['degradation_multiplier']
        severity *= env_cond['corrosion_factor']

        # Check for new fault
        fault_event = self.fault_sim.check_fault_initiation(
            1/3600, severity, datetime.now(), self.operating_hours,
            ['hgp', 'blade', 'bearing', 'fuel']
        )

        # Check for process upset
        upset_event = self.upset_sim.check_upset_initiation(
            60, datetime.now(), {'speed': self.speed}
        )

        # Advance health with faults
        health_state = self.health_model.step(severity)
        self.fault_sim.propagate_faults(1/3600, severity)
        health_state = self.fault_sim.adjust_health_for_faults(health_state)

        # Apply upset damage
        if self.upset_sim.active_upset:
            health_state = self.upset_sim.calculate_upset_damage(health_state)

        # Update base simulation
        self._update_speed()
        self._update_thermodynamics(health_state)

        # Enhanced vibration
        vib_signal, vib_metrics = self.vib_generator.generate_bearing_vibration(
            self.speed, health_state.get('bearing', 1.0), duration=1.0
        )

        # Check maintenance
        maint_type = self.maint_scheduler.check_maintenance_required(
            self.operating_hours, health_state, self.speed == 0
        )

        if maint_type:
            maint_action = self.maint_scheduler.perform_maintenance(
                maint_type, health_state, self.operating_hours, datetime.now()
            )
            self.health_model.health = maint_action.health_after

        # Build enhanced telemetry
        state = {
            'speed': self.speed,
            'egt': self.egt,
            'oil_temp': self.oil_temp,
            **vib_metrics,
            **env_cond,
            **thermal_state,
            **health_state,
            'active_faults': self.fault_sim.get_active_fault_summary(),
            'upset_active': self.upset_sim.active_upset is not None
        }

        # Format for ML
        state = self.output_formatter.format_record(state, datetime.now())

        self.elapsed_hours += 1/3600
        self.t += 1

        return state


# Usage
turbine = EnhancedGasTurbine('GT-001', location_type=LocationType.DESERT)
turbine.set_speed(10000)

for i in range(1000):
    state = turbine.next_state_enhanced()
    print(f"t={i}: EGT={state['egt']:.1f}°C, Vib={state['vibration_rms']:.3f} mm/s, "
          f"Mode={state['operating_mode']}")
```

---

## Performance Comparison

| Aspect | Original | Enhanced | Improvement |
|--------|----------|----------|-------------|
| Memory Usage | O(n_records) | O(batch_size) | 100-1000x |
| DB Insert Speed | 1-5k rows/s | 100-500k rows/s | 20-100x |
| Parallel Speedup | 1x (serial) | ~8x (8 cores) | 8x |
| Realism Score | Moderate | High | Qualitative |

---

## Migration Checklist

- [ ] Install new module dependencies (if any)
- [ ] Update equipment simulators with enhanced vibration
- [ ] Add thermal transient modeling
- [ ] Integrate maintenance scheduler
- [ ] Add environmental conditions
- [ ] Update database schema for new fields
- [ ] Switch to generator-based pipeline
- [ ] Implement parallel simulation
- [ ] Add bulk insertion (PostgreSQL COPY)
- [ ] Configure output modes for train/eval
- [ ] Enable incipient fault modeling
- [ ] Enable process upsets
- [ ] Update column mappings in data_ingestion.py
- [ ] Test end-to-end pipeline
- [ ] Validate data quality
- [ ] Benchmark performance improvements

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'vibration_enhanced'"

**Solution**: Ensure the new modules are in `src/data_generation/` and the path is in `sys.path`.

### Issue: "Memory usage still high with generators"

**Solution**: Ensure you're not accumulating results. Use streaming pipeline directly to database.

### Issue: "Multiprocessing not working on Windows"

**Solution**: Use `if __name__ == '__main__':` guard and pass equipment class as argument, not instance.

### Issue: "PostgreSQL COPY command fails"

**Solution**: Ensure you're using raw connection cursor and tab-delimited format. Check PostgreSQL permissions.

### Issue: "Thermal transient causes unrealistic temperatures"

**Solution**: Adjust time constants in `ThermalMassProperties` for your specific equipment.

---

## Next Steps

1. **Validate physics** - Compare simulated signals with real equipment data
2. **Benchmark ML models** - Train models on enhanced vs original data, compare performance
3. **Tune parameters** - Adjust fault rates, upset frequencies, maintenance intervals
4. **Extend to other equipment** - Apply enhancements to pumps and compressors
5. **Add more features** - Corrosion modeling, erosion tracking, etc.

For questions or issues, refer to individual module docstrings or contact the development team.