# Equipment Simulation Module

## Overview

The `equipment_sim.py` module provides equipment simulation with continuous operation through maintenance cycles. Unlike simple run-to-failure simulation (which stops at first failure), this module continues generating telemetry records after failures, modeling real-world maintenance and repair operations.

## Purpose

Real-world industrial equipment doesn't stop permanently after a failure - it gets repaired and continues operating. This module enables:

- **Continuous Telemetry Generation**: Equipment generates records through multiple failure-repair cycles
- **Targeted Component Repair**: Only the failed component is restored, not all components
- **Maintenance Period Modeling**: Equipment enters idle state during maintenance downtime
- **Parallel/Sequential Processing**: Multi-core simulation for large equipment fleets
- **Memory-Efficient Streaming**: Generator-based output for large datasets

## Key Features

- **Continuous Operation Model**: Equipment continues generating records after failures
- **Targeted Repair Logic**: Only failed component is repaired (85-92% health restoration)
- **Maintenance Downtime**: Configurable downtime period after each failure
- **Four Record Types**: telemetry, failure, maintenance_start, maintenance_complete
- **Parallel Processing**: Utilizes all CPU cores for fleet simulation
- **Streaming Output**: Memory-efficient generator-based architecture

## Module Components

### Core Functions

#### simulate_equipment()

Main simulation function for a single piece of equipment with maintenance/repair after failures.

```python
def simulate_equipment(
    equipment,                          # Equipment simulator instance
    equipment_id: int,                  # Equipment ID
    equipment_type: str,                # 'turbine', 'compressor', or 'pump'
    duration_days: int,                 # Simulation duration
    sample_interval_min: int,           # Sampling interval in minutes
    start_time: datetime = None,        # Optional start timestamp
    degradation_multiplier: float = 1.0,# Accelerate degradation (>1.0 = faster)
    include_equipment_type: bool = False,# Include type in records
    maintenance_downtime_hours: float = 24.0  # Repair duration
) -> Generator[Dict, None, None]
```

**Returns**: Generator yielding dictionaries with `type` field indicating record type.

#### _get_failed_component()

Parses failure codes to identify the failed component for targeted repair.

```python
def _get_failed_component(failure_code: str) -> str:
    """
    Parse failure code to identify the failed component.

    Failure codes follow pattern: F_COMPONENT or F_COMPONENT_SUBTYPE
    Examples: F_BEARING, F_FUEL, F_SEAL_PRIMARY, F_HIGH_VIBRATION
    """
```

**Failure Code Mappings**:

| Failure Code | Component |
|--------------|-----------|
| F_HIGH_VIBRATION | bearing |
| F_BEARING_TEMP | bearing |
| F_VIB_TRIP | bearing |
| F_SURGE | impeller |
| F_HGP | hgp |
| F_BLADE | blade |
| F_BEARING | bearing |
| F_FUEL | fuel |
| F_IMPELLER | impeller |
| F_SEAL | seal |
| F_SEAL_PRIMARY | seal_primary |
| F_SEAL_SECONDARY | seal_secondary |

#### _repair_equipment()

Repairs equipment after failure by restoring only the failed component's health.

```python
def _repair_equipment(
    equipment,                    # Equipment simulator instance
    equipment_type: str,          # 'turbine', 'compressor', or 'pump'
    failure_code: str             # Failure mode code that triggered repair
) -> dict                         # Health values after repair
```

**Repair Behavior**:
- **Target**: Only the failed component identified from failure code
- **Health Range**: 85-92% (randomized within range)
- **Other Components**: Retain current degraded health values
- **Post-Repair**: Equipment set to idle state (speed = 0)
- **Generator Reset**: Health degradation generators reinitialized

#### simulate_parallel()

Simulates multiple equipment in parallel using multiprocessing.

```python
def simulate_parallel(
    equipment_configs: List[Tuple],  # List of (class, id, config, params)
    num_processes: int = None        # Default: CPU count
) -> Tuple[List[Dict], List[Dict]]   # (telemetry, failures)
```

#### simulate_sequential()

Sequential fallback when parallel processing is not needed.

```python
def simulate_sequential(
    equipment_configs: List[Tuple]   # List of (class, id, config, params)
) -> Tuple[List[Dict], List[Dict]]   # (telemetry, failures)
```

## Record Types

### Telemetry Record

Normal operating telemetry generated during both operation and maintenance.

```python
{
    'type': 'telemetry',
    'equipment_id': 1,
    'sample_time': datetime(2025, 1, 15, 10, 30, 0),
    'operating_hours': 1520.5,
    'state': {
        'speed': 9500,
        'exhaust_gas_temp': 545.2,
        'health': {'hgp': 0.82, 'blade': 0.88, ...}
        # ... other sensor values
    },
    'in_maintenance': False  # True during maintenance period
}
```

### Failure Record

Generated when equipment fails (component health below threshold).

```python
{
    'type': 'failure',
    'equipment_id': 1,
    'equipment_type': 'turbine',
    'failure_time': datetime(2025, 3, 20, 14, 22, 0),
    'operating_hours_at_failure': 2845.3,
    'failure_mode_code': 'F_BEARING',
    'state': {
        'speed': 9200,
        'health': {'bearing': 0.34, ...}  # Below threshold
    }
}
```

### Maintenance Start Record

Generated immediately after failure when repair begins.

```python
{
    'type': 'maintenance_start',
    'equipment_id': 1,
    'equipment_type': 'turbine',
    'start_time': datetime(2025, 3, 20, 14, 22, 0),
    'failure_code': 'F_BEARING',
    'repaired_components': {'bearing': 0.89},  # New health values
    'downtime_hours': 24.0
}
```

### Maintenance Complete Record

Generated when maintenance period ends and equipment resumes operation.

```python
{
    'type': 'maintenance_complete',
    'equipment_id': 1,
    'equipment_type': 'turbine',
    'completion_time': datetime(2025, 3, 21, 14, 22, 0),
    'operating_hours': 2845.3
}
```

## Continuous Operation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Equipment Simulation Loop                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Normal Operation │
                    │   (yield telemetry)│
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Health < Threshold? │───No───┐
                    └──────────────────┘           │
                              │ Yes                │
                              ▼                    │
                    ┌──────────────────┐           │
                    │   yield 'failure'  │           │
                    └──────────────────┘           │
                              │                    │
                              ▼                    │
                    ┌──────────────────┐           │
                    │ _repair_equipment()│           │
                    │ (fix failed component)│        │
                    └──────────────────┘           │
                              │                    │
                              ▼                    │
                    ┌──────────────────┐           │
                    │yield 'maintenance_start'│     │
                    └──────────────────┘           │
                              │                    │
                              ▼                    │
                    ┌──────────────────┐           │
                    │  Maintenance Period │         │
                    │  (yield idle telemetry)│      │
                    └──────────────────┘           │
                              │                    │
                              ▼                    │
                    ┌──────────────────┐           │
                    │yield 'maintenance_complete'│  │
                    └──────────────────┘           │
                              │                    │
                              └────────────────────┘
                              │
                              ▼
                        (loop continues)
```

## Usage Examples

### Basic Single Equipment Simulation

```python
from src.ingestion.equipment_sim import simulate_equipment
from src.data_simulation.gas_turbine import GasTurbine

# Create equipment
turbine = GasTurbine(
    name='GT-001',
    initial_health={'hgp': 0.92, 'blade': 0.95, 'bearing': 0.90, 'fuel': 0.93}
)

# Simulate with maintenance cycles
telemetry = []
failures = []
maintenance_events = []

for record in simulate_equipment(
    equipment=turbine,
    equipment_id=1,
    equipment_type='turbine',
    duration_days=365,
    sample_interval_min=15,
    maintenance_downtime_hours=24.0
):
    if record['type'] == 'telemetry':
        telemetry.append(record)
    elif record['type'] == 'failure':
        failures.append(record)
        print(f"FAILURE: {record['failure_mode_code']} at {record['failure_time']}")
    elif record['type'] == 'maintenance_start':
        maintenance_events.append(record)
        print(f"REPAIR: {record['repaired_components']}")
    elif record['type'] == 'maintenance_complete':
        print(f"BACK ONLINE: {record['completion_time']}")

print(f"Total: {len(telemetry)} telemetry, {len(failures)} failures")
```

### Parallel Fleet Simulation

```python
from src.ingestion.equipment_sim import simulate_parallel
from src.data_simulation.gas_turbine import GasTurbine
from src.data_simulation.compressor import Compressor

# Configure equipment fleet
configs = []

# 10 Gas Turbines
for i in range(10):
    configs.append((
        GasTurbine,
        i + 1,  # equipment_id
        {
            'name': f'GT-{i+1:03d}',
            'initial_health': {'hgp': 0.92, 'blade': 0.95, 'bearing': 0.90, 'fuel': 0.93}
        },
        {
            'equipment_type': 'turbine',
            'duration_days': 180,
            'sample_interval_min': 10
        }
    ))

# 10 Compressors
for i in range(10):
    configs.append((
        Compressor,
        i + 11,  # equipment_id
        {
            'name': f'CC-{i+1:03d}',
            'initial_health': {'impeller': 0.88, 'bearing': 0.82}
        },
        {
            'equipment_type': 'compressor',
            'duration_days': 180,
            'sample_interval_min': 10
        }
    ))

# Run parallel simulation (uses all CPU cores)
all_telemetry, all_failures = simulate_parallel(configs)

print(f"Generated {len(all_telemetry)} telemetry records")
print(f"Captured {len(all_failures)} failures")
```

### Accelerated Degradation

```python
# Speed up degradation for faster failure generation (testing)
for record in simulate_equipment(
    equipment=turbine,
    equipment_id=1,
    equipment_type='turbine',
    duration_days=30,
    sample_interval_min=15,
    degradation_multiplier=5.0,  # 5x faster degradation
    maintenance_downtime_hours=8.0  # Shorter maintenance
):
    if record['type'] == 'failure':
        print(f"Failure: {record['failure_mode_code']}")
```

### Analyzing Maintenance History

```python
from collections import defaultdict

# Track repairs by component
repairs_by_component = defaultdict(list)

for record in simulate_equipment(turbine, 1, 'turbine', 365, 15):
    if record['type'] == 'maintenance_start':
        for component, new_health in record['repaired_components'].items():
            repairs_by_component[component].append({
                'time': record['start_time'],
                'new_health': new_health,
                'failure_code': record['failure_code']
            })

# Summary
for component, repairs in repairs_by_component.items():
    print(f"{component}: {len(repairs)} repairs")
    for repair in repairs:
        print(f"  - {repair['time']}: {repair['failure_code']} -> {repair['new_health']:.2%}")
```

## Configuration Parameters

### simulate_equipment() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| equipment | object | required | Equipment simulator instance (GasTurbine, Compressor, etc.) |
| equipment_id | int | required | Unique identifier for the equipment |
| equipment_type | str | required | 'turbine', 'compressor', or 'pump' |
| duration_days | int | required | Simulation duration in days |
| sample_interval_min | int | required | Time between telemetry samples (minutes) |
| start_time | datetime | None | Simulation start timestamp (default: now - duration_days) |
| degradation_multiplier | float | 1.0 | Speed up degradation (>1.0 = faster failures) |
| include_equipment_type | bool | False | Include equipment_type field in telemetry records |
| maintenance_downtime_hours | float | 24.0 | Hours of downtime after each failure |

### Repair Health Range

Repaired components are restored to health values in the range **85-92%** (randomly selected).

**Physical Basis**: Real maintenance doesn't restore components to perfect (100%) condition:
- Residual stress from operation
- Imperfect replacement parts
- Reassembly tolerances
- Material aging effects

## Design Decisions

### Why Targeted Repair?

Real-world maintenance typically addresses the specific failure mode:
- **Bearing replacement** restores bearing health, not impeller health
- **Seal replacement** restores seal health, not bearing health
- **Blade refurbishment** restores blade health, not fuel system health

This creates realistic multi-failure trajectories where different components fail at different times.

### Why Continue After Failure?

Industrial equipment is valuable (often $1M-$20M+) and is repaired rather than replaced:
- **Generates realistic datasets** with multiple failure-repair cycles
- **Models real maintenance operations** rather than idealized scenarios
- **Creates training data** for predictive maintenance systems that must operate through maintenance events

### Why Generator-Based Architecture?

Memory efficiency for large simulations:
- A 1-year simulation at 1-minute intervals = 525,600 records per equipment
- 100 equipment = 52+ million records
- Generator yields one record at a time, avoiding memory bloat

## Performance Considerations

### Computational Cost

| Mode | Cost per Equipment | Memory |
|------|-------------------|--------|
| Sequential | ~0.5ms per timestep | ~10 KB |
| Parallel (8 cores) | ~0.1ms per timestep (amortized) | ~80 KB per process |

### Recommended Settings

**Production Dataset Generation**:
```python
duration_days=180,          # 6 months
sample_interval_min=10,     # 10-minute intervals
degradation_multiplier=1.0, # Normal degradation
use_parallel=True           # Leverage all cores
```

**Quick Testing**:
```python
duration_days=30,           # 1 month
sample_interval_min=60,     # 1-hour intervals
degradation_multiplier=3.0, # Accelerated degradation
use_parallel=False          # Simpler debugging
```

## Integration with Data Pipeline

The `equipment_sim` module is designed to integrate with the data pipeline:

```python
from src.ingestion.data_pipeline import DataPipeline

# Pipeline automatically uses equipment_sim for simulation
pipeline = DataPipeline(db_url, duration_days=180, sample_interval_min=10)
pipeline.connect()
pipeline.seed_master_data(turbine_count=10, compressor_count=10, pump_count=30)
turbine_tel, compressor_tel, pump_tel, failures = pipeline.simulate_equipment(
    turbine_ids, compressor_ids, pump_ids, use_parallel=True
)
```

## See Also

- [data_pipeline.md](data_pipeline.md) - Pipeline orchestration
- [bulk_insert.md](bulk_insert.md) - Fast database insertion
- [gas_turbine.md](gas_turbine.md) - Gas turbine simulator
- [compressor.md](compressor.md) - Compressor simulator
- [maintenance_events.md](maintenance_events.md) - Maintenance scheduling module
