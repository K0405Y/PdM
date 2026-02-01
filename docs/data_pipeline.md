# Data Pipeline Orchestrator

## Overview

The `data_pipeline.py` module orchestrates the complete PdM data generation workflow, coordinating database operations, equipment simulation, and data ingestion into a single automated pipeline.

## Purpose

Generating a comprehensive predictive maintenance dataset requires multiple coordinated steps. This module provides:

- **End-to-End Orchestration**: Single entry point for complete data generation
- **Modular Architecture**: Clean separation of database, simulation, and insertion concerns
- **Production-Ready Logging**: Comprehensive progress tracking and error handling
- **Configurable Parameters**: Flexible simulation duration, sampling rates, and equipment counts
- **Data Verification**: Built-in integrity checks after data ingestion

## Pipeline Steps

The data pipeline executes six sequential steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PdM Data Pipeline                            │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ Step 1  │         │ Step 2  │         │ Step 3  │
    │ Connect │────────▶│ Schemas │────────▶│  Seed   │
    │   DB    │         │ Create  │         │ Master  │
    └─────────┘         └─────────┘         └─────────┘
                                                  │
         ┌────────────────────┬───────────────────┘
         │                    │
         ▼                    ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │ Step 6  │         │ Step 5  │         │ Step 4  │
    │ Verify  │◀────────│ Ingest  │◀────────│Simulate │
    │  Data   │         │  Data   │         │Equipment│
    └─────────┘         └─────────┘         └─────────┘
```

| Step | Description | Module Used |
|------|-------------|-------------|
| 1 | Connect to PostgreSQL database | `db_setup.Database` |
| 2 | Create database schemas | SQL scripts in `db schemas/` |
| 3 | Seed master data (equipment definitions) | `db_setup.MasterData` |
| 4 | Run equipment simulations | `equipment_sim` |
| 5 | Bulk insert telemetry and failures | `bulk_insert` |
| 6 | Verify data integrity | SQL queries |

## Module Components

### DataPipeline Class

Main orchestrator class coordinating all pipeline steps.

```python
class DataPipeline:
    def __init__(
        self,
        db_url: str,                    # PostgreSQL connection URL
        duration_days: int = 180,       # Simulation duration (6 months default)
        sample_interval_min: int = 10   # Sampling interval (10 minutes default)
    )
```

### Methods

#### connect()

Establishes database connection and initializes master data handler.

```python
pipeline.connect()
# Logs: "Connected to database"
```

#### create_schemas()

Executes SQL schema creation scripts in order.

```python
pipeline.create_schemas(schemas_dir=None)  # Uses default db schemas/ directory
```

**Behavior**:
- Executes `.sql` files in alphabetical order
- Skips files containing 'verification' in filename
- Creates schemas: `master_data`, `telemetry`, `failure_events`

#### seed_master_data()

Creates equipment master records with randomized initial health values.

```python
turbine_ids, compressor_ids, pump_ids = pipeline.seed_master_data(
    turbine_count=10,
    compressor_count=10,
    pump_count=80
)
```

**Returns**: Tuple of (turbine_ids, compressor_ids, pump_ids) lists.

#### simulate_equipment()

Runs physics-based simulation for all equipment.

```python
turbine_tel, compressor_tel, pump_tel, failures = pipeline.simulate_equipment(
    turbine_ids,
    compressor_ids,
    pump_ids,
    use_parallel=True  # Use multiprocessing
)
```

**Returns**: Tuple of (turbine_telemetry, compressor_telemetry, pump_telemetry, all_failures).

#### ingest_data()

Bulk inserts generated data into PostgreSQL.

```python
pipeline.ingest_data(
    turbine_telemetry,
    compressor_telemetry,
    pump_telemetry,
    failures
)
```

Uses PostgreSQL `COPY` command for 100x faster insertion than row-by-row `INSERT`.

#### verify_data()

Runs integrity verification queries and prints summary.

```python
pipeline.verify_data()
```

**Output Example**:
```
============================================================
DATA INTEGRITY SUMMARY
============================================================
Equipment:
  - Turbines: 10
  - Compressors: 10
  - Pumps: 80
  - Total: 100

Telemetry Records:
  - Turbine: 259,200
  - Compressor: 259,200
  - Pump: 2,073,600
  - Total: 2,592,000

Failure Events:
  - Turbine: 15
  - Compressor: 12
  - Pump: 89
  - Total: 116
============================================================
```

#### close()

Closes database connection and cleans up resources.

```python
pipeline.close()
```

## Usage Examples

### Basic Pipeline Execution

```python
from src.ingestion.data_pipeline import DataPipeline
import os

# Get database URL from environment
db_url = os.getenv('POSTGRES_URL')

# Initialize pipeline
pipeline = DataPipeline(
    db_url=db_url,
    duration_days=180,      # 6 months
    sample_interval_min=10  # 10-minute samples
)

try:
    # Execute pipeline steps
    pipeline.connect()
    pipeline.create_schemas()

    turbine_ids, compressor_ids, pump_ids = pipeline.seed_master_data(
        turbine_count=10,
        compressor_count=10,
        pump_count=30
    )

    turbine_tel, compressor_tel, pump_tel, failures = pipeline.simulate_equipment(
        turbine_ids, compressor_ids, pump_ids,
        use_parallel=True
    )

    pipeline.ingest_data(turbine_tel, compressor_tel, pump_tel, failures)
    pipeline.verify_data()

finally:
    pipeline.close()
```

### Command-Line Execution

```bash
# Set database URL
export POSTGRES_URL="postgresql://user:password@localhost:5432/pdm_db"

# Run pipeline
python -m src.ingestion.data_pipeline
```

### Custom Configuration

```python
# High-frequency sampling for detailed analysis
pipeline_detailed = DataPipeline(
    db_url=db_url,
    duration_days=30,       # 1 month
    sample_interval_min=1   # 1-minute samples
)

# Long-term study with standard sampling
pipeline_longterm = DataPipeline(
    db_url=db_url,
    duration_days=365,      # 1 year
    sample_interval_min=15  # 15-minute samples
)

# Quick test dataset
pipeline_test = DataPipeline(
    db_url=db_url,
    duration_days=7,        # 1 week
    sample_interval_min=60  # 1-hour samples
)
```

### Sequential vs Parallel Simulation

```python
# Parallel (default) - uses all CPU cores
turbine_tel, compressor_tel, pump_tel, failures = pipeline.simulate_equipment(
    turbine_ids, compressor_ids, pump_ids,
    use_parallel=True
)

# Sequential - for debugging or single-core environments
turbine_tel, compressor_tel, pump_tel, failures = pipeline.simulate_equipment(
    turbine_ids, compressor_ids, pump_ids,
    use_parallel=False
)
```

## Configuration Parameters

### DataPipeline Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| db_url | str | required | PostgreSQL connection URL |
| duration_days | int | 180 | Simulation duration (days) |
| sample_interval_min | int | 10 | Sampling interval (minutes) |

### seed_master_data() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| turbine_count | int | 10 | Number of gas turbines |
| compressor_count | int | 10 | Number of centrifugal compressors |
| pump_count | int | 80 | Number of centrifugal pumps |

### simulate_equipment() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| turbine_ids | List[int] | required | Turbine IDs to simulate |
| compressor_ids | List[int] | required | Compressor IDs to simulate |
| pump_ids | List[int] | required | Pump IDs to simulate |
| use_parallel | bool | True | Use multiprocessing |

## Data Generation Estimates

### Record Counts

Formula: `records = equipment_count * duration_days * 24 * 60 / sample_interval_min`

| Duration | Interval | Records/Equipment | 100 Equipment |
|----------|----------|-------------------|---------------|
| 30 days | 10 min | 4,320 | 432,000 |
| 180 days | 10 min | 25,920 | 2,592,000 |
| 365 days | 10 min | 52,560 | 5,256,000 |
| 180 days | 1 min | 259,200 | 25,920,000 |

### Processing Time (Approximate)

| Equipment | Duration | Parallel (8 cores) | Sequential |
|-----------|----------|-------------------|------------|
| 10 | 180 days | 2-3 minutes | 15-20 minutes |
| 100 | 180 days | 15-20 minutes | 2-3 hours |
| 100 | 365 days | 30-45 minutes | 5-6 hours |

### Storage Requirements

| Records | Compressed Size | Uncompressed Size |
|---------|-----------------|-------------------|
| 1 million | ~200 MB | ~800 MB |
| 10 million | ~2 GB | ~8 GB |
| 50 million | ~10 GB | ~40 GB |

## Database Schema

The pipeline creates three schemas:

### master_data Schema

- `gas_turbines`: Gas turbine equipment definitions
- `compressors`: Compressor equipment definitions
- `pumps`: Pump equipment definitions

### telemetry Schema

- `gas_turbine_telemetry`: Turbine sensor readings
- `compressor_telemetry`: Compressor sensor readings
- `pump_telemetry`: Pump sensor readings

### failure_events Schema

- `gas_turbine_failures`: Turbine failure records
- `compressor_failures`: Compressor failure records
- `pump_failures`: Pump failure records

## Error Handling

The pipeline includes comprehensive error handling:

```python
try:
    # Pipeline steps...
except Exception as e:
    logger.error(f"Pipeline failed: {e}", exc_info=True)
    sys.exit(1)
finally:
    pipeline.close()
```

**Common Errors**:

| Error | Cause | Solution |
|-------|-------|----------|
| `POSTGRES_URL not set` | Environment variable missing | Set `POSTGRES_URL` |
| `Connection refused` | Database not running | Start PostgreSQL server |
| `Permission denied` | Insufficient privileges | Check database user permissions |
| `Disk space` | Insufficient storage | Free disk space or reduce dataset |

## Logging

The pipeline uses Python's logging module with INFO level:

```
2025-01-15 10:30:00 - data_pipeline - INFO - Pipeline initialized: 180 days, 10 min intervals
2025-01-15 10:30:01 - data_pipeline - INFO - Connected to database
2025-01-15 10:30:02 - data_pipeline - INFO - Creating database schemas from: .../db schemas
2025-01-15 10:30:05 - data_pipeline - INFO - Database schemas created successfully
2025-01-15 10:30:05 - data_pipeline - INFO - Seeding master data: 10 turbines, 10 compressors, 30 pumps
...
```

## Integration Points

### Environment Variables

```bash
POSTGRES_URL=postgresql://user:password@host:port/database
```

### Module Dependencies

```python
# Database operations
from src.ingestion.db_setup import Database, MasterData

# Equipment simulation
from src.ingestion.equipment_sim import simulate_parallel, simulate_sequential

# Bulk data insertion
from src.ingestion.bulk_insert import bulk_insert_telemetry, insert_failures

# Equipment simulators
from src.data_simulation.gas_turbine import GasTurbine
from src.data_simulation.compressor import Compressor
from src.data_simulation.pump import Pump
```

## Best Practices

### Production Runs

1. **Test with small dataset first**: Use `duration_days=7` and `pump_count=10`
2. **Monitor disk space**: Large datasets can consume 10+ GB
3. **Use parallel processing**: Reduces generation time by 5-8x
4. **Verify data after generation**: Always run `verify_data()`

### Development/Testing

1. **Use sequential mode**: Easier to debug with `use_parallel=False`
2. **Short durations**: `duration_days=7-30` for quick iterations
3. **Fewer equipment**: Start with 5-10 of each type

### Performance Optimization

1. **Increase sample interval**: 15-minute samples reduce data 50% vs 10-minute
2. **Use SSD storage**: 3-5x faster bulk inserts than HDD
3. **Adequate RAM**: Allow 1 GB per parallel simulation process

## See Also

- [equipment_sim.md](equipment_sim.md) - Equipment simulation with maintenance cycles
- [bulk_insert.md](bulk_insert.md) - Fast PostgreSQL bulk insertion
- [gas_turbine.md](gas_turbine.md) - Gas turbine simulator documentation
- [compressor.md](compressor.md) - Compressor simulator documentation
- [db_setup.md](db_setup.md) - Database setup and master data management