# Bulk Insert Module

## Overview

The `bulk_insert.py` module provides high-performance database insertion using PostgreSQL's native `COPY` command, achieving 100x faster insertion compared to row-by-row `INSERT` statements.

## Purpose

Predictive maintenance simulations generate millions of telemetry records. Traditional row-by-row insertion would take hours; this module enables:

- **High-Speed Insertion**: PostgreSQL `COPY` command for bulk loading
- **Equipment-Type Mapping**: Automatic column mapping for turbines, compressors, and pumps
- **Type Safety**: Handles NumPy types and missing data gracefully
- **Memory Efficiency**: Streaming CSV buffer for large datasets

## Key Features

- **100x Performance**: `COPY` command vs individual `INSERT` statements
- **Column Mapping**: Maps simulation state keys to database column names
- **Default Values**: Automatic handling of missing data
- **NumPy Conversion**: Converts NumPy scalar types to Python native types
- **Failure Records**: Separate insertion path for failure events

## Performance Comparison

| Method | Records/Second | 1M Records Time |
|--------|---------------|-----------------|
| Row-by-row INSERT | ~500 | ~33 minutes |
| Batch INSERT (1000) | ~5,000 | ~3.3 minutes |
| PostgreSQL COPY | ~50,000+ | ~20 seconds |

## Module Components

### Column Mappings

Each equipment type has a mapping from simulation state keys to database column names.

#### Turbine Columns

```python
TURBINE_COLUMNS = {
    'equipment_id': 'turbine_id',
    'sample_time': 'sample_time',
    'operating_hours': 'operating_hours',
    'speed': 'speed_rpm',
    'exhaust_gas_temp': 'egt_celsius',
    'oil_temp': 'oil_temp_celsius',
    'fuel_flow': 'fuel_flow_kg_s',
    'compressor_discharge_temp': 'compressor_discharge_temp_celsius',
    'compressor_discharge_pressure': 'compressor_discharge_pressure_kpa',
    'vibration_rms': 'vibration_rms_mm_s',
    'vibration_peak': 'vibration_peak_mm_s',
    'efficiency': 'efficiency_fraction',
    'health_hgp': 'health_hgp',
    'health_blade': 'health_blade',
    'health_bearing': 'health_bearing',
    'health_fuel': 'health_fuel',
}
```

#### Compressor Columns

```python
COMPRESSOR_COLUMNS = {
    'equipment_id': 'compressor_id',
    'sample_time': 'sample_time',
    'operating_hours': 'operating_hours',
    'speed': 'speed_rpm',
    'flow': 'flow_m3h',
    'head': 'head_kj_kg',
    'discharge_pressure': 'discharge_pressure_kpa',
    'discharge_temp': 'discharge_temp_celsius',
    'surge_margin': 'surge_margin_percent',
    'vibration_amplitude': 'vibration_amplitude_mm',
    'average_gap': 'average_gap_mm',
    'sync_amplitude': 'sync_amplitude_mm',
    'bearing_temp_de': 'bearing_temp_de_celsius',
    'bearing_temp_nde': 'bearing_temp_nde_celsius',
    'thrust_bearing_temp': 'thrust_bearing_temp_celsius',
    'seal_health_primary': 'seal_health_primary',
    'seal_health_secondary': 'seal_health_secondary',
    'seal_leakage': 'seal_leakage_rate',
    'health_impeller': 'health_impeller',
    'health_bearing': 'health_bearing',
}
```

#### Pump Columns

```python
PUMP_COLUMNS = {
    'equipment_id': 'pump_id',
    'sample_time': 'sample_time',
    'operating_hours': 'operating_hours',
    'speed': 'speed_rpm',
    'flow': 'flow_m3h',
    'head': 'head_m',
    'efficiency': 'efficiency_fraction',
    'power': 'power_kw',
    'suction_pressure': 'suction_pressure_kpa',
    'discharge_pressure': 'discharge_pressure_kpa',
    'fluid_temp': 'fluid_temp_celsius',
    'npsh_available': 'npsh_available_m',
    'npsh_required': 'npsh_required_m',
    'cavitation_margin': 'cavitation_margin_m',
    'cavitation_severity': 'cavitation_severity',
    'vibration': 'vibration_mm_s',
    'bearing_temp_de': 'bearing_temp_de_celsius',
    'bearing_temp_nde': 'bearing_temp_nde_celsius',
    'motor_current': 'motor_current_amps',
    'seal_health': 'seal_health',
    'seal_leakage': 'seal_leakage_rate',
    'health_impeller': 'health_impeller',
    'health_seal': 'health_seal',
    'health_bearing_de': 'health_bearing_de',
    'health_bearing_nde': 'health_bearing_nde',
}
```

### Default Values

Missing data is handled with sensible defaults:

```python
TURBINE_DEFAULTS = {
    'speed_rpm': 0, 'egt_celsius': 0, 'oil_temp_celsius': 0, 'fuel_flow_kg_s': 0,
    'compressor_discharge_temp_celsius': 0, 'compressor_discharge_pressure_kpa': 0,
    'vibration_rms_mm_s': 0, 'vibration_peak_mm_s': 0, 'efficiency_fraction': 1.0,
    'health_hgp': 0, 'health_blade': 0, 'health_bearing': 0, 'health_fuel': 0,
}

COMPRESSOR_DEFAULTS = {
    'speed_rpm': 0, 'flow_m3h': 0, 'head_kj_kg': 0, 'discharge_pressure_kpa': 0,
    'discharge_temp_celsius': 0, 'surge_margin_percent': 0, 'vibration_amplitude_mm': 0,
    'average_gap_mm': 0, 'sync_amplitude_mm': 0, 'bearing_temp_de_celsius': 45,
    'bearing_temp_nde_celsius': 45, 'thrust_bearing_temp_celsius': 50,
    'seal_health_primary': 0, 'seal_health_secondary': 0, 'seal_leakage_rate': 0,
    'health_impeller': 0, 'health_bearing': 0,
}

PUMP_DEFAULTS = {
    'speed_rpm': 0, 'flow_m3h': 0, 'head_m': 0, 'efficiency_fraction': 0, 'power_kw': 0,
    'suction_pressure_kpa': 200, 'discharge_pressure_kpa': 200, 'fluid_temp_celsius': 40,
    'npsh_available_m': 8, 'npsh_required_m': 3, 'cavitation_margin_m': 5,
    'cavitation_severity': 0, 'vibration_mm_s': 0, 'bearing_temp_de_celsius': 50,
    'bearing_temp_nde_celsius': 50, 'motor_current_amps': 0, 'seal_health': 0,
    'seal_leakage_rate': 0, 'health_impeller': 0, 'health_seal': 0,
    'health_bearing_de': 0, 'health_bearing_nde': 0,
}
```

### Core Functions

#### bulk_insert_telemetry()

Fast bulk insert using PostgreSQL `COPY` command.

```python
def bulk_insert_telemetry(
    db,                          # Database connection object
    records: List[Dict],         # List of telemetry records
    equipment_type: str          # 'turbine', 'compressor', or 'pump'
) -> int                         # Number of records inserted
```

**Implementation**:
1. Builds CSV buffer in memory with tab-delimited values
2. Uses `copy_expert()` to execute PostgreSQL `COPY FROM STDIN`
3. Handles NULL values with `\N` marker

```python
# Internal flow
buffer = StringIO()
writer = csv.writer(buffer, delimiter='\t')

for record in records:
    row = [_get_value(record, key, defaults) for key in columns_map.keys()]
    writer.writerow(row)

buffer.seek(0)

# Execute COPY
cursor.copy_expert(
    f"COPY {table} ({columns}) FROM STDIN WITH CSV DELIMITER '\t' NULL '\\N'",
    buffer
)
```

#### insert_failures()

Inserts failure event records using parameterized SQL.

```python
def insert_failures(
    db,                          # Database connection object
    failures: List[Dict]         # List of failure records
) -> int                         # Number of records inserted
```

**Failure Tables**:
- `failure_events.gas_turbine_failures`
- `failure_events.centrifugal_compressor_failures`
- `failure_events.centrifugal_pump_failures`

**Failure Record Fields**:

| Field | Description |
|-------|-------------|
| failure_time | Timestamp of failure |
| operating_hours_at_failure | Cumulative operating hours |
| failure_mode_code | Failure code (e.g., F_BEARING, F_SURGE) |
| speed_rpm_at_failure | Speed at time of failure |
| Equipment-specific values | EGT, surge margin, vibration, etc. |

#### _get_value()

Internal helper to extract values from nested record structure.

```python
def _get_value(record: Dict, key: str, defaults: Dict) -> str
```

**Handles**:
- Top-level keys: `equipment_id`, `sample_time`, `operating_hours`
- Nested state keys: `state['speed']`, `state['exhaust_gas_temp']`
- Health keys: `state['health']['hgp']` → `health_hgp`
- NumPy scalar conversion
- Float rounding (2 decimal places)
- NULL marker for missing values

## Usage Examples

### Basic Telemetry Insertion

```python
from src.ingestion.bulk_insert import bulk_insert_telemetry
from src.ingestion.db_setup import Database

# Connect to database
db = Database('postgresql://user:pass@localhost:5432/pdm_db')
db.connect()

# Telemetry records from simulation
turbine_records = [
    {
        'equipment_id': 1,
        'sample_time': datetime(2025, 1, 15, 10, 0, 0),
        'operating_hours': 1500.5,
        'state': {
            'speed': 9500,
            'exhaust_gas_temp': 545.2,
            'oil_temp': 95.3,
            'fuel_flow': 2.1,
            'vibration_rms': 1.2,
            'vibration_peak': 3.5,
            'efficiency': 0.82,
            'health': {
                'hgp': 0.85,
                'blade': 0.90,
                'bearing': 0.78,
                'fuel': 0.88
            }
        }
    },
    # ... more records
]

# Bulk insert
count = bulk_insert_telemetry(db, turbine_records, 'turbine')
print(f"Inserted {count} records")
```

### Failure Record Insertion

```python
from src.ingestion.bulk_insert import insert_failures

failures = [
    {
        'equipment_id': 1,
        'equipment_type': 'turbine',
        'failure_time': datetime(2025, 3, 20, 14, 22, 0),
        'operating_hours_at_failure': 2845.3,
        'failure_mode_code': 'F_BEARING',
        'state': {
            'speed': 9200,
            'exhaust_gas_temp': 548.5,
            'vibration_rms': 2.8
        }
    },
    {
        'equipment_id': 5,
        'equipment_type': 'compressor',
        'failure_time': datetime(2025, 4, 10, 8, 15, 0),
        'operating_hours_at_failure': 3120.7,
        'failure_mode_code': 'F_SURGE',
        'state': {
            'speed': 11500,
            'surge_margin': 4.2,
            'vibration_amplitude': 0.08
        }
    }
]

count = insert_failures(db, failures)
print(f"Inserted {count} failure records")
```

### Integration with Pipeline

```python
from src.ingestion.data_pipeline import DataPipeline
from src.ingestion.bulk_insert import bulk_insert_telemetry, insert_failures

pipeline = DataPipeline(db_url, duration_days=180, sample_interval_min=10)
pipeline.connect()

# Generate data
turbine_tel, compressor_tel, pump_tel, failures = pipeline.simulate_equipment(
    turbine_ids, compressor_ids, pump_ids
)

# Bulk insert telemetry
bulk_insert_telemetry(pipeline.db, turbine_tel, 'turbine')
bulk_insert_telemetry(pipeline.db, compressor_tel, 'compressor')
bulk_insert_telemetry(pipeline.db, pump_tel, 'pump')

# Insert failures
insert_failures(pipeline.db, failures)
```

### Handling Large Datasets

For very large datasets, process in chunks to manage memory:

```python
def insert_in_chunks(db, records, equipment_type, chunk_size=100000):
    """Insert records in chunks to manage memory."""
    total_inserted = 0

    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        count = bulk_insert_telemetry(db, chunk, equipment_type)
        total_inserted += count
        print(f"Inserted chunk {i//chunk_size + 1}: {count} records")

    return total_inserted

# Process 10 million records in 100K chunks
total = insert_in_chunks(db, massive_dataset, 'turbine', chunk_size=100000)
```

## Technical Details

### PostgreSQL COPY Command

The `COPY` command bypasses SQL parsing and constraint checking overhead:

```sql
COPY gas_turbine_telemetry (turbine_id, sample_time, operating_hours, ...)
FROM STDIN WITH CSV DELIMITER '\t' NULL '\N'
```

**Advantages**:
- Direct data loading into table pages
- Minimal transaction overhead
- No SQL statement parsing per row
- Bulk WAL logging

### NumPy Type Handling

NumPy scalars (e.g., `np.float64`, `np.int32`) must be converted to Python native types for database insertion:

```python
if HAS_NUMPY and isinstance(value, (np.integer, np.floating)):
    value = float(value)
```

### NULL Handling

Missing values are represented as `\N` in the CSV stream, which PostgreSQL interprets as NULL:

```python
return str(value) if value is not None else '\\N'
```

### Schema Path

The module sets the search path before insertion to ensure table visibility:

```python
cursor.execute("SET search_path TO telemetry, master_data, failure_events, public")
```

## Error Handling

```python
try:
    cursor.copy_expert(sql, buffer)
    raw_conn.commit()
except Exception as e:
    logger.error(f"Bulk insert failed: {e}")
    raise
finally:
    session.close()
```

**Common Errors**:

| Error | Cause | Solution |
|-------|-------|----------|
| `column does not exist` | Schema mismatch | Check column mappings match table |
| `invalid input syntax` | Type conversion error | Check data types and NULL handling |
| `permission denied` | Insufficient privileges | Grant INSERT permission |
| `disk full` | Storage exhausted | Free space or use smaller chunks |

## Performance Optimization

### Memory Usage

The CSV buffer grows with record count. For very large inserts:

```python
# Process in chunks
for chunk in chunked(records, 100000):
    bulk_insert_telemetry(db, chunk, equipment_type)
    gc.collect()  # Force garbage collection
```

### Database Tuning

For maximum `COPY` performance:

```sql
-- Temporarily disable indexes (rebuild after)
ALTER TABLE telemetry.gas_turbine_telemetry SET UNLOGGED;

-- Increase checkpoint interval
SET checkpoint_timeout = '30min';

-- After insert, analyze for query optimization
ANALYZE telemetry.gas_turbine_telemetry;
```

### SSD vs HDD

| Storage | COPY Speed | 1M Records |
|---------|-----------|------------|
| SSD (NVMe) | 80-100K/sec | 10-12 seconds |
| SSD (SATA) | 50-70K/sec | 15-20 seconds |
| HDD | 20-30K/sec | 35-50 seconds |

## Database Tables

### Telemetry Tables

```sql
-- Gas Turbine Telemetry
CREATE TABLE telemetry.gas_turbine_telemetry (
    telemetry_id SERIAL PRIMARY KEY,
    turbine_id INTEGER REFERENCES master_data.gas_turbines(turbine_id),
    sample_time TIMESTAMP NOT NULL,
    operating_hours NUMERIC(10,2),
    speed_rpm NUMERIC(10,2),
    egt_celsius NUMERIC(8,2),
    oil_temp_celsius NUMERIC(6,2),
    fuel_flow_kg_s NUMERIC(8,4),
    vibration_rms_mm_s NUMERIC(8,4),
    vibration_peak_mm_s NUMERIC(8,4),
    efficiency_fraction NUMERIC(5,4),
    health_hgp NUMERIC(5,4),
    health_blade NUMERIC(5,4),
    health_bearing NUMERIC(5,4),
    health_fuel NUMERIC(5,4),
    -- ... additional columns
);
```

### Failure Tables

```sql
-- Gas Turbine Failures
CREATE TABLE failure_events.gas_turbine_failures (
    failure_id SERIAL PRIMARY KEY,
    turbine_id INTEGER REFERENCES master_data.gas_turbines(turbine_id),
    failure_time TIMESTAMP NOT NULL,
    operating_hours_at_failure NUMERIC(10,2),
    failure_mode_code VARCHAR(50),
    speed_rpm_at_failure NUMERIC(10,2),
    egt_celsius_at_failure NUMERIC(8,2),
    vibration_mm_s_at_failure NUMERIC(8,4)
);
```

## See Also

- [data_pipeline.md](data_pipeline.md) - Pipeline orchestration
- [equipment_sim.md](equipment_sim.md) - Equipment simulation
- [db_setup.md](db_setup.md) - Database setup and schema creation
