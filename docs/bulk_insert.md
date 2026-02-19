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
- **Derived Feature Extraction**: Parses JSON `features` field for ML-ready columns
- **NumPy Conversion**: Converts NumPy scalar types to Python native types
- **Failure Records**: Separate insertion path for failure events
- **Maintenance Records**: Insertion path for maintenance events with computed end times

## Performance Comparison

| Method | Records/Second | 1M Records Time |
|--------|---------------|-----------------|
| Row-by-row INSERT | ~500 | ~33 minutes |
| Batch INSERT (1000) | ~5,000 | ~3.3 minutes |
| PostgreSQL COPY | ~50,000+ | ~20 seconds |

## Module Components

### Column Mappings

Each equipment type has a mapping from simulation state keys to database column names.

#### Turbine Columns (31 fields)

```python
TURBINE_COLUMNS = {
    'equipment_id': 'turbine_id',
    'sample_time': 'sample_time',
    'operating_hours': 'operating_hours',
    'speed': 'speed_rpm',
    'speed_target': 'speed_target_rpm',
    'exhaust_gas_temp': 'egt_celsius',
    'oil_temp': 'oil_temp_celsius',
    'fuel_flow': 'fuel_flow_kg_s',
    'compressor_discharge_temp': 'compressor_discharge_temp_celsius',
    'compressor_discharge_pressure': 'compressor_discharge_pressure_kpa',
    'efficiency': 'efficiency_fraction',
    'ambient_temp': 'ambient_temp_celsius',
    'ambient_pressure': 'ambient_pressure_kpa',
    'vibration_rms': 'vibration_rms_mm_s',
    'vibration_peak': 'vibration_peak_mm_s',
    'vibration_crest_factor': 'vibration_crest_factor',
    'vibration_kurtosis': 'vibration_kurtosis',
    'health_hgp': 'health_hgp',
    'health_blade': 'health_blade',
    'health_bearing': 'health_bearing',
    'health_fuel': 'health_fuel',
    'num_active_faults': 'num_active_faults',
    'total_faults_initiated': 'total_faults_initiated',
    'upset_active': 'upset_active',
    'upset_type': 'upset_type',
    'upset_severity': 'upset_severity',
    # Derived features (extracted from JSON 'features' field)
    'vibration_trend_7d': 'vibration_trend_7d',
    'temp_variation_24h': 'temp_variation_24h',
    'speed_stability': 'speed_stability',
    'efficiency_degradation_rate': 'efficiency_degradation_rate',
    'load_factor': 'load_factor',
}
```

#### Compressor Columns (34 fields)

```python
COMPRESSOR_COLUMNS = {
    'equipment_id': 'compressor_id',
    'sample_time': 'sample_time',
    'operating_hours': 'operating_hours',
    'speed': 'speed_rpm',
    'speed_target': 'speed_target_rpm',
    'flow': 'flow_m3h',
    'head': 'head_kj_kg',
    'efficiency': 'efficiency_fraction',
    'power': 'power_kw',
    'suction_pressure': 'suction_pressure_kpa',
    'suction_temp': 'suction_temp_celsius',
    'discharge_pressure': 'discharge_pressure_kpa',
    'discharge_temp': 'discharge_temp_celsius',
    'surge_margin': 'surge_margin_percent',
    'surge_alarm': 'surge_alarm',
    'orbit_amplitude': 'vibration_amplitude_mm',
    'sync_amplitude': 'sync_amplitude_mm',
    'shaft_x_displacement': 'shaft_x_displacement_mm',
    'shaft_y_displacement': 'shaft_y_displacement_mm',
    'bearing_temp_de': 'bearing_temp_de_celsius',
    'bearing_temp_nde': 'bearing_temp_nde_celsius',
    'thrust_bearing_temp': 'thrust_bearing_temp_celsius',
    'health_seal_primary': 'seal_health_primary',
    'health_seal_secondary': 'seal_health_secondary',
    'primary_seal_leakage': 'primary_seal_leakage_kg_s',
    'secondary_seal_leakage': 'secondary_seal_leakage_kg_s',
    'health_impeller': 'health_impeller',
    'health_bearing': 'health_bearing',
    'num_active_faults': 'num_active_faults',
    'total_faults_initiated': 'total_faults_initiated',
    'upset_active': 'upset_active',
    'upset_type': 'upset_type',
    'upset_severity': 'upset_severity',
    # Derived features (extracted from JSON 'features' field)
    'vibration_trend_7d': 'vibration_trend_7d',
    'temp_variation_24h': 'temp_variation_24h',
    'speed_stability': 'speed_stability',
    'efficiency_degradation_rate': 'efficiency_degradation_rate',
    'pressure_ratio': 'pressure_ratio',
    'load_factor': 'load_factor',
}
```

#### Pump Columns (32 fields)

```python
PUMP_COLUMNS = {
    'equipment_id': 'pump_id',
    'sample_time': 'sample_time',
    'operating_hours': 'operating_hours',
    'speed': 'speed_rpm',
    'speed_target': 'speed_target_rpm',
    'flow': 'flow_m3h',
    'head': 'head_m',
    'efficiency': 'efficiency_fraction',
    'power': 'power_kw',
    'suction_pressure': 'suction_pressure_kpa',
    'discharge_pressure': 'discharge_pressure_kpa',
    'fluid_temp': 'fluid_temp_celsius',
    'npsh_available': 'npsh_available_m',
    'npsh_required': 'npsh_required_m',
    'npsh_margin': 'cavitation_margin_m',
    'cavitation_severity': 'cavitation_severity',
    'vibration_rms': 'vibration_rms_mm_s',
    'vibration_peak': 'vibration_peak_mm_s',
    'bearing_temp_de': 'bearing_temp_de_celsius',
    'bearing_temp_nde': 'bearing_temp_nde_celsius',
    'motor_current': 'motor_current_amps',
    'motor_current_ratio': 'motor_current_ratio',
    'seal_leakage': 'seal_leakage_rate',
    'bep_deviation': 'bep_deviation_percent',
    'health_impeller': 'health_impeller',
    'health_seal': 'health_seal',
    'health_bearing_de': 'health_bearing_de',
    'health_bearing_nde': 'health_bearing_nde',
    # Derived features (extracted from JSON 'features' field)
    'temp_variation_24h': 'temp_variation_24h',
    'vibration_trend_7d': 'vibration_trend_7d',
    'speed_stability': 'speed_stability',
    'efficiency_degradation_rate': 'efficiency_degradation_rate',
    'pressure_ratio': 'pressure_ratio',
    'load_factor': 'load_factor',
}
```

### Derived Feature Keys

Six derived features are extracted from the JSON `features` field nested inside each record's `state` dict, rather than from top-level state keys:

```python
DERIVED_FEATURE_KEYS = {
    'vibration_trend_7d',
    'temp_variation_24h',
    'speed_stability',
    'efficiency_degradation_rate',
    'pressure_ratio',
    'load_factor',
}
```

Not all equipment types produce all derived features. For example, turbines do not produce `pressure_ratio`. Missing derived features are inserted as NULL.

### NULL Handling

Missing values are inserted as PostgreSQL NULL. The module does not use default value substitution — any key absent from the record's state or features dict results in `\N` in the CSV stream, which PostgreSQL interprets as NULL.

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
- `failure_events.compressor_failures`
- `failure_events.pump_failures`

**Failure Record Fields**:

| Field | Description |
|-------|-------------|
| failure_time | Timestamp of failure |
| operating_hours_at_failure | Cumulative operating hours |
| failure_mode_code | Failure code (e.g., F_BEARING, F_IMPELLER) |
| speed_rpm_at_failure | Speed at time of failure |
| Equipment-specific values | EGT, surge margin, vibration, etc. |

#### _get_value()

Internal helper to extract values from record structure.

```python
def _get_value(record: Dict, key: str, defaults: Dict) -> str
```

**Value resolution order**:
1. **Record-level keys** (`_RECORD_KEYS`): `equipment_id`, `sample_time`, `operating_hours` — read directly from the record dict
2. **Derived feature keys** (`DERIVED_FEATURE_KEYS`): Extracted from `state['features']` JSON field, parsed if string
3. **All other keys**: Read flat from `state` dict (e.g., `state['speed']`, `state['health_hgp']`)

**Post-processing**:
- NumPy scalar conversion (`np.integer`, `np.floating` → `float`)
- Float rounding to 4 decimal places
- Returns `\N` (PostgreSQL NULL marker) for missing values

#### insert_maintenance()

Inserts maintenance event records using parameterized SQL.

```python
def insert_maintenance(
    db,                              # Database connection object
    maintenance_records: List[Dict]  # List of maintenance records
) -> int                             # Number of records inserted
```

**Maintenance Tables**:
- `maintenance_events.gas_turbine_maintenance`
- `maintenance_events.compressor_maintenance`
- `maintenance_events.pump_maintenance`

**Columns Inserted**:

| Column | Source | Description |
|--------|--------|-------------|
| `{equipment}_id` | `record['equipment_id']` | Equipment foreign key |
| `start_time` | `record['start_time']` | Maintenance start timestamp |
| `end_time` | Computed: `start_time + timedelta(hours=downtime_hours)` | Maintenance end timestamp |
| `failure_code` | `record['failure_code']` | Triggering failure mode code |
| `downtime_hours` | `record['downtime_hours']` | Maintenance duration in hours |
| `repaired_components` | `record['repaired_components']` | JSONB dict of component → new health value |

**Input Record Format** (from `equipment_sim.simulate_equipment()` `maintenance_start` records):

```python
{
    'equipment_type': 'turbine',
    'equipment_id': 1,
    'start_time': datetime(2025, 3, 20, 14, 22, 0),
    'failure_code': 'F_BEARING',
    'repaired_components': {'bearing': 0.89},
    'downtime_hours': 24.0
}
```

## Usage Examples

### Basic Telemetry Insertion

```python
from src.ingestion.bulk_insert import bulk_insert_telemetry
from src.ingestion.db_setup import Database

# Connect to database
db = Database('postgresql://user:pass@localhost:5432/pdm_db')
db.connect()

# Telemetry records from simulation (flat state dict)
turbine_records = [
    {
        'equipment_id': 1,
        'sample_time': datetime(2025, 1, 15, 10, 0, 0),
        'operating_hours': 1500.5,
        'state': {
            'speed': 9500,
            'speed_target': 9500,
            'exhaust_gas_temp': 545.2,
            'oil_temp': 95.3,
            'fuel_flow': 2.1,
            'efficiency': 0.82,
            'ambient_temp': 25.0,
            'ambient_pressure': 101.3,
            'vibration_rms': 1.2,
            'vibration_peak': 3.5,
            'vibration_crest_factor': 2.9,
            'vibration_kurtosis': 3.1,
            'health_hgp': 0.85,
            'health_blade': 0.90,
            'health_bearing': 0.78,
            'health_fuel': 0.88,
            'num_active_faults': 0,
            'total_faults_initiated': 0,
            'upset_active': False,
            'features': '{"vibration_trend_7d": 0.02, "temp_variation_24h": 1.5}'
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
        'failure_mode_code': 'F_IMPELLER',
        'state': {
            'speed': 11500,
            'surge_margin': 12.5,
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
from src.ingestion.bulk_insert import bulk_insert_telemetry, insert_failures, insert_maintenance

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

# Insert maintenance events
# Note: maintenance_start records are mixed into telemetry lists by _worker_simulate()
maintenance_records = [r for r in turbine_tel + compressor_tel + pump_tel
                       if r.get('type') == 'maintenance_start']
insert_maintenance(pipeline.db, maintenance_records)
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

The module sets the search path before insertion to ensure table visibility across all schemas:

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
-- Gas Turbine Telemetry (representative — see 03_create_telemetry_tables.sql for full DDL)
CREATE TABLE telemetry.gas_turbine_telemetry (
    telemetry_id BIGSERIAL PRIMARY KEY,
    turbine_id INT NOT NULL REFERENCES master_data.gas_turbines(turbine_id),
    sample_time TIMESTAMP NOT NULL,
    operating_hours FLOAT,
    -- Core measurements
    speed_rpm FLOAT,
    speed_target_rpm FLOAT,
    egt_celsius FLOAT,
    oil_temp_celsius FLOAT,
    fuel_flow_kg_s FLOAT,
    compressor_discharge_temp_celsius FLOAT,
    compressor_discharge_pressure_kpa FLOAT,
    efficiency_fraction FLOAT,
    -- Environmental conditions
    ambient_temp_celsius FLOAT,
    ambient_pressure_kpa FLOAT,
    -- Vibration metrics
    vibration_rms_mm_s FLOAT,
    vibration_peak_mm_s FLOAT,
    vibration_crest_factor FLOAT,
    vibration_kurtosis FLOAT,
    -- Health indicators
    health_hgp FLOAT,
    health_blade FLOAT,
    health_bearing FLOAT,
    health_fuel FLOAT,
    -- Fault tracking
    num_active_faults INT,
    total_faults_initiated INT,
    -- Process upset tracking
    upset_active BOOLEAN,
    upset_type VARCHAR(50),
    upset_severity FLOAT,
    -- Derived features
    vibration_trend_7d FLOAT,
    temp_variation_24h FLOAT,
    speed_stability FLOAT,
    efficiency_degradation_rate FLOAT,
    load_factor FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

### Maintenance Tables

```sql
-- Gas Turbine Maintenance (representative — compressor/pump tables follow same pattern)
CREATE TABLE maintenance_events.gas_turbine_maintenance (
    maintenance_id BIGSERIAL PRIMARY KEY,
    turbine_id INT NOT NULL REFERENCES master_data.gas_turbines(turbine_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    failure_code VARCHAR(50),
    downtime_hours FLOAT,
    repaired_components JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## See Also

- [data_pipeline.md](data_pipeline.md) - Pipeline orchestration
- [equipment_sim.md](equipment_sim.md) - Equipment simulation
- [db_setup.md](db_setup.md) - Database setup and schema creation
