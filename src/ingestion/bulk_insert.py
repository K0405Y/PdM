"""
Optimized Bulk Insertion

Fast database insertion using PostgreSQL COPY command:
- 100x faster than row-by-row INSERT
- Uses PostgreSQL's native binary protocol
- Handles all equipment types
"""

import csv
import json
import logging
import os
from datetime import timedelta
from io import StringIO
from typing import List, Dict
import yaml
from sqlalchemy import text

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_table_config():
    with open(os.path.join(_PROJECT_ROOT, "table_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

logger = logging.getLogger(__name__)
# Try to import numpy for type conversion
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _build_column_mappings() -> Dict[str, dict]:
    """Build column mapping dicts from YAML config for each equipment type."""
    cfg = load_table_config()
    return {
        eq_type: eq_cfg["telemetry"]["column_mappings"]
        for eq_type, eq_cfg in cfg["equipment_types"].items()
    }


# Column mappings: state_key -> database_column (loaded from YAML)
_COLUMN_MAPPINGS = _build_column_mappings()
TURBINE_COLUMNS = _COLUMN_MAPPINGS["turbine"]
COMPRESSOR_COLUMNS = _COLUMN_MAPPINGS["compressor"]
PUMP_COLUMNS = _COLUMN_MAPPINGS["pump"]

# Record-level keys (not from state dict)
_RECORD_KEYS = {'equipment_id', 'sample_time', 'operating_hours'}


def _get_value(record: Dict, key: str, defaults: Dict) -> str:
    """Extract value from record, checking state dict and JSON features.

    Tries flat state keys first, then falls back to the JSON 'features'
    field if present (for DERIVED_FEATURES output mode).
    """
    state = record.get('state', {})

    if key in _RECORD_KEYS:
        value = record.get(key)
    else:
        value = state.get(key)
        # Fall back to JSON 'features' field if key not found flat
        if value is None:
            features_raw = state.get('features')
            if features_raw:
                features = json.loads(features_raw) if isinstance(features_raw, str) else features_raw
                value = features.get(key)

    if value is None:
        value = defaults.get(key)

    if HAS_NUMPY and isinstance(value, (np.integer, np.floating)):
        value = float(value)

    if isinstance(value, float):
        value = round(value, 4)

    return str(value) if value is not None else '\\N'


def _get_test_column_mappings(eq_cfg: dict) -> dict:
    """Derive test_telemetry column mappings from telemetry mappings.

    Uses telemetry.column_mappings minus any keys whose DB column name
    appears in telemetry.health_columns.
    """
    health_cols = set(eq_cfg["telemetry"].get("health_columns", []))
    return {
        k: v for k, v in eq_cfg["telemetry"]["column_mappings"].items()
        if v not in health_cols
    }


def bulk_insert_telemetry(db, records: List[Dict], equipment_type: str,
                          use_test_schema: bool = False) -> int:
    """
    Fast bulk insert using PostgreSQL COPY command.

    Args:
        db: Database connection object (with get_session method)
        records: List of telemetry records from simulate_equipment
                 (each with 'equipment_id', 'sample_time', 'operating_hours', 'state')
        equipment_type: 'turbine', 'compressor', or 'pump'
        use_test_schema: If True, insert into test_data.* tables (sensor-only,
                         no health columns). Column mappings are derived from
                         telemetry mappings minus health_columns.

    Returns:
        Number of records inserted
    """
    if not records:
        return 0

    cfg = load_table_config()
    eq_cfg = cfg["equipment_types"][equipment_type]

    if use_test_schema:
        table = eq_cfg["test_telemetry"]["table"]
        columns_map = _get_test_column_mappings(eq_cfg)
    else:
        table = eq_cfg["telemetry"]["table"]
        columns_map = eq_cfg["telemetry"]["column_mappings"]

    defaults = {}

    # Build CSV buffer for COPY command
    buffer = StringIO()
    writer = csv.writer(buffer, delimiter='\t')

    for record in records:
        row = [_get_value(record, key, defaults) for key in columns_map.keys()]
        writer.writerow(row)

    buffer.seek(0)

    # Execute PostgreSQL COPY command
    schema = "test_data" if use_test_schema else "telemetry"
    session = db.get_session()
    try:
        raw_conn = session.connection().connection
        cursor = raw_conn.cursor()

        cursor.execute(f"SET search_path TO {schema}, master_data, failure_events, public")

        sql = f"COPY {table} ({','.join(columns_map.values())}) FROM STDIN WITH CSV DELIMITER '\t' NULL '\\N'"
        cursor.copy_expert(sql, buffer)

        raw_conn.commit()
    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        raise
    finally:
        session.close()

    return len(records)


def insert_failures(db, failures: List[Dict],
                    use_test_schema: bool = False) -> int:
    """Insert failure records with NumPy type cleaning.

    Args:
        db: Database connection object
        failures: List of failure records from simulation
        use_test_schema: If True, insert into test_data.* tables instead of
                         failure_events.* (preserves ground truth for evaluation)
    """
    if not failures:
        return 0

    def _clean(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    schema_label = "test_data" if use_test_schema else "failure_events"
    logger.info(f"Inserting {len(failures)} failure events into {schema_label}")

    yaml_cfg = load_table_config()

    session = db.get_session()
    try:
        for record in failures:
            eq_type = record['equipment_type']
            fcfg = yaml_cfg["equipment_types"][eq_type]["failures"]
            state = record.get('state', {})

            table = (yaml_cfg["equipment_types"][eq_type]["test_failure"]["table"]
                     if use_test_schema else fcfg["table"])
            id_col = fcfg["equipment_id_column"]
            insert_cols = fcfg["insert_columns"]
            state_mappings = fcfg["state_key_mappings"]

            all_cols = [id_col] + insert_cols
            placeholders = ', '.join([f':val{i}' for i in range(len(all_cols))])
            sql = f"INSERT INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders})"

            # Common values: equipment_id, failure_time, operating_hours, failure_mode_code
            values = {
                'val0': _clean(record['equipment_id']),
                'val1': record['failure_time'],
                'val2': _clean(record['operating_hours_at_failure']),
                'val3': record['failure_mode_code'],
            }

            # Equipment-specific state snapshot columns
            for idx, mapping in enumerate(state_mappings):
                values[f'val{4 + idx}'] = _clean(state.get(mapping['state_key'], 0))

            session.execute(text(sql), values)

        session.commit()
        logger.info(f"Inserted {len(failures)} failure records")
    except Exception as e:
        logger.error(f"Failure insert error: {e}")
        raise
    finally:
        session.close()

    return len(failures)


def insert_maintenance(db, maintenance_records: List[Dict],
                       use_test_schema: bool = False) -> int:
    """Insert maintenance event records with computed end_time.

    Args:
        db: Database connection object
        maintenance_records: List of maintenance records from simulation
        use_test_schema: If True, insert into test_data.* tables instead of
                         maintenance_events.* (preserves ground truth for evaluation)
    """
    if not maintenance_records:
        return 0

    def _clean(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    schema_label = "test_data" if use_test_schema else "maintenance_events"
    logger.info(f"Inserting {len(maintenance_records)} maintenance events into {schema_label}")

    yaml_cfg = load_table_config()
    columns = yaml_cfg["maintenance_insert_columns"]

    session = db.get_session()
    try:
        for record in maintenance_records:
            eq_type = record['equipment_type']
            mcfg = yaml_cfg["equipment_types"][eq_type]["maintenance"]

            table = (yaml_cfg["equipment_types"][eq_type]["test_maintenance"]["table"]
                     if use_test_schema else mcfg["table"])
            all_cols = [mcfg['equipment_id_column']] + columns
            placeholders = ', '.join([f':val{i}' for i in range(len(all_cols))])
            sql = f"INSERT INTO {table} ({', '.join(all_cols)}) VALUES ({placeholders})"

            start_time = record['start_time']
            downtime_hours = _clean(record['downtime_hours'])
            end_time = start_time + timedelta(hours=downtime_hours)

            repaired = record.get('repaired_components', {})
            repaired_json = json.dumps({k: round(float(v), 4) for k, v in repaired.items()})

            values = {
                'val0': _clean(record['equipment_id']),
                'val1': start_time,
                'val2': end_time,
                'val3': record['failure_code'],
                'val4': downtime_hours,
                'val5': repaired_json,
            }

            session.execute(text(sql), values)

        session.commit()
        logger.info(f"Inserted {len(maintenance_records)} maintenance records")
    except Exception as e:
        logger.error(f"Maintenance insert error: {e}")
        raise
    finally:
        session.close()

    return len(maintenance_records)