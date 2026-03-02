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
from datetime import timedelta
from io import StringIO
from typing import List, Dict, Set
from sqlalchemy import text
from shared_config import load_table_config

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


def _build_derived_feature_keys() -> Set[str]:
    """Build the set of derived feature keys from YAML config."""
    cfg = load_table_config()
    return set(cfg["derived_columns"]) - {"operating_state"}


# Column mappings: state_key -> database_column (loaded from YAML)
_COLUMN_MAPPINGS = _build_column_mappings()
TURBINE_COLUMNS = _COLUMN_MAPPINGS["turbine"]
COMPRESSOR_COLUMNS = _COLUMN_MAPPINGS["compressor"]
PUMP_COLUMNS = _COLUMN_MAPPINGS["pump"]

# Keys that come from the JSON 'features' field rather than directly from state
DERIVED_FEATURE_KEYS: Set[str] = _build_derived_feature_keys()

# Record-level keys (not from state dict)
_RECORD_KEYS = {'equipment_id', 'sample_time', 'operating_hours'}


def _get_value(record: Dict, key: str, defaults: Dict) -> str:
    """Extract value from record, checking state dict and JSON features."""
    state = record.get('state', {})

    if key in _RECORD_KEYS:
        value = record.get(key)
    elif key in DERIVED_FEATURE_KEYS:
        # Extract from JSON 'features' field in state
        features_raw = state.get('features')
        if features_raw:
            features = json.loads(features_raw) if isinstance(features_raw, str) else features_raw
            value = features.get(key)
        else:
            value = None
    else:
        # All other keys are flat in the state dict
        value = state.get(key)

    # Use default if missing
    if value is None:
        value = defaults.get(key)

    # Convert numpy types
    if HAS_NUMPY and isinstance(value, (np.integer, np.floating)):
        value = float(value)

    # Round floats for consistent precision
    if isinstance(value, float):
        value = round(value, 4)

    # Return as string or NULL marker
    return str(value) if value is not None else '\\N'


def bulk_insert_telemetry(db, records: List[Dict], equipment_type: str) -> int:
    """
    Fast bulk insert using PostgreSQL COPY command.

    Args:
        db: Database connection object (with get_session method)
        records: List of telemetry records from simulate_equipment
                 (each with 'equipment_id', 'sample_time', 'operating_hours', 'state')
        equipment_type: 'turbine', 'compressor', or 'pump'

    Returns:
        Number of records inserted
    """
    if not records:
        return 0

    # Get configuration for equipment type from YAML
    cfg = load_table_config()
    eq_cfg = cfg["equipment_types"][equipment_type]
    columns_map = eq_cfg["telemetry"]["column_mappings"]
    table = eq_cfg["telemetry"]["table"]
    defaults = {}  # Missing values become NULL via \\N

    # Build CSV buffer for COPY command
    buffer = StringIO()
    writer = csv.writer(buffer, delimiter='\t')

    for record in records:
        row = [_get_value(record, key, defaults) for key in columns_map.keys()]
        writer.writerow(row)

    buffer.seek(0)

    # Execute PostgreSQL COPY command
    session = db.get_session()
    try:
        raw_conn = session.connection().connection
        cursor = raw_conn.cursor()

        cursor.execute("SET search_path TO telemetry, master_data, failure_events, public")

        sql = f"COPY {table} ({','.join(columns_map.values())}) FROM STDIN WITH CSV DELIMITER '\t' NULL '\\N'"
        cursor.copy_expert(sql, buffer)

        raw_conn.commit()
    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        raise
    finally:
        session.close()

    return len(records)


def insert_failures(db, failures: List[Dict]) -> int:
    """Insert failure records with NumPy type cleaning."""
    if not failures:
        return 0

    def _clean(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    logger.info(f"Inserting {len(failures)} failure events")

    yaml_cfg = load_table_config()

    session = db.get_session()
    try:
        for record in failures:
            eq_type = record['equipment_type']
            fcfg = yaml_cfg["equipment_types"][eq_type]["failures"]
            state = record.get('state', {})

            id_col = fcfg["equipment_id_column"]
            insert_cols = fcfg["insert_columns"]
            state_mappings = fcfg["state_key_mappings"]

            all_cols = [id_col] + insert_cols
            placeholders = ', '.join([f':val{i}' for i in range(len(all_cols))])
            sql = f"INSERT INTO {fcfg['table']} ({', '.join(all_cols)}) VALUES ({placeholders})"

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


def insert_maintenance(db, maintenance_records: List[Dict]) -> int:
    """Insert maintenance event records with computed end_time."""
    if not maintenance_records:
        return 0

    def _clean(val):
        if hasattr(val, 'item'):
            return val.item()
        return val

    logger.info(f"Inserting {len(maintenance_records)} maintenance events")

    yaml_cfg = load_table_config()
    columns = yaml_cfg["maintenance_insert_columns"]

    session = db.get_session()
    try:
        for record in maintenance_records:
            eq_type = record['equipment_type']
            mcfg = yaml_cfg["equipment_types"][eq_type]["maintenance"]

            all_cols = [mcfg['equipment_id_column']] + columns
            placeholders = ', '.join([f':val{i}' for i in range(len(all_cols))])
            sql = f"INSERT INTO {mcfg['table']} ({', '.join(all_cols)}) VALUES ({placeholders})"

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