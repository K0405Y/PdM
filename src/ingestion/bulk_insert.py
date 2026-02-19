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

logger = logging.getLogger(__name__)
# Try to import numpy for type conversion
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Column mappings: state_key -> database_column
# Aligned with DB schema (03_create_telemetry_tables.sql) and simulator next_state() outputs

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
    'vibration_trend_7d': 'vibration_trend_7d',
    'speed_stability': 'speed_stability',
    'efficiency_degradation_rate': 'efficiency_degradation_rate',
    'pressure_ratio': 'pressure_ratio',
    'load_factor': 'load_factor',
}

# Keys that come from the JSON 'features' field rather than directly from state
DERIVED_FEATURE_KEYS: Set[str] = {
    'vibration_trend_7d', 'temp_variation_24h', 'speed_stability',
    'efficiency_degradation_rate', 'pressure_ratio', 'load_factor',
}

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

    # Get configuration for equipment type
    config = {
        'turbine': (TURBINE_COLUMNS, 'gas_turbine_telemetry'),
        'compressor': (COMPRESSOR_COLUMNS, 'compressor_telemetry'),
        'pump': (PUMP_COLUMNS, 'pump_telemetry')
    }

    columns_map, table = config[equipment_type]
    defaults = {}  # Missing values become NULL via \\N

    logger.info(f"Bulk inserting {len(records):,} {equipment_type} records into {table}")

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
        logger.info(f"Successfully inserted {len(records):,} records")
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

    failure_config = {
        'turbine': {
            'table': 'failure_events.gas_turbine_failures',
            'id_col': 'turbine_id',
            'columns': ['failure_time', 'operating_hours_at_failure', 'failure_mode_code',
                       'speed_rpm_at_failure', 'egt_celsius_at_failure', 'vibration_mm_s_at_failure']
        },
        'compressor': {
            'table': 'failure_events.compressor_failures',
            'id_col': 'compressor_id',
            'columns': ['failure_time', 'operating_hours_at_failure', 'failure_mode_code',
                       'speed_rpm_at_failure', 'surge_margin_at_failure', 'vibration_amplitude_at_failure']
        },
        'pump': {
            'table': 'failure_events.pump_failures',
            'id_col': 'pump_id',
            'columns': ['failure_time', 'operating_hours_at_failure', 'failure_mode_code',
                       'speed_rpm_at_failure', 'vibration_mm_s_at_failure', 'cavitation_margin_at_failure']
        }
    }

    session = db.get_session()
    try:
        for record in failures:
            eq_type = record['equipment_type']
            config = failure_config[eq_type]
            state = record.get('state', {})

            all_cols = [config['id_col']] + config['columns']
            placeholders = ', '.join([f':val{i}' for i in range(len(all_cols))])
            sql = f"INSERT INTO {config['table']} ({', '.join(all_cols)}) VALUES ({placeholders})"

            values = {
                'val0': _clean(record['equipment_id']),
                'val1': record['failure_time'],
                'val2': _clean(record['operating_hours_at_failure']),
                'val3': record['failure_mode_code'],
                'val4': _clean(state.get('speed', 0))
            }

            if eq_type == 'turbine':
                values['val5'] = _clean(state.get('exhaust_gas_temp', 0))
                values['val6'] = _clean(state.get('vibration_rms', 0))
            elif eq_type == 'compressor':
                values['val5'] = _clean(state.get('surge_margin', 0))
                values['val6'] = _clean(state.get('orbit_amplitude', 0))
            elif eq_type == 'pump':
                values['val5'] = _clean(state.get('vibration_rms', 0))
                values['val6'] = _clean(state.get('npsh_margin', 0))

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

    maintenance_config = {
        'turbine': {
            'table': 'maintenance_events.gas_turbine_maintenance',
            'id_col': 'turbine_id',
        },
        'compressor': {
            'table': 'maintenance_events.compressor_maintenance',
            'id_col': 'compressor_id',
        },
        'pump': {
            'table': 'maintenance_events.pump_maintenance',
            'id_col': 'pump_id',
        }
    }

    columns = ['start_time', 'end_time', 'failure_code', 'downtime_hours', 'repaired_components']

    session = db.get_session()
    try:
        for record in maintenance_records:
            eq_type = record['equipment_type']
            config = maintenance_config[eq_type]

            all_cols = [config['id_col']] + columns
            placeholders = ', '.join([f':val{i}' for i in range(len(all_cols))])
            sql = f"INSERT INTO {config['table']} ({', '.join(all_cols)}) VALUES ({placeholders})"

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
