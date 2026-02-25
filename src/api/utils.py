"""
Shared utilities for the API layer.
Pagination helpers, row conversion, operating state classification.
"""
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

# Operating state classification
def classify_operating_state(
    speed: float,
    speed_target: float,
    in_maintenance: bool = False,
) -> str:
    """Infer operating state from physical parameters.

    Returns one of: 'running', 'idle', 'startup', 'shutdown', 'maintenance'.
    """
    if in_maintenance:
        return "maintenance"
    if speed == 0 and speed_target == 0:
        return "idle"
    if speed_target > 0 and speed < speed_target * 0.5:
        return "startup"
    if speed_target == 0 and speed > 0:
        return "shutdown"
    return "running"


# Row conversion
def row_to_dict(row, columns: List[str]) -> Dict[str, Any]:
    """Convert a SQLAlchemy Row/tuple to a dict using column names."""
    return dict(zip(columns, row))


def rows_to_dicts(rows, columns: List[str]) -> List[Dict[str, Any]]:
    """Convert multiple SQLAlchemy rows to list of dicts."""
    return [dict(zip(columns, row)) for row in rows]



# Cursor-based pagination (for telemetry — millions of rows)
def build_cursor_query(
    base_sql: str,
    cursor_col: str,
    after_cursor: Optional[int],
    limit: int,
    extra_params: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """Build a keyset-paginated query.

    Returns (sql_string, params_dict).
    """
    params = dict(extra_params or {})
    if after_cursor is not None:
        base_sql += f" AND {cursor_col} > :after_cursor"
        params["after_cursor"] = after_cursor
    base_sql += f" ORDER BY {cursor_col} ASC LIMIT :limit"
    params["limit"] = limit
    return base_sql, params



# Offset-based pagination (for master data — small tables)
def build_offset_query(
    base_sql: str,
    count_sql: str,
    page: int,
    page_size: int,
    extra_params: Optional[Dict] = None,
) -> Tuple[str, str, Dict]:
    """Build an offset-paginated query with count.

    Returns (data_sql, count_sql, params_dict).
    """
    params = dict(extra_params or {})
    offset = (page - 1) * page_size
    data_sql = base_sql + " LIMIT :limit OFFSET :offset"
    params["limit"] = page_size
    params["offset"] = offset
    return data_sql, count_sql, params



# Equipment existence check
MASTER_TABLES = {
    "turbine": ("master_data.gas_turbines", "turbine_id"),
    "compressor": ("master_data.compressors", "compressor_id"),
    "pump": ("master_data.pumps", "pump_id"),
}


def validate_equipment_exists(
    session: Session,
    equipment_type: str,
    equipment_id: int,
) -> None:
    """Raise 404 if the equipment does not exist."""
    if equipment_type not in MASTER_TABLES:
        raise HTTPException(status_code=400, detail=f"Unknown equipment type: {equipment_type}")
    table, id_col = MASTER_TABLES[equipment_type]
    result = session.execute(
        text(f"SELECT 1 FROM {table} WHERE {id_col} = :id"),
        {"id": equipment_id},
    ).fetchone()
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"{equipment_type} with id {equipment_id} not found",
        )


# Telemetry table config 
TABLE_CONFIG = {
    "turbine": {
        "telemetry_table": "telemetry.gas_turbine_telemetry",
        "telemetry_id_col": "telemetry_id",
        "id_col": "turbine_id",
        "failure_table": "failure_events.gas_turbine_failures",
        "failure_id_col": "turbine_id",
        "maintenance_table": "maintenance_events.gas_turbine_maintenance",
        "maintenance_id_col": "turbine_id",
        "health_cols": ["health_hgp", "health_blade", "health_bearing", "health_fuel"],
        "key_numeric_cols": [
            "speed_rpm", "egt_celsius", "vibration_rms_mm_s",
            "efficiency_fraction", "fuel_flow_kg_s",
        ],
    },
    "compressor": {
        "telemetry_table": "telemetry.compressor_telemetry",
        "telemetry_id_col": "telemetry_id",
        "id_col": "compressor_id",
        "failure_table": "failure_events.compressor_failures",
        "failure_id_col": "compressor_id",
        "maintenance_table": "maintenance_events.compressor_maintenance",
        "maintenance_id_col": "compressor_id",
        "health_cols": ["health_impeller", "health_bearing"],
        "key_numeric_cols": [
            "speed_rpm", "flow_m3h", "surge_margin_percent",
            "efficiency_fraction", "orbit_amplitude_mm",
        ],
    },
    "pump": {
        "telemetry_table": "telemetry.pump_telemetry",
        "telemetry_id_col": "telemetry_id",
        "id_col": "pump_id",
        "failure_table": "failure_events.pump_failures",
        "failure_id_col": "pump_id",
        "maintenance_table": "maintenance_events.pump_maintenance",
        "maintenance_id_col": "pump_id",
        "health_cols": ["health_impeller", "health_seal", "health_bearing_de", "health_bearing_nde"],
        "key_numeric_cols": [
            "speed_rpm", "flow_m3h", "vibration_rms_mm_s",
            "efficiency_fraction", "cavitation_margin_m",
        ],
    },
}
