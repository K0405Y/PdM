"""
Shared utilities for the API layer.
Pagination helpers, row conversion, operating state classification
"""
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from api.config import load_table_config

# Operating state classification
def classify_operating_state(
    speed: float,
    speed_target: float,
) -> str:
    """Infer operating state from physical parameters.

    Returns one of: 'running', 'idle', 'startup', 'shutdown'.
    """
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



def _build_master_tables() -> Dict[str, tuple]:
    """Build MASTER_TABLES mapping from YAML config."""
    cfg = load_table_config()
    return {
        eq_type: (eq_cfg["master"]["table"], eq_cfg["master"]["id_column"])
        for eq_type, eq_cfg in cfg["equipment_types"].items()
    }


def _build_table_config() -> Dict[str, Dict[str, Any]]:
    """Build TABLE_CONFIG mapping from YAML config.

    Preserves the same dict key names used by all routers so downstream
    code requires no structural changes.
    """
    cfg = load_table_config()
    result = {}
    for eq_type, eq_cfg in cfg["equipment_types"].items():
        result[eq_type] = {
            "telemetry_table": eq_cfg["telemetry"]["table"],
            "telemetry_id_col": eq_cfg["telemetry"]["id_column"],
            "id_col": eq_cfg["telemetry"]["equipment_id_column"],
            "failure_table": eq_cfg["failures"]["table"],
            "failure_id_col": eq_cfg["failures"]["equipment_id_column"],
            "maintenance_table": eq_cfg["maintenance"]["table"],
            "maintenance_id_col": eq_cfg["maintenance"]["equipment_id_column"],
            "health_cols": list(eq_cfg["telemetry"]["health_columns"]),
            "key_numeric_cols": list(eq_cfg["telemetry"]["key_numeric_columns"]),
        }
    return result


# Equipment existence check
MASTER_TABLES = _build_master_tables()



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


# Telemetry table config (built from YAML)
TABLE_CONFIG = _build_table_config()
