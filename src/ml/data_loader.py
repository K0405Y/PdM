"""
Data Loader for ML Training

Queries telemetry and failure data from PostgreSQL, joins them
to create labeled datasets for failure mode classification.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import yaml
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_table_config() -> Dict:
    with open(os.path.join(_PROJECT_ROOT, "table_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_engine(db_url: str = None):
    """Create SQLAlchemy engine from URL or environment."""
    if db_url is None:
        db_url = os.getenv('POSTGRES_URL', '')
    if db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+psycopg2://', 1)
    return create_engine(db_url, echo=False)


def load_telemetry(engine, equipment_type: str) -> pd.DataFrame:
    """
    Load telemetry data for an equipment type.

    Args:
        engine: SQLAlchemy engine
        equipment_type: 'turbine', 'compressor', or 'pump'

    Returns:
        DataFrame with all telemetry columns, sorted by equipment_id and sample_time
    """
    cfg = load_table_config()
    eq_cfg = cfg['equipment_types'][equipment_type]
    table = eq_cfg['telemetry']['table']
    eq_id_col = eq_cfg['telemetry']['equipment_id_column']

    query = f"SELECT * FROM {table} ORDER BY {eq_id_col}, sample_time"
    logger.info(f"Loading telemetry from {table}")
    df = pd.read_sql(query, engine)

    # Normalize equipment ID column name
    if eq_id_col != 'equipment_id':
        df = df.rename(columns={eq_id_col: 'equipment_id'})

    logger.info(f"Loaded {len(df)} telemetry records for {equipment_type}")
    return df


def load_failures(engine, equipment_type: str) -> pd.DataFrame:
    """
    Load failure events for an equipment type.

    Args:
        engine: SQLAlchemy engine
        equipment_type: 'turbine', 'compressor', or 'pump'

    Returns:
        DataFrame with failure events
    """
    cfg = load_table_config()
    eq_cfg = cfg['equipment_types'][equipment_type]
    table = eq_cfg['failures']['table']
    eq_id_col = eq_cfg['failures']['equipment_id_column']

    query = f"SELECT * FROM {table} ORDER BY {eq_id_col}, failure_time"
    logger.info(f"Loading failures from {table}")
    df = pd.read_sql(query, engine)

    if eq_id_col != 'equipment_id':
        df = df.rename(columns={eq_id_col: 'equipment_id'})

    logger.info(f"Loaded {len(df)} failure events for {equipment_type}")
    return df


def load_equipment_ids(engine, equipment_type: str) -> List[int]:
    """Load active equipment IDs for a given type."""
    cfg = load_table_config()
    mcfg = cfg['equipment_types'][equipment_type]['master']
    table, id_col = mcfg['table'], mcfg['id_column']

    query = f"SELECT {id_col} FROM {table} WHERE status = 'active' ORDER BY {id_col}"
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return [row[0] for row in result]


def get_sensor_columns(equipment_type: str) -> List[str]:
    """
    Get sensor-only columns (no health/ground-truth) for an equipment type.

    Returns columns suitable for SENSOR_ONLY mode evaluation.
    """
    cfg = load_table_config()
    eq_cfg = cfg['equipment_types'][equipment_type]['telemetry']
    health_cols = set(eq_cfg.get('health_columns', []))

    # Get all mapped DB columns, exclude health and metadata
    mappings = eq_cfg['column_mappings']
    eq_id_db_col = eq_cfg.get('equipment_id_column', '')

    # Exclude by DB column name
    exclude_db = health_cols | {
        eq_id_db_col, 'sample_time',
        'num_active_faults', 'total_faults_initiated',
        'upset_active', 'upset_type', 'upset_severity',
    }
    # Also exclude by state key
    exclude_state = {
        'equipment_id', 'sample_time',
        'num_active_faults', 'total_faults_initiated',
        'upset_active', 'upset_type', 'upset_severity',
    }

    sensor_cols = []
    for state_key, db_col in mappings.items():
        if db_col not in exclude_db and state_key not in exclude_state:
            sensor_cols.append(db_col)

    return sensor_cols


def get_health_columns(equipment_type: str) -> List[str]:
    """Get ground-truth health columns for an equipment type."""
    cfg = load_table_config()
    return cfg['equipment_types'][equipment_type]['telemetry'].get('health_columns', [])


def get_failure_modes(equipment_type: str) -> List[str]:
    """Get valid failure mode codes for an equipment type."""
    cfg = load_table_config()
    modes = cfg.get('failure_modes', {}).get(
        {'turbine': 'gas_turbine', 'compressor': 'compressor', 'pump': 'pump'}[equipment_type],
        []
    )
    return [m['mode_code'] for m in modes]
