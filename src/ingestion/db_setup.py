"""
Database Connection and Master Data Management

Simple database operations:
- Connect to PostgreSQL
- Execute SQL scripts
- Seed equipment master data
- Fetch equipment configurations
"""
import os
import random
import logging
from typing import List, Dict
from datetime import datetime, timedelta
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_table_config():
    with open(os.path.join(_PROJECT_ROOT, "table_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database connection and operations."""

    def __init__(self, db_url: str):
        """Initialize database connection."""
        self.db_url = self._format_url(db_url)
        self.engine = None
        self.Session = None

    def _format_url(self, url: str) -> str:
        """Format PostgreSQL URL for SQLAlchemy."""
        if url.startswith('postgresql://'):
            url = url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        return url

    def connect(self):
        """Connect to database."""
        self.engine = create_engine(self.db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

        # Test connection
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("Connected to PostgreSQL")

    def get_session(self):
        """Get a new database session."""
        return self.Session()

    def execute_script(self, script_path: str):
        """Execute SQL script file."""
        with open(script_path, 'r') as f:
            sql = f.read()

        with self.engine.begin() as conn:
            for statement in sql.split(';'):
                clean = '\n'.join(
                    line for line in statement.split('\n')
                    if line.strip() and not line.strip().startswith('--')
                ).strip()

                if clean:
                    conn.execute(text(clean))

        logger.info(f"Executed: {script_path}")

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database closed")


def _tiered_health(i: int, n_total: int, fresh_range, mid_range, late_range) -> float:
    """Distribute equipment across three health tiers to ensure failure events in simulation window.

    Tier split: ~25% near end-of-life, ~35% mid-life, ~40% fresh.
    Guarantees slow-onset modes (rotor_crack, comp_fouling, wear_ring) produce enough
    failure events within the 4,320-hour simulation window for classifier training.
    """
    ratio = i / max(n_total - 1, 1)
    if ratio < 0.25:
        return random.uniform(*late_range)
    elif ratio < 0.60:
        return random.uniform(*mid_range)
    else:
        return random.uniform(*fresh_range)


class MasterData:
    """Equipment master data seeding and retrieval."""

    def __init__(self, db: Database):
        self.db = db

    def _get_existing_equipment(self, equipment_type: str) -> List[int]:
        """Get IDs of existing active equipment by type."""
        cfg = load_table_config()
        mcfg = cfg["equipment_types"][equipment_type]["master"]
        table, id_col = mcfg["table"], mcfg["id_column"]
        session = self.db.get_session()
        try:
            result = session.execute(text(
                f"SELECT {id_col} FROM {table} WHERE status = 'active' ORDER BY {id_col}"
            ))
            return [row[0] for row in result]
        finally:
            session.close()

    def get_existing_turbines(self) -> List[int]:
        """Get IDs of existing active turbines."""
        return self._get_existing_equipment("turbine")

    def get_existing_compressors(self) -> List[int]:
        """Get IDs of existing active compressors."""
        return self._get_existing_equipment("compressor")

    def get_existing_pumps(self) -> List[int]:
        """Get IDs of existing active pumps."""
        return self._get_existing_equipment("pump")

    def seed_turbines(self, count: int) -> List[int]:
        """Create gas turbine records"""
        cfg = load_table_config()
        mcfg = cfg["equipment_types"]["turbine"]["master"]
        table, id_col = mcfg["table"], mcfg["id_column"]
        cols = mcfg["insert_columns"]
        col_list = ", ".join(cols)
        placeholders = ", ".join(f":{c}" for c in cols)

        ids = []
        session = self.db.get_session()

        try:
            for i in range(count):
                name = f"GT-{1 + i}"
                values = {
                    'name': name,
                    'serial_number': f"SN-GT-{1 + i}",
                    'location': f"Platform-{i % 3 + 1}",
                    'installed_date': datetime(2024, 1, 1).date() + timedelta(days=random.randint(0, 364)),
                    'initial_health_hgp': random.uniform(0.70, 0.98),
                    'initial_health_blade_compressor': random.uniform(0.75, 0.98),
                    'initial_health_blade_turbine': random.uniform(0.75, 0.98),
                    'initial_health_bearing': random.uniform(0.70, 0.98),
                    'initial_health_fuel': random.uniform(0.70, 0.98),
                    'initial_health_compressor_fouling': _tiered_health(
                        i, count, fresh_range=(0.93, 0.99), mid_range=(0.78, 0.88), late_range=(0.65, 0.72)
                    ),
                }
                result = session.execute(text(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    f"ON CONFLICT (name) DO NOTHING RETURNING {id_col}"
                ), values)
                tid = result.scalar()
                if tid is None:
                    tid = session.execute(text(
                        f"SELECT {id_col} FROM {table} WHERE name = :name"
                    ), {'name': name}).scalar()
                ids.append(tid)

            session.commit()
            logger.info(f"Seeded {count} turbines")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return ids

    def seed_compressors(self, count: int) -> List[int]:
        """Create compressor records"""
        cfg = load_table_config()
        mcfg = cfg["equipment_types"]["compressor"]["master"]
        table, id_col = mcfg["table"], mcfg["id_column"]
        cols = mcfg["insert_columns"]
        col_list = ", ".join(cols)
        placeholders = ", ".join(f":{c}" for c in cols)

        ids = []
        session = self.db.get_session()

        try:
            for i in range(count):
                name = f"COMP-{1 + i}"
                values = {
                    'name': name,
                    'serial_number': f"SN-COMP-{1 + i}",
                    'location': f"Facility-{i % 4 + 1}",
                    'installed_date': datetime(2024, 1, 1).date() + timedelta(days=random.randint(0, 364)),
                    'design_flow_m3h': random.uniform(1200, 1800),
                    'design_head_kj_kg': random.uniform(7500, 8500),
                    'initial_health_impeller': random.uniform(0.88, 0.98),
                    'initial_health_bearing': random.uniform(0.85, 0.98),
                    'initial_health_seal_primary': random.uniform(0.90, 0.98),
                    'initial_health_seal_secondary': random.uniform(0.93, 0.99),
                    'initial_health_bearing_thrust': random.uniform(0.78, 0.95),
                    'initial_health_rotor_crack': _tiered_health(
                        i, count, fresh_range=(0.88, 0.97), mid_range=(0.60, 0.75), late_range=(0.42, 0.52)
                    ),
                }
                result = session.execute(text(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    f"ON CONFLICT (name) DO NOTHING RETURNING {id_col}"
                ), values)
                cid = result.scalar()
                if cid is None:
                    cid = session.execute(text(
                        f"SELECT {id_col} FROM {table} WHERE name = :name"
                    ), {'name': name}).scalar()
                ids.append(cid)

            session.commit()
            logger.info(f"Seeded {count} compressors")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return ids

    def seed_pumps(self, count: int) -> List[int]:
        """Create pump records"""
        cfg = load_table_config()
        mcfg = cfg["equipment_types"]["pump"]["master"]
        table, id_col = mcfg["table"], mcfg["id_column"]
        # npsh_available_m is in insert_columns but not provided by seed logic;
        # filter to only columns we supply values for
        seed_cols = [c for c in mcfg["insert_columns"] if c != "npsh_available_m"]
        col_list = ", ".join(seed_cols)
        placeholders = ", ".join(f":{c}" for c in seed_cols)

        ids = []
        services = [
            {'name': 'Crude Booster', 'flow': 200, 'head': 100, 'density': 850},
            {'name': 'Seawater Injection', 'flow': 300, 'head': 150, 'density': 1025},
            {'name': 'Process Water', 'flow': 100, 'head': 60, 'density': 1000},
            {'name': 'Methanol Pump', 'flow': 50, 'head': 80, 'density': 790},
            {'name': 'Fire Water', 'flow': 400, 'head': 120, 'density': 1000},
        ]

        session = self.db.get_session()

        try:
            for i in range(count):
                service = services[i % len(services)]
                name = f"PUMP-{1 + i}"
                values = {
                    'name': name,
                    'serial_number': f"SN-PUMP-{1 + i}",
                    'service_type': service['name'],
                    'location': f"Platform-{i % 5 + 1}",
                    'installed_date': datetime(2024, 1, 1).date() + timedelta(days=random.randint(0, 364)),
                    'design_flow_m3h': service['flow'],
                    'design_head_m': service['head'],
                    'design_speed_rpm': 3000,
                    'fluid_density_kg_m3': service['density'],
                    'initial_health_impeller': random.uniform(0.70, 0.98),
                    'initial_health_seal': random.uniform(0.70, 0.98),
                    'initial_health_bearing_de': random.uniform(0.70, 0.98),
                    'initial_health_bearing_nde': random.uniform(0.70, 0.98),
                    'initial_health_wear_ring': random.uniform(0.70, 0.98),
                }
                result = session.execute(text(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                    f"ON CONFLICT (name) DO NOTHING RETURNING {id_col}"
                ), values)
                pid = result.scalar()
                if pid is None:
                    pid = session.execute(text(
                        f"SELECT {id_col} FROM {table} WHERE name = :name"
                    ), {'name': name}).scalar()
                ids.append(pid)

            session.commit()
            logger.info(f"Seeded {count} pumps")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return ids

    def get_configs(self, equipment_ids: List[int], equipment_type: str) -> List[Dict]:
        """Fetch equipment configurations from database."""
        cfg = load_table_config()
        mcfg = cfg["equipment_types"][equipment_type]["master"]
        table, id_col = mcfg["table"], mcfg["id_column"]
        session = self.db.get_session()

        try:
            result = session.execute(
                text(f"SELECT * FROM {table} WHERE {id_col} = ANY(:ids)"),
                {'ids': equipment_ids}
            )
            columns = [col for col in result.keys()]
            return [dict(zip(columns, row)) for row in result.fetchall()]
        finally:
            session.close()