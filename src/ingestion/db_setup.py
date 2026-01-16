"""
Database Connection and Master Data Management

Simple database operations:
- Connect to PostgreSQL
- Execute SQL scripts
- Seed equipment master data
- Fetch equipment configurations
"""
import random
import logging
from typing import List, Dict
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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


class MasterData:
    """Equipment master data seeding and retrieval."""

    def __init__(self, db: Database):
        self.db = db

    def get_existing_turbines(self) -> List[int]:
        """Get IDs of existing active turbines."""
        session = self.db.get_session()
        try:
            result = session.execute(text("""
                SELECT turbine_id FROM master_data.gas_turbines
                WHERE status = 'active'
                ORDER BY turbine_id
            """))
            return [row[0] for row in result]
        finally:
            session.close()

    def get_existing_compressors(self) -> List[int]:
        """Get IDs of existing active compressors."""
        session = self.db.get_session()
        try:
            result = session.execute(text("""
                SELECT compressor_id FROM master_data.centrifugal_compressors
                WHERE status = 'active'
                ORDER BY compressor_id
            """))
            return [row[0] for row in result]
        finally:
            session.close()

    def get_existing_pumps(self) -> List[int]:
        """Get IDs of existing active pumps."""
        session = self.db.get_session()
        try:
            result = session.execute(text("""
                SELECT pump_id FROM master_data.centrifugal_pumps
                WHERE status = 'active'
                ORDER BY pump_id
            """))
            return [row[0] for row in result]
        finally:
            session.close()

    def seed_turbines(self, count: int) -> List[int]:
        """Create gas turbine records."""
        ids = []
        session = self.db.get_session()

        try:
            for i in range(count):
                result = session.execute(text("""
                    INSERT INTO master_data.gas_turbines
                    (name, serial_number, location, installed_date,
                     initial_health_hgp, initial_health_blade,
                     initial_health_bearing, initial_health_fuel)
                    VALUES (:name, :sn, :loc, :date, :hgp, :blade, :bearing, :fuel)
                    RETURNING turbine_id
                """), {
                    'name': f"GT-{1001 + i}",
                    'sn': f"SN-GT-{10001 + i}",
                    'loc': f"Platform-{i % 3 + 1}",
                    'date': datetime.now().date() - timedelta(days=random.randint(30, 365)),
                    'hgp': random.uniform(0.70, 0.98),
                    'blade': random.uniform(0.70, 0.98),
                    'bearing': random.uniform(0.70, 0.98),
                    'fuel': random.uniform(0.70, 0.98)
                })
                ids.append(result.scalar())

            session.commit()
            logger.info(f"Seeded {count} turbines")
        finally:
            session.close()

        return ids

    def seed_compressors(self, count: int) -> List[int]:
        """Create centrifugal compressor records."""
        ids = []
        session = self.db.get_session()

        try:
            for i in range(count):
                result = session.execute(text("""
                    INSERT INTO master_data.centrifugal_compressors
                    (name, serial_number, location, installed_date,
                     design_flow_m3h, design_head_kj_kg,
                     initial_health_impeller, initial_health_bearing)
                    VALUES (:name, :sn, :loc, :date, :flow, :head, :imp, :bear)
                    RETURNING compressor_id
                """), {
                    'name': f"CC-{2001 + i}",
                    'sn': f"SN-CC-{20001 + i}",
                    'loc': f"Facility-{i % 4 + 1}",
                    'date': datetime.now().date() - timedelta(days=random.randint(30, 365)),
                    'flow': random.uniform(1200, 1800),
                    'head': random.uniform(7500, 8500),
                    'imp': random.uniform(0.70, 0.98),
                    'bear': random.uniform(0.70, 0.98)
                })
                ids.append(result.scalar())

            session.commit()
            logger.info(f"Seeded {count} compressors")
        finally:
            session.close()

        return ids

    def seed_pumps(self, count: int) -> List[int]:
        """Create centrifugal pump records."""
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
                result = session.execute(text("""
                    INSERT INTO master_data.centrifugal_pumps
                    (name, serial_number, service_type, location, installed_date,
                     design_flow_m3h, design_head_m, design_speed_rpm, fluid_density_kg_m3,
                     initial_health_impeller, initial_health_seal,
                     initial_health_bearing_de, initial_health_bearing_nde)
                    VALUES (:name, :sn, :svc, :loc, :date, :flow, :head, :speed, :dens,
                            :imp, :seal, :bde, :bnde)
                    RETURNING pump_id
                """), {
                    'name': f"CP-{3001 + i}",
                    'sn': f"SN-CP-{30001 + i}",
                    'svc': service['name'],
                    'loc': f"Platform-{i % 5 + 1}",
                    'date': datetime.now().date() - timedelta(days=random.randint(30, 365)),
                    'flow': service['flow'],
                    'head': service['head'],
                    'speed': 3000,
                    'dens': service['density'],
                    'imp': random.uniform(0.70, 0.98),
                    'seal': random.uniform(0.70, 0.98),
                    'bde': random.uniform(0.70, 0.98),
                    'bnde': random.uniform(0.70, 0.98)
                })
                ids.append(result.scalar())

            session.commit()
            logger.info(f"Seeded {count} pumps")
        finally:
            session.close()

        return ids

    def get_configs(self, equipment_ids: List[int], equipment_type: str) -> List[Dict]:
        """Fetch equipment configurations from database."""
        tables = {
            'turbine': ('master_data.gas_turbines', 'turbine_id'),
            'compressor': ('master_data.centrifugal_compressors', 'compressor_id'),
            'pump': ('master_data.centrifugal_pumps', 'pump_id')
        }

        table, id_col = tables[equipment_type]
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