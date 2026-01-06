import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_generation.gas_turbine import GasTurbine
from src.data_generation.centrifugal_compressor import CentrifugalCompressor
from src.data_generation.centrifugal_pump import CentrifugalPump

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PdMDatabase:
    """Manages database connections and operations using SQLAlchemy."""
    
    def __init__(self, db_url: str):
        """Initialize database connection."""
        self.db_url = self._format_db_url(db_url)
        self.engine = None
        self.Session = None
        self.conn = None
    
    def _format_db_url(self, db_url: str) -> str:
        """Format database URL for SQLAlchemy compatibility."""
        # Convert postgresql:// to postgresql+psycopg2://
        if db_url.startswith('postgresql://'):
            db_url = db_url.replace('postgresql://', 'postgresql+psycopg2://', 1)
        
        # Handle options parameter if present
        if '?options=' in db_url:
            base_url, options_str = db_url.split('?options=', 1)
            # URL-decode the options
            import urllib.parse
            options_value = urllib.parse.unquote(options_str)
            # Add options as connect_args for SQLAlchemy
            db_url = base_url
            self.options = options_value
        else:
            self.options = None
        
        return db_url
        
    def connect(self):
        """Establish database connection."""
        try:
            connect_args = {}
            if self.options:
                connect_args['options'] = self.options
            
            self.engine = create_engine(
                self.db_url,
                connect_args=connect_args,
                echo=False
            )
            self.Session = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as test_conn:
                test_conn.execute(text("SELECT 1"))
            
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def execute_script(self, script_path: str):
        """Execute SQL script."""
        try:
            with open(script_path, 'r') as f:
                sql = f.read()
            
            # Split by semicolons and execute each statement
            statements = sql.split(';')
            
            with self.engine.connect() as conn:
                for statement in statements:
                    # Strip whitespace and remove comment-only lines
                    clean_statement = '\n'.join(
                        line for line in statement.split('\n')
                        if line.strip() and not line.strip().startswith('--')
                    ).strip()
                    
                    if clean_statement:
                        conn.execute(text(clean_statement))
                conn.commit()
            
            logger.info(f"Executed script: {script_path}")
        except Exception as e:
            logger.error(f"Failed to execute script {script_path}: {e}")
            raise
    
    def get_cursor(self):
        """Get database session (SQLAlchemy equivalent of cursor)."""
        return self.Session()
    
    def commit(self):
        """Commit transaction."""
        # Handled at session level
        pass
    
    def rollback(self):
        """Rollback transaction."""
        # Handled at session level
        pass


class MasterDataSeeder:
    """Seeds master data (equipment registry)."""
    
    def __init__(self, db: PdMDatabase):
        self.db = db
    
    def seed_gas_turbines(self, count: int = 10) -> List[int]:
        """
        Seed gas turbines with randomized initial health.
        
        Args:
            count: Number of turbines to create
            
        Returns:
            List of turbine IDs
        """
        turbine_ids = []
        session = self.db.get_cursor()
        
        try:
            for i in range(count):
                name = f"GT-{1001 + i}"
                serial_number = f"SN-GT-{10001 + i}"
                
                # Randomize initial health (0.70-0.98)
                initial_health = {
                    'hgp': random.uniform(0.70, 0.98),
                    'blade': random.uniform(0.70, 0.98),
                    'bearing': random.uniform(0.70, 0.98),
                    'fuel': random.uniform(0.70, 0.98)
                }
                
                result = session.execute(text("""
                    INSERT INTO master_data.gas_turbines 
                    (name, serial_number, location, installed_date,
                     initial_health_hgp, initial_health_blade,
                     initial_health_bearing, initial_health_fuel)
                    VALUES (:name, :serial_number, :location, :installed_date,
                            :hgp, :blade, :bearing, :fuel)
                    RETURNING turbine_id
                """), {
                    'name': name,
                    'serial_number': serial_number,
                    'location': f"Platform-{i % 3 + 1}",
                    'installed_date': datetime.now().date() - timedelta(days=random.randint(30, 365)),
                    'hgp': initial_health['hgp'],
                    'blade': initial_health['blade'],
                    'bearing': initial_health['bearing'],
                    'fuel': initial_health['fuel']
                })
                
                turbine_id = result.scalar()
                turbine_ids.append(turbine_id)
                logger.info(f"Seeded turbine {name} (ID: {turbine_id}) with health: {initial_health}")
            
            session.commit()
        finally:
            session.close()
        
        return turbine_ids
    
    def seed_centrifugal_compressors(self, count: int = 10) -> List[int]:
        """
        Seed centrifugal compressors with randomized initial health.
        
        Args:
            count: Number of compressors to create
            
        Returns:
            List of compressor IDs
        """
        compressor_ids = []
        session = self.db.get_cursor()
        
        try:
            for i in range(count):
                name = f"CC-{2001 + i}"
                serial_number = f"SN-CC-{20001 + i}"
                
                # Randomize initial health
                initial_health = {
                    'impeller': random.uniform(0.70, 0.98),
                    'bearing': random.uniform(0.70, 0.98)
                }
                
                result = session.execute(text("""
                    INSERT INTO master_data.centrifugal_compressors
                    (name, serial_number, location, installed_date,
                     design_flow_m3h, design_head_kj_kg,
                     initial_health_impeller, initial_health_bearing)
                    VALUES (:name, :serial_number, :location, :installed_date,
                            :flow, :head, :impeller, :bearing)
                    RETURNING compressor_id
                """), {
                    'name': name,
                    'serial_number': serial_number,
                    'location': f"Facility-{i % 4 + 1}",
                    'installed_date': datetime.now().date() - timedelta(days=random.randint(30, 365)),
                    'flow': random.uniform(1200, 1800),
                    'head': random.uniform(7500, 8500),
                    'impeller': initial_health['impeller'],
                    'bearing': initial_health['bearing']
                })
                
                compressor_id = result.scalar()
                compressor_ids.append(compressor_id)
                logger.info(f"Seeded compressor {name} (ID: {compressor_id}) with health: {initial_health}")
            
            session.commit()
        finally:
            session.close()
        
        return compressor_ids
    
    def seed_centrifugal_pumps(self, count: int = 80) -> List[int]:
        """
        Seed centrifugal pumps across different service types.
        
        Args:
            count: Number of pumps to create
            
        Returns:
            List of pump IDs
        """
        pump_ids = []
        
        service_types = [
            {'name': 'Crude Booster', 'design_flow': 200, 'design_head': 100, 'density': 850},
            {'name': 'Seawater Injection', 'design_flow': 300, 'design_head': 150, 'density': 1025},
            {'name': 'Process Water', 'design_flow': 100, 'design_head': 60, 'density': 1000},
            {'name': 'Methanol Pump', 'design_flow': 50, 'design_head': 80, 'density': 790},
            {'name': 'Fire Water', 'design_flow': 400, 'design_head': 120, 'density': 1000},
        ]
        
        session = self.db.get_cursor()
        
        try:
            for i in range(count):
                name = f"CP-{3001 + i}"
                serial_number = f"SN-CP-{30001 + i}"
                service = service_types[i % len(service_types)]
                
                # Randomize initial health
                initial_health = {
                    'impeller': random.uniform(0.70, 0.98),
                    'seal': random.uniform(0.70, 0.98),
                    'bearing_de': random.uniform(0.70, 0.98),
                    'bearing_nde': random.uniform(0.70, 0.98)
                }
                
                result = session.execute(text("""
                    INSERT INTO master_data.centrifugal_pumps
                    (name, serial_number, service_type, location, installed_date,
                     design_flow_m3h, design_head_m, design_speed_rpm,
                     fluid_density_kg_m3,
                     initial_health_impeller, initial_health_seal,
                     initial_health_bearing_de, initial_health_bearing_nde)
                    VALUES (:name, :serial_number, :service_type, :location, :installed_date,
                            :flow, :head, :speed, :density, :impeller, :seal, :bearing_de, :bearing_nde)
                    RETURNING pump_id
                """), {
                    'name': name,
                    'serial_number': serial_number,
                    'service_type': service['name'],
                    'location': f"Platform-{i % 5 + 1}",
                    'installed_date': datetime.now().date() - timedelta(days=random.randint(30, 365)),
                    'flow': service['design_flow'],
                    'head': service['design_head'],
                    'speed': 3000,
                    'density': service['density'],
                    'impeller': initial_health['impeller'],
                    'seal': initial_health['seal'],
                    'bearing_de': initial_health['bearing_de'],
                    'bearing_nde': initial_health['bearing_nde']
                })
                
                pump_id = result.scalar()
                pump_ids.append(pump_id)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Seeded {i + 1} pumps")
            
            session.commit()
        finally:
            session.close()
        
        logger.info(f"Seeded total {count} pumps")
        return pump_ids


class SimulationEngine:
    """Orchestrates equipment simulation."""
    
    def __init__(self, db: PdMDatabase):
        self.db = db
        self.simulation_duration_days = 180  # 6 months
        self.sample_interval_minutes = 10
    
    def simulate_turbines(self, turbine_ids: List[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate gas turbines.
        
        Returns:
            (telemetry_records, failure_records)
        """
        telemetry = []
        failures = []
        
        # Get turbine configs from database
        turbines_config = self._get_turbine_configs(turbine_ids)
        
        for config in turbines_config:
            logger.info(f"Simulating gas turbine {config['name']} (ID: {config['turbine_id']})")
            
            try:
                # Create turbine simulator
                turbine = GasTurbine(
                    name=config['name'],
                    initial_health={
                        'hgp': config['initial_health_hgp'],
                        'blade': config['initial_health_blade'],
                        'bearing': config['initial_health_bearing'],
                        'fuel': config['initial_health_fuel']
                    },
                    ambient_temp=config['ambient_temp_celsius'],
                    ambient_pressure=config['ambient_pressure_kpa']
                )
                
                # Simulate
                telem, fail = self._simulate_equipment(
                    turbine, config['turbine_id'], 'turbine'
                )
                telemetry.extend(telem)
                failures.extend(fail)
                
            except Exception as e:
                logger.error(f"Failed to simulate turbine {config['name']}: {e}")
        
        return telemetry, failures
    
    def simulate_compressors(self, compressor_ids: List[int]) -> Tuple[List[Dict], List[Dict]]:
        """Simulate centrifugal compressors."""
        telemetry = []
        failures = []
        
        compressors_config = self._get_compressor_configs(compressor_ids)
        
        for config in compressors_config:
            logger.info(f"Simulating centrifugal compressor {config['name']} (ID: {config['compressor_id']})")
            
            try:
                compressor = CentrifugalCompressor(
                    name=config['name'],
                    initial_health={
                        'impeller': config['initial_health_impeller'],
                        'bearing': config['initial_health_bearing']
                    },
                    design_flow=config['design_flow_m3h'],
                    design_head=config['design_head_kj_kg'],
                    suction_pressure=config['suction_pressure_kpa'],
                    suction_temp=config['suction_temp_celsius']
                )
                
                telem, fail = self._simulate_equipment(
                    compressor, config['compressor_id'], 'compressor'
                )
                telemetry.extend(telem)
                failures.extend(fail)
                
            except Exception as e:
                logger.error(f"Failed to simulate compressor {config['name']}: {e}")
        
        return telemetry, failures
    
    def simulate_pumps(self, pump_ids: List[int]) -> Tuple[List[Dict], List[Dict]]:
        """Simulate centrifugal pumps."""
        telemetry = []
        failures = []
        
        pumps_config = self._get_pump_configs(pump_ids)
        
        for config in pumps_config:
            logger.info(f"Simulating centrifugal pump {config['name']} (ID: {config['pump_id']})")
            
            try:
                pump = CentrifugalPump(
                    name=config['name'],
                    initial_health={
                        'impeller': config['initial_health_impeller'],
                        'seal': config['initial_health_seal'],
                        'bearing_de': config['initial_health_bearing_de'],
                        'bearing_nde': config['initial_health_bearing_nde']
                    },
                    design_flow=config['design_flow_m3h'],
                    design_head=config['design_head_m'],
                    design_speed=config['design_speed_rpm'],
                    fluid_density=config['fluid_density_kg_m3'],
                    npsh_available=config['npsh_available_m']
                )
                
                telem, fail = self._simulate_equipment(
                    pump, config['pump_id'], 'pump'
                )
                telemetry.extend(telem)
                failures.extend(fail)
                
            except Exception as e:
                logger.error(f"Failed to simulate pump {config['name']}: {e}")
        
        return telemetry, failures
    
    def _simulate_equipment(self, equipment, equipment_id: int, 
                           equipment_type: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Generic equipment simulation.
        
        Generates 6 months of 10-minute interval data.
        """
        telemetry = []
        failures = []
        
        start_time = datetime.now() - timedelta(days=self.simulation_duration_days)
        num_samples = int(
            self.simulation_duration_days * 24 * 60 / self.sample_interval_minutes
        )
        
        # Typical duty cycle: 90% online, 10% idle
        duty_cycle = [random.random() < 0.9 for _ in range(num_samples)]
        
        for i in range(num_samples):
            sample_time = start_time + timedelta(minutes=i * self.sample_interval_minutes)
            
            try:
                if duty_cycle[i]:
                    # Operating - set speed to 70-100% of rated
                    rated_speed = getattr(equipment, 'LIMITS', {}).get('speed_rated', 9500)
                    target_speed = rated_speed * random.uniform(0.7, 1.0)
                    equipment.set_speed(target_speed)
                else:
                    # Idle
                    equipment.set_speed(0)
                
                # Advance simulation
                state = equipment.next_state()
                equipment.t += 1
                equipment.operating_hours += self.sample_interval_minutes / 60.0
                
                # Format telemetry record
                record = {
                    'equipment_id': equipment_id,
                    'sample_time': sample_time,
                    'operating_hours': equipment.operating_hours,
                    'state': state
                }
                telemetry.append(record)
                
            except Exception as e:
                # Equipment failed
                failure_code = str(e)
                logger.warning(
                    f"{equipment_type} {equipment_id} failed at {sample_time}: {failure_code}"
                )
                
                failures.append({
                    'equipment_id': equipment_id,
                    'equipment_type': equipment_type,
                    'failure_time': sample_time,
                    'operating_hours_at_failure': equipment.operating_hours,
                    'failure_mode_code': failure_code,
                    'state': state if 'state' in locals() else {}
                })
                
                # Stop simulating this equipment
                break
        
        logger.info(f"Generated {len(telemetry)} telemetry records for {equipment_type} {equipment_id}")
        return telemetry, failures
    
    def _get_turbine_configs(self, turbine_ids: List[int]) -> List[Dict]:
        """Fetch turbine configurations from database."""
        configs = []
        session = self.db.get_cursor()
        try:
            result = session.execute(text(
                "SELECT * FROM master_data.gas_turbines WHERE turbine_id = ANY(:ids)"
            ), {'ids': turbine_ids})
            columns = [col for col in result.keys()]
            for row in result.fetchall():
                configs.append(dict(zip(columns, row)))
        finally:
            session.close()
        return configs
    
    def _get_compressor_configs(self, compressor_ids: List[int]) -> List[Dict]:
        """Fetch compressor configurations from database."""
        configs = []
        session = self.db.get_cursor()
        try:
            result = session.execute(text(
                "SELECT * FROM master_data.centrifugal_compressors WHERE compressor_id = ANY(:ids)"
            ), {'ids': compressor_ids})
            columns = [col for col in result.keys()]
            for row in result.fetchall():
                configs.append(dict(zip(columns, row)))
        finally:
            session.close()
        return configs
    
    def _get_pump_configs(self, pump_ids: List[int]) -> List[Dict]:
        """Fetch pump configurations from database."""
        configs = []
        session = self.db.get_cursor()
        try:
            result = session.execute(text(
                "SELECT * FROM master_data.centrifugal_pumps WHERE pump_id = ANY(:ids)"
            ), {'ids': pump_ids})
            columns = [col for col in result.keys()]
            for row in result.fetchall():
                configs.append(dict(zip(columns, row)))
        finally:
            session.close()
        return configs