import os
import sys
from typing import Dict, List
import logging
from sqlalchemy import text

# Import numpy at module level for efficiency
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ingestion.data_pipeline import PdMDatabase, MasterDataSeeder, SimulationEngine 


class DataIngestion:
    """Handles batch insertion of telemetry and failure data."""
    
    # Column mappings: maps state dict keys to database column names
    TURBINE_TELEMETRY_MAPPING = {
        'equipment_id': 'turbine_id','sample_time': 'sample_time','operating_hours': 'operating_hours',
        'speed': 'speed_rpm','egt': 'egt_celsius','oil_temp': 'oil_temp_celsius',
        'fuel_flow': 'fuel_flow_kg_s', 'compressor_discharge_temp': 'compressor_discharge_temp_celsius','compressor_discharge_pressure': 'compressor_discharge_pressure_kpa',
        'vib_rms': 'vibration_rms_mm_s','vib_peak': 'vibration_peak_mm_s', 'efficiency': 'efficiency_fraction',
        'health.hgp': 'health_hgp','health.blade': 'health_blade','health.bearing': 'health_bearing','health.fuel': 'health_fuel',
    }
    
    COMPRESSOR_TELEMETRY_MAPPING = {'equipment_id': 'compressor_id', 'sample_time': 'sample_time', 'operating_hours': 'operating_hours', 'speed': 'speed_rpm',
        'flow': 'flow_m3h','head': 'head_kj_kg', 'discharge_pressure': 'discharge_pressure_kpa','discharge_temp': 'discharge_temp_celsius',
        'surge_margin': 'surge_margin_percent','vibration_amplitude': 'vibration_amplitude_mm','average_gap': 'average_gap_mm',
        'sync_amplitude': 'sync_amplitude_mm','bearing_temp_de': 'bearing_temp_de_celsius','bearing_temp_nde': 'bearing_temp_nde_celsius',
        'thrust_bearing_temp': 'thrust_bearing_temp_celsius','seal_health_primary': 'seal_health_primary','seal_health_secondary': 'seal_health_secondary',
        'seal_leakage': 'seal_leakage_rate','health.impeller': 'health_impeller','health.bearing': 'health_bearing',
    }
    
    PUMP_TELEMETRY_MAPPING = {
        'equipment_id': 'pump_id', 'sample_time': 'sample_time', 'operating_hours': 'operating_hours',
        'speed': 'speed_rpm', 'flow': 'flow_m3h', 'head': 'head_m', 'efficiency': 'efficiency_fraction',
        'power': 'power_kw', 'suction_pressure': 'suction_pressure_kpa',
        'discharge_pressure': 'discharge_pressure_kpa', 'fluid_temp': 'fluid_temp_celsius',
        'npsh_available': 'npsh_available_m', 'npsh_required': 'npsh_required_m',
        'cavitation_margin': 'cavitation_margin_m', 'cavitation_severity': 'cavitation_severity',
        'vibration': 'vibration_mm_s', 'bearing_temp_de': 'bearing_temp_de_celsius',
        'bearing_temp_nde': 'bearing_temp_nde_celsius', 'motor_current': 'motor_current_amps',
        'seal_health': 'seal_health', 'seal_leakage': 'seal_leakage_rate',
        'health.impeller': 'health_impeller', 'health.seal': 'health_seal',
        'health.bearing_de': 'health_bearing_de', 'health.bearing_nde': 'health_bearing_nde',
    }

    TURBINE_DEFAULTS = {
        'speed_rpm': 0, 'egt_celsius': 0, 'oil_temp_celsius': 0, 'fuel_flow_kg_s': 0,
        'compressor_discharge_temp_celsius': 0, 'compressor_discharge_pressure_kpa': 0,
        'vibration_rms_mm_s': 0, 'vibration_peak_mm_s': 0, 'efficiency_fraction': 1.0,
        'health_hgp': 0, 'health_blade': 0, 'health_bearing': 0, 'health_fuel': 0,
    }

    COMPRESSOR_DEFAULTS = {
        'speed_rpm': 0, 'flow_m3h': 0, 'head_kj_kg': 0, 'discharge_pressure_kpa': 0,
        'discharge_temp_celsius': 0, 'surge_margin_percent': 0,
        'vibration_amplitude_mm': 0, 'average_gap_mm': 0, 'sync_amplitude_mm': 0,
        'bearing_temp_de_celsius': 45, 'bearing_temp_nde_celsius': 45,
        'thrust_bearing_temp_celsius': 50, 'seal_health_primary': 0,
        'seal_health_secondary': 0, 'seal_leakage_rate': 0,
        'health_impeller': 0, 'health_bearing': 0,
    }

    PUMP_DEFAULTS = {
        'speed_rpm': 0, 'flow_m3h': 0, 'head_m': 0, 'efficiency_fraction': 0, 'power_kw': 0,
        'suction_pressure_kpa': 200, 'discharge_pressure_kpa': 200, 'fluid_temp_celsius': 40,
        'npsh_available_m': 8, 'npsh_required_m': 3, 'cavitation_margin_m': 5,
        'cavitation_severity': 0, 'vibration_mm_s': 0,
        'bearing_temp_de_celsius': 50, 'bearing_temp_nde_celsius': 50,
        'motor_current_amps': 0, 'seal_health': 0, 'seal_leakage_rate': 0,
        'health_impeller': 0, 'health_seal': 0,
        'health_bearing_de': 0, 'health_bearing_nde': 0,
    }

    
    def __init__(self, db: PdMDatabase):
        self.db = db
        self.batch_size = 1000
    
    def _get_nested_value(self, data: Dict, key: str, default=None):
        """Extract nested values from dict using dot notation."""
        keys = key.split('.')
        value = data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def _convert_numpy_types(self, value):
        """Convert numpy types to native Python types and round floats to 2 decimal places."""
        if HAS_NUMPY and np is not None:
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return round(float(value), 2)
            elif isinstance(value, np.ndarray):
                return value.tolist()

        # Round regular Python floats to 2 decimal places
        if isinstance(value, float):
            return round(value, 2)
        return value    
        
    def _extract_values(self, record: Dict, column_mapping: Dict, defaults: Dict) -> List:
        """Extract values from record according to column mapping."""
        values = []
        for state_key, db_column in column_mapping.items():
            if state_key == 'equipment_id':
                value = record.get('equipment_id')
            elif state_key == 'sample_time':
                value = record.get('sample_time')
            elif state_key == 'operating_hours':
                value = record.get('operating_hours')
            else:
                # Try to get from state dict with dot notation support
                state = record.get('state', {})
                value = self._get_nested_value(state, state_key)
                if value is None:
                    value = defaults.get(db_column)
            
            # Convert numpy types to native Python types
            value = self._convert_numpy_types(value)
            values.append(value)
        
        return values
    
    def ingest_turbine_telemetry(self, records: List[Dict]):
        """Batch insert gas turbine telemetry using dynamic column mapping."""
        if not records:
            return
        
        logger.info(f"Ingesting {len(records)} gas turbine telemetry records...")
        
        session = self.db.get_cursor()
        try:
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                values = [
                    self._extract_values(r, self.TURBINE_TELEMETRY_MAPPING, self.TURBINE_DEFAULTS)
                    for r in batch
                ]
                
                # Build insert statement
                columns = list(self.TURBINE_TELEMETRY_MAPPING.values())
                columns_str = ', '.join(columns)
                placeholders = ', '.join([':val' + str(j) for j in range(len(columns))])
                sql_template = f"INSERT INTO telemetry.gas_turbine_telemetry ({columns_str}) VALUES ({placeholders})"
                
                for val_list in values:
                    params = {f'val{j}': val_list[j] for j in range(len(val_list))}
                    session.execute(text(sql_template), params)
            
            session.commit()
        finally:
            session.close()
        
        logger.info(f"Ingested {len(records)} turbine records")
    
    def ingest_compressor_telemetry(self, records: List[Dict]):
        """Batch insert centrifugal compressor telemetry using dynamic column mapping."""
        if not records:
            return
        
        logger.info(f"Ingesting {len(records)} compressor telemetry records...")
        
        session = self.db.get_cursor()
        try:
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                values = [
                    self._extract_values(r, self.COMPRESSOR_TELEMETRY_MAPPING, self.COMPRESSOR_DEFAULTS)
                    for r in batch
                ]
                
                columns = list(self.COMPRESSOR_TELEMETRY_MAPPING.values())
                columns_str = ', '.join(columns)
                placeholders = ', '.join([':val' + str(j) for j in range(len(columns))])
                sql_template = f"INSERT INTO telemetry.centrifugal_compressor_telemetry ({columns_str}) VALUES ({placeholders})"
                
                for val_list in values:
                    params = {f'val{j}': val_list[j] for j in range(len(val_list))}
                    session.execute(text(sql_template), params)
            
            session.commit()
        finally:
            session.close()
        
        logger.info(f"Ingested {len(records)} compressor records")
    
    def ingest_pump_telemetry(self, records: List[Dict]):
        """Batch insert centrifugal pump telemetry using dynamic column mapping."""
        if not records:
            return
        
        logger.info(f"Ingesting {len(records)} pump telemetry records...")
        
        session = self.db.get_cursor()
        try:
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                values = [
                    self._extract_values(r, self.PUMP_TELEMETRY_MAPPING, self.PUMP_DEFAULTS)
                    for r in batch
                ]
                
                columns = list(self.PUMP_TELEMETRY_MAPPING.values())
                columns_str = ', '.join(columns)
                placeholders = ', '.join([':val' + str(j) for j in range(len(columns))])
                sql_template = f"INSERT INTO telemetry.centrifugal_pump_telemetry ({columns_str}) VALUES ({placeholders})"
                
                for val_list in values:
                    params = {f'val{j}': val_list[j] for j in range(len(val_list))}
                    session.execute(text(sql_template), params)
            
            session.commit()
        finally:
            session.close()
        
        logger.info(f"Ingested {len(records)} pump records")
    
    def ingest_failures(self, failure_records: List[Dict]):
        """Batch insert failure events using dynamic column mapping."""
        if not failure_records:
            return
        
        logger.info(f"Ingesting {len(failure_records)} failure events...")
        
        # Define failure event column mappings per equipment type
        turbine_failure_mapping = {
            'equipment_id': 'turbine_id',
            'failure_time': 'failure_time',
            'operating_hours_at_failure': 'operating_hours_at_failure',
            'failure_mode_code': 'failure_mode_code',
            'speed': 'speed_rpm_at_failure',
            'egt': 'egt_celsius_at_failure',
            'vib_rms': 'vibration_mm_s_at_failure',
        }
        
        compressor_failure_mapping = {
            'equipment_id': 'compressor_id',
            'failure_time': 'failure_time',
            'operating_hours_at_failure': 'operating_hours_at_failure',
            'failure_mode_code': 'failure_mode_code',
            'speed': 'speed_rpm_at_failure',
            'surge_margin': 'surge_margin_at_failure',
            'vibration_amplitude': 'vibration_amplitude_at_failure',
        }
        
        pump_failure_mapping = {
            'equipment_id': 'pump_id',
            'failure_time': 'failure_time',
            'operating_hours_at_failure': 'operating_hours_at_failure',
            'failure_mode_code': 'failure_mode_code',
            'speed': 'speed_rpm_at_failure',
            'vibration': 'vibration_mm_s_at_failure',
            'cavitation_margin': 'cavitation_margin_at_failure',
        }
        
        session = self.db.get_cursor()
        try:
            for record in failure_records:
                equipment_type = record['equipment_type']
                
                if equipment_type == 'turbine':
                    mapping = turbine_failure_mapping
                    table = 'failure_events.gas_turbine_failures'
                    values = self._extract_failure_values(record, mapping)
                elif equipment_type == 'compressor':
                    mapping = compressor_failure_mapping
                    table = 'failure_events.centrifugal_compressor_failures'
                    values = self._extract_failure_values(record, mapping)
                elif equipment_type == 'pump':
                    mapping = pump_failure_mapping
                    table = 'failure_events.centrifugal_pump_failures'
                    values = self._extract_failure_values(record, mapping)
                else:
                    logger.warning(f"Unknown equipment type: {equipment_type}")
                    continue
                
                columns = list(mapping.values())
                columns_str = ', '.join(columns)
                placeholders = ', '.join([':val' + str(j) for j in range(len(columns))])
                sql_template = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
                
                params = {f'val{j}': values[j] for j in range(len(values))}
                session.execute(text(sql_template), params)
            
            session.commit()
        finally:
            session.close()
        
        logger.info(f"Ingested {len(failure_records)} failure records")
    
    def _extract_failure_values(self, record: Dict, column_mapping: Dict) -> List:
        """Extract failure event values from record according to column mapping."""
        values = []
        state = record.get('state', {})
        
        for state_key, db_column in column_mapping.items():
            if state_key == 'equipment_id':
                value = record.get('equipment_id')
            elif state_key == 'failure_time':
                value = record.get('failure_time')
            elif state_key == 'operating_hours_at_failure':
                value = record.get('operating_hours_at_failure')
            elif state_key == 'failure_mode_code':
                value = record.get('failure_mode_code')
            else:
                # Extract from state dict with dot notation support
                value = self._get_nested_value(state, state_key)
            
            # Convert numpy types to native Python types
            value = self._convert_numpy_types(value)
            values.append(value)
        
        return values
    

def verify_data_integrity(db: PdMDatabase):
    """Run verification queries on ingested data."""
    logger.info("Running data integrity verification...")
    
    session = db.get_cursor()
    try:
        # Count records per equipment type
        result = session.execute(text("SELECT COUNT(*) FROM master_data.gas_turbines WHERE status = 'active'"))
        turbine_count = result.scalar()
        logger.info(f"Active turbines: {turbine_count}")
        
        result = session.execute(text("SELECT COUNT(*) FROM master_data.centrifugal_compressors WHERE status = 'active'"))
        compressor_count = result.scalar()
        logger.info(f"Active compressors: {compressor_count}")
        
        result = session.execute(text("SELECT COUNT(*) FROM master_data.centrifugal_pumps WHERE status = 'active'"))
        pump_count = result.scalar()
        logger.info(f"Active pumps: {pump_count}")
        
        # Count telemetry records
        result = session.execute(text("SELECT COUNT(*) FROM telemetry.gas_turbine_telemetry"))
        gt_telemetry = result.scalar()
        logger.info(f"Gas turbine telemetry records: {gt_telemetry}")
        
        result = session.execute(text("SELECT COUNT(*) FROM telemetry.centrifugal_compressor_telemetry"))
        cc_telemetry = result.scalar()
        logger.info(f"Compressor telemetry records: {cc_telemetry}")
        
        result = session.execute(text("SELECT COUNT(*) FROM telemetry.centrifugal_pump_telemetry"))
        cp_telemetry = result.scalar()
        logger.info(f"Pump telemetry records: {cp_telemetry}")
        
        # Count failures
        result = session.execute(text("SELECT COUNT(*) FROM failure_events.gas_turbine_failures"))
        gt_failures = result.scalar()
        logger.info(f"Gas turbine failures: {gt_failures}")
        
        result = session.execute(text("SELECT COUNT(*) FROM failure_events.centrifugal_compressor_failures"))
        cc_failures = result.scalar()
        logger.info(f"Compressor failures: {cc_failures}")
        
        result = session.execute(text("SELECT COUNT(*) FROM failure_events.centrifugal_pump_failures"))
        cp_failures = result.scalar()
        logger.info(f"Pump failures: {cp_failures}")
        
        # Data integrity summary
        logger.info("\n" + "="*60)
        logger.info("DATA INTEGRITY SUMMARY")
        logger.info("="*60)
        logger.info(f"Total equipment: {turbine_count + compressor_count + pump_count}")
        logger.info(f"Total telemetry records: {gt_telemetry + cc_telemetry + cp_telemetry}")
        logger.info(f"Total failure events: {gt_failures + cc_failures + cp_failures}")
        logger.info("="*60)
    finally:
        session.close()


def main():
    """Main orchestration function."""
    logger.info("Starting PdM Data Generation and Ingestion System")
    
    # Build connection string
    db_url = os.getenv('POSTGRES_URL')

    print(f"Connecting to database at {db_url}")
    
    # Initialize database
    db = PdMDatabase(db_url)
    db.connect()
    
    try:
        # Step 1: Create database schemas
        logger.info("\n--- STEP 1: Creating database schemas ---")
        schemas_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'database', 'schemas')
        for schema_file in sorted([f for f in os.listdir(schemas_dir) if f.endswith('.sql')]):
            if schema_file == '05_verification_queries.sql':
                # logger.info(f"Skipped script: {schema_file} (informational only)")
                continue
            db.execute_script(os.path.join(schemas_dir, schema_file))
        
        # Step 2: Seed master data
        logger.info("\n--- STEP 2: Seeding master data ---")
        seeder = MasterDataSeeder(db)
        turbine_ids = seeder.seed_gas_turbines(count=10)
        compressor_ids = seeder.seed_centrifugal_compressors(count=10)
        pump_ids = seeder.seed_centrifugal_pumps(count=80)
        
        # Step 3: Run simulation
        logger.info("\n--- STEP 3: Running simulation (6 months of data) ---")
        engine = SimulationEngine(db)
        
        turbine_telemetry, turbine_failures = engine.simulate_turbines(turbine_ids)
        compressor_telemetry, compressor_failures = engine.simulate_compressors(compressor_ids)
        pump_telemetry, pump_failures = engine.simulate_pumps(pump_ids)
        
        # Step 4: Ingest data
        logger.info("\n--- STEP 4: Ingesting data into PostgreSQL ---")
        ingestion = DataIngestion(db)
        ingestion.ingest_turbine_telemetry(turbine_telemetry)
        ingestion.ingest_compressor_telemetry(compressor_telemetry)
        ingestion.ingest_pump_telemetry(pump_telemetry)
        ingestion.ingest_failures(
            turbine_failures + compressor_failures + pump_failures
        )
        
        # # Step 5: Verify data integrity
        # logger.info("\n--- STEP 5: Verifying data integrity ---")
        # verify_data_integrity(db)
        
        # logger.info("\nPdM Data Generation and Ingestion Complete!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()
if __name__ == '__main__':
    main()