"""
Data Pipeline Orchestrator

Orchestrates the complete data generation pipeline:
1. Database schema creation
2. Master data seeding
3. Equipment simulation
4. Bulk data insertion
5. Data verification

Uses modular components:
- db_setup: Database connection and master data management
- equipment_sim: Parallel equipment simulation
- bulk_insert: Fast PostgreSQL bulk insertion
"""

import os
import sys
import logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import new modular components
from src.ingestion.db_setup import Database, MasterData
from src.ingestion.equipment_sim import simulate_parallel, simulate_sequential
from src.ingestion.bulk_insert import bulk_insert_telemetry, insert_failures

# Import equipment simulators
from src.data_simulation.gas_turbine import GasTurbine
from src.data_simulation.centrifugal_compressor import CentrifugalCompressor
from src.data_simulation.centrifugal_pump import CentrifugalPump

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataPipeline:
    """
    Main pipeline orchestrator for PdM data generation.

    Coordinates all steps from schema creation to data verification.
    """

    def __init__(self, db_url: str, duration_days: int = 180, sample_interval_min: int = 10):
        """
        Initialize data pipeline.

        Args:
            db_url: PostgreSQL connection URL
            duration_days: Simulation duration (default: 180 days / 6 months)
            sample_interval_min: Sampling interval in minutes (default: 10)
        """
        self.db = Database(db_url)
        self.master_data = None
        self.duration_days = duration_days
        self.sample_interval_min = sample_interval_min

        logger.info(f"Pipeline initialized: {duration_days} days, {sample_interval_min} min intervals")

    def connect(self):
        """Connect to database."""
        self.db.connect()
        self.master_data = MasterData(self.db)
        logger.info("Connected to database")

    def create_schemas(self, schemas_dir: str = None):
        """
        Execute database schema creation scripts.

        Args:
            schemas_dir: Path to SQL schema files (default: project db schemas directory)
        """
        if schemas_dir is None:
            schemas_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'db schemas'
            )

        logger.info(f"Creating database schemas from: {schemas_dir}")

        # Execute schema scripts in order
        schema_files = sorted([f for f in os.listdir(schemas_dir) if f.endswith('.sql')])

        for schema_file in schema_files:
            # Skip verification queries (not executable)
            if 'verification' in schema_file.lower():
                logger.info(f"Skipped: {schema_file} (informational only)")
                continue

            script_path = os.path.join(schemas_dir, schema_file)
            self.db.execute_script(script_path)

        logger.info("Database schemas created successfully")

    def seed_master_data(self, turbine_count: int = 10, compressor_count: int = 10,
                        pump_count: int = 80) -> Tuple[List[int], List[int], List[int]]:
        """
        Seed equipment master data.

        Args:
            turbine_count: Number of gas turbines to create
            compressor_count: Number of centrifugal compressors to create
            pump_count: Number of centrifugal pumps to create

        Returns:
            (turbine_ids, compressor_ids, pump_ids)
        """
        logger.info(f"Seeding master data: {turbine_count} turbines, "
                   f"{compressor_count} compressors, {pump_count} pumps")

        turbine_ids = self.master_data.seed_turbines(turbine_count)
        compressor_ids = self.master_data.seed_compressors(compressor_count)
        pump_ids = self.master_data.seed_pumps(pump_count)

        logger.info(f"Master data seeded: {len(turbine_ids)} turbines, "
                   f"{len(compressor_ids)} compressors, {len(pump_ids)} pumps")

        return turbine_ids, compressor_ids, pump_ids

    def simulate_equipment(self, turbine_ids: List[int], compressor_ids: List[int],
                          pump_ids: List[int], use_parallel: bool = True) -> Tuple[Dict, Dict, Dict, List[Dict]]:
        """
        Simulate all equipment and generate telemetry data.

        Args:
            turbine_ids: List of turbine IDs to simulate
            compressor_ids: List of compressor IDs to simulate
            pump_ids: List of pump IDs to simulate
            use_parallel: Use parallel processing (default: True)

        Returns:
            (turbine_telemetry, compressor_telemetry, pump_telemetry, all_failures)
        """
        logger.info(f"Starting simulation: {len(turbine_ids)} turbines, "
                   f"{len(compressor_ids)} compressors, {len(pump_ids)} pumps")

        # Prepare simulation configurations
        configs = []

        # Turbines
        turbine_configs = self.master_data.get_configs(turbine_ids, 'turbine')
        for config in turbine_configs:
            equipment_config = {
                'name': config['name'],
                'initial_health': {
                    'hgp': config['initial_health_hgp'],
                    'blade': config['initial_health_blade'],
                    'bearing': config['initial_health_bearing'],
                    'fuel': config['initial_health_fuel']
                },
                'ambient_temp': config['ambient_temp_celsius'],
                'ambient_pressure': config['ambient_pressure_kpa']
            }
            params = {
                'equipment_type': 'turbine',
                'duration_days': self.duration_days,
                'sample_interval_min': self.sample_interval_min
            }
            configs.append((GasTurbine, config['turbine_id'], equipment_config, params))

        # Compressors
        compressor_configs = self.master_data.get_configs(compressor_ids, 'compressor')
        for config in compressor_configs:
            equipment_config = {
                'name': config['name'],
                'initial_health': {
                    'impeller': config['initial_health_impeller'],
                    'bearing': config['initial_health_bearing']
                },
                'design_flow': config['design_flow_m3h'],
                'design_head': config['design_head_kj_kg'],
                'suction_pressure': config['suction_pressure_kpa'],
                'suction_temp': config['suction_temp_celsius']
            }
            params = {
                'equipment_type': 'compressor',
                'duration_days': self.duration_days,
                'sample_interval_min': self.sample_interval_min
            }
            configs.append((CentrifugalCompressor, config['compressor_id'], equipment_config, params))

        # Pumps
        pump_configs = self.master_data.get_configs(pump_ids, 'pump')
        for config in pump_configs:
            equipment_config = {
                'name': config['name'],
                'initial_health': {
                    'impeller': config['initial_health_impeller'],
                    'seal': config['initial_health_seal'],
                    'bearing_de': config['initial_health_bearing_de'],
                    'bearing_nde': config['initial_health_bearing_nde']
                },
                'design_flow': config['design_flow_m3h'],
                'design_head': config['design_head_m'],
                'design_speed': config['design_speed_rpm'],
                'fluid_density': config['fluid_density_kg_m3'],
                'npsh_available': config['npsh_available_m']
            }
            params = {
                'equipment_type': 'pump',
                'duration_days': self.duration_days,
                'sample_interval_min': self.sample_interval_min
            }
            configs.append((CentrifugalPump, config['pump_id'], equipment_config, params))

        # Run simulation (parallel or sequential)
        if use_parallel:
            all_telemetry, all_failures = simulate_parallel(configs)
        else:
            all_telemetry, all_failures = simulate_sequential(configs)

        # Separate telemetry by equipment type
        turbine_telemetry = [r for r in all_telemetry if r['equipment_id'] in turbine_ids]
        compressor_telemetry = [r for r in all_telemetry if r['equipment_id'] in compressor_ids]
        pump_telemetry = [r for r in all_telemetry if r['equipment_id'] in pump_ids]

        logger.info(f"Simulation complete: {len(turbine_telemetry)} turbine records, "
                   f"{len(compressor_telemetry)} compressor records, "
                   f"{len(pump_telemetry)} pump records, {len(all_failures)} failures")

        return turbine_telemetry, compressor_telemetry, pump_telemetry, all_failures

    def ingest_data(self, turbine_telemetry: List[Dict], compressor_telemetry: List[Dict],
                   pump_telemetry: List[Dict], failures: List[Dict]):
        """
        Bulk insert telemetry and failure data into database.

        Args:
            turbine_telemetry: List of turbine telemetry records
            compressor_telemetry: List of compressor telemetry records
            pump_telemetry: List of pump telemetry records
            failures: List of failure event records
        """
        logger.info("Starting bulk data insertion")

        # Bulk insert telemetry
        turbine_count = bulk_insert_telemetry(self.db, turbine_telemetry, 'turbine')
        logger.info(f"Inserted {turbine_count} turbine telemetry records")

        compressor_count = bulk_insert_telemetry(self.db, compressor_telemetry, 'compressor')
        logger.info(f"Inserted {compressor_count} compressor telemetry records")

        pump_count = bulk_insert_telemetry(self.db, pump_telemetry, 'pump')
        logger.info(f"Inserted {pump_count} pump telemetry records")

        # Insert failures
        failure_count = insert_failures(self.db, failures)
        logger.info(f"Inserted {failure_count} failure events")

        logger.info(f"Data ingestion complete: {turbine_count + compressor_count + pump_count} "
                   f"telemetry records, {failure_count} failures")

    def verify_data(self):
        """Run data integrity verification queries."""
        logger.info("Running data integrity verification")

        session = self.db.get_session()

        try:
            from sqlalchemy import text

            # Count equipment
            turbine_count = session.execute(
                text("SELECT COUNT(*) FROM master_data.gas_turbines WHERE status = 'active'")
            ).scalar()

            compressor_count = session.execute(
                text("SELECT COUNT(*) FROM master_data.centrifugal_compressors WHERE status = 'active'")
            ).scalar()

            pump_count = session.execute(
                text("SELECT COUNT(*) FROM master_data.centrifugal_pumps WHERE status = 'active'")
            ).scalar()

            # Count telemetry
            gt_telemetry = session.execute(
                text("SELECT COUNT(*) FROM telemetry.gas_turbine_telemetry")
            ).scalar()

            cc_telemetry = session.execute(
                text("SELECT COUNT(*) FROM telemetry.centrifugal_compressor_telemetry")
            ).scalar()

            cp_telemetry = session.execute(
                text("SELECT COUNT(*) FROM telemetry.centrifugal_pump_telemetry")
            ).scalar()

            # Count failures
            gt_failures = session.execute(
                text("SELECT COUNT(*) FROM failure_events.gas_turbine_failures")
            ).scalar()

            cc_failures = session.execute(
                text("SELECT COUNT(*) FROM failure_events.centrifugal_compressor_failures")
            ).scalar()

            cp_failures = session.execute(
                text("SELECT COUNT(*) FROM failure_events.centrifugal_pump_failures")
            ).scalar()

            # Print summary
            logger.info("\n" + "="*60)
            logger.info("DATA INTEGRITY SUMMARY")
            logger.info("="*60)
            logger.info(f"Equipment:")
            logger.info(f"  - Turbines: {turbine_count}")
            logger.info(f"  - Compressors: {compressor_count}")
            logger.info(f"  - Pumps: {pump_count}")
            logger.info(f"  - Total: {turbine_count + compressor_count + pump_count}")
            logger.info(f"\nTelemetry Records:")
            logger.info(f"  - Turbine: {gt_telemetry:,}")
            logger.info(f"  - Compressor: {cc_telemetry:,}")
            logger.info(f"  - Pump: {cp_telemetry:,}")
            logger.info(f"  - Total: {gt_telemetry + cc_telemetry + cp_telemetry:,}")
            logger.info(f"\nFailure Events:")
            logger.info(f"  - Turbine: {gt_failures}")
            logger.info(f"  - Compressor: {cc_failures}")
            logger.info(f"  - Pump: {cp_failures}")
            logger.info(f"  - Total: {gt_failures + cc_failures + cp_failures}")
            logger.info("="*60)

        finally:
            session.close()

    def close(self):
        """Close database connection."""
        self.db.close()
        logger.info("Pipeline closed")


def main():
    """
    Main execution function.

    Runs the complete data generation pipeline:
    1. Connect to database
    2. Create schemas
    3. Seed master data
    4. Run equipment simulation
    5. Bulk insert data
    6. Verify data integrity
    """
    logger.info("PdM DATA GENERATION PIPELINE")
    
    # Get database URL from environment
    db_url = os.getenv('POSTGRES_URL')
    if not db_url:
        logger.error("POSTGRES_URL environment variable not set")
        sys.exit(1)

    logger.info(f"Database URL: {db_url.split('@')[1] if '@' in db_url else db_url}")

    # Initialize pipeline
    pipeline = DataPipeline(
        db_url=db_url,
        duration_days=180,  # 6 months
        sample_interval_min=10
    )

    try:
        # Step 1: Connect
        logger.info("\n--- STEP 1: Connecting to Database ---")
        pipeline.connect()

        # Step 2: Create schemas
        logger.info("\n--- STEP 2: Creating Database Schemas ---")
        pipeline.create_schemas()

        # Step 3: Seed master data
        logger.info("\n--- STEP 3: Seeding Master Data ---")
        turbine_ids, compressor_ids, pump_ids = pipeline.seed_master_data(
            turbine_count=10,
            compressor_count=10,
            pump_count=10
        )

        # Step 4: Run simulation
        logger.info("\n--- STEP 4: Running Equipment Simulation ---")
        turbine_tel, compressor_tel, pump_tel, failures = pipeline.simulate_equipment(
            turbine_ids, compressor_ids, pump_ids,
            use_parallel=True  # Use parallel processing for speed
        )

        # Step 5: Ingest data
        logger.info("\n--- STEP 5: Bulk Inserting Data ---")
        pipeline.ingest_data(turbine_tel, compressor_tel, pump_tel, failures)

        # Step 6: Verify data
        logger.info("\n--- STEP 6: Verifying Data Integrity ---")
        pipeline.verify_data()
        logger.info("PIPELINE EXECUTION COMPLETE")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        pipeline.close()


if __name__ == '__main__':
    main()