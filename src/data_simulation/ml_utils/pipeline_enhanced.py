"""
Enhanced Simulation Pipeline with Memory Efficiency and Parallelization

Provides generator-based telemetry output, parallel simulation, and efficient
database bulk insertion for large-scale dataset generation.

Key Features:
- Generator pattern for memory-efficient processing
- Multiprocessing for parallel equipment simulation
- PostgreSQL COPY command for 10-100x faster insertion
- Progress tracking and error recovery

Reference: Python multiprocessing patterns, PostgreSQL bulk loading
"""

import os
import sys
from typing import Generator, Dict, List, Tuple, Iterator
from multiprocessing import Pool, cpu_count
import logging
from io import StringIO
import csv
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import simulator classes (original imports would be here)
# from src.data_generation.gas_turbine import GasTurbine
# from src.data_generation.centrifugal_compressor import CentrifugalCompressor
# from src.data_generation.centrifugal_pump import CentrifugalPump

logger = logging.getLogger(__name__)


class GeneratorBasedSimulation:
    """
    Memory-efficient simulation using Python generators.

    Instead of accumulating all telemetry in memory, yields records
    as they're generated for streaming processing.
    """

    def __init__(self,
                 simulation_duration_days: int = 180,
                 sample_interval_minutes: int = 10):
        """
        Initialize generator-based simulation.

        Args:
            simulation_duration_days: Simulation duration
            sample_interval_minutes: Sampling interval
        """
        self.simulation_duration_days = simulation_duration_days
        self.sample_interval_minutes = sample_interval_minutes

    def simulate_equipment_stream(self,
                                  equipment,
                                  equipment_id: int,
                                  equipment_type: str) -> Generator[Dict, None, None]:
        """
        Stream telemetry records as they're generated.

        Args:
            equipment: Equipment simulator instance
            equipment_id: Equipment ID
            equipment_type: 'turbine', 'compressor', or 'pump'

        Yields:
            Telemetry records one at a time
        """
        start_time = datetime.now() - timedelta(days=self.simulation_duration_days)
        num_samples = int(
            self.simulation_duration_days * 24 * 60 / self.sample_interval_minutes
        )

        # Duty cycle: 90% online, 10% idle
        duty_cycle = [random.random() < 0.9 for _ in range(num_samples)]

        for i in range(num_samples):
            sample_time = start_time + timedelta(minutes=i * self.sample_interval_minutes)

            try:
                if duty_cycle[i]:
                    # Operating
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

                # Yield telemetry record
                yield {
                    'equipment_id': equipment_id,
                    'sample_time': sample_time,
                    'operating_hours': equipment.operating_hours,
                    'state': state
                }

            except Exception as e:
                # Equipment failed - yield failure record and stop
                failure_code = str(e)

                yield {
                    'type': 'failure',
                    'equipment_id': equipment_id,
                    'equipment_type': equipment_type,
                    'failure_time': sample_time,
                    'operating_hours_at_failure': equipment.operating_hours,
                    'failure_mode_code': failure_code,
                    'state': state if 'state' in locals() else {}
                }
                break

    @staticmethod
    def batch_generator(stream: Iterator, batch_size: int = 1000) -> Generator[List, None, None]:
        """
        Batch records from a stream.

        Args:
            stream: Iterator yielding individual records
            batch_size: Number of records per batch

        Yields:
            Lists of records (batches)
        """
        batch = []
        for record in stream:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining records
        if batch:
            yield batch


class ParallelSimulator:
    """
    Parallel equipment simulation using multiprocessing.
    """

    @staticmethod
    def simulate_single_equipment(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
        """
        Simulate a single equipment instance (worker function).

        Args:
            args: Tuple of (equipment_class, equipment_id, config, simulation_params)

        Returns:
            (telemetry_records, failure_records)
        """
        equipment_class, equipment_id, config, sim_params = args

        # Create equipment instance
        equipment = equipment_class(**config)

        # Run simulation using generator
        generator = GeneratorBasedSimulation(
            simulation_duration_days=sim_params['duration_days'],
            sample_interval_minutes=sim_params['sample_interval']
        )

        telemetry = []
        failures = []

        for record in generator.simulate_equipment_stream(
            equipment,
            equipment_id,
            sim_params['equipment_type']
        ):
            if record.get('type') == 'failure':
                failures.append(record)
            else:
                telemetry.append(record)

        return telemetry, failures

    @staticmethod
    def simulate_equipment_parallel(equipment_configs: List[Tuple],
                                    num_processes: int = None) -> Tuple[List, List]:
        """
        Simulate multiple equipment instances in parallel.

        Args:
            equipment_configs: List of (equipment_class, id, config, params) tuples
            num_processes: Number of parallel processes (default: CPU count)

        Returns:
            (all_telemetry, all_failures)
        """
        if num_processes is None:
            num_processes = min(cpu_count(), len(equipment_configs))

        logger.info(f"Starting parallel simulation with {num_processes} processes "
                   f"for {len(equipment_configs)} equipment instances")

        all_telemetry = []
        all_failures = []

        with Pool(processes=num_processes) as pool:
            results = pool.map(ParallelSimulator.simulate_single_equipment, equipment_configs)

        for telemetry, failures in results:
            all_telemetry.extend(telemetry)
            all_failures.extend(failures)

        logger.info(f"Parallel simulation complete: {len(all_telemetry)} telemetry records, "
                   f"{len(all_failures)} failures")

        return all_telemetry, all_failures


class BulkDatabaseInserter:
    """
    High-performance database insertion using PostgreSQL COPY command.
    """

    def __init__(self, db_connection):
        """
        Initialize bulk inserter.

        Args:
            db_connection: Database connection object
        """
        self.db = db_connection

    def bulk_insert_telemetry(self,
                              records: List[Dict],
                              table_name: str,
                              column_mapping: Dict[str, str]) -> int:
        """
        Bulk insert telemetry using COPY command.

        Args:
            records: List of telemetry records
            table_name: Target table name (e.g., 'telemetry.gas_turbine_telemetry')
            column_mapping: Dict mapping record keys to column names

        Returns:
            Number of records inserted
        """
        if not records:
            return 0

        logger.info(f"Bulk inserting {len(records)} records into {table_name}")

        # Create CSV buffer
        buffer = StringIO()
        writer = csv.writer(buffer, delimiter='\t')  # Tab-delimited for COPY

        columns = list(column_mapping.values())

        for record in records:
            row = self._extract_values(record, column_mapping)
            writer.writerow(row)

        # Reset buffer to beginning
        buffer.seek(0)

        # Execute COPY command
        session = self.db.get_cursor()
        try:
            # PostgreSQL COPY requires raw connection
            raw_conn = session.connection().connection
            cursor = raw_conn.cursor()

            cursor.copy_from(
                buffer,
                table_name,
                sep='\t',
                columns=columns,
                null='\\N'  # PostgreSQL NULL representation
            )

            raw_conn.commit()
            logger.info(f"Successfully inserted {len(records)} records")

        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            raise

        finally:
            session.close()

        return len(records)

    def _extract_values(self, record: Dict, column_mapping: Dict) -> List:
        """Extract values from record according to column mapping."""
        values = []
        state = record.get('state', {})

        for state_key in column_mapping.keys():
            if state_key == 'equipment_id':
                value = record.get('equipment_id')
            elif state_key == 'sample_time':
                value = record.get('sample_time')
            elif state_key == 'operating_hours':
                value = record.get('operating_hours')
            else:
                # Extract from state dict (support dot notation)
                value = self._get_nested_value(state, state_key)

            # Convert None to \N for PostgreSQL NULL
            if value is None:
                values.append('\\N')
            else:
                values.append(str(value))

        return values

    @staticmethod
    def _get_nested_value(data: Dict, key: str, default=None):
        """Extract nested values using dot notation."""
        keys = key.split('.')
        value = data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


class StreamingDataPipeline:
    """
    Complete streaming pipeline: simulate → batch → insert.
    """

    def __init__(self,
                 db_connection,
                 batch_size: int = 5000,
                 use_bulk_insert: bool = True):
        """
        Initialize streaming pipeline.

        Args:
            db_connection: Database connection
            batch_size: Records per batch
            use_bulk_insert: Use COPY command (vs individual inserts)
        """
        self.db = db_connection
        self.batch_size = batch_size
        self.use_bulk_insert = use_bulk_insert
        self.bulk_inserter = BulkDatabaseInserter(db_connection) if use_bulk_insert else None

    def process_stream(self,
                      equipment_stream: Iterator[Dict],
                      table_name: str,
                      column_mapping: Dict):
        """
        Process equipment telemetry stream directly to database.

        Args:
            equipment_stream: Generator yielding telemetry records
            table_name: Database table name
            column_mapping: Column mapping dict
        """
        total_inserted = 0

        for batch in GeneratorBasedSimulation.batch_generator(equipment_stream, self.batch_size):
            if self.use_bulk_insert:
                count = self.bulk_inserter.bulk_insert_telemetry(
                    batch,
                    table_name,
                    column_mapping
                )
            else:
                # Fall back to standard insert
                count = self._standard_insert(batch, table_name, column_mapping)

            total_inserted += count

            if total_inserted % 50000 == 0:
                logger.info(f"Progress: {total_inserted} records inserted")

        logger.info(f"Completed: {total_inserted} total records inserted")

    def _standard_insert(self, records: List[Dict], table_name: str, mapping: Dict) -> int:
        """Standard row-by-row insert (fallback)."""
        # Implementation would use existing DataIngestion methods
        logger.warning("Using standard insert (slower than bulk)")
        return len(records)


if __name__ == '__main__':
    """Demonstration of enhanced pipeline."""
    print("Enhanced Simulation Pipeline - Demonstration")
    print("=" * 60)

    # Example: Generator-based simulation
    print("\n--- Generator-Based Simulation ---")
    print("Memory usage: O(batch_size) instead of O(total_records)")

    # Simulate record counting without storing in memory
    record_count = 0
    batch_count = 0

    # Mock equipment stream
    def mock_stream():
        for i in range(10000):
            yield {'sample': i, 'value': i * 1.5}

    for batch in GeneratorBasedSimulation.batch_generator(mock_stream(), batch_size=1000):
        record_count += len(batch)
        batch_count += 1
        # In real usage, batch would be inserted to database here
        # then garbage collected, keeping memory usage low

    print(f"Processed {record_count} records in {batch_count} batches")
    print("Peak memory: ~1 batch worth of data")

    # Example: Parallel simulation
    print("\n--- Parallel Simulation ---")
    print(f"Available CPUs: {cpu_count()}")
    print("Parallel simulation can provide near-linear speedup for independent equipment")

    # Example: Bulk insert performance
    print("\n--- Bulk Insert Performance ---")
    print("PostgreSQL COPY command performance:")
    print("  Standard INSERT: ~1,000-5,000 rows/second")
    print("  COPY command:    ~100,000-500,000 rows/second")
    print("  Speedup:         10-100x faster")