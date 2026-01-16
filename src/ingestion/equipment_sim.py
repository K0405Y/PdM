"""
Equipment Simulation

Simple simulation of equipment:
- Stream telemetry records (memory efficient)
- Parallel processing (multi-core)
- Sequential fallback
"""

import random
import logging
from typing import Dict, List, Tuple, Generator
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)


def simulate_equipment(equipment, equipment_id: int, equipment_type: str,
                      duration_days: int, sample_interval_min: int,
                      start_time: datetime = None,
                      degradation_multiplier: float = 1.0,
                      include_equipment_type: bool = False) -> Generator[Dict, None, None]:
    """
    Simulate a single piece of equipment.

    Yields telemetry or failure records one at a time.

    Args:
        equipment: Equipment simulator instance (GasTurbine, CentrifugalCompressor, etc.)
        equipment_id: Equipment ID
        equipment_type: 'turbine', 'compressor', or 'pump'
        duration_days: Simulation duration in days
        sample_interval_min: Sample interval in minutes
        start_time: Optional start time (default: now - duration_days)
        degradation_multiplier: Multiplier for degradation rate (>1.0 = faster degradation)
        include_equipment_type: If True, include equipment_type in telemetry records

    Yields:
        Dict with 'type' = 'telemetry' or 'failure'
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(days=duration_days)

    num_samples = int(duration_days * 24 * 60 / sample_interval_min)

    # Duty cycle: 90% running, 10% idle
    duty_cycle = [random.random() < 0.9 for _ in range(num_samples)]

    for i in range(num_samples):
        sample_time = start_time + timedelta(minutes=i * sample_interval_min)

        try:
            # Set operating state
            if duty_cycle[i]:
                # Running - 70-100% of rated speed
                rated_speed = getattr(equipment, 'LIMITS', {}).get('speed_rated', 9500)
                target_speed = rated_speed * random.uniform(0.7, 1.0)
                equipment.set_speed(target_speed)
            else:
                # Idle
                equipment.set_speed(0)

            # Advance simulation
            state = equipment.next_state()
            equipment.t += 1
            equipment.operating_hours += sample_interval_min / 60.0

            # Update timestamps for weather API integration
            if hasattr(equipment, 'elapsed_hours'):
                equipment.elapsed_hours += sample_interval_min / 60.0
            if hasattr(equipment, 'current_timestamp'):
                equipment.current_timestamp = sample_time

            # Apply additional degradation for increased failure rate
            if degradation_multiplier > 1.0 and hasattr(equipment, 'health_model'):
                for _ in range(int(degradation_multiplier) - 1):
                    if random.random() < (degradation_multiplier % 1.0):
                        equipment.health_model.step(1.0)

            # Yield telemetry record
            telemetry_record = {
                'type': 'telemetry',
                'equipment_id': equipment_id,
                'sample_time': sample_time,
                'operating_hours': equipment.operating_hours,
                'state': state
            }
            if include_equipment_type:
                telemetry_record['equipment_type'] = equipment_type

            yield telemetry_record

        except Exception as e:
            # Equipment failed - yield failure and stop
            yield {
                'type': 'failure',
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'failure_time': sample_time,
                'operating_hours_at_failure': equipment.operating_hours,
                'failure_mode_code': str(e),
                'state': state if 'state' in locals() else {}
            }
            break


def _worker_simulate(args: Tuple) -> Tuple[List[Dict], List[Dict]]:
    """
    Worker function for parallel simulation.

    Args:
        args: (equipment_class, equipment_id, config, params)

    Returns:
        (telemetry_list, failures_list)
    """
    equipment_class, equipment_id, config, params = args

    # Create equipment
    equipment = equipment_class(**config)

    # Simulate
    telemetry = []
    failures = []

    for record in simulate_equipment(
        equipment, equipment_id, params['equipment_type'],
        params['duration_days'], params['sample_interval_min']
    ):
        if record['type'] == 'failure':
            failures.append(record)
        else:
            telemetry.append(record)

    return telemetry, failures


def simulate_parallel(equipment_configs: List[Tuple], num_processes: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Simulate multiple equipment in parallel (uses all CPU cores).

    Args:
        equipment_configs: List of (equipment_class, id, config, params) tuples
        num_processes: Number of parallel processes (default: CPU count)

    Returns:
        (all_telemetry, all_failures)
    """
    if num_processes is None:
        num_processes = min(cpu_count(), len(equipment_configs))

    logger.info(f"Parallel simulation: {len(equipment_configs)} equipment on {num_processes} cores")

    all_telemetry = []
    all_failures = []

    with Pool(processes=num_processes) as pool:
        results = pool.map(_worker_simulate, equipment_configs)

    for telemetry, failures in results:
        all_telemetry.extend(telemetry)
        all_failures.extend(failures)

    logger.info(f"Complete: {len(all_telemetry)} telemetry, {len(all_failures)} failures")

    return all_telemetry, all_failures


def simulate_sequential(equipment_configs: List[Tuple]) -> Tuple[List[Dict], List[Dict]]:
    """
    Simulate equipment sequentially (fallback if parallel not needed).

    Args:
        equipment_configs: List of (equipment_class, id, config, params) tuples

    Returns:
        (all_telemetry, all_failures)
    """
    logger.info(f"Sequential simulation: {len(equipment_configs)} equipment")

    all_telemetry = []
    all_failures = []

    for config in equipment_configs:
        telemetry, failures = _worker_simulate(config)
        all_telemetry.extend(telemetry)
        all_failures.extend(failures)

    logger.info(f"Complete: {len(all_telemetry)} telemetry, {len(all_failures)} failures")

    return all_telemetry, all_failures