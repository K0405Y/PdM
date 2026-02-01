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


def _get_failed_component(failure_code: str) -> str:
    """
    Parse failure code to identify the failed component.

    Failure codes follow pattern: F_COMPONENT or F_COMPONENT_SUBTYPE
    Examples: F_BEARING, F_FUEL, F_SEAL_PRIMARY, F_HIGH_VIBRATION

    Returns:
        Component name in lowercase (e.g., 'bearing', 'fuel', 'seal_primary')
    """
    # Remove F_ prefix and convert to lowercase
    if failure_code.startswith('F_'):
        component = failure_code[2:].lower()
    else:
        component = failure_code.lower()

    # Map special failure codes to components
    failure_to_component = {
        'high_vibration': 'bearing',      # Vibration issues typically mean bearing
        'bearing_temp': 'bearing',        # High bearing temp = bearing issue
        'vib_trip': 'bearing',            # Vibration trip = bearing
        'surge': 'impeller',              # Surge damages impeller
        'hgp': 'hgp',                     # Hot gas path
        'blade': 'blade',
        'bearing': 'bearing',
        'fuel': 'fuel',
        'impeller': 'impeller',
        'seal': 'seal',
        'seal_primary': 'seal_primary',
        'seal_secondary': 'seal_secondary',
        'bearing_de': 'bearing_de',
        'bearing_nde': 'bearing_nde',
    }

    return failure_to_component.get(component, component)


def _repair_equipment(equipment, equipment_type: str, failure_code: str):
    """
    Repair equipment after failure by restoring health of the failed component.

    Only the failed component is restored to high health (85-95%).
    Other components retain their current degraded health values.

    Args:
        equipment: Equipment simulator instance
        equipment_type: 'turbine', 'compressor', or 'pump'
        failure_code: The failure mode code that triggered repair

    Returns:
        dict: Health values after repair (showing what was repaired)
    """
    failed_component = _get_failed_component(failure_code)
    repair_health_range = (0.85, 0.92)

    repaired = {}

    # Repair main health model component
    if hasattr(equipment, 'health_model') and hasattr(equipment.health_model, 'health'):
        health_dict = equipment.health_model.health

        # Check if the failed component is in the health model
        if failed_component in health_dict:
            new_value = random.uniform(*repair_health_range)
            health_dict[failed_component] = new_value
            repaired[failed_component] = new_value
        else:
            # Try partial match (e.g., 'bearing' matches 'bearing_de')
            for comp in health_dict:
                if failed_component in comp or comp in failed_component:
                    new_value = random.uniform(*repair_health_range)
                    health_dict[comp] = new_value
                    repaired[comp] = new_value

        # Reinitialize degradation generators with updated health values
        if hasattr(equipment.health_model, '_init_generators'):
            equipment.health_model._init_generators()

    # Repair seal model for compressors if seal failed
    if 'seal' in failed_component:
        if hasattr(equipment, 'seal_model') and hasattr(equipment.seal_model, 'health'):
            seal_health = equipment.seal_model.health
            if 'primary' in failed_component and 'primary' in seal_health:
                seal_health['primary'] = random.uniform(*repair_health_range)
                repaired['seal_primary'] = seal_health['primary']
            elif 'secondary' in failed_component and 'secondary' in seal_health:
                seal_health['secondary'] = random.uniform(*repair_health_range)
                repaired['seal_secondary'] = seal_health['secondary']
            else:
                # Repair both seals if generic seal failure
                for seal_type in seal_health:
                    seal_health[seal_type] = random.uniform(*repair_health_range)
                    repaired[f'seal_{seal_type}'] = seal_health[seal_type]

    # Reset equipment to idle state after repair
    equipment.set_speed(0)

    return repaired


def simulate_equipment(equipment, equipment_id: int, equipment_type: str,
                      duration_days: int, sample_interval_min: int,
                      start_time: datetime = None,
                      degradation_multiplier: float = 1.0,
                      include_equipment_type: bool = False,
                      maintenance_downtime_hours: float = 24.0) -> Generator[Dict, None, None]:
    """
    Simulate a single piece of equipment with maintenance/repair after failures.

    Yields telemetry or failure records one at a time. When equipment fails,
    the failed component is repaired and the equipment continues operation.

    Args:
        equipment: Equipment simulator instance (GasTurbine, Compressor, Pump, etc.)
        equipment_id: Equipment ID
        equipment_type: 'turbine', 'compressor', or 'pump'
        duration_days: Simulation duration in days
        sample_interval_min: Sample interval in minutes
        start_time: Optional start time (default: now - duration_days)
        degradation_multiplier: Multiplier for degradation rate (>1.0 = faster degradation)
        include_equipment_type: If True, include equipment_type in telemetry records
        maintenance_downtime_hours: Hours of downtime after failure for repairs

    Yields:
        Dict with 'type' = 'telemetry', 'failure', or 'maintenance'
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(days=duration_days)

    num_samples = int(duration_days * 24 * 60 / sample_interval_min)
    maintenance_samples = int(maintenance_downtime_hours * 60 / sample_interval_min)

    # Duty cycle: 90% running, 10% idle
    duty_cycle = [random.random() < 0.9 for _ in range(num_samples)]

    # Track maintenance periods
    in_maintenance = False
    maintenance_remaining = 0
    state = {}

    for i in range(num_samples):
        sample_time = start_time + timedelta(minutes=i * sample_interval_min)

        # Handle maintenance period
        if in_maintenance:
            maintenance_remaining -= 1
            if maintenance_remaining <= 0:
                in_maintenance = False
                # Yield maintenance complete record
                yield {
                    'type': 'maintenance_complete',
                    'equipment_id': equipment_id,
                    'equipment_type': equipment_type,
                    'completion_time': sample_time,
                    'operating_hours': equipment.operating_hours
                }
            else:
                # Still in maintenance - yield idle telemetry
                equipment.set_speed(0)
                try:
                    state = equipment.next_state()
                except:
                    pass  # Ignore failures during maintenance

                telemetry_record = {
                    'type': 'telemetry',
                    'equipment_id': equipment_id,
                    'sample_time': sample_time,
                    'operating_hours': equipment.operating_hours,
                    'state': state,
                    'in_maintenance': True
                }
                if include_equipment_type:
                    telemetry_record['equipment_type'] = equipment_type
                yield telemetry_record
                continue

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
            failure_code = str(e)

            # Yield failure record
            yield {
                'type': 'failure',
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'failure_time': sample_time,
                'operating_hours_at_failure': equipment.operating_hours,
                'failure_mode_code': failure_code,
                'state': state if state else {}
            }

            # Repair the failed component and enter maintenance period
            repaired_components = _repair_equipment(equipment, equipment_type, failure_code)

            # Yield maintenance start record
            yield {
                'type': 'maintenance_start',
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'start_time': sample_time,
                'failure_code': failure_code,
                'repaired_components': repaired_components,
                'downtime_hours': maintenance_downtime_hours
            }

            # Enter maintenance period
            in_maintenance = True
            maintenance_remaining = maintenance_samples


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