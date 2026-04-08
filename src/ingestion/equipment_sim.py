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
        'bearing_overtemp': 'bearing',    # Bearing overtemperature = bearing issue
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
        'bearing_drive_end': 'bearing_de',          # Pump DE bearing
        'bearing_non_drive_end': 'bearing_nde',     # Pump NDE bearing
        'cavitation': 'impeller',                   # Cavitation damages impeller
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

    if hasattr(equipment, 'impeller_health') and failed_component == 'impeller':
        new_health = random.uniform(*repair_health_range)
        equipment.impeller_health = new_health
        repaired['impeller'] = new_health

    if hasattr(equipment, 'wear_ring_health') and failed_component == 'wear_ring':
        new_health = random.uniform(*repair_health_range)
        equipment.wear_ring_health = new_health
        repaired['wear_ring'] = new_health

    # Repair seal model if seal failed
    if 'seal' in failed_component:
        if hasattr(equipment, 'seal_model') and hasattr(equipment.seal_model, 'health'):
            seal_health = equipment.seal_model.health
            if isinstance(seal_health, dict):
                # Compressor-style seal model with primary/secondary seals
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
            else:
                # Pump-style seal model with single health value
                new_health = random.uniform(*repair_health_range)
                equipment.seal_model.health = new_health
                repaired['seal'] = new_health

    # Repair pump bearing model (pumps use bearing_model, not health_model)
    if failed_component in ('bearing', 'bearing_de', 'bearing_nde'):
        if hasattr(equipment, 'bearing_model') and hasattr(equipment.bearing_model, 'health'):
            bearing_health = equipment.bearing_model.health
            if failed_component == 'bearing_de' and 'drive_end' in bearing_health:
                bearing_health['drive_end'] = random.uniform(*repair_health_range)
                repaired['bearing_de'] = bearing_health['drive_end']
            elif failed_component == 'bearing_nde' and 'non_drive_end' in bearing_health:
                bearing_health['non_drive_end'] = random.uniform(*repair_health_range)
                repaired['bearing_nde'] = bearing_health['non_drive_end']
            elif failed_component == 'bearing':
                # Generic bearing failure (high_vibration, overtemp) - repair both
                for key in bearing_health:
                    bearing_health[key] = random.uniform(*repair_health_range)
                    repaired[key] = bearing_health[key]

    # Motor overload - repair the most degraded contributing component
    # (motor overload is caused by both bearing friction and impeller loss)
    if failed_component == 'motor_overload':
        worst_comp = None
        worst_health = 1.0
        if hasattr(equipment, 'impeller_health'):
            if equipment.impeller_health < worst_health:
                worst_comp = 'impeller'
                worst_health = equipment.impeller_health
        if hasattr(equipment, 'bearing_model') and hasattr(equipment.bearing_model, 'health'):
            for bearing, health in equipment.bearing_model.health.items():
                if health < worst_health:
                    worst_comp = bearing
                    worst_health = health
        if worst_comp == 'impeller' and hasattr(equipment, 'impeller_health'):
            equipment.impeller_health = random.uniform(*repair_health_range)
            repaired['impeller'] = equipment.impeller_health
        elif worst_comp and hasattr(equipment, 'bearing_model'):
            equipment.bearing_model.health[worst_comp] = random.uniform(*repair_health_range)
            repaired[worst_comp] = equipment.bearing_model.health[worst_comp]

    # Reset surge model state after surge failure (prevent re-trigger loop)
    if hasattr(equipment, 'surge_model') and 'surge' in failure_code.lower():
        equipment.surge_model.surge_active = False
        equipment.surge_model.surge_cycles = 0
        equipment.surge_model.surge_phase = 0.0
        equipment.surge_model._surge_evaluated = False

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
            if degradation_multiplier > 1.0:
                full_extra = int(degradation_multiplier - 1.0)
                fractional_extra = (degradation_multiplier - 1.0) - full_extra

                if hasattr(equipment, 'health_model'):
                    for _ in range(full_extra):
                        equipment.health_model.step(1.0)
                    if fractional_extra > 0 and random.random() < fractional_extra:
                        equipment.health_model.step(1.0)

                if hasattr(equipment, 'seal_model'):
                    for _ in range(full_extra):
                        equipment.seal_model.step(1.0)
                    if fractional_extra > 0 and random.random() < fractional_extra:
                        equipment.seal_model.step(1.0)

                if hasattr(equipment, 'bearing_model'):
                    speed = getattr(equipment, 'speed', 0)
                    for _ in range(full_extra):
                        equipment.bearing_model.step(speed)
                    if fractional_extra > 0 and random.random() < fractional_extra:
                        equipment.bearing_model.step(speed)

                if hasattr(equipment, 'impeller_degradation_rate'):
                    for _ in range(full_extra):
                        equipment.impeller_health -= equipment.impeller_degradation_rate
                    if fractional_extra > 0 and random.random() < fractional_extra:
                        equipment.impeller_health -= equipment.impeller_degradation_rate

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

            bearing_alarm = state.get('bearing_alarm') if state else None
            if bearing_alarm:
                telemetry_record['alarm'] = bearing_alarm

            yield telemetry_record

        except Exception as e:
            failure_code = str(e)

            yield {
                'type': 'failure',
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'failure_time': sample_time,
                'operating_hours_at_failure': equipment.operating_hours,
                'failure_mode_code': failure_code,
                'state': state if state else {}
            }

            repaired_components = _repair_equipment(equipment, equipment_type, failure_code)
            yield {
                'type': 'maintenance_start',
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'start_time': sample_time,
                'failure_code': failure_code,
                'repaired_components': repaired_components,
                'downtime_hours': maintenance_downtime_hours
            }
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