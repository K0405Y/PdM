"""
Maintenance Event Modeling System

Simulates maintenance interventions including routine service, component replacement,
and major overhauls. Models imperfect maintenance and post-maintenance infant mortality.

Key Features:
- Multiple maintenance types (routine, minor, major)
- Probabilistic health restoration
- Infant mortality after maintenance
- Maintenance scheduling (time-based, condition-based, opportunistic)

Reference: CMMS integration patterns, maintenance effectiveness studies
"""

import numpy as np
import random
from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta


class MaintenanceType(Enum):
    """Types of maintenance interventions."""
    ROUTINE = "routine"              # Oil/filter change
    MINOR_OVERHAUL = "minor_overhaul"  # Seal/bearing replacement
    MAJOR_OVERHAUL = "major_overhaul"  # Full refurbishment
    EMERGENCY = "emergency"            # Unplanned repair


@dataclass
class MaintenanceAction:
    """Details of a maintenance intervention."""
    maintenance_type: MaintenanceType
    timestamp: datetime
    components_affected: List[str]
    health_before: Dict[str, float]
    health_after: Dict[str, float]
    cost: float
    duration_hours: float
    quality_factor: float  # 0.0 to 1.0 (workmanship quality)


class MaintenanceScheduler:
    """
    Manages maintenance scheduling based on multiple triggers.
    """

    def __init__(self,
                 enable_time_based: bool = True,
                 enable_condition_based: bool = True,
                 enable_opportunistic: bool = True):
        """
        Initialize maintenance scheduler.

        Args:
            enable_time_based: Enable scheduled maintenance
            enable_condition_based: Enable condition-triggered maintenance
            enable_opportunistic: Enable opportunistic maintenance
        """
        self.enable_time_based = enable_time_based
        self.enable_condition_based = enable_condition_based
        self.enable_opportunistic = enable_opportunistic

        # Scheduling parameters (hours)
        self.routine_interval = 2000      # Every 2000 operating hours
        self.minor_interval = 8000        # Every 8000 operating hours
        self.major_interval = 24000       # Every 24000 operating hours

        # Condition-based thresholds
        self.condition_thresholds = {
            'routine': 0.85,    # Trigger routine at health < 0.85
            'minor': 0.70,      # Trigger minor overhaul at health < 0.70
            'major': 0.55       # Trigger major overhaul at health < 0.55
        }

        # Maintenance effectiveness (restoration amount)
        self.restoration_factors = {
            MaintenanceType.ROUTINE: 0.05,       # Small improvement
            MaintenanceType.MINOR_OVERHAUL: 0.25, # Moderate restoration
            MaintenanceType.MAJOR_OVERHAUL: 0.85, # Near-complete restoration
            MaintenanceType.EMERGENCY: 0.15      # Quick fix, limited restoration
        }

        # Tracking
        self.last_routine = 0.0
        self.last_minor = 0.0
        self.last_major = 0.0
        self.maintenance_history: List[MaintenanceAction] = []

    def check_maintenance_required(self,
                                   operating_hours: float,
                                   health_state: Dict[str, float],
                                   is_planned_shutdown: bool = False) -> Optional[MaintenanceType]:
        """
        Check if maintenance is required.

        Args:
            operating_hours: Total operating hours
            health_state: Current component health values
            is_planned_shutdown: True if planned shutdown (opportunistic window)

        Returns:
            MaintenanceType if maintenance needed, None otherwise
        """
        # Check condition-based triggers (highest priority)
        if self.enable_condition_based:
            min_health = min(health_state.values())

            if min_health < self.condition_thresholds['major']:
                return MaintenanceType.MAJOR_OVERHAUL
            elif min_health < self.condition_thresholds['minor']:
                return MaintenanceType.MINOR_OVERHAUL
            elif min_health < self.condition_thresholds['routine']:
                return MaintenanceType.ROUTINE

        # Check time-based triggers
        if self.enable_time_based:
            hours_since_major = operating_hours - self.last_major
            hours_since_minor = operating_hours - self.last_minor
            hours_since_routine = operating_hours - self.last_routine

            if hours_since_major >= self.major_interval:
                return MaintenanceType.MAJOR_OVERHAUL
            elif hours_since_minor >= self.minor_interval:
                return MaintenanceType.MINOR_OVERHAUL
            elif hours_since_routine >= self.routine_interval:
                return MaintenanceType.ROUTINE

        # Check opportunistic maintenance
        if self.enable_opportunistic and is_planned_shutdown:
            # Use shutdown window for overdue routine maintenance
            hours_since_routine = operating_hours - self.last_routine
            if hours_since_routine > self.routine_interval * 0.8:  # Within 80% of interval
                return MaintenanceType.ROUTINE

        return None

    def perform_maintenance(self,
                           maintenance_type: MaintenanceType,
                           current_health: Dict[str, float],
                           operating_hours: float,
                           timestamp: datetime) -> MaintenanceAction:
        """
        Execute maintenance and update component health.

        Args:
            maintenance_type: Type of maintenance to perform
            current_health: Current health state
            operating_hours: Operating hours at maintenance
            timestamp: Timestamp of maintenance event

        Returns:
            MaintenanceAction with details and updated health
        """
        # Determine components affected
        components = self._get_affected_components(maintenance_type, current_health)

        # Calculate quality factor (workmanship variability)
        quality_factor = np.random.normal(0.9, 0.1)  # Mean 0.9, std 0.1
        quality_factor = np.clip(quality_factor, 0.6, 1.0)

        # Calculate new health values
        new_health = {}
        for component, old_health in current_health.items():
            if component in components:
                # Apply restoration
                restoration = self.restoration_factors[maintenance_type]
                restoration *= quality_factor  # Quality affects restoration

                # New health = old + restoration + random variation
                improvement = restoration + np.random.normal(0, 0.02)
                new_health[component] = min(1.0, old_health + improvement)
            else:
                # Component not maintained
                new_health[component] = old_health

        # Calculate cost and duration
        cost, duration = self._calculate_cost_duration(maintenance_type)

        # Create maintenance action record
        action = MaintenanceAction(
            maintenance_type=maintenance_type,
            timestamp=timestamp,
            components_affected=components,
            health_before=current_health.copy(),
            health_after=new_health,
            cost=cost,
            duration_hours=duration,
            quality_factor=quality_factor
        )

        # Update last maintenance times
        if maintenance_type == MaintenanceType.ROUTINE:
            self.last_routine = operating_hours
        elif maintenance_type == MaintenanceType.MINOR_OVERHAUL:
            self.last_minor = operating_hours
            self.last_routine = operating_hours  # Minor includes routine
        elif maintenance_type == MaintenanceType.MAJOR_OVERHAUL:
            self.last_major = operating_hours
            self.last_minor = operating_hours
            self.last_routine = operating_hours

        # Add to history
        self.maintenance_history.append(action)

        return action

    def check_infant_mortality(self,
                              hours_since_maintenance: float,
                              quality_factor: float) -> bool:
        """
        Check for post-maintenance infant mortality.

        Improper reassembly, contamination, or wrong parts can cause
        early failures after maintenance.

        Args:
            hours_since_maintenance: Hours since last maintenance
            quality_factor: Quality of the maintenance work

        Returns:
            True if infant mortality failure occurs
        """
        # Infant mortality window: first 100 hours after maintenance
        if hours_since_maintenance > 100:
            return False

        # Failure probability inversely related to quality
        # Poor quality (0.6) → 2% failure rate
        # Good quality (1.0) → 0.2% failure rate
        base_rate = 0.02
        quality_multiplier = 2.0 - quality_factor  # 1.4 to 1.0
        failure_rate = base_rate * quality_multiplier

        # Time-dependent hazard (higher risk immediately after maintenance)
        time_factor = np.exp(-hours_since_maintenance / 20)  # Decays over ~20 hours

        # Probabilistic check
        effective_rate = failure_rate * time_factor
        return random.random() < (effective_rate / 100)  # Per-hour probability

    def _get_affected_components(self,
                                 maintenance_type: MaintenanceType,
                                 current_health: Dict) -> List[str]:
        """Determine which components are affected by maintenance type."""
        all_components = list(current_health.keys())

        if maintenance_type == MaintenanceType.ROUTINE:
            # Routine: filters, oil, minor adjustments (affects all slightly)
            return all_components

        elif maintenance_type == MaintenanceType.MINOR_OVERHAUL:
            # Minor: replace worst 1-2 components
            sorted_components = sorted(current_health.items(), key=lambda x: x[1])
            return [comp for comp, _ in sorted_components[:2]]

        elif maintenance_type == MaintenanceType.MAJOR_OVERHAUL:
            # Major: all components
            return all_components

        elif maintenance_type == MaintenanceType.EMERGENCY:
            # Emergency: only the worst component
            worst_component = min(current_health.items(), key=lambda x: x[1])[0]
            return [worst_component]

        return []

    def _calculate_cost_duration(self, maintenance_type: MaintenanceType) -> tuple:
        """
        Calculate maintenance cost and downtime.

        Returns:
            (cost_USD, duration_hours)
        """
        base_costs = {
            MaintenanceType.ROUTINE: (2000, 4),
            MaintenanceType.MINOR_OVERHAUL: (25000, 48),
            MaintenanceType.MAJOR_OVERHAUL: (150000, 240),
            MaintenanceType.EMERGENCY: (15000, 24)
        }

        base_cost, base_duration = base_costs[maintenance_type]

        # Add variability (±20%)
        cost = base_cost * np.random.uniform(0.8, 1.2)
        duration = base_duration * np.random.uniform(0.8, 1.2)

        return round(cost, 2), round(duration, 1)

    def get_maintenance_summary(self) -> Dict:
        """Get summary statistics of maintenance history."""
        if not self.maintenance_history:
            return {'total_events': 0, 'total_cost': 0, 'total_downtime': 0}

        total_cost = sum(m.cost for m in self.maintenance_history)
        total_downtime = sum(m.duration_hours for m in self.maintenance_history)

        by_type = {}
        for mtype in MaintenanceType:
            events = [m for m in self.maintenance_history if m.maintenance_type == mtype]
            by_type[mtype.value] = {
                'count': len(events),
                'cost': sum(m.cost for m in events),
                'downtime': sum(m.duration_hours for m in events)
            }

        return {
            'total_events': len(self.maintenance_history),
            'total_cost': round(total_cost, 2),
            'total_downtime': round(total_downtime, 1),
            'by_type': by_type
        }


if __name__ == '__main__':
    """Demonstration of maintenance scheduling."""
    print("Maintenance Event Modeling - Demonstration")
    print("=" * 60)

    scheduler = MaintenanceScheduler()

    # Simulate equipment degradation and maintenance
    health = {'bearing': 0.95, 'seal': 0.92, 'impeller': 0.90}
    operating_hours = 0
    timestamp = datetime.now()

    print("\n--- SIMULATION ---")
    print(f"Initial health: {health}")

    for i in range(100):  # 100 time steps (e.g., weeks)
        operating_hours += 200  # 200 hours per week

        # Gradual degradation
        for component in health:
            health[component] -= np.random.uniform(0.005, 0.015)

        # Check maintenance
        maintenance_needed = scheduler.check_maintenance_required(
            operating_hours, health
        )

        if maintenance_needed:
            action = scheduler.perform_maintenance(
                maintenance_needed, health, operating_hours, timestamp
            )

            health = action.health_after

            print(f"\n[{operating_hours:.0f} hrs] {maintenance_needed.value.upper()}")
            print(f"  Components: {action.components_affected}")
            print(f"  Health improvement: {action.health_before} → {action.health_after}")
            print(f"  Cost: ${action.cost:,.0f} | Downtime: {action.duration_hours:.1f} hrs")
            print(f"  Quality factor: {action.quality_factor:.2f}")

        timestamp += timedelta(days=7)

    print("\n--- MAINTENANCE SUMMARY ---")
    summary = scheduler.get_maintenance_summary()
    print(f"Total events: {summary['total_events']}")
    print(f"Total cost: ${summary['total_cost']:,.0f}")
    print(f"Total downtime: {summary['total_downtime']:.1f} hours")
    print("\nBy type:")
    for mtype, stats in summary['by_type'].items():
        if stats['count'] > 0:
            print(f"  {mtype}: {stats['count']} events, "
                  f"${stats['cost']:,.0f}, {stats['downtime']:.1f} hrs")

    print(f"\nFinal health: {health}")