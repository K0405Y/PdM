"""
Incipient Fault Modeling with Discrete Initiation

Models fault lifecycle from discrete initiation through propagation to failure.
Provides more realistic precursor signatures for ML model development.

Fault Lifecycle:
1. Initiation: Discrete random event (crack nucleation, contamination ingress)
2. Propagation: Gradual growth phase
3. Acceleration: Positive feedback (wear debris causes more wear)
4. Failure: Functional loss

Reference: Failure physics, Paris law for crack growth, bearing wear mechanisms
"""

import numpy as np
import random
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


class FaultType(Enum):
    """Types of discrete fault initiations."""
    BEARING_SPALL = "bearing_spall"              # Surface spall on bearing race
    CRACK_FATIGUE = "crack_fatigue"              # Fatigue crack initiation
    SEAL_DAMAGE = "seal_damage"                  # Seal face damage
    CONTAMINATION = "contamination"              # Debris contamination
    CORROSION_PIT = "corrosion_pit"             # Localized corrosion
    BLADE_FOD = "blade_fod"                      # Foreign object damage (turbine/compressor)
    CAVITATION_EROSION = "cavitation_erosion"    # Pump cavitation damage


@dataclass
class FaultEvent:
    """Record of a discrete fault initiation event."""
    fault_type: FaultType
    initiation_time: datetime
    initiation_operating_hours: float
    affected_component: str
    severity: float  # Initial fault severity (0.0 to 1.0)
    location: str    # Physical location (e.g., "inner_race", "blade_leading_edge")


class FaultGrowthModel:
    """
    Models fault propagation from initiation to failure.
    """

    def __init__(self, fault_event: FaultEvent):
        """
        Initialize fault growth model.

        Args:
            fault_event: The initiating fault event
        """
        self.fault = fault_event
        self.current_size = fault_event.severity  # Normalized fault size (0-1)
        self.growth_rate = self._calculate_initial_growth_rate()
        self.operating_hours_since_init = 0.0

    def _calculate_initial_growth_rate(self) -> float:
        """Calculate initial fault growth rate based on fault type."""
        # Different faults grow at different rates
        base_rates = {
            FaultType.BEARING_SPALL: 0.001,        # Slow initial growth
            FaultType.CRACK_FATIGUE: 0.0005,       # Very slow until acceleration
            FaultType.SEAL_DAMAGE: 0.003,          # Faster growth
            FaultType.CONTAMINATION: 0.002,        # Moderate growth
            FaultType.CORROSION_PIT: 0.0008,       # Slow growth
            FaultType.BLADE_FOD: 0.0015,           # Moderate growth
            FaultType.CAVITATION_EROSION: 0.0025   # Faster growth
        }

        base_rate = base_rates.get(self.fault.fault_type, 0.001)

        # Scale by initial severity
        return base_rate * (0.5 + self.fault.severity)

    def propagate(self, operating_hours_increment: float, stress_factor: float = 1.0) -> float:
        """
        Advance fault growth.

        Args:
            operating_hours_increment: Operating time increment
            stress_factor: Operating stress multiplier (1.0 = normal)

        Returns:
            Current fault size (0-1)
        """
        self.operating_hours_since_init += operating_hours_increment

        # Growth models vary by fault type
        if self.fault.fault_type == FaultType.CRACK_FATIGUE:
            # Paris law: da/dN = C * (ΔK)^m
            # Simplified: exponential acceleration
            growth_increment = self._crack_growth(stress_factor, operating_hours_increment)

        elif self.fault.fault_type == FaultType.BEARING_SPALL:
            # Bearing spalls accelerate as debris damages more surface
            growth_increment = self._spall_growth(stress_factor, operating_hours_increment)

        else:
            # Generic linear-to-exponential growth
            growth_increment = self._generic_growth(stress_factor, operating_hours_increment)

        self.current_size = min(1.0, self.current_size + growth_increment)

        return self.current_size

    def _crack_growth(self, stress_factor: float, dt: float) -> float:
        """
        Fatigue crack growth (Paris law approximation).

        Slow growth initially, accelerates as crack lengthens.
        """
        # Effective stress intensity factor
        stress_intensity = stress_factor * (1.0 + self.current_size)

        # Paris law exponent (typically 2-4)
        m = 3.0

        growth = self.growth_rate * (stress_intensity ** m) * dt

        return growth

    def _spall_growth(self, stress_factor: float, dt: float) -> float:
        """
        Bearing spall propagation.

        Accelerates due to positive feedback: spall → debris → more spalls.
        """
        # Debris generation increases with spall size
        debris_factor = 1.0 + 2.0 * self.current_size

        growth = self.growth_rate * stress_factor * debris_factor * dt

        return growth

    def _generic_growth(self, stress_factor: float, dt: float) -> float:
        """Generic fault growth with acceleration phase."""
        # Quadratic acceleration
        acceleration = 1.0 + self.current_size ** 2

        growth = self.growth_rate * stress_factor * acceleration * dt

        return growth

    def calculate_health_impact(self, baseline_health: float) -> float:
        """
        Calculate component health reduction due to fault.

        Args:
            baseline_health: Baseline health from general degradation

        Returns:
            Adjusted health (reduced by fault impact)
        """
        # Fault size maps to health reduction
        # Small faults (< 0.2) have minimal impact
        # Large faults (> 0.8) cause severe degradation

        if self.current_size < 0.2:
            health_reduction = self.current_size * 0.05  # Max 1% reduction
        elif self.current_size < 0.5:
            health_reduction = 0.01 + (self.current_size - 0.2) * 0.15  # Up to 5.5%
        else:
            # Accelerating impact
            health_reduction = 0.055 + (self.current_size - 0.5) ** 2 * 0.4

        adjusted_health = baseline_health * (1.0 - health_reduction)

        return max(0.0, adjusted_health)


class IncipientFaultSimulator:
    """
    Manages discrete fault initiation events using Poisson process.
    """

    def __init__(self,
                 enable_incipient_faults: bool = True,
                 fault_rate_per_1000hrs: float = 0.5):
        """
        Initialize incipient fault simulator.

        Args:
            enable_incipient_faults: Enable discrete fault events
            fault_rate_per_1000hrs: Expected faults per 1000 operating hours
        """
        self.enable = enable_incipient_faults
        self.fault_rate = fault_rate_per_1000hrs / 1000.0  # Per hour
        self.active_faults: Dict[str, FaultGrowthModel] = {}  # component → fault model
        self.fault_history: list[FaultEvent] = []

    def check_fault_initiation(self,
                               operating_hours_increment: float,
                               stress_factor: float,
                               timestamp: datetime,
                               operating_hours: float,
                               component_list: list[str]) -> Optional[FaultEvent]:
        """
        Check for new fault initiation (Poisson process).

        Args:
            operating_hours_increment: Operating time increment
            stress_factor: Stress multiplier affecting initiation rate
            timestamp: Current timestamp
            operating_hours: Total operating hours
            component_list: List of components that can fail

        Returns:
            FaultEvent if initiated, None otherwise
        """
        if not self.enable:
            return None

        # Probability of fault in this time increment
        # P(event) = λ × dt × stress_factor
        effective_rate = self.fault_rate * stress_factor
        p_fault = 1 - np.exp(-effective_rate * operating_hours_increment)

        if random.random() < p_fault:
            # Fault initiated!
            fault_event = self._create_fault_event(
                timestamp, operating_hours, component_list
            )

            # Add to active faults
            if fault_event.affected_component not in self.active_faults:
                growth_model = FaultGrowthModel(fault_event)
                self.active_faults[fault_event.affected_component] = growth_model
                self.fault_history.append(fault_event)

                return fault_event

        return None

    def propagate_faults(self,
                        operating_hours_increment: float,
                        stress_factor: float) -> Dict[str, float]:
        """
        Propagate all active faults.

        Args:
            operating_hours_increment: Operating time increment
            stress_factor: Operating stress multiplier

        Returns:
            Dict of component → current fault size
        """
        fault_sizes = {}

        for component, fault_model in self.active_faults.items():
            size = fault_model.propagate(operating_hours_increment, stress_factor)
            fault_sizes[component] = size

        return fault_sizes

    def adjust_health_for_faults(self, baseline_health: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust component health values for active faults.

        Args:
            baseline_health: Health from general degradation

        Returns:
            Adjusted health with fault impacts
        """
        adjusted = baseline_health.copy()

        for component, fault_model in self.active_faults.items():
            if component in adjusted:
                adjusted[component] = fault_model.calculate_health_impact(
                    baseline_health[component]
                )

        return adjusted

    def _create_fault_event(self,
                           timestamp: datetime,
                           operating_hours: float,
                           components: list[str]) -> FaultEvent:
        """Create a new fault initiation event."""
        # Randomly select fault type
        fault_type = random.choice(list(FaultType))

        # Randomly select affected component
        affected_component = random.choice(components)

        # Initial severity (most faults start small)
        severity = np.random.beta(2, 5)  # Skewed toward small values

        # Location
        locations = {
            FaultType.BEARING_SPALL: ['inner_race', 'outer_race', 'ball'],
            FaultType.CRACK_FATIGUE: ['shaft', 'casing', 'blade_root'],
            FaultType.SEAL_DAMAGE: ['seal_face', 'seal_ring'],
            FaultType.BLADE_FOD: ['leading_edge', 'trailing_edge', 'tip'],
        }
        location = random.choice(locations.get(fault_type, ['unspecified']))

        return FaultEvent(
            fault_type=fault_type,
            initiation_time=timestamp,
            initiation_operating_hours=operating_hours,
            affected_component=affected_component,
            severity=severity,
            location=location
        )

    def get_active_fault_summary(self) -> Dict:
        """Get summary of active faults."""
        return {
            'num_active_faults': len(self.active_faults),
            'faults_by_component': {
                comp: {
                    'type': fault.fault.fault_type.value,
                    'size': fault.current_size,
                    'hours_since_init': fault.operating_hours_since_init
                }
                for comp, fault in self.active_faults.items()
            },
            'total_initiated': len(self.fault_history)
        }


if __name__ == '__main__':
    """Demonstration of incipient fault modeling."""
    print("Incipient Fault Modeling - Demonstration")
    print("=" * 60)

    simulator = IncipientFaultSimulator(
        enable_incipient_faults=True,
        fault_rate_per_1000hrs=1.0  # 1 fault per 1000 hours (high rate for demo)
    )

    # Simulate 5000 hours of operation
    components = ['bearing', 'seal', 'impeller']
    baseline_health = {'bearing': 0.95, 'seal': 0.92, 'impeller': 0.90}
    timestamp = datetime.now()
    operating_hours = 0.0

    print("\n--- SIMULATION ---")

    for i in range(5000):  # 5000 hours
        operating_hours += 1.0
        timestamp += timedelta(hours=1)

        # Check for fault initiation
        fault_event = simulator.check_fault_initiation(
            operating_hours_increment=1.0,
            stress_factor=1.2,  # Moderate stress
            timestamp=timestamp,
            operating_hours=operating_hours,
            component_list=components
        )

        if fault_event:
            print(f"\n[{operating_hours:.0f} hrs] FAULT INITIATED:")
            print(f"  Type: {fault_event.fault_type.value}")
            print(f"  Component: {fault_event.affected_component}")
            print(f"  Location: {fault_event.location}")
            print(f"  Initial severity: {fault_event.severity:.3f}")

        # Propagate existing faults
        fault_sizes = simulator.propagate_faults(1.0, stress_factor=1.2)

        # General degradation (simplified)
        for comp in baseline_health:
            baseline_health[comp] -= 0.00001

        # Adjust for faults
        adjusted_health = simulator.adjust_health_for_faults(baseline_health)

        # Report every 1000 hours
        if (i + 1) % 1000 == 0:
            print(f"\n[{operating_hours:.0f} hrs] Status:")
            print(f"  Baseline health: {baseline_health}")
            print(f"  Adjusted health: {adjusted_health}")
            summary = simulator.get_active_fault_summary()
            print(f"  Active faults: {summary['num_active_faults']}")
            print(f"  Total initiated: {summary['total_initiated']}")

    print("\n--- FINAL STATE ---")
    print(simulator.get_active_fault_summary())