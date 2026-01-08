"""
Process Upset Events for Edge Case Coverage

Simulates abnormal operating conditions and process disturbances that
stress equipment and create rich edge cases for ML model training.

Key Features:
- Liquid carryover to compressor (surge risk)
- Pump cavitation events (rapid damage)
- Process trips and thermal shocks
- Feed composition changes
- Equipment overload scenarios

Reference: Process safety management, abnormal situation management
"""

import numpy as np
import random
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


class UpsetType(Enum):
    """Types of process upsets."""
    LIQUID_CARRYOVER = "liquid_carryover"      # Liquid enters gas compressor
    PUMP_RUNOUT = "pump_runout"                # Pump operating beyond BEP
    CAVITATION_EVENT = "cavitation_event"      # Pump cavitation
    THERMAL_SHOCK = "thermal_shock"            # Rapid temperature change
    FEED_COMPOSITION_SHIFT = "feed_composition_shift"  # Gas/fluid property change
    OVERLOAD = "overload"                      # Sustained overload operation
    TRIP_EVENT = "trip_event"                  # Emergency shutdown
    LUBE_OIL_CONTAMINATION = "lube_oil_contamination"  # Oil quality degradation


@dataclass
class UpsetEvent:
    """Record of a process upset."""
    upset_type: UpsetType
    timestamp: datetime
    duration_seconds: int
    severity: float  # 0.0 to 1.0
    damage_potential: float  # Expected damage (health reduction)
    description: str


class ProcessUpsetSimulator:
    """
    Generates and manages process upset events.
    """

    def __init__(self,
                 enable_upsets: bool = True,
                 upset_rate_per_month: float = 2.0):
        """
        Initialize process upset simulator.

        Args:
            enable_upsets: Enable upset events
            upset_rate_per_month: Expected upsets per month (30 days)
        """
        self.enable = enable_upsets
        self.upset_rate = upset_rate_per_month / (30 * 24)  # Per hour
        self.active_upset: Optional[UpsetEvent] = None
        self.upset_remaining_seconds = 0
        self.upset_history = []

    def check_upset_initiation(self,
                               timestep_seconds: int,
                               timestamp: datetime,
                               operating_state: Dict) -> Optional[UpsetEvent]:
        """
        Check for new upset event initiation (Poisson process).

        Args:
            timestep_seconds: Simulation timestep
            timestamp: Current timestamp
            operating_state: Current equipment operating state

        Returns:
            UpsetEvent if initiated, None otherwise
        """
        if not self.enable or self.active_upset is not None:
            return None  # Only one upset at a time

        # Probability of upset in this timestep
        p_upset = 1 - np.exp(-self.upset_rate * timestep_seconds / 3600)

        if random.random() < p_upset:
            upset = self._create_upset_event(timestamp, operating_state)
            self.active_upset = upset
            self.upset_remaining_seconds = upset.duration_seconds
            self.upset_history.append(upset)
            return upset

        return None

    def step(self, timestep_seconds: int) -> Optional[UpsetEvent]:
        """
        Advance upset simulation.

        Args:
            timestep_seconds: Simulation timestep

        Returns:
            Active upset or None
        """
        if self.active_upset is None:
            return None

        self.upset_remaining_seconds -= timestep_seconds

        if self.upset_remaining_seconds <= 0:
            # Upset ended
            self.active_upset = None
            return None

        return self.active_upset

    def apply_upset_effects(self,
                           normal_state: Dict,
                           equipment_type: str) -> Dict:
        """
        Apply upset effects to equipment state.

        Args:
            normal_state: Normal operating state
            equipment_type: 'turbine', 'compressor', or 'pump'

        Returns:
            Modified state during upset
        """
        if self.active_upset is None:
            return normal_state

        upset_state = normal_state.copy()
        upset_type = self.active_upset.upset_type
        severity = self.active_upset.severity

        # Apply type-specific effects
        if upset_type == UpsetType.LIQUID_CARRYOVER and equipment_type == 'compressor':
            upset_state = self._apply_liquid_carryover(upset_state, severity)

        elif upset_type == UpsetType.CAVITATION_EVENT and equipment_type == 'pump':
            upset_state = self._apply_cavitation(upset_state, severity)

        elif upset_type == UpsetType.THERMAL_SHOCK:
            upset_state = self._apply_thermal_shock(upset_state, severity)

        elif upset_type == UpsetType.OVERLOAD:
            upset_state = self._apply_overload(upset_state, severity)

        elif upset_type == UpsetType.LUBE_OIL_CONTAMINATION:
            upset_state = self._apply_oil_contamination(upset_state, severity)

        return upset_state

    def calculate_upset_damage(self,
                              baseline_health: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate health reduction due to active upset.

        Args:
            baseline_health: Current health state

        Returns:
            Health after upset damage
        """
        if self.active_upset is None:
            return baseline_health

        damaged_health = baseline_health.copy()

        # Damage rate depends on upset type and severity
        damage_per_second = self.active_upset.damage_potential / self.active_upset.duration_seconds

        # Apply to all components (some upsets affect specific components more)
        for component in damaged_health:
            damaged_health[component] = max(0.0,
                damaged_health[component] - damage_per_second
            )

        return damaged_health

    def _create_upset_event(self,
                           timestamp: datetime,
                           operating_state: Dict) -> UpsetEvent:
        """Create a new upset event."""
        # Select upset type randomly
        upset_type = random.choice(list(UpsetType))

        # Duration varies by type
        duration_ranges = {
            UpsetType.LIQUID_CARRYOVER: (30, 300),      # 30s to 5min
            UpsetType.PUMP_RUNOUT: (60, 1800),          # 1-30min
            UpsetType.CAVITATION_EVENT: (10, 120),      # 10s to 2min
            UpsetType.THERMAL_SHOCK: (5, 60),           # 5s to 1min
            UpsetType.FEED_COMPOSITION_SHIFT: (300, 3600),  # 5min to 1hr
            UpsetType.OVERLOAD: (600, 7200),            # 10min to 2hrs
            UpsetType.TRIP_EVENT: (1, 10),              # 1-10s
            UpsetType.LUBE_OIL_CONTAMINATION: (3600, 86400)  # 1hr to 1 day
        }

        duration = random.randint(*duration_ranges[upset_type])

        # Severity (most upsets are moderate)
        severity = np.random.beta(3, 3)  # Centered around 0.5

        # Damage potential
        damage_potentials = {
            UpsetType.LIQUID_CARRYOVER: 0.05,    # 5% health loss
            UpsetType.PUMP_RUNOUT: 0.02,
            UpsetType.CAVITATION_EVENT: 0.08,    # High damage rate
            UpsetType.THERMAL_SHOCK: 0.03,
            UpsetType.FEED_COMPOSITION_SHIFT: 0.01,
            UpsetType.OVERLOAD: 0.04,
            UpsetType.TRIP_EVENT: 0.02,
            UpsetType.LUBE_OIL_CONTAMINATION: 0.06
        }

        damage = damage_potentials[upset_type] * severity

        description = self._generate_description(upset_type, severity)

        return UpsetEvent(
            upset_type=upset_type,
            timestamp=timestamp,
            duration_seconds=duration,
            severity=severity,
            damage_potential=damage,
            description=description
        )

    def _apply_liquid_carryover(self, state: Dict, severity: float) -> Dict:
        """Liquid entering compressor causes surge risk and blade damage."""
        modified = state.copy()

        # Reduce surge margin dramatically
        if 'surge_margin' in modified:
            modified['surge_margin'] *= (1.0 - severity * 0.8)

        # Increase vibration (liquid impacts)
        if 'vibration_amplitude' in modified:
            modified['vibration_amplitude'] *= (1.0 + severity * 2.0)

        # Temperature fluctuations
        if 'discharge_temp' in modified:
            modified['discharge_temp'] += severity * 20

        return modified

    def _apply_cavitation(self, state: Dict, severity: float) -> Dict:
        """Cavitation causes vibration, noise, and rapid erosion."""
        modified = state.copy()

        # NPSH margin drops
        if 'npsh_margin' in modified:
            modified['npsh_margin'] = max(0, modified['npsh_margin'] - severity * 3.0)

        # Vibration spikes
        if 'vibration_rms' in modified:
            modified['vibration_rms'] *= (1.0 + severity * 5.0)

        # Flow instability
        if 'flow' in modified:
            modified['flow'] *= (0.85 + random.random() * 0.3)  # Fluctuating

        # Cavitation severity indicator
        modified['cavitation_severity'] = min(3, int(severity * 4))

        return modified

    def _apply_thermal_shock(self, state: Dict, severity: float) -> Dict:
        """Rapid temperature change causes thermal stress."""
        modified = state.copy()

        # Temperature jump
        temp_keys = [k for k in state.keys() if 'temp' in k.lower()]
        for key in temp_keys:
            # Random direction (hot or cold shock)
            direction = random.choice([-1, 1])
            modified[key] = state[key] + direction * severity * 50

        # Stress manifests as increased vibration
        vib_keys = [k for k in state.keys() if 'vib' in k.lower()]
        for key in vib_keys:
            modified[key] = state[key] * (1.0 + severity * 0.5)

        return modified

    def _apply_overload(self, state: Dict, severity: float) -> Dict:
        """Sustained overload operation."""
        modified = state.copy()

        # Speed/power increase
        if 'speed' in modified:
            modified['speed'] *= (1.0 + severity * 0.3)

        if 'power' in modified:
            modified['power'] *= (1.0 + severity * 0.4)

        # Temperature rise
        temp_keys = [k for k in state.keys() if 'temp' in k.lower()]
        for key in temp_keys:
            modified[key] = state[key] + severity * 15

        # Motor current increase (for pumps)
        if 'motor_current' in modified:
            modified['motor_current'] *= (1.0 + severity * 0.3)

        return modified

    def _apply_oil_contamination(self, state: Dict, severity: float) -> Dict:
        """Contaminated lube oil increases bearing wear."""
        modified = state.copy()

        # Bearing temperatures rise
        bearing_keys = [k for k in state.keys() if 'bearing' in k.lower() and 'temp' in k.lower()]
        for key in bearing_keys:
            modified[key] = state[key] + severity * 10

        # Vibration increases
        vib_keys = [k for k in state.keys() if 'vib' in k.lower()]
        for key in vib_keys:
            modified[key] = state[key] * (1.0 + severity * 0.8)

        return modified

    def _generate_description(self, upset_type: UpsetType, severity: float) -> str:
        """Generate human-readable description."""
        severity_desc = ['minor', 'moderate', 'significant', 'severe'][int(severity * 3.99)]

        descriptions = {
            UpsetType.LIQUID_CARRYOVER: f"{severity_desc} liquid carryover event - slug of liquid entered compressor",
            UpsetType.PUMP_RUNOUT: f"{severity_desc} pump runout - operating far from BEP",
            UpsetType.CAVITATION_EVENT: f"{severity_desc} cavitation event - NPSH critically low",
            UpsetType.THERMAL_SHOCK: f"{severity_desc} thermal shock - rapid temperature transient",
            UpsetType.FEED_COMPOSITION_SHIFT: f"{severity_desc} feed composition change",
            UpsetType.OVERLOAD: f"{severity_desc} overload condition - sustained operation above rating",
            UpsetType.TRIP_EVENT: f"{severity_desc} trip event - emergency shutdown",
            UpsetType.LUBE_OIL_CONTAMINATION: f"{severity_desc} lube oil contamination detected"
        }

        return descriptions.get(upset_type, f"{severity_desc} process upset")


if __name__ == '__main__':
    """Demonstration of process upset simulation."""
    print("Process Upset Events - Demonstration")
    print("=" * 60)

    simulator = ProcessUpsetSimulator(
        enable_upsets=True,
        upset_rate_per_month=5.0  # High rate for demonstration
    )

    # Simulate 30 days (720 hours)
    hours_to_simulate = 720
    timestep_seconds = 60  # 1-minute timesteps
    timestamp = datetime.now()

    normal_state = {
        'speed': 3000,
        'vibration_rms': 2.5,
        'bearing_temp_de': 75.0,
        'npsh_margin': 5.0,
        'surge_margin': 25.0
    }

    health = {'bearing': 0.90, 'seal': 0.85, 'impeller': 0.92}

    print("\n--- SIMULATION ---")

    for hour in range(hours_to_simulate):
        timestamp += timedelta(hours=1)

        for minute in range(60):
            # Check for upset initiation
            upset_event = simulator.check_upset_initiation(
                timestep_seconds,
                timestamp,
                normal_state
            )

            if upset_event:
                print(f"\n[{hour} hrs] UPSET INITIATED:")
                print(f"  Type: {upset_event.upset_type.value}")
                print(f"  Duration: {upset_event.duration_seconds}s")
                print(f"  Severity: {upset_event.severity:.2f}")
                print(f"  Description: {upset_event.description}")

            # Advance upset
            active_upset = simulator.step(timestep_seconds)

            # Apply effects
            if active_upset:
                current_state = simulator.apply_upset_effects(
                    normal_state,
                    equipment_type='pump'
                )

                # Apply damage
                health = simulator.calculate_upset_damage(health)
            else:
                current_state = normal_state

    print(f"\n--- SUMMARY ---")
    print(f"Total upsets simulated: {len(simulator.upset_history)}")
    print(f"Final health: {health}")

    print(f"\nUpset types encountered:")
    for upset in simulator.upset_history:
        print(f"  - {upset.upset_type.value}: severity {upset.severity:.2f}")