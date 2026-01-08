"""
Thermal Transient Modeling for Rotating Equipment

Models temperature dynamics during startup, shutdown, and transient events.
Captures differential expansion stress and thermal fatigue that occurs during
rapid temperature changes.

Key Features:
- Operating mode state machine (COLD → STARTUP → STEADY → SHUTDOWN)
- Component-specific thermal time constants
- Differential expansion stress calculation
- Transient-based degradation multipliers

Reference: API 670, turbine startup procedures, thermal stress analysis
"""

import numpy as np
from enum import Enum
from typing import Dict
from dataclasses import dataclass


class OperatingMode(Enum):
    """Equipment operating states."""
    COLD_STANDBY = "cold_standby"
    STARTUP = "startup"
    LOADING = "loading"
    STEADY_STATE = "steady_state"
    UNLOADING = "unloading"
    SHUTDOWN = "shutdown"
    HOT_STANDBY = "hot_standby"


@dataclass
class ThermalMassProperties:
    """Thermal properties for equipment components."""
    # Time constants (minutes) - how fast component reaches thermal equilibrium
    tau_bearing: float = 8.0        
    tau_casing: float = 25.0        
    tau_rotor: float = 45.0       
    # Maximum safe differential temperature (°C)
    max_differential: float = 80.0

    # Thermal stress factor (dimensionless stress per °C differential)
    stress_per_deg: float = 0.015

class ThermalTransientModel:
    """
    Models thermal behavior during equipment transients.
    """
    def __init__(self,
                 ambient_temp: float = 25.0,
                 thermal_properties: ThermalMassProperties = None):
        """
        Initialize thermal transient model.

        Args:
            ambient_temp: Ambient temperature (°C)
            thermal_properties: Component thermal properties
        """
        self.ambient_temp = ambient_temp 
        self.props = thermal_properties or ThermalMassProperties()

        # Current component temperatures
        self.temp_bearing = ambient_temp
        self.temp_casing = ambient_temp
        self.temp_rotor = ambient_temp

        # Operating mode tracking
        self.operating_mode = OperatingMode.COLD_STANDBY
        self.mode_duration = 0  # Minutes in current mode

    def step(self,
             target_speed: float,
             rated_speed: float,
             timestep_minutes: float = 1/60) -> Dict:
        """
        Advance thermal model by one timestep.

        Args:
            target_speed: Target rotor speed (RPM)
            rated_speed: Rated speed for load calculation (RPM)
            timestep_minutes: Timestep duration (minutes)

        Returns:
            Dict with thermal state and degradation factors
        """
        # Update operating mode
        self._update_operating_mode(target_speed, rated_speed)

        # Calculate target temperatures based on load
        load_fraction = target_speed / rated_speed if rated_speed > 0 else 0
        target_temps = self._calculate_target_temperatures(load_fraction)

        # Update component temperatures (exponential approach)
        self.temp_bearing = self._thermal_approach(
            self.temp_bearing,
            target_temps['bearing'],
            self.props.tau_bearing,
            timestep_minutes
        )

        self.temp_casing = self._thermal_approach(
            self.temp_casing,
            target_temps['casing'],
            self.props.tau_casing,
            timestep_minutes
        )

        self.temp_rotor = self._thermal_approach(
            self.temp_rotor,
            target_temps['rotor'],
            self.props.tau_rotor,
            timestep_minutes
        )

        # Calculate differential expansion
        differential_temp = abs(self.temp_rotor - self.temp_casing)

        # Calculate thermal stress
        thermal_stress = min(1.0, differential_temp / self.props.max_differential)

        # Calculate degradation multiplier based on operating mode
        degradation_mult = self._calculate_degradation_multiplier(
            thermal_stress, differential_temp
        )

        # Increment mode duration
        self.mode_duration += timestep_minutes

        return {
            'operating_mode': self.operating_mode.value,
            'mode_duration_min': round(self.mode_duration, 2),
            'temp_bearing': round(self.temp_bearing, 2),
            'temp_casing': round(self.temp_casing, 2),
            'temp_rotor': round(self.temp_rotor, 2),
            'differential_temp': round(differential_temp, 2),
            'thermal_stress': round(thermal_stress, 4),
            'degradation_multiplier': round(degradation_mult, 3),
            'startup_cycles': self._get_startup_cycle_count()
        }

    def _update_operating_mode(self, target_speed: float, rated_speed: float):
        """Update operating mode based on speed transitions."""
        previous_mode = self.operating_mode
        speed_ratio = target_speed / rated_speed if rated_speed > 0 else 0

        # State transition logic
        if speed_ratio < 0.05:  # Essentially stopped
            if self.temp_rotor < (self.ambient_temp + 20):
                new_mode = OperatingMode.COLD_STANDBY
            else:
                new_mode = OperatingMode.HOT_STANDBY

        elif speed_ratio < 0.3:  # Starting up or shutting down
            if previous_mode in [OperatingMode.COLD_STANDBY, OperatingMode.HOT_STANDBY]:
                new_mode = OperatingMode.STARTUP
            elif previous_mode in [OperatingMode.STEADY_STATE, OperatingMode.LOADING]:
                new_mode = OperatingMode.SHUTDOWN
            else:
                new_mode = OperatingMode.STARTUP

        elif speed_ratio < 0.6:  # Loading or unloading
            if previous_mode == OperatingMode.STARTUP:
                new_mode = OperatingMode.LOADING
            elif previous_mode in [OperatingMode.STEADY_STATE, OperatingMode.LOADING]:
                new_mode = OperatingMode.UNLOADING
            else:
                new_mode = OperatingMode.LOADING

        else:  # Normal operation
            new_mode = OperatingMode.STEADY_STATE

        # Reset duration counter on mode change
        if new_mode != previous_mode:
            self.mode_duration = 0
            if new_mode == OperatingMode.STARTUP:
                self._startup_count += 1

        self.operating_mode = new_mode

    _startup_count = 0  # Class variable to track startup cycles

    def _get_startup_cycle_count(self) -> int:
        """Get total number of startup cycles."""
        return self._startup_count

    def _calculate_target_temperatures(self, load_fraction: float) -> Dict:
        """
        Calculate target component temperatures based on load.

        Args:
            load_fraction: Operating load (0.0 to 1.0)

        Returns:
            Dict with target temperatures
        """
        if load_fraction < 0.05:
            # Stopped - cool to ambient
            return {
                'bearing': self.ambient_temp + 5,
                'casing': self.ambient_temp + 10,
                'rotor': self.ambient_temp + 15
            }

        # Operating temperatures scale with load
        return {
            'bearing': self.ambient_temp + 30 + 60 * load_fraction,
            'casing': self.ambient_temp + 25 + 80 * load_fraction,
            'rotor': self.ambient_temp + 35 + 120 * load_fraction
        }

    def _thermal_approach(self,
                         current_temp: float,
                         target_temp: float,
                         tau: float,
                         dt: float) -> float:
        """
        First-order thermal response (exponential approach).

        dT/dt = (T_target - T_current) / tau

        Args:
            current_temp: Current temperature (°C)
            target_temp: Target temperature (°C)
            tau: Time constant (minutes)
            dt: Timestep (minutes)

        Returns:
            New temperature (°C)
        """
        # Rate constant
        k = 1.0 - np.exp(-dt / tau)

        # New temperature
        new_temp = current_temp + k * (target_temp - current_temp)

        return new_temp

    def _calculate_degradation_multiplier(self,
                                         thermal_stress: float,
                                         differential_temp: float) -> float:
        """
        Calculate degradation rate multiplier based on thermal conditions.

        Research shows:
        - 60-70% of thermal fatigue occurs during transients
        - Rapid startups cause 2-3x normal degradation
        - Differential expansion causes mechanical stress

        Args:
            thermal_stress: Normalized thermal stress (0.0 to 1.0)
            differential_temp: Temperature difference rotor-casing (°C)

        Returns:
            Degradation multiplier (1.0 = normal, >1.0 = accelerated)
        """
        base_multiplier = 1.0

        # Mode-specific multipliers
        if self.operating_mode == OperatingMode.STARTUP:
            # High stress during startup, especially first 30 minutes
            if self.mode_duration < 30:
                startup_factor = 2.5 - (self.mode_duration / 30) * 1.5  # 2.5 → 1.0
            else:
                startup_factor = 1.2
            base_multiplier *= startup_factor

        elif self.operating_mode == OperatingMode.SHUTDOWN:
            # Moderate stress during shutdown
            base_multiplier *= 1.3

        elif self.operating_mode == OperatingMode.LOADING:
            # Elevated stress while loading
            base_multiplier *= 1.4

        elif self.operating_mode == OperatingMode.UNLOADING:
            # Moderate stress during unloading
            base_multiplier *= 1.2

        # Add thermal stress component
        # Exponential increase with differential temperature
        thermal_factor = 1.0 + 2.0 * (thermal_stress ** 2)
        base_multiplier *= thermal_factor

        return min(base_multiplier, 5.0)  # Cap at 5x normal degradation

    def get_thermal_state(self) -> Dict:
        """Get current thermal state for diagnostics."""
        return {
            'mode': self.operating_mode.value,
            'mode_duration_min': self.mode_duration,
            'temp_bearing_C': self.temp_bearing,
            'temp_casing_C': self.temp_casing,
            'temp_rotor_C': self.temp_rotor,
            'differential_C': abs(self.temp_rotor - self.temp_casing),
            'startup_cycles': self._startup_count
        }

if __name__ == '__main__':
    """Demonstration of thermal transient behavior."""
    print("Thermal Transient Model - Demonstration")
    print("=" * 60)

    model = ThermalTransientModel(ambient_temp=25.0)

    # Simulate startup sequence
    print("\n--- STARTUP SEQUENCE ---")
    rated_speed = 10000

    # Initial idle
    for i in range(10):
        state = model.step(0, rated_speed, timestep_minutes=1)

    print(f"Initial: {model.get_thermal_state()}")

    # Rapid startup to 50% load
    print("\nRapid startup to 5000 RPM:")
    for i in range(30):
        state = model.step(5000, rated_speed, timestep_minutes=1)
        if i % 10 == 9:
            print(f"  {i+1} min: Mode={state['operating_mode']}, "
                  f"ΔT={state['differential_temp']:.1f}°C, "
                  f"Mult={state['degradation_multiplier']:.2f}x")

    # Increase to full load
    print("\nLoading to 10000 RPM:")
    for i in range(20):
        state = model.step(10000, rated_speed, timestep_minutes=1)
        if i % 10 == 9:
            print(f"  {i+1} min: Mode={state['operating_mode']}, "
                  f"ΔT={state['differential_temp']:.1f}°C, "
                  f"Mult={state['degradation_multiplier']:.2f}x")

    # Steady state
    print("\nSteady state operation:")
    for i in range(60):
        state = model.step(10000, rated_speed, timestep_minutes=1)
    print(f"  60 min: Mode={state['operating_mode']}, "
          f"ΔT={state['differential_temp']:.1f}°C, "
          f"Mult={state['degradation_multiplier']:.2f}x")

    # Shutdown
    print("\nShutdown:")
    for i in range(30):
        state = model.step(0, rated_speed, timestep_minutes=1)
        if i % 10 == 9:
            print(f"  {i+1} min: Mode={state['operating_mode']}, "
                  f"ΔT={state['differential_temp']:.1f}°C, "
                  f"Mult={state['degradation_multiplier']:.2f}x")

    print(f"\nFinal state: {model.get_thermal_state()}")
    print(f"Total startup cycles: {model._startup_count}")