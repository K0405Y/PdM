"""
Centrifugal Compressor Data Simulator 

This module simulates an industrial centrifugal compressor typical of pipeline
networks, LNG facilities, and refinery process units. Centrifugal compressors 
are critical high-speed rotating equipment requiring sophisticated surge 
protection and condition monitoring.

Key Features:
- Surge margin monitoring and anti-surge control simulation
- Dry gas seal health tracking (primary and secondary seals)
- Shaft orbit and displacement simulation via proximity probes
- Multi-mode degradation: seal wear, impeller fouling, bearing degradation
- Performance map simulation for thermodynamic efficiency

Reference: API 670, API 617, Bently Nevada System 1, industry operational data
"""

import numpy as np
import random
import math
from datetime import datetime, timedelta
from typing import Tuple

# Try to import enhancement modules
ENHANCEMENTS_AVAILABLE = False
try:
    from .physics import (
        EnhancedVibrationGenerator, BearingGeometry,
        ThermalTransientModel, ThermalMassProperties,
        EnvironmentalConditions, LocationType
    )
    from .simulation import (
        MaintenanceScheduler, IncipientFaultSimulator, ProcessUpsetSimulator
    )
    from .ml_utils import DataOutputFormatter, OutputMode
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    pass

class SurgeModel:
    """
    Models the surge characteristics of a centrifugal compressor.
    
    Surge is the most dangerous operational event - a violent flow reversal
    that can damage impellers and thrust bearings. This model tracks the
    surge margin and simulates approach to surge conditions.
    """
    
    def __init__(self, 
                 design_flow: float = 1000.0,
                 design_head: float = 8000.0,
                 surge_margin_alarm: float = 10.0,
                 surge_margin_trip: float = 5.0):
        """
        Initialize surge model.
        
        Args:
            design_flow: Design volumetric flow rate (m³/hr)
            design_head: Design polytropic head (kJ/kg)
            surge_margin_alarm: Alarm threshold (% from surge line)
            surge_margin_trip: Trip threshold (% from surge line)
        """
        self.design_flow = design_flow
        self.design_head = design_head
        self.surge_margin_alarm = surge_margin_alarm
        self.surge_margin_trip = surge_margin_trip
        
        # Surge line coefficients (parabolic approximation)
        # head_surge = a * flow^2 + b * flow + c
        self._a = 0.005
        self._b = -2.0
        self._c = design_head * 1.2
        
    def calculate_surge_flow(self, head: float) -> float:
        """
        Calculate the flow rate at surge for a given head.
        
        Args:
            head: Current polytropic head (kJ/kg)
            
        Returns:
            float: Flow rate at surge point (m³/hr)
        """
        # Solve quadratic: a*Q^2 + b*Q + (c - head) = 0
        discriminant = self._b**2 - 4 * self._a * (self._c - head)
        if discriminant < 0:
            return 0.0
        return (-self._b - math.sqrt(discriminant)) / (2 * self._a)
    
    def calculate_surge_margin(self, flow: float, head: float) -> float:
        """
        Calculate current surge margin as percentage.
        
        Args:
            flow: Current volumetric flow rate (m³/hr)
            head: Current polytropic head (kJ/kg)
            
        Returns:
            float: Surge margin (%) - positive is safe, <0 is in surge
        """
        surge_flow = self.calculate_surge_flow(head)
        if surge_flow <= 0:
            return 100.0  # Far from surge
        return ((flow - surge_flow) / surge_flow) * 100.0
    
    def is_surge_alarm(self, margin: float) -> bool:
        """Check if surge margin is in alarm condition."""
        return margin < self.surge_margin_alarm
    
    def is_surge_trip(self, margin: float) -> bool:
        """Check if surge margin requires trip."""
        return margin < self.surge_margin_trip

class DryGasSealModel:
    """
    Models dry gas seal health and leakage for centrifugal compressors.
    
    Dry gas seals are critical for preventing process gas leakage along
    the compressor shaft. Primary and secondary seal health are tracked
    independently.
    """
    
    SEAL_TYPES = ['primary', 'secondary']
    
    def __init__(self, initial_health: dict = None):
        """
        Initialize dry gas seal model.
        
        Args:
            initial_health: Dict with 'primary' and 'secondary' health (0-1)
        """
        self.health = initial_health or {
            'primary': 0.95,
            'secondary': 0.98
        }
        
        # Degradation rates (fraction per operating hour at severity 1.0)
        self.degradation_rates = {
            'primary': 0.00005,   # Primary degrades faster
            'secondary': 0.00002
        }
        
        # Leakage model parameters
        self.base_leakage = {
            'primary': 2.0,    # Nm³/hr baseline
            'secondary': 0.5
        }
        
        # Failure thresholds
        self.failure_threshold = 0.25
        
    def step(self, 
             operating_severity: float = 1.0,
             contamination_factor: float = 1.0) -> dict:
        """
        Advance seal degradation by one hour.
        
        Args:
            operating_severity: Multiplier for degradation rate
            contamination_factor: Additional factor for dirty gas
            
        Returns:
            dict: Current health and leakage values
            
        Raises:
            Exception: When seal fails below threshold
        """
        result = {}
        
        for seal_type in self.SEAL_TYPES:
            # Apply degradation
            rate = self.degradation_rates[seal_type]
            effective_rate = rate * operating_severity * contamination_factor
            self.health[seal_type] -= effective_rate
            
            # Check for failure
            if self.health[seal_type] < self.failure_threshold:
                raise Exception(f"F_SEAL_{seal_type.upper()}")
            
            # Calculate leakage based on health
            health_factor = 1.0 / max(self.health[seal_type], 0.1)
            leakage = self.base_leakage[seal_type] * health_factor
            
            result[f'{seal_type}_health'] = self.health[seal_type]
            result[f'{seal_type}_leakage'] = leakage
            
        return result


class ShaftOrbitModel:
    """
    Simulates shaft displacement and orbit patterns measured by 
    orthogonal proximity probes.
    
    Shaft orbit analysis reveals rotor unbalance, misalignment, rub,
    and bearing condition through characteristic patterns.
    """
    
    def __init__(self, 
                 bearing_clearance: float = 0.15,  # mm
                 sample_rate: int = 1024):
        """
        Initialize shaft orbit model.
        
        Args:
            bearing_clearance: Radial bearing clearance (mm)
            sample_rate: Samples per second (Hz)
        """
        self.bearing_clearance = bearing_clearance
        self.sample_rate = sample_rate
        self.phase = 0.0
        
    def generate_orbit(self,
                       rpm: float,
                       health_state: dict,
                       duration: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate X-Y displacement signals for shaft orbit.
        
        Args:
            rpm: Current rotor speed (RPM)
            health_state: Dict with health indicators
            duration: Signal duration in seconds
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X and Y displacement signals (mm)
        """
        if rpm <= 0:
            n_samples = int(self.sample_rate * duration)
            noise_x = np.random.normal(0, 0.001, n_samples)
            noise_y = np.random.normal(0, 0.001, n_samples)
            return noise_x, noise_y
            
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        omega = 2 * np.pi * rpm / 60.0  # rad/s
        
        # Base orbit (healthy machine - slightly elliptical)
        base_radius = 0.02  # mm - small orbit when healthy
        ellipticity = 0.2   # Slight ellipse
        
        # Scale orbit with degradation
        impeller_health = health_state.get('impeller', 1.0)
        bearing_health = health_state.get('bearing', 1.0)
        
        # Unbalance increases orbit size
        unbalance_factor = 1.0 + 2.0 * (1.0 - impeller_health)
        
        # Bearing wear increases orbit size and introduces harmonics
        wear_factor = 1.0 + 3.0 * (1.0 - bearing_health)
        
        orbit_radius = base_radius * unbalance_factor * wear_factor
        
        # Generate orbit signals
        x = orbit_radius * np.cos(omega * t + self.phase)
        y = orbit_radius * (1 - ellipticity) * np.sin(omega * t + self.phase)
        
        # Add sub-synchronous whirl if bearing is degraded (oil whirl/whip)
        if bearing_health < 0.6:
            whirl_freq = 0.43 * omega  # Characteristic oil whirl frequency
            whirl_amp = (0.6 - bearing_health) * 0.05
            x += whirl_amp * np.cos(whirl_freq * t)
            y += whirl_amp * np.sin(whirl_freq * t)
            
        # Add super-synchronous content if impeller damaged
        if impeller_health < 0.7:
            # Blade pass frequency effects
            n_blades = 8  # Typical impeller blade count
            blade_freq = n_blades * omega
            blade_amp = (0.7 - impeller_health) * 0.01
            x += blade_amp * np.cos(blade_freq * t)
            y += blade_amp * np.sin(blade_freq * t)
            
        # Add noise
        x += np.random.normal(0, 0.002, n_samples)
        y += np.random.normal(0, 0.002, n_samples)
        
        # Update phase for continuity
        self.phase = (self.phase + omega * duration) % (2 * np.pi)
        
        return x, y
    
    def compute_metrics(self, 
                        x: np.ndarray, 
                        y: np.ndarray) -> dict:
        """
        Compute shaft orbit metrics.
        
        Args:
            x: X-axis displacement signal
            y: Y-axis displacement signal
            
        Returns:
            dict: Orbit metrics including amplitude, gap, and synchronous amplitude
        """
        # Direct orbit amplitude (peak-to-peak)
        smax = np.sqrt(x**2 + y**2).max()
        
        # Average position
        x_avg = np.mean(x)
        y_avg = np.mean(y)
        gap = np.sqrt(x_avg**2 + y_avg**2)
        
        # 1X synchronous amplitude (approximate)
        sync_amp = np.sqrt(np.var(x) + np.var(y)) * 2
        
        return {
            'orbit_amplitude': round(smax * 2, 4),  # Peak-to-peak (mm)
            'average_gap': round(gap, 4),           # mm
            'sync_amplitude': round(sync_amp, 4)    # mm
        }


class CentrifugalCompressorHealthModel:
    """
    Manages degradation pathways specific to centrifugal compressors.
    """
    
    FAILURE_MODES = {
        'F_IMPELLER': 'Impeller Degradation - Erosion or fouling',
        'F_BEARING': 'Bearing Failure - Journal or thrust bearing damage',
        'F_SEAL_PRIMARY': 'Primary Dry Gas Seal Failure',
        'F_SEAL_SECONDARY': 'Secondary Dry Gas Seal Failure',
        'F_SURGE': 'Surge Event - Violent flow reversal',
        'F_HIGH_VIBRATION': 'High Vibration Trip - Shaft orbit amplitude exceeded safety limits',
        'F_BEARING_TEMP': 'High Bearing Temperature Trip - Temperature exceeded limit'
    }
    
    def __init__(self, initial_health: dict = None):
        """
        Initialize health model.
        
        Args:
            initial_health: Dict with 'impeller', 'bearing' health values
        """
        self.health = initial_health or {
            'impeller': 0.92,
            'bearing': 0.88
        }
        
        # Degradation parameters (d, a, b) for h(t) = 1 - d - exp(a*t^b)
        self.degradation_params = {
            'impeller': (0.04, -0.28, 0.21),
            'bearing': (0.06, -0.32, 0.24)
        }
        
        self.failure_thresholds = {
            'impeller': 0.42,
            'bearing': 0.38
        }
        
        self._init_generators()
        
    def _init_generators(self):
        """Initialize health trajectory generators."""
        self._generators = {}
        for mode, (d, a, b) in self.degradation_params.items():
            current_h = self.health[mode]
            threshold = self.failure_thresholds[mode]
            self._generators[mode] = self._health_generator(current_h, d, a, b, threshold)

    def _health_generator(self, initial_health, d, a, b, threshold):
        """Generator yielding health values over time using linear degradation."""
        # Use simpler linear degradation from initial health to threshold
        # Rate is calibrated so equipment with health=0.92 lasts ~50000 steps
        base_rate = 0.00001  # Health loss per step
        current_health = initial_health

        while current_health >= threshold:
            yield current_health
            current_health -= base_rate * (1.0 + random.gauss(0, 0.1))  # Small random variation
            current_health = max(current_health, 0)  # Don't go negative
            
    def step(self, operating_severity: float = 1.0) -> dict:
        """
        Advance health model by one time step.

        Returns:
            dict: Current health values

        Raises:
            Exception: With failure mode code on failure
        """
        updated_health = {}

        for mode, gen in self._generators.items():
            try:
                # Apply extra degradation steps for high severity
                if operating_severity > 1.0:
                    extra_steps = int(operating_severity - 1.0)
                    for _ in range(extra_steps):
                        next(gen)
                    if random.random() < (operating_severity % 1.0):
                        next(gen)
                h = next(gen)
                self.health[mode] = h
                updated_health[mode] = h
            except StopIteration:
                raise Exception(f"F_{mode.upper()}")

        return updated_health


class CentrifugalCompressor:
    """
    Industrial Centrifugal Compressor Simulator for Predictive Maintenance.
    
    Simulates a centrifugal compressor typical of pipeline networks and LNG
    facilities with realistic operating parameters, surge protection, and
    dry gas seal monitoring.
    
    Operating Envelope (based on industrial data):
    - Speed: 5,000 - 25,000 RPM (varies by design)
    - Suction Pressure: 500 - 5,000 kPa
    - Discharge Pressure: 2,000 - 15,000 kPa
    - Flow Rate: 100 - 5,000 m³/hr
    - Vibration (shaft): 0.025 - 0.075 mm (API 617)
    """
    
    LIMITS = {
        'speed_min': 5000,
        'speed_max': 25000,
        'speed_rated': 12000,
        'suction_pressure_min': 500,    # kPa
        'suction_pressure_max': 5000,
        'discharge_pressure_max': 15000,
        'flow_min': 100,                 # m³/hr
        'flow_max': 3000,
        'flow_rated': 1500,
        'vibration_alarm': 0.050,        # mm
        'vibration_trip': 0.075,         # mm (API 617)
        'bearing_temp_max': 110,         # °C
        'seal_leakage_alarm': 5.0,       # Nm³/hr
    }
    
    def __init__(self,
                 name: str,
                 initial_health: dict = None,
                 design_flow: float = 1500.0,
                 design_head: float = 8000.0,
                 suction_pressure: float = 2000.0,
                 suction_temp: float = 35.0,
                 location_type = None,
                 env_model = None,
                 enable_enhanced_vibration: bool = True,
                 enable_thermal_transients: bool = True,
                 enable_environmental: bool = True,
                 enable_maintenance: bool = True,
                 enable_incipient_faults: bool = True,
                 enable_process_upsets: bool = True,
                 output_mode = None):
        """
        Initialize centrifugal compressor simulator.

        Args:
            name: Unique identifier
            initial_health: Dict with initial health values
            design_flow: Design volumetric flow (m³/hr)
            design_head: Design polytropic head (kJ/kg)
            suction_pressure: Suction pressure (kPa)
            suction_temp: Suction temperature (°C)
            location_type: LocationType enum for synthetic environmental modeling
            env_model: Custom environmental data source (real weather API or synthetic)
                      If provided, takes precedence over location_type
            enable_enhanced_vibration: Use envelope-modulated vibration
            enable_thermal_transients: Model thermal stress
            enable_environmental: Include environmental variability
            enable_maintenance: Enable maintenance events
            enable_incipient_faults: Enable discrete fault initiation
            enable_process_upsets: Enable process upset events
            output_mode: Data output format mode
        """
        self.name = name
        self.design_flow = design_flow
        self.design_head = design_head
        
        # Process conditions
        self.suction_pressure = suction_pressure
        self.suction_temp = suction_temp
        self.discharge_pressure = suction_pressure  # Will be calculated
        self.discharge_temp = suction_temp
        
        # Operating state
        self.speed = 0.0
        self.speed_target = 0.0
        self.flow = 0.0
        self.head = 0.0
        
        # Component models
        self.health_model = CentrifugalCompressorHealthModel(initial_health)
        self.surge_model = SurgeModel(design_flow, design_head)
        self.seal_model = DryGasSealModel()
        self.orbit_model = ShaftOrbitModel()
        
        # Bearing temperatures
        self.bearing_temp_de = 45.0  # Drive end
        self.bearing_temp_nde = 45.0  # Non-drive end
        self.thrust_bearing_temp = 50.0
        
        # Performance
        self.efficiency = 0.82
        self.power = 0.0
        
        # Time tracking
        self.operating_hours = 0.0
        self.t = 0
        self.elapsed_hours = 0.0
        self.current_timestamp = datetime.now()
        
        # Initialize enhancement features
        if ENHANCEMENTS_AVAILABLE:
            # Enhanced vibration generator
            if enable_enhanced_vibration:
                try:
                    bearing_geom = BearingGeometry(
                        n_balls=14, ball_diameter=18.0,
                        pitch_diameter=90.0, contact_angle=0.0
                    )
                    self.vib_generator_enhanced = EnhancedVibrationGenerator(
                        sample_rate=10240, resonance_freq=3000,
                        bearing_geometry=bearing_geom
                    )
                    self.use_enhanced_vibration = True
                except:
                    self.use_enhanced_vibration = False
            else:
                self.use_enhanced_vibration = False
            
            # Thermal transient model
            if enable_thermal_transients:
                try:
                    self.thermal_model = ThermalTransientModel(
                        ambient_temp=25.0,
                        thermal_properties=ThermalMassProperties(
                            tau_bearing=6.0, tau_casing=20.0, tau_rotor=35.0
                        )
                    )
                    self.use_thermal_model = True
                except:
                    self.use_thermal_model = False
            else:
                self.use_thermal_model = False
            
            # Environmental model
            if enable_environmental:
                try:
                    # Priority: env_model > location_type
                    if env_model is not None:
                        # Use custom environmental source (real weather API or custom synthetic)
                        self.env_model = env_model
                        self.use_environmental = True
                    elif location_type is not None:
                        # Use synthetic location profile
                        self.env_model = EnvironmentalConditions(location_type=location_type)
                        self.use_environmental = True
                    else:
                        # Default to temperate if enhancements available but no location specified
                        self.env_model = EnvironmentalConditions(location_type=LocationType.TEMPERATE)
                        self.use_environmental = True
                except Exception as e:
                    print(f"Warning: Environmental initialization failed: {e}")
                    self.use_environmental = False
            else:
                self.use_environmental = False
            
            # Maintenance scheduler
            if enable_maintenance:
                try:
                    self.maint_scheduler = MaintenanceScheduler(
                        enable_time_based=True,
                        enable_condition_based=True,
                        enable_opportunistic=True
                    )
                    self.use_maintenance = True
                except:
                    self.use_maintenance = False
            else:
                self.use_maintenance = False
            
            # Incipient fault simulator
            if enable_incipient_faults:
                try:
                    self.fault_sim = IncipientFaultSimulator(
                        enable_incipient_faults=True, fault_rate_per_1000hrs=0.4
                    )
                    self.use_faults = True
                except:
                    self.use_faults = False
            else:
                self.use_faults = False
            
            # Process upset simulator
            if enable_process_upsets:
                try:
                    self.upset_sim = ProcessUpsetSimulator(
                        enable_upsets=True, upset_rate_per_month=2.0
                    )
                    self.use_upsets = True
                except:
                    self.use_upsets = False
            else:
                self.use_upsets = False
            
            # Output formatter
            if output_mode is not None:
                try:
                    self.output_formatter = DataOutputFormatter(output_mode=output_mode)
                    self.use_output_formatter = True
                except:
                    self.use_output_formatter = False
            else:
                self.use_output_formatter = False
        else:
            # Enhancements not available
            self.use_enhanced_vibration = False
            self.use_thermal_model = False
            self.use_environmental = False
            self.use_maintenance = False
            self.use_faults = False
            self.use_upsets = False
            self.use_output_formatter = False
        
    def set_speed(self, target_rpm: float):
        """Set target operating speed."""
        self.speed_target = max(0, min(target_rpm, self.LIMITS['speed_max']))
        
    def set_flow(self, target_flow: float):
        """Set target flow rate (will adjust speed accordingly)."""
        self.flow_target = max(0, min(target_flow, self.LIMITS['flow_max']))
        
    def _calculate_operating_severity(self) -> float:
        """Calculate severity based on operating conditions."""
        speed_factor = self.speed / self.LIMITS['speed_rated']
        flow_factor = self.flow / self.LIMITS['flow_rated'] if self.LIMITS['flow_rated'] > 0 else 1.0
        
        severity = 1.0
        if speed_factor > 1.0:
            severity *= (1.0 + 0.4 * (speed_factor - 1.0)**2)
        
        # Operating near surge increases severity
        surge_margin = self.surge_model.calculate_surge_margin(self.flow, self.head)
        if surge_margin < 20:
            severity *= (1.0 + 0.3 * (20 - surge_margin) / 20)
            
        return severity
    
    def _update_process(self, health_state: dict):
        """Update process conditions based on operating state and health."""
        if self.speed <= 0:
            self.flow = 0
            self.head = 0
            self.discharge_pressure = self.suction_pressure
            self.discharge_temp = self.suction_temp
            self.power = 0
            return
            
        # Flow is proportional to speed (affinity laws)
        speed_ratio = self.speed / self.LIMITS['speed_rated']
        self.flow = self.design_flow * speed_ratio
        
        # Head is proportional to speed squared (affinity laws)
        impeller_health = health_state.get('impeller', 1.0)
        head_degradation = 0.9 + 0.1 * impeller_health  # Up to 10% head loss
        self.head = self.design_head * (speed_ratio ** 2) * head_degradation
        
        # Calculate discharge conditions
        # Simplified: assume polytropic process
        gamma = 1.3  # Typical for natural gas
        pressure_ratio = 1 + (self.head / (1000 * gamma / (gamma - 1) * (self.suction_temp + 273.15) * 8.314 / 18))
        pressure_ratio = max(1.0, min(pressure_ratio, 5.0))  # Limit ratio
        
        self.discharge_pressure = self.suction_pressure * pressure_ratio
        self.discharge_temp = (self.suction_temp + 273.15) * (pressure_ratio ** ((gamma - 1) / gamma)) - 273.15
        
        # Efficiency degrades with impeller condition
        self.efficiency = 0.75 + 0.10 * impeller_health
        
        # Power calculation (simplified)
        mass_flow = self.flow * 0.8  # Approximate mass flow (kg/hr assuming ~0.8 kg/m³)
        self.power = (mass_flow * self.head) / (3600 * self.efficiency)  # kW
        
    def _update_bearings(self, health_state: dict):
        """Update bearing temperatures based on load and health."""
        if self.speed <= 0:
            ambient = 35.0
            self.bearing_temp_de = self._approach(self.bearing_temp_de, ambient, 0.02)
            self.bearing_temp_nde = self._approach(self.bearing_temp_nde, ambient, 0.02)
            self.thrust_bearing_temp = self._approach(self.thrust_bearing_temp, ambient, 0.02)
            return
            
        bearing_health = health_state.get('bearing', 1.0)
        load_factor = self.speed / self.LIMITS['speed_rated']
        
        # Base temperature rise from load
        base_temp = 45 + 30 * load_factor
        
        # Friction heating from degraded bearings
        friction_penalty = (1.0 - bearing_health) * 40
        
        target_de = base_temp + friction_penalty + random.gauss(0, 1)
        target_nde = base_temp + friction_penalty * 0.8 + random.gauss(0, 1)
        target_thrust = base_temp + friction_penalty * 1.2 + random.gauss(0, 1)
        
        self.bearing_temp_de = self._approach(self.bearing_temp_de, target_de, 0.1)
        self.bearing_temp_nde = self._approach(self.bearing_temp_nde, target_nde, 0.1)
        self.thrust_bearing_temp = self._approach(self.thrust_bearing_temp, target_thrust, 0.1)
        
    def _approach(self, current: float, target: float, rate: float) -> float:
        """Exponential approach to target."""
        return current + (target - current) * rate
    
    def _add_noise(self, value: float, magnitude: float) -> float:
        """Add measurement noise."""
        return value + random.gauss(0, magnitude)
        
    def next_state(self) -> dict:
        """
        Advance simulation by one time step.
        
        Conditionally applies enhancements based on initialization flags.
        
        Returns:
            dict: Current telemetry values
            
        Raises:
            Exception: With failure code on critical failure
        """
        # Update speed toward target
        speed_rate = 0.08 if self.speed_target > self.speed else 0.12
        self.speed = self._approach(self.speed, self.speed_target, speed_rate)
        
        # 1. Apply environmental conditions if enabled
        if self.use_environmental:
            try:
                env_cond = self.env_model.get_conditions(self.elapsed_hours)
                self.suction_temp = env_cond.get('ambient_temp_C', self.suction_temp)
                self.suction_pressure = env_cond.get('pressure_kPa', self.suction_pressure)
            except:
                pass
        
        # 2. Calculate operating severity
        severity = self._calculate_operating_severity()
        
        # 3. Apply thermal transients if enabled
        thermal_multiplier = 1.0
        if self.use_thermal_model:
            try:
                thermal_state = self.thermal_model.step(
                    target_speed=self.speed_target,
                    rated_speed=self.LIMITS['speed_max'],
                    timestep_minutes=1/60
                )
                thermal_multiplier = thermal_state.get('degradation_multiplier', 1.0)
                severity *= thermal_multiplier
            except:
                pass
        
        # 4. Check for incipient faults if enabled
        if self.use_faults:
            try:
                fault_event = self.fault_sim.check_fault_initiation(
                    operating_hours_increment=1/3600,
                    stress_factor=severity,
                    timestamp=self.current_timestamp,
                    operating_hours=self.operating_hours,
                    component_list=['impeller', 'bearing', 'seal_primary', 'seal_secondary']
                )
                self.fault_sim.propagate_faults(1/3600, severity)
            except:
                pass
        
        # 5. Check for process upsets if enabled
        if self.use_upsets:
            try:
                upset_event = self.upset_sim.check_upset_initiation(
                    timestep_seconds=1,
                    timestamp=self.current_timestamp,
                    operating_state={'speed': self.speed, 'flow': self.flow}
                )
            except:
                pass
        
        # 6. Advance health model
        health_state = self.health_model.step(severity)
        
        # 7. Adjust health for active faults if enabled
        if self.use_faults:
            try:
                health_state = self.fault_sim.adjust_health_for_faults(health_state)
            except:
                pass
        
        # 8. Apply upset damage if enabled
        if self.use_upsets:
            try:
                if self.upset_sim.active_upset:
                    health_state = self.upset_sim.calculate_upset_damage(health_state)
            except:
                pass
        
        # 9. Update process conditions
        self._update_process(health_state)
        
        # Check surge
        surge_margin = self.surge_model.calculate_surge_margin(self.flow, self.head)
        if self.surge_model.is_surge_trip(surge_margin) and self.speed > 0:
            raise Exception("F_SURGE")
            
        # Update seal condition (convert time step to hours)
        seal_state = self.seal_model.step(severity / 3600)
        
        # Update bearings
        self._update_bearings(health_state)
        
        # Check bearing temperature trip
        if max(self.bearing_temp_de, self.bearing_temp_nde, self.thrust_bearing_temp) > self.LIMITS['bearing_temp_max']:
            raise Exception("F_BEARING_TEMP")
        
        # 10. Generate vibration metrics
        # Always use orbit model for displacement (mm) - used for API 617 trip checks
        x_disp, y_disp = self.orbit_model.generate_orbit(self.speed, health_state)
        orbit_metrics = self.orbit_model.compute_metrics(x_disp, y_disp)
        orbit_amplitude = orbit_metrics['orbit_amplitude']
        sync_amplitude = orbit_metrics['sync_amplitude']

        # Optionally get enhanced velocity metrics for ML features
        vib_rms = None
        vib_peak = None
        if self.use_enhanced_vibration:
            try:
                vib_signal, vib_metrics = self.vib_generator_enhanced.generate_bearing_vibration(
                    rpm=self.speed,
                    bearing_health=health_state.get('bearing', 1.0),
                    duration=1.0
                )
                vib_rms = vib_metrics.get('rms', 0)
                vib_peak = vib_metrics.get('peak', 0)
            except:
                pass

        # Check vibration trip - orbit_amplitude is displacement in mm (API 617)
        if orbit_amplitude > self.LIMITS['vibration_trip'] * 2:
            raise Exception("F_HIGH_VIBRATION")
        
        # 11. Check maintenance required if enabled
        if self.use_maintenance:
            try:
                maint_type = self.maint_scheduler.check_maintenance_required(
                    operating_hours=self.operating_hours,
                    health_state=health_state,
                    is_planned_shutdown=(self.speed == 0)
                )
                
                if maint_type:
                    maint_action = self.maint_scheduler.perform_maintenance(
                        maint_type,
                        current_health=health_state,
                        operating_hours=self.operating_hours,
                        timestamp=self.current_timestamp
                    )
                    self.health_model.health = maint_action.health_after
                    health_state = maint_action.health_after
            except:
                pass
        
        # Update operating hours
        if self.speed > 0:
            self.operating_hours += 1/3600
        
        self.t += 1
        if hasattr(self, 'elapsed_hours'):
            self.elapsed_hours += 1/3600
        if hasattr(self, 'current_timestamp'):
            self.current_timestamp += timedelta(seconds=1)
        
        # Build telemetry message
        state = {
            'speed': round(self._add_noise(self.speed, 10), 2),
            'speed_target': round(self.speed_target, 2),
            'flow': round(self._add_noise(self.flow, 5), 2),
            'head': round(self._add_noise(self.head, 20), 2),
            'suction_pressure': round(self._add_noise(self.suction_pressure, 5), 2),
            'discharge_pressure': round(self._add_noise(self.discharge_pressure, 10), 2),
            'suction_temp': round(self._add_noise(self.suction_temp, 0.2), 2),
            'discharge_temp': round(self._add_noise(self.discharge_temp, 0.5), 2),
            'surge_margin': round(surge_margin, 2),
            'surge_alarm': self.surge_model.is_surge_alarm(surge_margin),
            'bearing_temp_de': round(self._add_noise(self.bearing_temp_de, 0.3), 2),
            'bearing_temp_nde': round(self._add_noise(self.bearing_temp_nde, 0.3), 2),
            'thrust_bearing_temp': round(self._add_noise(self.thrust_bearing_temp, 0.3), 2),
            'shaft_x_displacement': round(orbit_amplitude / 2, 4),
            'shaft_y_displacement': round(orbit_amplitude / 2 * 0.95, 4),
            'orbit_amplitude': round(orbit_amplitude, 4),
            'sync_amplitude': round(sync_amplitude, 4),
            'primary_seal_leakage': round(seal_state['primary_leakage'], 3),
            'secondary_seal_leakage': round(seal_state['secondary_leakage'], 3),
            'efficiency': round(self.efficiency, 4),
            'power': round(self._add_noise(self.power, 5), 2),
            'operating_hours': round(self.operating_hours, 2),
            'health_impeller': round(health_state['impeller'], 4),
            'health_bearing': round(health_state['bearing'], 4),
            'health_seal_primary': round(seal_state['primary_health'], 4),
            'health_seal_secondary': round(seal_state['secondary_health'], 4),
        }

        # Add enhanced velocity metrics if available (for ML features)
        if vib_rms is not None:
            state['vibration_rms'] = round(vib_rms, 4)
        if vib_peak is not None:
            state['vibration_peak'] = round(vib_peak, 4)

        # Add fault information if enabled
        if self.use_faults:
            try:
                fault_summary = self.fault_sim.get_active_fault_summary()
                state['num_active_faults'] = fault_summary.get('num_active_faults', 0)
                state['total_faults_initiated'] = fault_summary.get('total_initiated', 0)
            except:
                pass
        
        # Add upset information if enabled
        if self.use_upsets:
            try:
                state['upset_active'] = self.upset_sim.active_upset is not None
                if self.upset_sim.active_upset:
                    state['upset_type'] = self.upset_sim.active_upset.upset_type.value
                    state['upset_severity'] = self.upset_sim.active_upset.severity
            except:
                pass
        
        # Format output if formatter is available
        if self.use_output_formatter:
            try:
                state = self.output_formatter.format_record(state, self.current_timestamp)
            except:
                pass
        
        return state


def generate_compressor_dataset(
    n_machines: int = 5,
    n_cycles_per_machine: int = 100,
    cycle_duration_range: tuple = (120, 600),
    random_seed: int = None
) -> tuple:
    """
    Generate run-to-failure dataset for multiple centrifugal compressors.
    
    Args:
        n_machines: Number of compressors to simulate
        n_cycles_per_machine: Operating cycles per compressor
        cycle_duration_range: (min, max) seconds per cycle
        random_seed: Seed for reproducibility
        
    Returns:
        tuple: (telemetry_records, failure_records)
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    telemetry = []
    failures = []
    
    for m in range(n_machines):
        machine_id = f"CC-{m+1:03d}"
        
        initial_health = {
            'impeller': random.uniform(0.75, 0.98),
            'bearing': random.uniform(0.70, 0.95)
        }
        
        compressor = CentrifugalCompressor(
            machine_id,
            initial_health,
            design_flow=random.uniform(1200, 1800),
            design_head=random.uniform(7000, 9000)
        )
        timestamp = datetime.now()
        
        try:
            for cycle in range(n_cycles_per_machine):
                duration = random.randint(*cycle_duration_range)
                target_speed = random.uniform(9000, 15000)
                
                compressor.set_speed(target_speed)
                
                for _ in range(duration):
                    state = compressor.next_state()
                    state['timestamp'] = timestamp.isoformat()
                    state['machineID'] = machine_id
                    state['cycle'] = cycle
                    telemetry.append(state)
                    timestamp = timestamp + timedelta(seconds=1)
                    
                # Shutdown period
                compressor.set_speed(0)
                for _ in range(60):
                    state = compressor.next_state()
                    state['timestamp'] = timestamp.isoformat()
                    state['machineID'] = machine_id
                    state['cycle'] = cycle
                    telemetry.append(state)
                    
        except Exception as e:
            failures.append({
                'timestamp': timestamp.isoformat(),
                'machineID': machine_id,
                'level': 'CRITICAL',
                'code': str(e),
                'message': CentrifugalCompressorHealthModel.FAILURE_MODES.get(
                    str(e), 'Unknown failure'
                )
            })
            
    return telemetry, failures


if __name__ == '__main__':
    print("Centrifugal Compressor Simulator - Example Run")
    
    cc = CentrifugalCompressor(
        name="CC-001",
        initial_health={
            'impeller': 0.88,
            'bearing': 0.82
        },
        design_flow=1500,
        design_head=8000
    )
    
    print("\nStarting compressor to 12000 RPM...")
    cc.set_speed(12000)
    
    for i in range(10):
        try:
            state = cc.next_state()
            print(f"t={i}: Speed={state['speed']:.0f} RPM, "
                  f"Flow={state['flow']:.0f} m³/hr, "
                  f"Surge Margin={state['surge_margin']:.1f}%, "
                  f"Orbit={state['orbit_amplitude']:.4f} mm")
        except Exception as e:
            print(f"FAILURE: {e}")
            break
            
    print("\nDone.")
