"""
Gas Turbine Data Simulator

This module simulates an industrial gas turbine typical of offshore platforms and 
LNG facilities. 
Key Features:
- Multi-mode degradation
- Realistic parameter ranges based on industrial standards
- Physics-inspired degradation trajectories using exponential wear models
- Thermodynamic performance monitoring simulation
- Vibration signature generation with fault-specific harmonics

Reference: API 670, Bently Nevada monitoring standards, industry operational data
"""

import numpy as np
import random
import math
from datetime import datetime, timedelta
from typing import Optional
try:
    from .physics.vibration_enhanced import EnhancedVibrationGenerator, BearingGeometry
    from .physics.thermal_transient import ThermalTransientModel, ThermalMassProperties
    from .physics.environmental_conditions import EnvironmentalConditions, LocationType
    from .simulation.maintenance_events import MaintenanceScheduler
    from .simulation.incipient_faults import IncipientFaultSimulator
    from .simulation.process_upsets import ProcessUpsetSimulator
    from .ml_utils.ml_output_modes import DataOutputFormatter, OutputMode
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    LocationType = None
    OutputMode = None

class GasTurbineHealthModel:
    """
    Manages multiple degradation pathways for gas turbine components.
    Uses the generalized wear equation: h(t) = 1 - d - exp(a * t^b)
    Each failure mode has independent degradation trajectories.
    """
    
    # Failure mode codes
    FAILURE_MODES = {
        'F_HGP': 'Hot Gas Path Degradation - Combustion liner cracking',
        'F_BLADE_COMPRESSOR': 'Compressor Blade Fouling/Erosion - Discharge temp loss',
        'F_BLADE_TURBINE': 'Turbine Blade Rub/Erosion - Tip clearance increase',
        'F_BEARING': 'Bearing Failure - Lubrication/mechanical degradation',
        'F_FUEL': 'Fuel System Fouling - Nozzle blockage',
        'F_COMPRESSOR_FOULING': 'Compressor Fouling - Airborne deposit buildup',
        'F_VIB_TRIP': 'High Vibration Trip - Vibration exceeded trip limit',
    }

    def __init__(self,
                 initial_health: dict = None,
                 degradation_params: dict = None):
        """
        Initialize health model for all degradation pathways.

        Args:
            initial_health: Dict with keys 'hgp', 'blade_compressor', 'blade_turbine',
                           'bearing', 'fuel', 'compressor_fouling'
                           Values between 0.0 (failed) and 1.0 (new)
            degradation_params: Dict of (d, a, b) tuples per mode
        """
        # Default initial health states (slightly degraded from new)
        self.health = initial_health or {
            'hgp': 0.92,
            'blade_compressor': 0.95,
            'blade_turbine': 0.95,
            'bearing': 0.90,
            'fuel': 0.93,
            'compressor_fouling': 0.98,
        }

        # Default degradation parameters (d, a, b) for h(t) = 1 - d - exp(a*t^b)
        # Different rates reflect real-world component lifespans
        self.degradation_params = degradation_params or {
            'hgp':              (0.05, -0.25, 0.22),
            'blade_compressor': (0.03, -0.30, 0.20),
            'blade_turbine':    (0.025, -0.28, 0.19),
            'bearing':          (0.08, -0.35, 0.25),
            'fuel':             (0.04, -0.20, 0.18),
            # compressor_fouling: slower onset; reaches threshold ~3,000-4,000 hrs from mid-life start
            'compressor_fouling':     (0.025, -0.20, 0.18),
        }

        # Failure thresholds - below these, component is considered failed
        self.failure_thresholds = {
            'hgp': 0.45,
            'blade_compressor': 0.40,
            'blade_turbine': 0.40,
            'bearing': 0.35,
            'fuel': 0.40,         
            'compressor_fouling': 0.60,  
        }
        
        # Initialize time-to-failure generators
        self._init_generators()
        
    def _init_generators(self):
        """Initialize health trajectory generators for each mode."""
        self._generators = {}
        for mode, (d, a, b) in self.degradation_params.items():
            current_h = self.health[mode]
            threshold = self.failure_thresholds[mode]

            # Maximum health the exponential formula can represent
            max_formula_health = 1 - d

            if current_h > max_formula_health:
                # Health exceeds formula maximum - use hybrid approach
                # Linear degradation until we reach the formula's range
                self._generators[mode] = self._hybrid_health_generator(
                    current_h, max_formula_health, d, a, b, threshold
                )
            else:
                # Normal case - calculate ttf from inverse health function
                # h = 1 - d - exp(a*t^b) => t = (ln(1-d-h)/a)^(1/b)
                try:
                    ttf = math.pow(math.log(1 - d - current_h) / a, 1 / b)
                except (ValueError, ZeroDivisionError):
                    ttf = 10000  # Large default if calculation fails

                self._generators[mode] = self._health_generator(
                    ttf, d, a, b, threshold
                )
    
    def _health_generator(self, ttf, d, a, b, threshold):
        """
        Generator yielding health values over time until failure.

        Yields:
            tuple: (time_remaining, health_value)
        """
        for t in range(int(ttf), -1, -1):
            h = 1 - d - math.exp(a * t**b)
            if h < threshold:
                break
            yield t, h

    def _hybrid_health_generator(self, initial_health, max_formula_health, d, a, b, threshold):
        """
        Generator for cases where initial health exceeds formula maximum.

        Uses linear degradation until health reaches the formula's valid range,
        then switches to the exponential degradation curve.

        Args:
            initial_health: Starting health value (> max_formula_health)
            max_formula_health: Maximum health the formula can represent (1 - d)
            d, a, b: Degradation curve parameters
            threshold: Failure threshold

        Yields:
            tuple: (time_remaining, health_value)
        """
        # Phase 1: Linear degradation from initial_health to max_formula_health
        # Use a rate that provides smooth transition (calibrated to ~5000 steps)
        linear_rate = 0.00002
        current_health = initial_health
        t_virtual = 100000  # Virtual time counter for phase 1

        while current_health > max_formula_health:
            if current_health < threshold:
                return
            yield t_virtual, current_health
            current_health -= linear_rate * (1.0 + random.gauss(0, 0.05))
            t_virtual -= 1

        # Phase 2: Switch to exponential formula
        # Find the ttf value that corresponds to max_formula_health (minus small epsilon)
        transition_health = max_formula_health - 0.001
        try:
            ttf = math.pow(math.log(1 - d - transition_health) / a, 1 / b)
        except (ValueError, ZeroDivisionError):
            ttf = 10000

        # Continue with exponential degradation
        for t in range(int(ttf), -1, -1):
            h = 1 - d - math.exp(a * t**b)
            if h < threshold:
                break
            yield t, h

    def step(self, operating_severity: float = 1.0):
        """
        Advance health model by one time step.

        Args:
            operating_severity: Multiplier for degradation rate (1.0 = normal)
                               Higher values accelerate degradation

        Returns:
            dict: Current health values for all modes and failure status
        """
        updated_health = {'failed_mode': None}

        for mode, gen in self._generators.items():
            try:
                # Apply severity factor by potentially skipping steps
                if operating_severity > 1.0 and random.random() < (operating_severity - 1.0):
                    next(gen)  # Extra degradation step

                t_remaining, h = next(gen)
                self.health[mode] = h
                updated_health[mode] = h
            except StopIteration:
                # Track which mode failed (if any)
                if updated_health['failed_mode'] is None:
                    updated_health['failed_mode'] = mode
                # Set health to threshold so we have a valid value
                updated_health[mode] = self.failure_thresholds.get(mode, 0.4)
                self.health[mode] = updated_health[mode]

        return updated_health

class VibrationSignalGenerator:
    """
    Generates realistic vibration signals for gas turbine monitoring.
    
    Simulates accelerometer output at bearing housings with fault-specific
    harmonic content based on rotor dynamics and machine condition.
    """
    
    # Standard harmonics for healthy turbine (multiples of shaft frequency)
    BASE_HARMONICS = [1, 2, 3]  # 1x, 2x, 3x shaft speed
    
    # Fault-specific harmonic patterns
    FAULT_SIGNATURES = {
        'unbalance': {'harmonics': [1], 'amplitude_mult': 3.0},
        'misalignment': {'harmonics': [2], 'amplitude_mult': 2.5},
        'blade_rub': {'harmonics': [1, 3, 5, 7], 'amplitude_mult': 2.0},
        'bearing_defect': {'harmonics': [3.5, 7, 10.5], 'amplitude_mult': 1.5}
    }
    
    def __init__(self, sample_rate: int = 1024):
        """
        Initialize vibration signal generator.
        
        Args:
            sample_rate: Samples per second (Hz). 1024 is minimum for
                        detecting bearing defects per Nyquist theorem.
        """
        self.sample_rate = sample_rate
        self._phase = 0.0
        
    def generate(self, 
                 rpm: float, 
                 health_state: dict, 
                 duration: float = 1.0) -> np.ndarray:
        """
        Generate vibration signal based on current operating state.
        
        Args:
            rpm: Current rotor speed in revolutions per minute
            health_state: Dict with health values for degradation modes
            duration: Signal duration in seconds
            
        Returns:
            np.ndarray: Vibration velocity signal in mm/s
        """
        if rpm <= 0:
            return np.zeros(int(self.sample_rate * duration))
            
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Fundamental frequency (Hz)
        f0 = rpm / 60.0
        
        # Start with baseline healthy signal
        signal = np.zeros(n_samples)
        
        # Add base harmonics (healthy machine signature)
        base_amplitudes = [0.3, 0.15, 0.05]  # Typical healthy amplitudes (mm/s)
        for harm, amp in zip(self.BASE_HARMONICS, base_amplitudes):
            signal += amp * np.sin(2 * np.pi * harm * f0 * t + self._phase)
            
        # Add fault signatures based on degraded health
        # Non-linear amplitude growth enables F_VIB_TRIP before health failures

        # Bearing health affects bearing defect signatures
        bearing_health = health_state.get('bearing', 1.0)
        if bearing_health < 0.75:
            bearing_deg = 0.75 - bearing_health
            # Non-linear: at health=0.35, fault_amp = 0.4*4 + 0.16*8 = 2.88 mm/s
            fault_amp = bearing_deg * 4.0 + bearing_deg ** 2 * 8.0
            for harm in self.FAULT_SIGNATURES['bearing_defect']['harmonics']:
                signal += fault_amp * np.sin(2 * np.pi * harm * f0 * t)

        # Turbine blade rub — tip clearance increase produces [1,3,5,7]x harmonics
        # Note: BPFO (~4x) overlaps 3x/5x rub at low RPM; signatures separate above ~7,000 RPM
        # Discriminator: rub is broadband integer harmonics; BPFO is narrowband with sidebands
        blade_turbine_health = health_state.get('blade_turbine', 1.0)
        if blade_turbine_health < 0.90:
            blade_t_deg = 0.90 - blade_turbine_health
            # Non-linear: at health=0.40, fault_amp = 0.5*2.5 + 0.25*5 = 2.5 mm/s
            fault_amp = blade_t_deg * 2.5 + blade_t_deg ** 2 * 5.0
            for harm in self.FAULT_SIGNATURES['blade_rub']['harmonics']:
                signal += fault_amp * np.sin(2 * np.pi * harm * f0 * t)

        # General unbalance from hot gas path degradation
        hgp_health = health_state.get('hgp', 1.0)
        if hgp_health < 0.85:
            hgp_deg = 0.85 - hgp_health
            # Non-linear: at health=0.45, unbal_amp = 0.4*3.5 + 0.16*5 = 2.2 mm/s
            unbal_amp = hgp_deg * 3.5 + hgp_deg ** 2 * 5.0
            signal += unbal_amp * np.sin(2 * np.pi * f0 * t)

        # Add noise floor (instrumentation noise, increases with degradation)
        avg_health = (bearing_health + blade_turbine_health + hgp_health) / 3
        noise_level = 0.05 + 0.15 * (1.0 - avg_health)
        noise = np.random.normal(0, noise_level, n_samples)
        signal += noise
        
        # Update phase for continuity
        self._phase = (self._phase + 2 * np.pi * f0 * duration) % (2 * np.pi)
        
        return signal
    
    def compute_rms(self, signal: np.ndarray) -> float:
        """Compute RMS velocity (mm/s) - primary vibration metric."""
        return np.sqrt(np.mean(signal**2))
    
    def compute_peak(self, signal: np.ndarray) -> float:
        """Compute peak velocity (mm/s)."""
        return np.max(np.abs(signal))

class GasTurbine:
    """
    Industrial Gas Turbine Simulator for Predictive Maintenance.
    
    Simulates a gas turbine typical of offshore platforms and LNG facilities
    with realistic operating parameters, thermodynamic performance, and
    multiple degradation pathways.
    
    Operating Envelope (based on industrial data):
    - Speed: 3,000 - 15,000 RPM (varies by design)
    - Exhaust Gas Temperature: 450 - 600°C
    - Bearing Vibration: 0.5 - 3.0 mm/s (alarm at 2.2 mm/s)
    - Lube Oil Temperature: 90 - 130°C
    - Fuel Flow Rate: 1.5 - 3.5 kg/s
    """
    
    # Operating limits based on API 670 and industry standards
    LIMITS = {
        'speed_min': 3000,       # RPM
        'speed_max': 15000,      # RPM
        'speed_rated': 9500,     # Typical rated speed
        'egt_min': 400,          # °C (idle)
        'egt_max': 620,          # °C (alarm threshold)
        'egt_nominal': 520,      # °C (normal full load)
        'vib_alarm': 2.2,        # mm/s (API 670)
        'vib_trip': 3.0,         # mm/s
        'oil_temp_min': 70,      # °C
        'oil_temp_max': 130,     # °C (alarm)
        'oil_temp_nominal': 95,  # °C
        'fuel_flow_min': 0.5,    # kg/s (idle)
        'fuel_flow_max': 3.8,    # kg/s
    }
    
    def __init__(self,
                 name: str,
                 initial_health: Optional[dict] = None,
                 ambient_temp: float = 25.0,
                 ambient_pressure: float = 101.3,
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
        Initialize gas turbine simulator with optional enhancements.

        Args:
            name: Unique identifier for this turbine
            initial_health: Dict with initial health values per component
            ambient_temp: Ambient temperature in °C
            ambient_pressure: Ambient pressure in kPa
            location_type: Installation location type (synthetic, requires enhancements)
            env_model: Custom environmental data source (real weather API or synthetic)
                      If provided, takes precedence over location_type
            enable_enhanced_vibration: Use envelope-modulated vibration
            enable_thermal_transients: Model startup/shutdown thermal stress
            enable_environmental: Include environmental variability
            enable_maintenance: Enable maintenance events
            enable_incipient_faults: Enable discrete fault initiation
            enable_process_upsets: Enable process upset events
            output_mode: Data output format (requires enhancements)
        """
        self.name = name
        self.ambient_temp = ambient_temp
        self.ambient_pressure = ambient_pressure

        # Initialize health model
        self.health_model = GasTurbineHealthModel(initial_health)

        # Initialize base vibration generator
        self.vib_generator = VibrationSignalGenerator()

        # Operating state
        self.speed = 0.0           # Current RPM
        self.speed_target = 0.0   # Target RPM
        self.egt = ambient_temp   # Exhaust gas temperature
        self.oil_temp = ambient_temp
        self.fuel_flow = 0.0

        # Thermodynamic state
        self.compressor_discharge_temp = ambient_temp
        self.compressor_discharge_pressure = ambient_pressure

        # Time tracking
        self.operating_hours = 0.0
        self.t = 0
        self.elapsed_hours = 0.0
        self.current_timestamp = datetime.now()

        # Performance degradation factor (1.0 = new, <1.0 = degraded)
        self.efficiency = 1.0

        # Initialize enhancements if available and enabled
        self.enhancements_enabled = ENHANCEMENTS_AVAILABLE

        if ENHANCEMENTS_AVAILABLE:
            # Enhanced vibration generator
            if enable_enhanced_vibration:
                try:
                    bearing_geom = BearingGeometry(n_balls=12, ball_diameter=15.0,
                                                  pitch_diameter=75.0, contact_angle=0.0)
                    self.vib_generator_enhanced = EnhancedVibrationGenerator(
                        sample_rate=10240, resonance_freq=2500, bearing_geometry=bearing_geom)
                    self.use_enhanced_vibration = True
                except:
                    self.use_enhanced_vibration = False
            else:
                self.use_enhanced_vibration = False

            # Thermal transient model
            if enable_thermal_transients:
                try:
                    self.thermal_model = ThermalTransientModel(
                        ambient_temp=ambient_temp,
                        thermal_properties=ThermalMassProperties(
                            tau_bearing=8.0, tau_casing=25.0, tau_rotor=45.0))
                    self.use_thermal_model = True
                except:
                    self.use_thermal_model = False
            else:
                self.use_thermal_model = False

            # Environmental conditions
            if enable_environmental:
                try:
                    # Priority: env_model > location_type
                    if env_model is not None:
                        # Use custom environmental source (real weather API or custom synthetic)
                        self.env_model = env_model
                        self.use_environmental = True
                    elif location_type is not None:
                        # Use synthetic location profile
                        self.env_model = EnvironmentalConditions(
                            location_type=location_type, start_day_of_year=1)
                        self.use_environmental = True
                    else:
                        # Neither provided - environmental disabled
                        self.use_environmental = False
                except Exception as e:
                    print(f"Warning: Environmental initialization failed: {e}")
                    self.use_environmental = False
            else:
                self.use_environmental = False

            # Maintenance scheduler
            if enable_maintenance:
                try:
                    self.maint_scheduler = MaintenanceScheduler(
                        enable_time_based=True, enable_condition_based=True,
                        enable_opportunistic=True)
                    self.use_maintenance = True
                except:
                    self.use_maintenance = False
            else:
                self.use_maintenance = False
            self._maintenance_until_hours = 0.0

            # Incipient fault simulator
            if enable_incipient_faults:
                try:
                    self.fault_sim = IncipientFaultSimulator(
                        enable_incipient_faults=True, fault_rate_per_1000hrs=0.3)
                    self.use_faults = True
                except:
                    self.use_faults = False
            else:
                self.use_faults = False

            # Process upset simulator
            if enable_process_upsets:
                try:
                    self.upset_sim = ProcessUpsetSimulator(
                        enable_upsets=True, upset_rate_per_month=1.5)
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
            # Enhancements not available - use base functionality
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
        
    def _calculate_operating_severity(self) -> float:
        """
        Calculate operating severity factor based on current conditions.
        
        High temperatures and speeds accelerate degradation.
        """
        speed_factor = self.speed / self.LIMITS['speed_rated']
        temp_factor = self.egt / self.LIMITS['egt_nominal']
        
        # Severity increases exponentially above rated conditions
        severity = 1.0
        if speed_factor > 1.0:
            severity *= (1.0 + 0.5 * (speed_factor - 1.0)**2)
        if temp_factor > 1.0:
            severity *= (1.0 + 0.3 * (temp_factor - 1.0)**2)
            
        return severity
    
    def _update_thermodynamics(self, health_state: dict):
        """
        Update thermodynamic parameters based on operating conditions and health.
        
        Hot gas path degradation increases EGT for same power output.
        Blade erosion reduces efficiency.
        """
        if self.speed <= 0:
            # Cooldown dynamics
            self.egt = self._approach(self.egt, self.ambient_temp, 0.02)
            self.oil_temp = self._approach(self.oil_temp, self.ambient_temp, 0.01)
            self.fuel_flow = 0.0
            self.compressor_discharge_temp = self.ambient_temp
            self.compressor_discharge_pressure = self.ambient_pressure
            return
            
        # Decouple efficiency losses by component:
        #   HGP           → thermal loss (combustor/TIT degradation)
        #   blade_turbine → aerodynamic loss (tip clearance increase)
        #   compressor_fouling  → compressor aero loss (airborne deposit buildup)
        hgp_health = health_state.get('hgp', 1.0)
        blade_turbine_health = health_state.get('blade_turbine', 1.0)
        compressor_fouling_health = health_state.get('compressor_fouling', 1.0)
        fuel_health = health_state.get('fuel', 1.0)
        thermal_loss = (1.0 - hgp_health) * 0.10
        aero_loss = (1.0 - blade_turbine_health) * 0.05
        fouling_loss = (1.0 - compressor_fouling_health) * 0.08
        fuel_loss = (1.0 - fuel_health) * 0.06   # nozzle fouling reduces combustion efficiency
        self.efficiency = max(0.85, 1.0 - thermal_loss - aero_loss - fouling_loss - fuel_loss)

        # EGT increases with load and degradation
        load_fraction = self.speed / self.LIMITS['speed_rated']
        base_egt = self.LIMITS['egt_min'] + (
            self.LIMITS['egt_nominal'] - self.LIMITS['egt_min']
        ) * load_fraction

        # HGP is dominant EGT driver; turbine blade has minor effect
        # Fouling does NOT raise EGT — key discriminator (deposits are upstream of combustor)
        # Fuel nozzle fouling → uneven combustion → hot spots → elevated EGT
        egt_penalty = (1.0 - hgp_health) * 50 + (1.0 - blade_turbine_health) * 10 + (1.0 - fuel_health) * 20
        target_egt = base_egt + egt_penalty

        self.egt = self._approach(self.egt, target_egt, 0.1)

        # Fuel flow correlates with load and inverse of efficiency
        base_fuel = self.LIMITS['fuel_flow_min'] + (
            self.LIMITS['fuel_flow_max'] - self.LIMITS['fuel_flow_min']
        ) * load_fraction

        self.fuel_flow = base_fuel / (self.efficiency * fuel_health)

        # Oil temperature correlates with load and bearing health
        bearing_health = health_state.get('bearing', 1.0)
        target_oil = self.LIMITS['oil_temp_min'] + (
            self.LIMITS['oil_temp_nominal'] - self.LIMITS['oil_temp_min']
        ) * load_fraction
        target_oil += (1.0 - bearing_health) * 25  # Friction heating

        self.oil_temp = self._approach(self.oil_temp, target_oil, 0.05)

        # Compressor discharge (simplified model)
        pressure_ratio = 10 + 5 * load_fraction  # Typical for industrial GT
        # Fouling reduces pressure ratio (deposits restrict flow area)
        pressure_ratio *= (1.0 - 0.06 * (1.0 - compressor_fouling_health))
        self.compressor_discharge_pressure = self.ambient_pressure * pressure_ratio
        self.compressor_discharge_temp = (self.ambient_temp + 273.15) * (
            pressure_ratio ** 0.286
        ) - 273.15  # Isentropic compression

        # Compressor blade erosion reduces discharge temp (unique blade_compressor signal)
        blade_compressor_health = health_state.get('blade_compressor', 1.0)
        self.compressor_discharge_temp *= (1.0 - 0.04 * (1.0 - blade_compressor_health))
        
    def _approach(self, current: float, target: float, rate: float) -> float:
        """Exponential approach to target value."""
        return current + (target - current) * rate
    
    def _add_noise(self, value: float, magnitude: float) -> float:
        """Add realistic measurement noise."""
        return value + random.gauss(0, magnitude)
        
    def next_state(self) -> dict:
        """
        Advance simulation by one time step and return current state.
        
        Conditionally applies enhancements based on initialization flags.
        
        Returns:
            dict: Current telemetry values
            
        Raises:
            Exception: With failure code when critical failure occurs
        """
        # Enforce maintenance downtime — override speed before any physics
        if self.use_maintenance and self._maintenance_until_hours > 0 and self.operating_hours < self._maintenance_until_hours:
            self.set_speed(0)
            self.speed_target = 0

        # Update speed toward target
        speed_rate = 0.1 if self.speed_target > self.speed else 0.15
        self.speed = self._approach(self.speed, self.speed_target, speed_rate)
        
        # 1. Apply environmental conditions if enabled
        if self.use_environmental:
            try:
                env_cond = self.env_model.get_conditions(self.elapsed_hours, self.current_timestamp)
                self.ambient_temp = env_cond.get('ambient_temp_C', self.ambient_temp)
                self.ambient_pressure = env_cond.get('pressure_kPa', self.ambient_pressure)
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
                    rated_speed=self.LIMITS['speed_rated'],
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
                    component_list=['hgp', 'blade_compressor', 'blade_turbine', 'bearing', 'fuel', 'compressor_fouling']
                )
                # Propagate existing faults
                self.fault_sim.propagate_faults(1/3600, severity)
            except:
                pass
        
        # 5. Check for process upsets if enabled
        if self.use_upsets:
            try:
                upset_event = self.upset_sim.check_upset_initiation(
                    timestep_seconds=1,
                    timestamp=self.current_timestamp,
                    operating_state={'speed': self.speed, 'egt': self.egt}
                )
            except:
                pass
        
        # 6. Advance health model - returns dict with 'failed_mode' key
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

        # 9. Update thermodynamic state
        self._update_thermodynamics(health_state)

        # 10. Generate vibration signal and compute metrics
        bpfo_order = 0.0
        bpfi_order = 0.0
        if self.use_enhanced_vibration:
            try:
                vib_signal, vib_metrics = self.vib_generator_enhanced.generate_bearing_vibration(
                    rpm=self.speed,
                    bearing_health=health_state.get('bearing', 1.0),
                    duration=1.0
                )
                vib_rms = vib_metrics.get('rms', 0)
                vib_peak = vib_metrics.get('peak', 0)
                vib_crest = vib_metrics.get('crest_factor', 0)
                vib_kurtosis = vib_metrics.get('kurtosis', 0)
                # Convert bearing defect frequencies to speed-invariant order ratios
                shaft_freq = self.speed / 60.0
                if shaft_freq > 0:
                    bpfo_order = round(vib_metrics.get('bpfo_freq', 0) / shaft_freq, 3)
                    bpfi_order = round(vib_metrics.get('bpfi_freq', 0) / shaft_freq, 3)
            except:
                # Fallback to base vibration generator
                vib_signal = self.vib_generator.generate(
                    self.speed, health_state, duration=1.0
                )
                vib_rms = self.vib_generator.compute_rms(vib_signal)
                vib_peak = self.vib_generator.compute_peak(vib_signal)
                vib_crest = 0
                vib_kurtosis = 0
        else:
            vib_signal = self.vib_generator.generate(
                self.speed, health_state, duration=1.0
            )
            vib_rms = self.vib_generator.compute_rms(vib_signal)
            vib_peak = self.vib_generator.compute_peak(vib_signal)
            vib_crest = 0
            vib_kurtosis = 0

        # Check component health failures first (health-based)
        if health_state.get('failed_mode'):
            raise Exception(f"F_{health_state['failed_mode'].upper()}")

        # Vibration trip (process-based, secondary — caught by equipment_sim as non-recordable)
        if vib_rms > self.LIMITS['vib_trip']:
            raise Exception("F_HIGH_VIBRATION")
        
        # 11. Check maintenance required if enabled
        if self.use_maintenance:
            if self._maintenance_until_hours > 0 and self.operating_hours < self._maintenance_until_hours:
                self.set_speed(0)
            else:
                self._maintenance_until_hours = 0.0
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
                        self._maintenance_until_hours = self.operating_hours + maint_action.duration_hours
                        self.set_speed(0)
                except Exception:
                    pass
        
        self.t += 1
        
        # Build telemetry message
        state = {
            'speed': round(self._add_noise(self.speed, 5), 2),
            'speed_target': round(self.speed_target, 2),
            'exhaust_gas_temp': round(self._add_noise(self.egt, 2), 2),
            'oil_temp': round(self._add_noise(self.oil_temp, 0.5), 2),
            'fuel_flow': round(self._add_noise(self.fuel_flow, 0.05), 3),
            'vibration_rms': round(vib_rms, 3),
            'vibration_peak': round(vib_peak, 3),
            'compressor_discharge_temp': round(
                self._add_noise(self.compressor_discharge_temp, 1), 2
            ),
            'compressor_discharge_pressure': round(
                self._add_noise(self.compressor_discharge_pressure, 5), 2
            ),
            'ambient_temp': round(self._add_noise(self.ambient_temp, 0.1), 2),
            'ambient_pressure': round(self._add_noise(self.ambient_pressure, 0.1), 2),
            'efficiency': round(self.efficiency, 4),
            'operating_hours': round(self.operating_hours, 2),
            'health_hgp': round(health_state['hgp'], 4),
            'health_blade_compressor': round(health_state.get('blade_compressor', 0.95), 4),
            'health_blade_turbine': round(health_state.get('blade_turbine', 0.95), 4),
            'health_bearing': round(health_state['bearing'], 4),
            'health_fuel': round(health_state['fuel'], 4),
            'health_compressor_fouling': round(health_state.get('compressor_fouling', 0.98), 4),
        }

        # Add enhanced vibration metrics if available
        if self.use_enhanced_vibration and vib_crest > 0:
            state['vibration_crest_factor'] = round(vib_crest, 3)
            state['vibration_kurtosis'] = round(vib_kurtosis, 3)
            if bpfo_order > 0:
                state['bpfo_order'] = bpfo_order
                state['bpfi_order'] = bpfi_order
        
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

# Convenience function for batch data generation
def generate_turbine_dataset(
    n_machines: int = 5,
    n_cycles_per_machine: int = 100,
    cycle_duration_range: tuple = (60, 300),
    random_seed: int = None
) -> tuple:
    """
    Generate run-to-failure dataset for multiple gas turbines.
    
    Args:
        n_machines: Number of turbines to simulate
        n_cycles_per_machine: Operating cycles per turbine
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
        machine_id = f"GT-{m+1:03d}"
        
        # Random initial health to get varied failure times
        initial_health = {
            'hgp': random.uniform(0.75, 0.98),
            'blade_compressor': random.uniform(0.80, 0.98),
            'blade_turbine': random.uniform(0.80, 0.98),
            'bearing': random.uniform(0.70, 0.95),
            'fuel': random.uniform(0.75, 0.98),
            'compressor_fouling': random.uniform(0.85, 0.99),
        }
        
        turbine = GasTurbine(machine_id, initial_health)
        timestamp = datetime.now()
        
        try:
            for cycle in range(n_cycles_per_machine):
                duration = random.randint(*cycle_duration_range)
                target_speed = random.uniform(7000, 12000)
                
                turbine.set_speed(target_speed)
                
                for _ in range(duration):
                    state = turbine.next_state()
                    state['timestamp'] = timestamp.isoformat()
                    state['machineID'] = machine_id
                    state['cycle'] = cycle
                    telemetry.append(state)
                    timestamp = timestamp + timedelta(seconds=1)
                    
                # Cooldown period
                turbine.set_speed(0)
                for _ in range(30):
                    state = turbine.next_state()
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
                'message': GasTurbineHealthModel.FAILURE_MODES.get(
                    str(e), 'Unknown failure'
                )
            })
            
    return telemetry, failures

if __name__ == '__main__':
    # Example usage
    print("Gas Turbine Simulator")
    
    # Create a turbine with slightly degraded initial state
    gt = GasTurbine(
        name="GT-001",
        initial_health={
            'hgp': 0.85,
            'blade_compressor': 0.90,
            'blade_turbine': 0.90,
            'bearing': 0.80,
            'fuel': 0.88,
            'compressor_fouling': 0.95,
        }
    )
    
    # Run a short simulation
    print("\nStarting turbine to 9500 RPM...")
    gt.set_speed(9500)
    
    for i in range(10):
        try:
            state = gt.next_state()
            print(f"t={i}: Speed={state['speed']:.0f} RPM, "
                  f"EGT={state['exhaust_gas_temp']:.1f}°C, "
                  f"Vib={state['vibration_rms']:.3f} mm/s, "
                  f"Health(bearing)={state['health_bearing']:.3f}")
        except Exception as e:
            print(f"FAILURE: {e}")
            break
            
    print("\nDone.")