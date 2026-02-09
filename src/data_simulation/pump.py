"""
Pump Data Simulator

This module simulates industrial pumps typical of offshore platforms,
refineries, and process facilities. Pumps are the most numerous rotating equipment
in oil & gas operations, critical for production continuity and safety.

Key Features:
- Cavitation detection through NPSH monitoring and acoustic signatures
- Seal health tracking (mechanical seals, gland packing)
- Bearing degradation with temperature and vibration correlation
- Hydraulic performance modeling (BEP, efficiency curves)
- Motor current signature for electrical/mechanical anomaly detection

Reference: API 610, Hydraulic Institute Standards, Bently Nevada guidelines
"""

import numpy as np
import random
import math
from datetime import datetime, timedelta

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


class CavitationModel:
    """
    Models cavitation phenomena in pumps.
    
    Cavitation occurs when Net Positive Suction Head available (NPSHa) drops below NPSH required (NPSHr),
    causing vapor bubbles to form and collapse. This is detected through:
    - NPSH margin monitoring
    - High-frequency acoustic emissions (15-100 kHz)
    - Characteristic vibration patterns
    """
    
    SEVERITY_NONE = 0
    SEVERITY_INCIPIENT = 1
    SEVERITY_MODERATE = 2
    SEVERITY_SEVERE = 3
    
    def __init__(self, 
                 npsh_required: float = 3.0,
                 npsh_margin_alarm: float = 1.0,
                 npsh_margin_trip: float = 0.3):
        """
        Initialize cavitation model.
        
        Args:
            npsh_required: Base NPSHr at BEP (meters)
            npsh_margin_alarm: Margin for alarm (meters)
            npsh_margin_trip: Margin for trip (meters)
        """
        self.npsh_required = npsh_required
        self.npsh_margin_alarm = npsh_margin_alarm
        self.npsh_margin_trip = npsh_margin_trip
        
    def calculate_npsh_required(self,
                                 flow_ratio: float,
                                 speed_ratio: float = 1.0,
                                 impeller_health: float = 1.0) -> float:
        """Calculate NPSHr based on operating point and impeller health.

        Degraded impellers have rougher surfaces and altered blade geometry,
        increasing turbulence and NPSHr. This can trigger cavitation trips.
        """
        base_npsh = self.npsh_required
        flow_factor = 1.0 + 0.5 * (flow_ratio - 1.0)**2
        speed_factor = speed_ratio ** 2

        # Impeller degradation increases NPSH required
        # At health=1.0: factor=1.0, at health=0.40: factor=2.16, at health=0.35: factor=2.31
        impeller_factor = 1.0 + 2.5 * (1.0 - impeller_health) ** 1.5

        return base_npsh * flow_factor * speed_factor * impeller_factor
    
    def calculate_margin(self, npsh_available: float, npsh_required: float) -> float:
        """Calculate NPSH margin (NPSHa - NPSHr)."""
        return npsh_available - npsh_required
    
    def get_severity(self, margin: float) -> int:
        """Determine cavitation severity from NPSH margin."""
        if margin > self.npsh_margin_alarm:
            return self.SEVERITY_NONE
        elif margin > self.npsh_margin_trip:
            return self.SEVERITY_INCIPIENT
        elif margin > 0:
            return self.SEVERITY_MODERATE
        else:
            return self.SEVERITY_SEVERE
            
    def is_trip_condition(self, margin: float) -> bool:
        """Check if cavitation requires trip."""
        return margin < self.npsh_margin_trip


class MechanicalSealModel:
    """
    Models mechanical seal health and leakage in pumps.
    """
    
    def __init__(self,
                 initial_health: float = 0.95,
                 degradation_rate: float = 0.000015):
        """
        Initialize mechanical seal model.
        
        Args:
            initial_health: Initial seal condition (0-1)
            degradation_rate: Base degradation per step
        """
        self.health = initial_health
        self.degradation_rate = degradation_rate
        self.failure_threshold = 0.40
        self.base_leakage = 0.5
        
    def step(self,
             operating_severity: float = 1.0,
             temperature_factor: float = 1.0,
             contamination_factor: float = 1.0) -> dict:
        """
        Advance seal degradation by one hour.

        Returns:
            dict: Seal health, leakage rate, and failure status
        """
        effective_rate = (self.degradation_rate *
                         operating_severity *
                         temperature_factor *
                         contamination_factor)

        self.health -= effective_rate

        leakage_factor = math.exp(2 * (1 - self.health))
        leakage = self.base_leakage * leakage_factor

        return {
            'seal_health': self.health,
            'leakage_rate': leakage,
            'failed': self.health < self.failure_threshold
        }


class PumpBearingModel:
    """
    Models pump bearing health with temperature and vibration correlation.
    """
    
    BEARING_TYPES = ['drive_end', 'non_drive_end']
    
    DEFECT_FREQS = {
        'outer_race': 3.58,
        'inner_race': 5.42,
        'ball': 2.37,
        'cage': 0.42
    }
    
    def __init__(self, initial_health: dict = None):
        """
        Initialize bearing model.
        
        Args:
            initial_health: Dict with 'drive_end' and 'non_drive_end' health
        """
        self.health = initial_health or {
            'drive_end': 0.92,
            'non_drive_end': 0.94
        }
        
        self.degradation_rates = {
            'drive_end': 0.00006,
            'non_drive_end': 0.00004
        }
        
        self.failure_threshold = 0.28
        self.ambient_temp = 35.0
        self.current_temps = {
            'drive_end': self.ambient_temp,
            'non_drive_end': self.ambient_temp
        }

        # NDE-specific stochastic failure mechanisms
        # Lubrication starvation (Ornstein-Uhlenbeck process):
        # NDE is farther from oil supply, so lube effectiveness drifts over time
        self.nde_lube_effectiveness = 1.0   # 1.0 = fully lubricated, 0.3 = starved
        self._lube_mean = 0.92              # Long-run mean (slight impairment)
        self._lube_reversion_rate = 0.002   # Mean-reversion speed per step
        self._lube_volatility = 0.003       # Noise magnitude per step
        # Contamination and thrust factors are computed per-step from external inputs
        self.nde_contamination_factor = 1.0
        self.nde_thrust_factor = 1.0
        
    def step(self,
             rpm: float,
             load_factor: float = 1.0,
             lubrication_factor: float = 1.0,
             seal_health: float = 1.0,
             impeller_health: float = 1.0) -> dict:
        """
        Advance bearing degradation.

        Args:
            rpm: Shaft speed
            load_factor: Hydraulic load relative to design
            lubrication_factor: Global lubrication quality
            seal_health: Mechanical seal health (0-1), affects NDE contamination
            impeller_health: Impeller health (0-1), affects NDE axial thrust

        Returns:
            dict: Bearing health, temperature values, and failure status
        """
        result = {'failed_bearing': None}
        speed_factor = (rpm / 3000) ** 0.5 if rpm > 0 else 0

        # Update NDE-specific stochastic factors
        # 1. Lubrication starvation (Ornstein-Uhlenbeck process)
        if rpm > 0:
            noise = random.gauss(0, self._lube_volatility)
            self.nde_lube_effectiveness += (
                self._lube_reversion_rate * (self._lube_mean - self.nde_lube_effectiveness)
                + noise
            )
            self.nde_lube_effectiveness = max(0.3, min(1.0, self.nde_lube_effectiveness))

        # 2. Contamination from seal leakage (NDE closer to process seal)
        self.nde_contamination_factor = 1.0 + 1.5 * (1.0 - seal_health) ** 1.3

        # 3. Axial thrust shift from impeller wear (loads NDE more)
        self.nde_thrust_factor = 1.0 + 1.5 * (1.0 - impeller_health) ** 1.5

        for bearing in self.BEARING_TYPES:
            rate = self.degradation_rates[bearing]

            if bearing == 'non_drive_end':
                # NDE gets per-bearing stochastic multipliers
                bearing_load = load_factor * self.nde_thrust_factor
                bearing_lube = lubrication_factor * (2.0 - self.nde_lube_effectiveness)
                bearing_contam = self.nde_contamination_factor
                effective_rate = rate * speed_factor * bearing_load * bearing_lube * bearing_contam
            else:
                effective_rate = rate * speed_factor * load_factor * lubrication_factor

            self.health[bearing] -= effective_rate

            # Track which bearing failed (if any)
            if self.health[bearing] < self.failure_threshold and result['failed_bearing'] is None:
                result['failed_bearing'] = bearing

            if rpm > 0:
                base_temp = self.ambient_temp + 20 * speed_factor
                degradation = 1.0 - self.health[bearing]
                friction_heat = (35 * degradation + 55 * degradation ** 2) * speed_factor

                # Cross-heating: degraded adjacent bearing adds heat via shared housing
                other = 'non_drive_end' if bearing == 'drive_end' else 'drive_end'
                other_degradation = 1.0 - self.health[other]
                cross_heat = 40 * other_degradation * speed_factor

                load_heat = (load_factor - 1.0) * 15 if load_factor > 1 else 0

                if bearing == 'non_drive_end':
                    # NDE-specific heating from stochastic factors
                    lube_heat = 15 * (1.0 - self.nde_lube_effectiveness) * speed_factor
                    contam_heat = 10 * (self.nde_contamination_factor - 1.0) * speed_factor
                    thrust_heat = 8 * (self.nde_thrust_factor - 1.0) * speed_factor
                    target_temp = base_temp + friction_heat + cross_heat + load_heat + lube_heat + contam_heat + thrust_heat
                else:
                    target_temp = base_temp + friction_heat + cross_heat + load_heat
            else:
                target_temp = self.ambient_temp

            # Asymmetric approach: fast heating (friction response), slow cooling (thermal mass)
            if target_temp > self.current_temps[bearing]:
                approach_rate = 0.12
            else:
                approach_rate = 0.04
            self.current_temps[bearing] = self._approach(
                self.current_temps[bearing], target_temp, approach_rate
            )

            result[f'{bearing}_health'] = self.health[bearing]
            result[f'{bearing}_temp'] = self.current_temps[bearing]

        return result
    
    def _approach(self, current: float, target: float, rate: float) -> float:
        """Exponential approach to target."""
        return current + (target - current) * rate
    
    def generate_vibration(self,
                           rpm: float,
                           duration: float = 1.0,
                           sample_rate: int = 10240) -> np.ndarray:
        """
        Generate bearing vibration signal with defect frequencies.
        
        Returns:
            np.ndarray: Vibration velocity signal (mm/s)
        """
        n_samples = int(sample_rate * duration)
        
        if rpm <= 0:
            return np.random.normal(0, 0.05, n_samples)
            
        t = np.linspace(0, duration, n_samples)
        f_shaft = rpm / 60.0
        
        signal = np.zeros(n_samples)
        
        # Base running vibration
        signal += 0.5 * np.sin(2 * np.pi * f_shaft * t)
        signal += 0.2 * np.sin(2 * np.pi * 2 * f_shaft * t)
        
        # Add defect frequencies based on health
        # Vibration increases non-linearly as bearings degrade toward failure
        for bearing in self.BEARING_TYPES:
            health = self.health[bearing]
            degradation = 1.0 - health

            # Outer race defect - appears early, grows with degradation
            # Scaled so combined bearing vibration stays below trip (7.0 mm/s)
            # until bearing health drops below failure threshold (0.28)
            if health < 0.85:
                f_bpfo = self.DEFECT_FREQS['outer_race'] * f_shaft
                amp = degradation * 3.5 + degradation ** 2 * 5.5
                signal += amp * np.sin(2 * np.pi * f_bpfo * t)
                signal += amp * 0.4 * np.sin(2 * np.pi * (f_bpfo - f_shaft) * t)
                signal += amp * 0.4 * np.sin(2 * np.pi * (f_bpfo + f_shaft) * t)

            # Inner race defect - appears later, severe amplitude
            if health < 0.6:
                f_bpfi = self.DEFECT_FREQS['inner_race'] * f_shaft
                inner_deg = 0.6 - health
                amp = inner_deg * 4.5 + inner_deg ** 2 * 9.0
                signal += amp * np.sin(2 * np.pi * f_bpfi * t)

            # Ball/roller defect and broadband noise increase near failure
            if health < 0.45:
                f_ball = self.DEFECT_FREQS['ball'] * f_shaft
                ball_deg = 0.45 - health
                amp = ball_deg * 6.0
                signal += amp * np.sin(2 * np.pi * f_ball * t)
                # Increased broadband noise from surface damage
                signal += np.random.normal(0, ball_deg * 2.0, n_samples)

        signal += np.random.normal(0, 0.15, n_samples)
        
        return signal


class HydraulicPerformanceModel:
    """
    Models pump hydraulic performance including BEP tracking and efficiency.
    """
    
    def __init__(self,
                 design_flow: float = 100.0,
                 design_head: float = 50.0,
                 design_efficiency: float = 0.80,
                 design_speed: float = 3000):
        """
        Initialize hydraulic performance model.
        """
        self.design_flow = design_flow
        self.design_head = design_head
        self.design_efficiency = design_efficiency
        self.design_speed = design_speed
        
        self._a = -design_head / (1.5 * design_flow)**2
        self._b = 0
        self._c = design_head * 1.1
        
    def calculate_head(self, 
                       flow: float, 
                       speed: float,
                       impeller_health: float = 1.0) -> float:
        """Calculate pump head at given flow and speed."""
        speed_ratio = speed / self.design_speed
        flow_at_design_speed = flow / speed_ratio if speed_ratio > 0 else 0
        head_design = self._a * flow_at_design_speed**2 + self._b * flow_at_design_speed + self._c
        head = head_design * speed_ratio**2
        degradation_factor = 0.85 + 0.15 * impeller_health
        head *= degradation_factor
        return max(0, head)
    
    def calculate_efficiency(self,
                              flow: float,
                              speed: float,
                              impeller_health: float = 1.0) -> float:
        """Calculate pump efficiency at operating point."""
        if speed <= 0 or flow <= 0:
            return 0.0
            
        speed_ratio = speed / self.design_speed
        flow_ratio = flow / (self.design_flow * speed_ratio)
        eta = self.design_efficiency * (1 - 0.5 * (flow_ratio - 1.0)**2)
        eta *= (0.9 + 0.1 * impeller_health)
        return max(0.1, min(eta, 0.95))
    
    def calculate_bep_deviation(self, flow: float, speed: float) -> float:
        """Calculate deviation from BEP as percentage."""
        speed_ratio = speed / self.design_speed if self.design_speed > 0 else 0
        bep_flow = self.design_flow * speed_ratio
        if bep_flow <= 0:
            return 100.0
        return abs((flow - bep_flow) / bep_flow) * 100


class Pump:
    """
    Industrial Pump Simulator for Predictive Maintenance.
    
    Operating Envelope (based on API 610):
    - Speed: 1,000 - 6,000 RPM
    - Flow: 10 - 500 m³/hr
    - Head: 20 - 200 m
    - Vibration: 2.5 - 7.0 mm/s (API 610 limits)
    """
    
    LIMITS = {
        'speed_min': 1000,
        'speed_max': 6000,
        'speed_rated': 3000,
        'vibration_alarm': 4.5,
        'vibration_trip': 7.0,
        'bearing_temp_max': 110,
        'seal_leakage_alarm': 10,
        'motor_current_max': 1.15,
    }
    
    def __init__(self,
                 name: str,
                 initial_health: dict = None,
                 design_flow: float = 150.0,
                 design_head: float = 80.0,
                 design_speed: float = 3000,
                 fluid_density: float = 850.0,
                 npsh_available: float = 8.0,
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
        Initialize pump simulator with optional enhancements.

        Args:
            name: Equipment identifier
            initial_health: Dict with initial component health values
            design_flow: Pump design flow rate (m³/hr)
            design_head: Pump design head (m)
            design_speed: Pump design speed (RPM)
            fluid_density: Process fluid density (kg/m³)
            npsh_available: Net positive suction head (m)
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
        self.design_speed = design_speed
        self.fluid_density = fluid_density
        self.npsh_available = npsh_available
        
        # Component models
        self.hydraulic_model = HydraulicPerformanceModel(
            design_flow, design_head, 0.78, design_speed
        )
        self.cavitation_model = CavitationModel(npsh_required=3.5)
        self.seal_model = MechanicalSealModel(
            initial_health=(initial_health or {}).get('seal', 0.93)
        )
        self.bearing_model = PumpBearingModel(
            {
                'drive_end': (initial_health or {}).get('bearing_de', 0.90),
                'non_drive_end': (initial_health or {}).get('bearing_nde', 0.92)
            }
        )
        
        # Impeller health
        self.impeller_health = (initial_health or {}).get('impeller', 0.94)
        self.impeller_degradation_rate = 0.00001
        
        # Operating state
        self.speed = 0.0
        self.speed_target = 0.0
        self.flow = 0.0
        self.head = 0.0
        self.efficiency = 0.0
        self.power = 0.0
        
        # Electrical state - properly size motor for pump power
        # Use actual BEP head from hydraulic model (not design_head which is shutoff-based)
        bep_head = self.hydraulic_model.calculate_head(design_flow, design_speed, 1.0)
        design_power = (fluid_density * 9.81 * design_flow * bep_head) / (0.78 * 3600)
        self.motor_rated_power = design_power * 1.10  # 10% margin
        # Calculate rated current from motor power (480V, 3-phase, 0.85 PF, 0.92 eff)
        self.motor_rated_current = (self.motor_rated_power * 1000) / (math.sqrt(3) * 480 * 0.85 * 0.92)
        self.motor_current = 0.0
        
        # Process conditions
        self.suction_pressure = 200.0
        self.discharge_pressure = 200.0
        self.fluid_temp = 40.0
        
        # Time tracking
        self.operating_hours = 0.0
        self.elapsed_hours = 0.0
        self.current_timestamp = datetime.now()
        self.t = 0
        
        # Enhancement feature flags and modules
        self.use_enhanced_vibration = ENHANCEMENTS_AVAILABLE and enable_enhanced_vibration
        self.use_thermal_model = ENHANCEMENTS_AVAILABLE and enable_thermal_transients
        self.use_environmental = ENHANCEMENTS_AVAILABLE and enable_environmental
        self.use_maintenance = ENHANCEMENTS_AVAILABLE and enable_maintenance
        self.use_faults = ENHANCEMENTS_AVAILABLE and enable_incipient_faults
        self.use_upsets = ENHANCEMENTS_AVAILABLE and enable_process_upsets
        self.use_output_formatter = ENHANCEMENTS_AVAILABLE and output_mode is not None
        
        # Initialize enhancement modules
        self.vibration_generator = None
        self.thermal_model = None
        self.environmental_conditions = None
        self.maintenance_scheduler = None
        self.fault_simulator = None
        self.upset_simulator = None
        self.output_formatter = None
        
        if self.use_enhanced_vibration:
            try:
                self.vibration_generator = EnhancedVibrationGenerator(
                    bearing_type='cylindrical_roller',
                    modulation_type='amplitude'
                )
            except Exception as e:
                self.use_enhanced_vibration = False

        if self.use_thermal_model:
            try:
                self.thermal_model = ThermalTransientModel(
                    material='steel',
                    initial_temp=40.0
                )
            except Exception as e:
                self.use_thermal_model = False

        if self.use_environmental:
            try:
                self.environmental_conditions = EnvironmentalConditions(
                    location_type=location_type,
                    base_temp=40.0,
                    base_pressure=101.325
                )
            except Exception as e:
                self.use_environmental = False

        if self.use_maintenance:
            try:
                self.maintenance_scheduler = MaintenanceScheduler(
                    equipment_type='pump',
                    base_mtbf=40000,
                    base_cbm_threshold=0.85
                )
            except Exception as e:
                self.use_maintenance = False

        if self.use_faults:
            try:
                self.fault_simulator = IncipientFaultSimulator(
                    equipment_type='pump',
                    fault_components=['impeller', 'seal', 'bearing_de', 'bearing_nde'],
                    degradation_rates={'impeller': 0.00003, 'seal': 0.00002,
                                     'bearing_de': 0.00015, 'bearing_nde': 0.00015}
                )
            except Exception as e:
                self.use_faults = False

        if self.use_upsets:
            try:
                self.upset_simulator = ProcessUpsetSimulator(
                    equipment_type='pump',
                    base_frequency=0.1,
                    duration_range=(100, 500)
                )
            except Exception as e:
                self.use_upsets = False

        if self.use_output_formatter:
            try:
                self.output_formatter = DataOutputFormatter(
                    mode=output_mode,
                    include_raw=True,
                    include_derived=True
                )
            except Exception as e:
                self.use_output_formatter = False
        
    def set_speed(self, target_rpm: float):
        """Set target operating speed."""
        self.speed_target = max(0, min(target_rpm, self.LIMITS['speed_max']))
        
    def _calculate_operating_severity(self) -> float:
        """Calculate severity based on operating conditions."""
        severity = 1.0
        
        speed_factor = self.speed / self.LIMITS['speed_rated'] if self.LIMITS['speed_rated'] > 0 else 1.0
        if speed_factor > 1.0:
            severity *= (1.0 + 0.5 * (speed_factor - 1.0)**2)
            
        bep_deviation = self.hydraulic_model.calculate_bep_deviation(self.flow, self.speed)
        if bep_deviation > 20:
            severity *= (1.0 + 0.02 * (bep_deviation - 20))
            
        flow_ratio = self.flow / self.design_flow if self.design_flow > 0 else 0
        speed_ratio = self.speed / self.design_speed if self.design_speed > 0 else 1.0
        npsh_r = self.cavitation_model.calculate_npsh_required(
            flow_ratio, speed_ratio, self.impeller_health
        )
        margin = self.cavitation_model.calculate_margin(self.npsh_available, npsh_r)
        if margin < 2.0:
            severity *= (1.0 + 0.5 * (2.0 - margin))

        return severity
    
    def _update_impeller(self, severity: float) -> tuple:
        """Update impeller health and return (health, failed) tuple."""
        if self.speed > 0:
            self.impeller_health -= self.impeller_degradation_rate * severity

        failed = self.impeller_health < 0.35
        return self.impeller_health, failed
    
    def _update_hydraulics(self):
        """Update hydraulic state based on current operating conditions."""
        if self.speed <= 0:
            self.flow = 0
            self.head = 0
            self.efficiency = 0
            self.power = 0
            self.discharge_pressure = self.suction_pressure
            return
            
        speed_ratio = self.speed / self.design_speed
        self.flow = self.design_flow * speed_ratio * random.uniform(0.85, 1.15)
        
        self.head = self.hydraulic_model.calculate_head(
            self.flow, self.speed, self.impeller_health
        )
        
        self.efficiency = self.hydraulic_model.calculate_efficiency(
            self.flow, self.speed, self.impeller_health
        )
        
        if self.efficiency > 0:
            self.power = (self.fluid_density * 9.81 * self.flow * self.head) / (self.efficiency * 3600)
        else:
            self.power = 0
            
        self.discharge_pressure = self.suction_pressure + (self.fluid_density * 9.81 * self.head / 1000)
        
    def _update_motor(self):
        """Update motor electrical state with health-based current increase.

        Degraded bearings and impellers increase mechanical friction and hydraulic
        losses, requiring more motor current. This can trigger motor overload.
        """
        if self.speed <= 0:
            self.motor_current = 0
            return

        if self.power > 0:
            # Calculate base motor current from power
            # Account for motor efficiency (~0.92) and power factor (~0.85)
            base_current = (self.power * 1000) / (math.sqrt(3) * 480 * 0.85 * 0.92)

            # Bearing degradation increases friction -> higher current
            avg_bearing_health = sum(self.bearing_model.health.values()) / len(self.bearing_model.health)
            bearing_friction_factor = 1.0 + 0.50 * (1.0 - avg_bearing_health) ** 1.5

            # Impeller degradation increases hydraulic losses -> higher current
            impeller_loss_factor = 1.0 + 0.40 * (1.0 - self.impeller_health) ** 1.5

            # Combined effect
            self.motor_current = base_current * bearing_friction_factor * impeller_loss_factor
        else:
            self.motor_current = 0

        # Add small random variation
        self.motor_current += random.uniform(-1, 2)
        
    def _approach(self, current: float, target: float, rate: float) -> float:
        """Exponential approach to target."""
        return current + (target - current) * rate
    
    def _add_noise(self, value: float, magnitude: float) -> float:
        """Add measurement noise."""
        return value + random.gauss(0, magnitude)
        
    def next_state(self) -> dict:
        """
        Advance simulation by one time step with optional physics enhancements.
        
        Returns:
            dict: Current telemetry values
            
        Raises:
            Exception: With failure code on critical failure
        """
        # Apply environmental conditions if enabled
        if self.use_environmental:
            try:
                env_state = self.environmental_conditions.get_state()
                ambient_temp = env_state.get('ambient_temp', 40.0)
                ambient_pressure = env_state.get('ambient_pressure', 101.325)
                corrosion_factor = env_state.get('corrosion_factor', 1.0)
            except Exception as e:
                ambient_temp = 40.0
                ambient_pressure = 101.325
                corrosion_factor = 1.0
        else:
            ambient_temp = 40.0
            ambient_pressure = 101.325
            corrosion_factor = 1.0
        
        # Update speed toward target
        # Fast rate when already running (speed changes are near-instant at 15-min intervals)
        # Slower rate during startup from idle for realistic ramp-up
        if self.speed < self.design_speed * 0.3:
            speed_rate = 0.15 if self.speed_target > self.speed else 0.20
        else:
            speed_rate = 0.70 if self.speed_target > self.speed else 0.80
        self.speed = self._approach(self.speed, self.speed_target, speed_rate)
        
        # Calculate operating severity (base level)
        severity = self._calculate_operating_severity()
        
        # Apply thermal transients if enabled
        thermal_multiplier = 1.0
        if self.use_thermal_model:
            try:
                thermal_state = self.thermal_model.step(
                    ambient_temp=ambient_temp,
                    operating_temp=self.fluid_temp,
                    severity=severity
                )
                thermal_multiplier = thermal_state.get('degradation_multiplier', 1.0)
                severity *= thermal_multiplier
            except Exception as e:
                pass
        
        #Check for incipient fault initiation if enabled
        if self.use_faults:
            try:
                fault_state = self.fault_simulator.step(
                    operating_hours=self.elapsed_hours if hasattr(self, 'elapsed_hours') else self.operating_hours,
                    current_severity=severity
                )
                # Check for new faults and apply degradation
                for component, active in fault_state.get('active_faults', {}).items():
                    if active:
                        if component == 'impeller' and self.impeller_health > 0.35:
                            self.impeller_health -= fault_state.get('propagation_rate', 0.001)
                        elif component == 'seal':
                            self.seal_model._health -= fault_state.get('propagation_rate', 0.001)
                        elif component.startswith('bearing'):
                            bearing_type = 'drive_end' if component == 'bearing_de' else 'non_drive_end'
                            if bearing_type in self.bearing_model.health:
                                self.bearing_model.health[bearing_type] -= fault_state.get('propagation_rate', 0.001)
            except Exception as e:
                pass
        
        # Check for process upsets if enabled
        upset_damage = 1.0
        if self.use_upsets:
            try:
                upset_state = self.upset_simulator.step()
                if upset_state.get('active', False):
                    upset_damage = upset_state.get('damage_multiplier', 1.0)
                    severity *= upset_damage
            except Exception as e:
                pass
        
        # Update impeller (with adjusted severity) - returns (health, failed)
        impeller_health, impeller_failed = self._update_impeller(severity)

        # Update hydraulics
        self._update_hydraulics()

        # Update motor
        self._update_motor()

        # Update seals - returns dict with 'failed' key
        seal_state = self.seal_model.step(severity)

        # Update bearings - returns dict with 'failed_bearing' key
        load_factor = self.flow / self.design_flow if self.design_flow > 0 else 1.0
        bearing_state = self.bearing_model.step(
            self.speed, load_factor,
            seal_health=seal_state.get('seal_health', 1.0),
            impeller_health=impeller_health
        )

        # Generate vibration signal
        if self.use_enhanced_vibration:
            try:
                vib_metrics = self.vibration_generator.generate(
                    speed=self.speed,
                    load=self.flow / self.design_flow if self.design_flow > 0 else 0,
                    degradation=1.0 - impeller_health
                )
                vib_rms = vib_metrics.get('rms', 0.0)
                vib_peak = vib_metrics.get('peak', 0.0)
                vib_enhanced = vib_metrics
            except Exception as e:
                vib_signal = self.bearing_model.generate_vibration(self.speed)
                vib_rms = np.sqrt(np.mean(vib_signal**2))
                vib_peak = np.max(np.abs(vib_signal))
                vib_enhanced = None
        else:
            vib_signal = self.bearing_model.generate_vibration(self.speed)
            vib_rms = np.sqrt(np.mean(vib_signal**2))
            vib_peak = np.max(np.abs(vib_signal))
            vib_enhanced = None

        # Calculate cavitation parameters
        flow_ratio = self.flow / self.design_flow if self.design_flow > 0 else 0
        speed_ratio = self.speed / self.design_speed if self.design_speed > 0 else 1.0
        npsh_r = self.cavitation_model.calculate_npsh_required(
            flow_ratio, speed_ratio, impeller_health
        )
        npsh_margin = self.cavitation_model.calculate_margin(self.npsh_available, npsh_r)
        cav_severity = self.cavitation_model.get_severity(npsh_margin)

        # Calculate bearing temperature
        max_bearing_temp = max(
            bearing_state['drive_end_temp'],
            bearing_state['non_drive_end_temp']
        )


        # Check component health failures (health-based)
        if impeller_failed:
            raise Exception("F_IMPELLER")

        if seal_state.get('failed', False):
            raise Exception("F_SEAL")

        if bearing_state.get('failed_bearing'):
            raise Exception(f"F_BEARING_{bearing_state['failed_bearing'].upper()}")

        # Process-based trips (specific conditions checked before general vibration)
        if max_bearing_temp > self.LIMITS['bearing_temp_max'] and self.speed > self.design_speed * 0.95:
            raise Exception("F_BEARING_OVERTEMP")

        if vib_rms > self.LIMITS['vibration_trip']:
            raise Exception("F_HIGH_VIBRATION")

        if self.motor_current > self.motor_rated_current * self.LIMITS['motor_current_max']:
            raise Exception("F_MOTOR_OVERLOAD")

        if self.cavitation_model.is_trip_condition(npsh_margin) and self.speed > 0:
            raise Exception("F_CAVITATION")

        # Update operating hours
        if self.speed > 0:
            self.operating_hours += 1/3600
            
        # Apply maintenance scheduling if enabled
        if self.use_maintenance:
            try:
                maint_state = self.maintenance_scheduler.step(
                    operating_hours=self.elapsed_hours if hasattr(self, 'elapsed_hours') else self.operating_hours,
                    component_health={
                        'impeller': impeller_health,
                        'seal': seal_state.get('seal_health', 0.93),
                        'bearing_de': bearing_state['drive_end_health'],
                        'bearing_nde': bearing_state['non_drive_end_health']
                    }
                )
                # Apply maintenance if performed
                if maint_state.get('performed', False):
                    for component in maint_state.get('maintained_components', []):
                        if component == 'impeller':
                            self.impeller_health = min(1.0, self.impeller_health + 0.05)
                        elif component == 'seal' and hasattr(self.seal_model, '_health'):
                            self.seal_model._health = min(1.0, self.seal_model._health + 0.05)
                        elif component.startswith('bearing'):
                            bearing_type = 'drive_end' if component == 'bearing_de' else 'non_drive_end'
                            if bearing_type in self.bearing_model.health:
                                self.bearing_model.health[bearing_type] = min(
                                    1.0, self.bearing_model.health[bearing_type] + 0.05
                                )
            except Exception as e:
                pass
        
        # Update time tracking
        if hasattr(self, 'elapsed_hours'):
            self.elapsed_hours += 1/3600
        if hasattr(self, 'current_timestamp'):
            self.current_timestamp += timedelta(seconds=1)
        self.t += 1
        
        # Build telemetry message
        state = {
            'speed': round(self._add_noise(self.speed, 3), 2),
            'speed_target': round(self.speed_target, 2),
            'flow': round(self._add_noise(self.flow, 0.5), 2),
            'head': round(self._add_noise(self.head, 0.3), 2),
            'efficiency': round(self.efficiency, 4),
            'power': round(self._add_noise(self.power, 0.2), 2),
            'suction_pressure': round(self._add_noise(self.suction_pressure, 2), 2),
            'discharge_pressure': round(self._add_noise(self.discharge_pressure, 3), 2),
            'fluid_temp': round(self._add_noise(self.fluid_temp, 0.2), 2),
            'motor_current': round(self._add_noise(self.motor_current, 0.5), 2),
            'motor_current_ratio': round(self.motor_current / self.motor_rated_current, 3),
            'vibration_rms': round(vib_rms, 3),
            'vibration_peak': round(vib_peak, 3),
            'bearing_temp_de': round(self._add_noise(bearing_state['drive_end_temp'], 0.3), 2),
            'bearing_temp_nde': round(self._add_noise(bearing_state['non_drive_end_temp'], 0.3), 2),
            'npsh_available': round(self.npsh_available, 2),
            'npsh_required': round(npsh_r, 2),
            'npsh_margin': round(npsh_margin, 2),
            'cavitation_severity': cav_severity,
            'seal_leakage': round(seal_state['leakage_rate'], 3),
            'bep_deviation': round(self.hydraulic_model.calculate_bep_deviation(self.flow, self.speed), 2),
            'operating_hours': round(self.operating_hours, 2),
            'health_impeller': round(impeller_health, 4),
            'health_seal': round(seal_state['seal_health'], 4),
            'health_bearing_de': round(bearing_state['drive_end_health'], 4),
            'health_bearing_nde': round(bearing_state['non_drive_end_health'], 4),
        }
        
        #Apply output formatting if enabled
        if self.use_output_formatter and vib_enhanced:
            try:
                formatted_state = self.output_formatter.format(state, vib_enhanced)
                state.update(formatted_state)
            except Exception as e:
                pass
        
        return state


class PumpFailureModes:
    """Catalog of pump failure modes for classification tasks."""
    
    FAILURE_MODES = {
        'F_IMPELLER': 'Impeller Degradation - Erosion, corrosion, or damage',
        'F_SEAL': 'Mechanical Seal Failure - Wear, thermal damage, or contamination',
        'F_BEARING_DRIVE_END': 'Drive End Bearing Failure - Fatigue, lubrication, or contamination',
        'F_BEARING_NON_DRIVE_END': 'Non-Drive End Bearing Failure',
        'F_BEARING_OVERTEMP': 'Bearing Overtemperature - Excessive friction or cooling failure',
        'F_HIGH_VIBRATION': 'High Vibration Trip - Mechanical instability',
        'F_CAVITATION': 'Severe Cavitation - NPSH margin critical',
        'F_MOTOR_OVERLOAD': 'Motor Overload - Excessive current draw'
    }


def generate_pump_dataset(
    n_machines: int = 10,
    n_cycles_per_machine: int = 200,
    cycle_duration_range: tuple = (60, 300),
    random_seed: int = None
) -> tuple:
    """
    Generate run-to-failure dataset for multiple pumps.
    
    Args:
        n_machines: Number of pumps to simulate
        n_cycles_per_machine: Operating cycles per pump
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
    
    pump_services = [
        {'name': 'Crude Booster', 'design_flow': 200, 'design_head': 100, 'density': 850},
        {'name': 'Seawater Injection', 'design_flow': 300, 'design_head': 150, 'density': 1025},
        {'name': 'Process Water', 'design_flow': 100, 'design_head': 60, 'density': 1000},
        {'name': 'Methanol Pump', 'design_flow': 50, 'design_head': 80, 'density': 790},
        {'name': 'Fire Water', 'design_flow': 400, 'design_head': 120, 'density': 1000},
    ]
    
    for m in range(n_machines):
        service = pump_services[m % len(pump_services)]
        machine_id = f"CP-{m+1:03d}"
        
        initial_health = {
            'impeller': random.uniform(0.75, 0.98),
            'seal': random.uniform(0.80, 0.98),
            'bearing_de': random.uniform(0.72, 0.95),
            'bearing_nde': random.uniform(0.75, 0.96)
        }
        
        pump = Pump(
            machine_id,
            initial_health,
            design_flow=service['design_flow'] * random.uniform(0.9, 1.1),
            design_head=service['design_head'] * random.uniform(0.9, 1.1),
            design_speed=3000 + random.randint(-500, 500),
            fluid_density=service['density'],
            npsh_available=random.uniform(6, 12)
        )
        timestamp = datetime.now()
        
        try:
            for cycle in range(n_cycles_per_machine):
                duration = random.randint(*cycle_duration_range)
                target_speed = pump.design_speed * random.uniform(0.85, 1.05)
                
                pump.set_speed(target_speed)
                
                for _ in range(duration):
                    state = pump.next_state()
                    state['timestamp'] = timestamp.isoformat()
                    state['machineID'] = machine_id
                    state['service'] = service['name']
                    state['cycle'] = cycle
                    telemetry.append(state)
                    timestamp = timestamp + timedelta(seconds=1)
                    
                pump.set_speed(0)
                for _ in range(30):
                    state = pump.next_state()
                    state['timestamp'] = timestamp.isoformat()
                    state['machineID'] = machine_id
                    state['service'] = service['name']
                    state['cycle'] = cycle
                    telemetry.append(state)
                    
        except Exception as e:
            failures.append({
                'timestamp': timestamp.isoformat(),
                'machineID': machine_id,
                'service': service['name'],
                'level': 'CRITICAL',
                'code': str(e),
                'message': PumpFailureModes.FAILURE_MODES.get(
                    str(e), 'Unknown failure'
                )
            })
            
    return telemetry, failures


if __name__ == '__main__':
    print("Pump Simulator - Example Run")
    
    pump = Pump(
        name="CP-001",
        initial_health={
            'impeller': 0.88,
            'seal': 0.85,
            'bearing_de': 0.82,
            'bearing_nde': 0.86
        },
        design_flow=150,
        design_head=80,
        design_speed=3000,
        fluid_density=850,
        npsh_available=8.0
    )
    
    print("\nStarting pump to 3000 RPM...")
    pump.set_speed(3000)
    
    for i in range(10):
        try:
            state = pump.next_state()
            print(f"t={i}: Speed={state['speed']:.0f} RPM, "
                  f"Flow={state['flow']:.1f} m³/hr, "
                  f"Head={state['head']:.1f} m, "
                  f"Vib={state['vibration_rms']:.2f} mm/s, "
                  f"NPSH margin={state['npsh_margin']:.1f} m")
        except Exception as e:
            print(f"FAILURE: {e}")
            break
            
    print("\nDone.")
