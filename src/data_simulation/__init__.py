"""
Data Generation Package.

Equipment Simulators:
- gas_turbine: Industrial gas turbine simulation
- compressor: Compressor simulation
- pump: Pump simulation

Physics Enhancements (physics/):
- vibration_enhanced: Realistic bearing vibration with envelope modulation
- thermal_transient: Startup/shutdown thermal stress modeling
- environmental_conditions: Location-specific environmental effects

Simulation Events (simulation/):
- maintenance_events: Maintenance scheduling and interventions
- incipient_faults: Discrete fault initiation and growth
- process_upsets: Process abnormalities and edge cases

ML Utilities (ml_utils/):
- ml_output_modes: Output formatting for ML training/evaluation
- pipeline_enhanced: Memory-efficient and parallel processing
"""

from .gas_turbine import GasTurbine, GasTurbineHealthModel, VibrationSignalGenerator
from .compressor import (
    Compressor,
    CompressorHealthModel,
    SurgeModel,
    DryGasSealModel,
    ShaftOrbitModel
)
from .pump import (
    Pump,
    CavitationModel,
    MechanicalSealModel,
    PumpBearingModel,
    HydraulicPerformanceModel,
    PumpFailureModes
)

# Import subpackages
from . import physics
from . import simulation
from . import ml_utils

__all__ = [
    # Gas Turbine
    'GasTurbine',
    'GasTurbineHealthModel',
    'VibrationSignalGenerator',
    # Compressor
    'Compressor',
    'CompressorHealthModel',
    'SurgeModel',
    'DryGasSealModel',
    'ShaftOrbitModel',
    # Pump
    'Pump',
    'CavitationModel',
    'MechanicalSealModel',
    'PumpBearingModel',
    'HydraulicPerformanceModel',
    'PumpFailureModes',
    # Subpackages
    'physics',
    'simulation',
    'ml_utils'
]