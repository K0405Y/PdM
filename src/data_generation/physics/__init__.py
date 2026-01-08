"""
Physics-based simulation enhancements.

This package contains modules for realistic physical modeling:
- vibration_enhanced: Envelope-modulated bearing vibration
- thermal_transient: Startup/shutdown thermal stress
- environmental_conditions: Location-specific environmental variability
"""

from .vibration_enhanced import (
    EnhancedVibrationGenerator,
    BearingGeometry,
    BackwardCompatibleVibrationGenerator
)
from .thermal_transient import (
    ThermalTransientModel,
    ThermalMassProperties,
    OperatingMode
)
from .environmental_conditions import (
    EnvironmentalConditions,
    LocationType,
    EnvironmentalProfile,
    SeasonalPattern,
    LOCATION_PROFILES
)

__all__ = [
    'EnhancedVibrationGenerator',
    'BearingGeometry',
    'BackwardCompatibleVibrationGenerator',
    'ThermalTransientModel',
    'ThermalMassProperties',
    'OperatingMode',
    'EnvironmentalConditions',
    'LocationType',
    'EnvironmentalProfile',
    'SeasonalPattern',
    'LOCATION_PROFILES'
]