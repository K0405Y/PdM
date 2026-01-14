"""
Environmental Conditions Modeling

Simulates realistic environmental variations including daily temperature cycles,
seasonal patterns, weather events, and location-specific characteristics.

Key Features:
- Daily and seasonal temperature cycles
- Location-based environmental profiles (offshore, desert, arctic)
- Humidity and pressure variations
- Impact propagation to equipment performance

Reference: ISO atmospheric conditions, offshore environmental data
"""

import numpy as np
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass, field


class LocationType(Enum):
    """Installation location types with distinct environmental characteristics."""
    OFFSHORE = "offshore"
    DESERT = "desert"
    ARCTIC = "arctic"
    TROPICAL = "tropical"
    TEMPERATE = "temperate"
    SAHEL = "sahel"                          
    HIGHLAND_TROPICAL = "highland_tropical"  
    SAVANNA = "savanna"                      


@dataclass
class SeasonalPattern:
    """
    Defines seasonal behavior for a location.
    Allows modeling of 2-season (monsoon), 3-season, or 4-season patterns
    using multiple sinusoidal components with location-specific peaks.
    """
    hemisphere: str = "northern"  # "northern" or "southern"
    season_peaks: List[int] = field(default_factory=list)  # Days of year for temperature peaks/troughs
    season_amplitudes: List[float] = field(default_factory=list)  # Amplitude for each component (°C)

    def __post_init__(self):
        """Validate seasonal pattern configuration."""
        if len(self.season_peaks) != len(self.season_amplitudes):
            raise ValueError("season_peaks and season_amplitudes must have same length")
        if self.hemisphere not in ["northern", "southern"]:
            raise ValueError("hemisphere must be 'northern' or 'southern'")


@dataclass
class EnvironmentalProfile:
    """Environmental characteristics for a location type."""
    # Temperature parameters (°C)
    temp_annual_mean: float
    temp_daily_amplitude: float

    # Seasonal pattern (replaces simple temp_seasonal_amplitude)
    seasonal_pattern: SeasonalPattern = None

    # Humidity parameters (% relative humidity)
    humidity_mean: float = 65.0
    humidity_variation: float = 15.0

    # Pressure parameters (kPa)
    pressure_mean: float = 101.3
    pressure_variation: float = 3.0

    # Special factors
    salt_exposure: float = 0.1
    dust_exposure: float = 0.1
    ice_risk: float = 0.1                

# Predefined location profiles
LOCATION_PROFILES = {
    LocationType.OFFSHORE: EnvironmentalProfile(
        temp_annual_mean=15.0,
        temp_daily_amplitude=3.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="northern",
            season_peaks=[355, 182],  # Winter minimum (late Dec), Summer maximum (early July)
            season_amplitudes=[-10.0, 10.0]  # Cold winter, warm summer
        ),
        humidity_mean=75.0,
        humidity_variation=10.0,
        pressure_mean=101.3,
        pressure_variation=3.0,
        salt_exposure=0.9,
        dust_exposure=0.1,
        ice_risk=0.2
    ),

    LocationType.DESERT: EnvironmentalProfile(
        temp_annual_mean=30.0,
        temp_daily_amplitude=18.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="northern",
            season_peaks=[15, 195],  # Winter (mid-Jan), Summer (mid-July)
            season_amplitudes=[-15.0, 15.0]  # Cold winter nights, extreme summer heat
        ),
        humidity_mean=20.0,
        humidity_variation=15.0,
        pressure_mean=100.5,
        pressure_variation=2.0,
        salt_exposure=0.0,
        dust_exposure=0.95,
        ice_risk=0.0
    ),

    LocationType.ARCTIC: EnvironmentalProfile(
        temp_annual_mean=-15.0,
        temp_daily_amplitude=5.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="northern",
            season_peaks=[1, 172],  # Polar winter (Jan 1), Short summer (Jun 21)
            season_amplitudes=[-25.0, 25.0]  # Extreme: -40°C winter, +10°C summer
        ),
        humidity_mean=60.0,
        humidity_variation=20.0,
        pressure_mean=102.0,
        pressure_variation=5.0,
        salt_exposure=0.3,
        dust_exposure=0.1,
        ice_risk=0.95
    ),

    LocationType.TROPICAL: EnvironmentalProfile(
        temp_annual_mean=28.0,
        temp_daily_amplitude=7.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="northern",
            season_peaks=[60, 240],  # Wet season (Mar), Dry season (Sep)
            season_amplitudes=[3.0, -3.0]  # Minimal temperature change, humidity-driven seasons
        ),
        humidity_mean=85.0,
        humidity_variation=10.0,
        pressure_mean=101.0,
        pressure_variation=1.5,
        salt_exposure=0.4,
        dust_exposure=0.3,
        ice_risk=0.0
    ),

    LocationType.TEMPERATE: EnvironmentalProfile(
        temp_annual_mean=12.0,
        temp_daily_amplitude=8.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="northern",
            season_peaks=[20, 110, 200, 290],  # Winter, Spring, Summer, Fall
            season_amplitudes=[-18.0, 0.0, 18.0, 0.0]  # Traditional 4-season pattern
        ),
        humidity_mean=65.0,
        humidity_variation=15.0,
        pressure_mean=101.3,
        pressure_variation=4.0,
        salt_exposure=0.1,
        dust_exposure=0.3,
        ice_risk=0.3
    ),

    LocationType.SAHEL: EnvironmentalProfile(
        temp_annual_mean=30.0,
        temp_daily_amplitude=14.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="northern",
            season_peaks=[80, 260],  # Wet season (Mar-May), Dry season (Sep-Nov, Harmattan)
            season_amplitudes=[5.0, -5.0]  # Moderate seasonal variation
        ),
        humidity_mean=35.0,
        humidity_variation=25.0,  # Large variation between wet/dry seasons
        pressure_mean=101.0,
        pressure_variation=2.0,
        salt_exposure=0.0,
        dust_exposure=0.80,  # High dust, especially during Harmattan
        ice_risk=0.0
    ),

    LocationType.HIGHLAND_TROPICAL: EnvironmentalProfile(
        temp_annual_mean=18.0,  # 10°C cooler than lowland tropical
        temp_daily_amplitude=10.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="northern",
            season_peaks=[60, 240],  # Wet season (Mar-May), Dry season (Sep)
            season_amplitudes=[2.0, -2.0]  # Minimal temperature change, moderate altitude
        ),
        humidity_mean=70.0,
        humidity_variation=15.0,
        pressure_mean=98.0,  # Lower pressure at altitude (~600m-2000m)
        pressure_variation=2.0,
        salt_exposure=0.0,
        dust_exposure=0.2,
        ice_risk=0.1  # Occasional frost at high altitude
    ),

    LocationType.SAVANNA: EnvironmentalProfile(
        temp_annual_mean=25.0,
        temp_daily_amplitude=12.0,
        seasonal_pattern=SeasonalPattern(
            hemisphere="southern",
            season_peaks=[15, 195],  # Summer (Jan), Winter (Jul)
            season_amplitudes=[8.0, -8.0]  # Moderate seasonal variation
        ),
        humidity_mean=55.0,
        humidity_variation=20.0,
        pressure_mean=101.0,
        pressure_variation=2.5,
        salt_exposure=0.0,
        dust_exposure=0.5,  # Moderate dust during dry season
        ice_risk=0.0
    )
}


class EnvironmentalConditions:
    """
    Models time-varying environmental conditions.
    """

    def __init__(self,
                 location_type: LocationType = LocationType.TEMPERATE,
                 start_day_of_year: int = 1):
        """
        Initialize environmental model.

        Args:
            location_type: Installation location
            start_day_of_year: Starting day (1-365) for seasonal alignment
        """
        self.location_type = location_type
        self.profile = LOCATION_PROFILES[location_type]
        self.day_of_year = start_day_of_year
        self.hour_of_day = 0.0

    def get_conditions(self, elapsed_hours: float) -> Dict:
        """
        Get current environmental conditions.

        Args:
            elapsed_hours: Hours since simulation start

        Returns:
            Dict with environmental parameters
        """
        # Update time tracking
        self.hour_of_day = (elapsed_hours % 24)
        self.day_of_year = int(elapsed_hours / 24) % 365

        # Calculate cyclic components
        ambient_temp = self._calculate_temperature()
        humidity = self._calculate_humidity()
        pressure = self._calculate_pressure()

        # Calculate derived impacts
        impacts = self._calculate_equipment_impacts(ambient_temp, humidity, pressure)

        return {
            'ambient_temp_C': round(ambient_temp, 2),
            'humidity_percent': round(humidity, 2),
            'pressure_kPa': round(pressure, 2),
            'hour_of_day': self.hour_of_day,
            'day_of_year': self.day_of_year,
            'location': self.location_type.value,
            **impacts 
        }

    def _calculate_temperature(self) -> float:
        """
        Calculate ambient temperature with daily and seasonal cycles.

        T(t) = T_mean + A_daily*sin(2π*hour/24) + Σ[A_i*sin(2π*(day-peak_i)/365)]

        Uses location-specific seasonal patterns with multiple sinusoidal components
        to model 2-season (monsoon), 3-season, or 4-season patterns.
        """
        # Daily cycle (peaks at ~2pm, minimum at ~4am)
        daily_phase = (self.hour_of_day - 4) / 24 * 2 * np.pi
        daily_component = self.profile.temp_daily_amplitude * np.sin(daily_phase)

        # Seasonal cycle - sum of multiple sinusoidal components
        seasonal_component = 0.0
        if self.profile.seasonal_pattern:
            pattern = self.profile.seasonal_pattern
            for peak_day, amplitude in zip(pattern.season_peaks, pattern.season_amplitudes):
                phase = (self.day_of_year - peak_day) / 365 * 2 * np.pi
                seasonal_component += amplitude * np.sin(phase)

        # Random variation
        noise = np.random.normal(0, 1.0)

        temperature = (self.profile.temp_annual_mean +
                      daily_component +
                      seasonal_component +
                      noise)

        return temperature

    def _calculate_humidity(self) -> float:
        """
        Calculate relative humidity.

        Humidity inversely correlates with temperature (roughly).
        """
        # Base humidity
        humidity = self.profile.humidity_mean

        # Inverse correlation with temperature (when temp high, humidity often lower)
        current_temp = self._calculate_temperature()
        temp_effect = -(current_temp - self.profile.temp_annual_mean) * 0.3

        # Random variation
        noise = np.random.normal(0, self.profile.humidity_variation * 0.3)

        humidity = humidity + temp_effect + noise

        # Clamp to physical limits
        return np.clip(humidity, 10, 100)

    def _calculate_pressure(self) -> float:
        """Calculate atmospheric pressure with weather variation."""
        # Slow pressure changes (weather systems)
        # Use multi-day cycle to simulate weather patterns
        weather_cycle_days = 7
        weather_phase = (self.day_of_year % weather_cycle_days) / weather_cycle_days * 2 * np.pi
        pressure_variation = self.profile.pressure_variation * np.sin(weather_phase)

        # Random fluctuations
        noise = np.random.normal(0, 0.5)

        pressure = self.profile.pressure_mean + pressure_variation + noise

        return max(pressure, 95.0) 

    def _calculate_equipment_impacts(self,
                                    temp: float,
                                    humidity: float,
                                    pressure: float) -> Dict:
        """
        Calculate how environmental conditions affect equipment.

        Returns:
            Dict with impact multipliers and factors
        """
        # Temperature impact on gas turbine power
        # ISO conditions: 15°C, power decreases ~0.7% per °C above ISO
        temp_derating_factor = 1.0 - ((temp - 15.0) * 0.007)
        temp_derating_factor = np.clip(temp_derating_factor, 0.7, 1.15)

        # Compressor performance affected by inlet temperature
        # Density ratio affects mass flow
        iso_density = 101.325 / (0.287 * (15 + 273.15))  # kg/m³ at ISO conditions
        actual_density = pressure / (0.287 * (temp + 273.15))
        density_ratio = actual_density / iso_density

        # Corrosion rate factor (salt + humidity)
        corrosion_factor = 1.0 + (self.profile.salt_exposure * 0.5) * (humidity / 100)

        # Fouling rate factor (dust + humidity)
        fouling_factor = 1.0 + (self.profile.dust_exposure * 0.8)

        # Ice formation risk (temperature + humidity)
        ice_formation = 0.0
        if temp < 5.0 and humidity > 70:
            ice_formation = self.profile.ice_risk * ((5.0 - temp) / 5.0)

        return {
            'temp_derating_factor': round(temp_derating_factor, 4),
            'density_ratio': round(density_ratio, 4),
            'corrosion_factor': round(corrosion_factor, 3),
            'fouling_factor': round(fouling_factor, 3),
            'ice_formation_risk': round(ice_formation, 3)
        }

    def simulate_weather_event(self, event_type: str) -> Dict:
        """
        Simulate extreme weather event.

        Args:
            event_type: 'storm', 'heatwave', 'coldsnap', 'dust_storm'

        Returns:
            Modified environmental conditions during event
        """
        conditions = self.get_conditions(self.day_of_year * 24 + self.hour_of_day)

        if event_type == 'storm':
            conditions['pressure_kPa'] -= 5.0
            conditions['humidity_percent'] = min(100, conditions['humidity_percent'] + 20)
            conditions['ambient_temp_C'] -= 5.0

        elif event_type == 'heatwave':
            conditions['ambient_temp_C'] += 15.0
            conditions['humidity_percent'] = max(10, conditions['humidity_percent'] - 30)

        elif event_type == 'coldsnap':
            conditions['ambient_temp_C'] -= 20.0
            conditions['ice_formation_risk'] = min(1.0, conditions['ice_formation_risk'] + 0.5)

        elif event_type == 'dust_storm':
            conditions['fouling_factor'] *= 3.0

        return conditions


if __name__ == '__main__':
    print("Environmental Conditions Simulation")

    # Test each location type
    locations = [LocationType.OFFSHORE, LocationType.DESERT, LocationType.ARCTIC, 
                 LocationType.TROPICAL, LocationType.TEMPERATE, LocationType.SAHEL, 
                 LocationType.HIGHLAND_TROPICAL, LocationType.SAVANNA]

    for loc_type in locations:
        print(f"\n- {loc_type.value.upper()} -")
        env = EnvironmentalConditions(loc_type, start_day_of_year=180)  # Mid-summer

        # Sample throughout a day
        print(f"Daily cycle (summer, day 180):")
        for hour in [0, 6, 12, 18]:
            conditions = env.get_conditions(hour)
            print(f"  {hour:02d}:00 - Temp: {conditions['ambient_temp_C']:5.1f}°C, "
                  f"Humidity: {conditions['humidity_percent']:4.1f}%, "
                  f"Derating: {conditions['temp_derating_factor']:.3f}")

        # Sample across seasons
        print(f"\nSeasonal variation:")
        for day in [1, 91, 182, 273]:  # Winter, spring, summer, fall
            conditions = env.get_conditions(day * 24 + 12)  # Noon
            season = ['Winter', 'Spring', 'Summer', 'Fall'][day // 91]
            print(f"  {season:8s} - Temp: {conditions['ambient_temp_C']:5.1f}°C, "
                  f"Corrosion: {conditions['corrosion_factor']:.2f}x, "
                  f"Fouling: {conditions['fouling_factor']:.2f}x")

    # Demonstrate weather event
    print("\nWEATHER EVENT SIMULATION")
    env = EnvironmentalConditions(LocationType.OFFSHORE)
    normal = env.get_conditions(1000)
    storm = env.simulate_weather_event('storm')

    print(f"Normal conditions:")
    print(f"  Temp: {normal['ambient_temp_C']:.1f}°C, "
          f"Pressure: {normal['pressure_kPa']:.1f} kPa, "
          f"Humidity: {normal['humidity_percent']:.1f}%")

    print(f"\nDuring storm:")
    print(f"  Temp: {storm['ambient_temp_C']:.1f}°C, "
          f"Pressure: {storm['pressure_kPa']:.1f} kPa, "
          f"Humidity: {storm['humidity_percent']:.1f}%")