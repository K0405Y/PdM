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
from typing import Dict
from dataclasses import dataclass


class LocationType(Enum):
    """Installation location types with distinct environmental characteristics."""
    OFFSHORE = "offshore"          
    DESERT = "desert"              
    ARCTIC = "arctic"             
    TROPICAL = "tropical"          
    TEMPERATE = "temperate"    


@dataclass
class EnvironmentalProfile:
    """Environmental characteristics for a location type."""
    # Temperature parameters (°C)
    temp_annual_mean: float        
    temp_daily_amplitude: float    
    temp_seasonal_amplitude: float 

    # Humidity parameters (% relative humidity)
    humidity_mean: float
    humidity_variation: float

    # Pressure parameters (kPa)
    pressure_mean: float
    pressure_variation: float

    # Special factors
    salt_exposure: float           
    dust_exposure: float           
    ice_risk: float                

# Predefined location profiles
LOCATION_PROFILES = {
    LocationType.OFFSHORE: EnvironmentalProfile(
        temp_annual_mean=15.0,
        temp_daily_amplitude=3.0,      
        temp_seasonal_amplitude=10.0,  
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
        temp_seasonal_amplitude=15.0,  
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
        temp_seasonal_amplitude=25.0,  # Extreme seasonal variation
        humidity_mean=60.0,
        humidity_variation=20.0,
        pressure_mean=102.0,
        pressure_variation=5.0,
        salt_exposure=0.3,
        dust_exposure=0.1,
        ice_risk=0.95                  # Extreme ice risk
    ),

    LocationType.TROPICAL: EnvironmentalProfile(
        temp_annual_mean=28.0,
        temp_daily_amplitude=7.0,
        temp_seasonal_amplitude=3.0,   # Minimal seasonal variation
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
        temp_seasonal_amplitude=18.0,
        humidity_mean=65.0,
        humidity_variation=15.0,
        pressure_mean=101.3,
        pressure_variation=4.0,
        salt_exposure=0.1,
        dust_exposure=0.3,
        ice_risk=0.3
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

        T(t) = T_mean + A_daily*sin(2π*hour/24) + A_seasonal*sin(2π*day/365)
        """
        # Daily cycle (peaks at ~2pm, minimum at ~4am)
        daily_phase = (self.hour_of_day - 4) / 24 * 2 * np.pi
        daily_component = self.profile.temp_daily_amplitude * np.sin(daily_phase)

        # Seasonal cycle (peaks mid-summer, day ~200, minimum mid-winter, day ~20)
        seasonal_phase = (self.day_of_year - 20) / 365 * 2 * np.pi
        seasonal_component = self.profile.temp_seasonal_amplitude * np.sin(seasonal_phase)

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
                 LocationType.TROPICAL, LocationType.TEMPERATE]

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
    print("\n--- WEATHER EVENT SIMULATION ---")
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