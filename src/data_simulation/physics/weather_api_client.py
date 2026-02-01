"""
Weather API Integration with Caching Layer

Provides hybrid environmental data sources combining synthetic models
with real-world weather data from API providers.

Key Features:
- Abstract base class for environmental data sources
- Multiple weather API provider support (OpenWeatherMap, WeatherAPI.com, Visual Crossing)
- SQLite-based caching for offline/reproducible simulations
- Rate limiting and cost optimization
- Automatic fallback to synthetic data

Reference: Weather API services, industrial SCADA integration patterns
"""

import os
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import re
from data_simulation.physics.environmental_conditions import EnvironmentalConditions, LocationType
import requests
import dotenv

@dataclass
class WeatherConfig:
    """Configuration for weather API integration."""
    api_provider: str = "weatherapi"  # "openweathermap", "weatherapi", "visualcrossing"
    api_key: str = ""

    # Location specification (use EITHER location_name OR lat/lon)
    location_name: str = ""  # e.g., "Lagos, Nigeria" or "Port Harcourt"
    country: str = ""         # Optional country code (e.g., "NG", "KE")
    latitude: float = 0.0     # Alternative: specify exact coordinates
    longitude: float = 0.0

    # Cache configuration
    cache_enabled: bool = True
    cache_db_path: str = "weather_cache.db"
    cache_ttl_hours: int = 24

    # Rate limiting
    rate_limit_calls_per_minute: int = 60
    timeout_seconds: int = 10

    def get_location_query(self) -> str:
        """
        Get location query string for API calls.

        Returns:
            Location string (either "city,country" or "lat,lon")
        """
        if self.location_name:
            if self.country:
                return f"{self.location_name},{self.country}"
            return self.location_name
        elif self.latitude != 0.0 or self.longitude != 0.0:
            return f"{self.latitude},{self.longitude}"
        else:
            raise ValueError("Must specify either location_name or latitude/longitude")

class EnvironmentalDataSource(ABC):
    """
    Abstract base class for environmental data sources.

    Allows switching between synthetic models and real weather APIs
    without changing equipment simulator code.
    """

    @abstractmethod
    def get_conditions(self, elapsed_hours: float, timestamp: Optional[datetime] = None) -> Dict:
        """
        Get environmental conditions at a given time.

        Args:
            elapsed_hours: Hours since simulation start
            timestamp: Optional absolute timestamp for real weather lookup

        Returns:
            Dict with environmental parameters
        """
        pass


class WeatherAPIClient(EnvironmentalDataSource):
    """
    Real-time weather API client with multiple provider support.

    Fetches actual weather data from third-party APIs with rate limiting
    and error handling.
    """

    def __init__(self, config: WeatherConfig):
        """
        Initialize weather API client.

        Args:
            config: Weather API configuration
        """
        # if not HAS_REQUESTS:
        #     raise ImportError("requests library required for WeatherAPIClient. Install with: pip install requests")

        self.config = config
        self.last_call_time = 0.0
        self.call_count = 0
        self.minute_start = time.time()

        # Validate configuration
        if not config.api_key:
            raise ValueError("API key required for weather API access")

        # Validate location specification
        try:
            config.get_location_query()
        except ValueError as e:
            raise ValueError(f"Invalid location configuration: {e}")

    def get_conditions(self, elapsed_hours: float, timestamp: Optional[datetime] = None) -> Dict:
        """
        Fetch current weather conditions from API.

        Args:
            elapsed_hours: Ignored for real weather (uses timestamp instead)
            timestamp: Timestamp for weather lookup (defaults to now)

        Returns:
            Dict with environmental parameters
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Apply rate limiting
        self._apply_rate_limit()

        try:
            if self.config.api_provider == "openweathermap":
                raw_data = self._fetch_openweathermap(timestamp)
            elif self.config.api_provider == "weatherapi":
                raw_data = self._fetch_weatherapi(timestamp)
            elif self.config.api_provider == "visualcrossing":
                raw_data = self._fetch_visualcrossing(timestamp)
            else:
                raise ValueError(f"Unknown API provider: {self.config.api_provider}")

            # Convert to standard format
            return self._standardize_conditions(raw_data, timestamp)

        except Exception as e:
            raise RuntimeError(f"Weather API fetch failed: {e}")

    def _apply_rate_limit(self):
        """Apply rate limiting to avoid exceeding API quotas."""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.call_count = 0
            self.minute_start = current_time

        # Check rate limit
        if self.call_count >= self.config.rate_limit_calls_per_minute:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.call_count = 0
            self.minute_start = time.time()

        self.call_count += 1

    def _fetch_openweathermap(self, timestamp: datetime) -> Dict:
        """Fetch from OpenWeatherMap API (2.5 Current Weather)."""
        url = "https://api.openweathermap.org/data/2.5/weather"

        # Use location query (supports both city names and lat/lon)
        location_query = self.config.get_location_query()

        # OpenWeatherMap prefers separate lat/lon params, but also supports 'q' for city
        if self.config.location_name:
            params = {
                "q": location_query,
                "appid": self.config.api_key,
                "units": "metric"
            }
        else:
            params = {
                "lat": self.config.latitude,
                "lon": self.config.longitude,
                "appid": self.config.api_key,
                "units": "metric"
            }

        response = requests.get(url, params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _fetch_weatherapi(self, timestamp: datetime) -> Dict:
        """Fetch from WeatherAPI.com (Current Weather API)."""
        url = "https://api.weatherapi.com/v1/current.json"

        # WeatherAPI supports city name or lat,lon in the 'q' parameter
        params = {
            "key": self.config.api_key,
            "q": self.config.get_location_query()
        }

        response = requests.get(url, params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _fetch_visualcrossing(self, timestamp: datetime) -> Dict:
        """Fetch from Visual Crossing Timeline API."""
        date_str = timestamp.strftime("%Y-%m-%d")

        # Visual Crossing supports location name or lat,lon in URL path
        location_query = self.config.get_location_query()
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location_query}/{date_str}"

        params = {
            "key": self.config.api_key,
            "unitGroup": "metric",
            "include": "current"
        }

        response = requests.get(url, params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _standardize_conditions(self, raw_data: Dict, timestamp: datetime) -> Dict:
        """
        Convert provider-specific format to standard environmental conditions.

        Args:
            raw_data: Raw API response
            timestamp: Request timestamp

        Returns:
            Standardized environmental conditions dict
        """
        if self.config.api_provider == "openweathermap":
            temp = raw_data['main']['temp']
            humidity = raw_data['main']['humidity']
            pressure = raw_data['main']['pressure'] / 10.0  # hPa to kPa
            wind_speed = raw_data.get('wind', {}).get('speed', 0.0)  # m/s

        elif self.config.api_provider == "weatherapi":
            temp = raw_data['current']['temp_c']
            humidity = raw_data['current']['humidity']
            pressure = raw_data['current']['pressure_mb'] / 10.0  # mb to kPa
            wind_speed = raw_data['current'].get('wind_kph', 0.0) / 3.6  # kph to m/s

        elif self.config.api_provider == "visualcrossing":
            current = raw_data.get('currentConditions', raw_data['days'][0])
            temp = current['temp']
            humidity = current['humidity']
            pressure = current['pressure'] / 10.0  # mb to kPa
            wind_speed = current.get('windspeed', 0.0) / 3.6  # kph to m/s

        # Calculate derived impacts (same as EnvironmentalConditions)
        temp_derating_factor = 1.0 - ((temp - 15.0) * 0.007)
        temp_derating_factor = np.clip(temp_derating_factor, 0.7, 1.15)

        iso_density = 101.325 / (0.287 * (15 + 273.15))
        actual_density = pressure / (0.287 * (temp + 273.15))
        density_ratio = actual_density / iso_density

        return {
            'ambient_temp_C': round(temp, 2),
            'humidity_percent': round(humidity, 2),
            'pressure_kPa': round(pressure, 2),
            'wind_speed_m_s': round(wind_speed, 2),
            'hour_of_day': timestamp.hour,
            'day_of_year': timestamp.timetuple().tm_yday,
            'location': f"real_weather_{self.config.api_provider}",
            'temp_derating_factor': round(temp_derating_factor, 4),
            'density_ratio': round(density_ratio, 4),
            'corrosion_factor': 1.0,  # Would need additional data
            'fouling_factor': 1.0,
            'ice_formation_risk': 0.0
        }


class CachedWeatherEnvironment(EnvironmentalDataSource):
    """
    Cached weather environment with SQLite storage.

    Provides offline simulation capability and cost optimization
    by caching weather API responses locally.
    """

    def __init__(self,
                 weather_client: Optional[WeatherAPIClient] = None,
                 fallback_source: Optional[EnvironmentalDataSource] = None,
                 config: Optional[WeatherConfig] = None):
        """
        Initialize cached weather environment.

        Args:
            weather_client: Optional API client for cache misses
            fallback_source: Fallback synthetic source when API unavailable
            config: Weather configuration (required if weather_client provided)
        """
        self.weather_client = weather_client
        self.fallback_source = fallback_source
        self.config = config or WeatherConfig()

        # Initialize cache database
        if self.config.cache_enabled:
            self._init_cache_db()

    def _init_cache_db(self):
        """Initialize SQLite cache database with location-specific filename."""
        # Determine sanitized suffix from location
        location_query = self.config.get_location_query()
        suffix = re.sub(r'[^0-9A-Za-z_]', '_', location_query)[:50]
        if re.match(r'^[0-9]', suffix):
            suffix = f"t_{suffix}"

        # Create location-specific database path
        base_path = self.config.cache_db_path
        if base_path.endswith('.db'):
            self._cache_db_path = base_path[:-3] + f"_{suffix}.db"
        else:
            self._cache_db_path = f"{base_path}_{suffix}"

        conn = sqlite3.connect(self._cache_db_path)
        cursor = conn.cursor()

        self._cache_table_name = "weather_cache"

        # Create the cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_cache (
            location_query TEXT,
            timestamp TEXT,
            ambient_temp_C REAL,
            humidity_percent REAL,
            pressure_kPa REAL,
            cached_at TEXT,
            PRIMARY KEY (location_query, timestamp)
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_location_time
            ON weather_cache(location_query, timestamp)
        """)

        conn.commit()
        conn.close()

    def get_conditions(self, elapsed_hours: float, timestamp: Optional[datetime] = None) -> Dict:
        """
        Get weather conditions with caching.

        Args:
            elapsed_hours: Hours since simulation start
            timestamp: Absolute timestamp for weather lookup

        Returns:
            Environmental conditions dict
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Round timestamp to nearest hour for cache efficiency
        timestamp = timestamp.replace(minute=0, second=0, microsecond=0)

        # Try cache first
        if self.config.cache_enabled:
            cached = self._get_from_cache(timestamp)
            if cached:
                return cached

        # Cache miss - try API
        if self.weather_client:
            try:
                conditions = self.weather_client.get_conditions(elapsed_hours, timestamp)

                # Store in cache
                if self.config.cache_enabled:
                    self._store_in_cache(timestamp, conditions)

                return conditions

            except Exception as e:
                # API failed - use fallback
                if self.fallback_source:
                    return self.fallback_source.get_conditions(elapsed_hours, timestamp)
                raise

        # No API client - use fallback
        if self.fallback_source:
            return self.fallback_source.get_conditions(elapsed_hours, timestamp)

        raise RuntimeError("No weather data source available")

    def _get_from_cache(self, timestamp: datetime) -> Optional[Dict]:
        """Retrieve conditions from cache if available and fresh."""
        conn = sqlite3.connect(self._cache_db_path)
        cursor = conn.cursor()

        location_query = self.config.get_location_query()

        cursor.execute("""
            SELECT ambient_temp_C, humidity_percent, pressure_kPa, cached_at
            FROM weather_cache
            WHERE location_query = ? AND timestamp = ?
        """, (location_query, timestamp.isoformat()))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Check cache freshness
        cached_at = datetime.fromisoformat(row[3])
        age_hours = (datetime.now() - cached_at).total_seconds() / 3600

        if age_hours > self.config.cache_ttl_hours:
            return None  # Cache expired

        # Reconstruct conditions dict
        temp, humidity, pressure = row[0:3]

        temp_derating_factor = 1.0 - ((temp - 15.0) * 0.007)
        temp_derating_factor = np.clip(temp_derating_factor, 0.7, 1.15)

        iso_density = 101.325 / (0.287 * (15 + 273.15))
        actual_density = pressure / (0.287 * (temp + 273.15))
        density_ratio = actual_density / iso_density

        return {
            'ambient_temp_C': temp,
            'humidity_percent': humidity,
            'pressure_kPa': pressure,
            'hour_of_day': timestamp.hour,
            'day_of_year': timestamp.timetuple().tm_yday,
            'location': 'cached_weather',
            'temp_derating_factor': round(temp_derating_factor, 4),
            'density_ratio': round(density_ratio, 4),
            'corrosion_factor': 1.0,
            'fouling_factor': 1.0,
            'ice_formation_risk': 0.0
        }

    def _store_in_cache(self, timestamp: datetime, conditions: Dict):
        """Store conditions in cache."""
        conn = sqlite3.connect(self._cache_db_path)
        cursor = conn.cursor()

        location_query = self.config.get_location_query()

        cursor.execute("""
            INSERT OR REPLACE INTO weather_cache
            (location_query, timestamp, ambient_temp_C, humidity_percent,
             pressure_kPa, cached_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            location_query,
            timestamp.isoformat(),
            conditions['ambient_temp_C'],
            conditions['humidity_percent'],
            conditions['pressure_kPa'],
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def preload_cache(self, start_date: datetime, end_date: datetime, interval_hours: int = 1):
        """
        Pre-populate cache for a date range.

        Useful for running reproducible offline simulations with real weather data.

        Args:
            start_date: Start of simulation period
            end_date: End of simulation period
            interval_hours: Hours between cached data points
        """
        if not self.weather_client:
            raise ValueError("Weather client required for cache preloading")

        current = start_date
        cached_count = 0

        print(f"Preloading weather cache from {start_date} to {end_date}...")

        while current <= end_date:
            try:
                conditions = self.weather_client.get_conditions(0, current)
                self._store_in_cache(current, conditions)
                cached_count += 1

                if cached_count % 100 == 0:
                    print(f"  Cached {cached_count} data points...")

            except Exception as e:
                print(f"  Warning: Failed to cache {current}: {e}")

            current += timedelta(hours=interval_hours)

        print(f"Cache preload complete: {cached_count} data points stored")


# Convenience function for creating hybrid environment
def create_hybrid_environment(
    use_real_weather: bool = False,
    api_provider: str = "weatherapi",
    api_key: str = "",
    location_name: str = "",
    country: str = "",
    latitude: float = 0.0,
    longitude: float = 0.0,
    fallback_source: Optional[EnvironmentalDataSource] = None,
    cache_enabled: bool = True
) -> EnvironmentalDataSource:
    """
    Factory function for creating hybrid environmental data source.

    Args:
        use_real_weather: If True, use weather API; otherwise use fallback
        api_provider: Weather API provider name
        api_key: API key for weather service
        location_name: City/location name (e.g., "Lagos", "Port Harcourt")
        country: Optional country code for disambiguation (e.g., "NG", "KE")
        latitude: Alternative: specify exact latitude
        longitude: Alternative: specify exact longitude
        fallback_source: Synthetic environment for fallback
        cache_enabled: Enable SQLite caching

    Returns:
        EnvironmentalDataSource instance
    """
    if not use_real_weather:
        if fallback_source:
            return fallback_source
        else:
            # Import here to avoid circular dependency
            from .environmental_conditions import EnvironmentalConditions, LocationType
            return EnvironmentalConditions(location_type=LocationType.TEMPERATE)

    # Create weather API client
    config = WeatherConfig(
        api_provider=api_provider,
        api_key=api_key,
        location_name=location_name,
        country=country,
        latitude=latitude,
        longitude=longitude,
        cache_enabled=cache_enabled
    )

    weather_client = WeatherAPIClient(config)

    # Wrap in caching layer
    cached_env = CachedWeatherEnvironment(
        weather_client=weather_client,
        fallback_source=fallback_source,
        config=config
    )

    return cached_env


if __name__ == '__main__':
    """Demonstration of weather API integration."""
    print("Weather API Integration - Demonstration")

    # Example 1: Synthetic fallback (no API key needed)
    print("\n SYNTHETIC ENVIRONMENT")

    synthetic = EnvironmentalConditions(location_type=LocationType.SAHEL)
    conditions = synthetic.get_conditions(elapsed_hours=1000)
    print(f"Sahel location (synthetic):")
    print(f"  Temp: {conditions['ambient_temp_C']:.1f}°C")
    print(f"  Humidity: {conditions['humidity_percent']:.1f}%")
    print(f"  Dust exposure factor: {conditions['fouling_factor']:.2f}x")

    # Example 2: Hybrid with real API (requires API key)
    print("\nHYBRID ENVIRONMENT")
    print("To use real weather API:")
    print("1. Sign up for free API key:")
    print("   - WeatherAPI.com: https://www.weatherapi.com/signup.aspx")
    print("   - OpenWeatherMap: https://openweathermap.org/api")
    print("2. Set environment variable: WEATHER_API_KEY=your_key_here")
    print("3. Run: python weather_api_client.py")

    dotenv.load_dotenv()
    api_key = os.getenv('WEATHER_API_KEY')

    if api_key:
        print("\nAPI key found - testing real weather fetch...")

        # Example with location name (more user-friendly)
        hybrid_env = create_hybrid_environment(
            use_real_weather=True,
            api_provider="weatherapi",
            api_key=api_key,
            location_name="Lagos",
            country="Nigeria",
            cache_enabled=True
        )

        try:
            real_conditions = hybrid_env.get_conditions(0, datetime.now())
            print(f"\nLagos, Nigeria (real weather):")
            print(f"  Temp: {real_conditions['ambient_temp_C']:.1f}°C")
            print(f"  Humidity: {real_conditions['humidity_percent']:.1f}%")
            print(f"  Pressure: {real_conditions['pressure_kPa']:.1f} kPa")

            # Test cache
            cached_conditions = hybrid_env.get_conditions(0, datetime.now())
            print(f"\n  (Second call retrieved from cache)")

        except Exception as e:
            print(f"\nAPI call failed: {e}")
            print("Falling back to synthetic data would occur automatically")
    else:
        print("\n(No API key configured - skipping real weather test)")
        print("\nExample usage with location name:")
        print("  hybrid_env = create_hybrid_environment(")
        print("      use_real_weather=True,")
        print("      api_provider='weatherapi',")
        print("      api_key='your_key',")
        print("      location_name='Port Harcourt',")
        print("      country='Nigeria'")
        print("  )")

    # Example 3: Cache preloading for offline simulation
    print("\CACHE PRELOADING")
    print("For reproducible offline simulations:")
    print("  1. Preload cache with historical weather")
    print("  2. Run simulation offline using cached data")
    print("  3. Cache persists across runs for reproducibility")