# Weather API Client Module

## Overview

The `weather_api_client.py` module provides real-world weather data integration for the PdM simulation framework through a hybrid architecture that seamlessly combines synthetic environmental models with third-party weather APIs. This enables site-specific accuracy while maintaining the flexibility of deterministic synthetic data for testing and development.

## Purpose

Industrial equipment simulators benefit from both synthetic and real weather data:

**Synthetic Weather** (via `environmental_conditions.py`):
- Fast, deterministic, reproducible
- No external dependencies or API costs
- Ideal for algorithm development and testing
- Location-type approximations (Offshore, Desert, Arctic, etc.)

**Real Weather** (via `weather_api_client.py`):
- Site-specific actual conditions
- Real temperature, humidity, pressure, wind patterns
- Historical weather for run-to-failure simulations
- Validation against real-world installations

The weather API client bridges these approaches with:
- Abstract base class for unified interface
- Multiple API provider support (OpenWeatherMap, WeatherAPI.com, Visual Crossing)
- SQLite caching for cost optimization and offline use
- Automatic fallback to synthetic on API failures

## Key Features

- **Hybrid Architecture**: Single interface for both synthetic and real weather
- **Multiple Providers**: Support for 3 major weather API services
- **Location Flexibility**: Specify sites by city name or coordinates
- **Intelligent Caching**: SQLite-based persistent cache with configurable TTL
- **Cost Optimization**: Hourly caching reduces API calls by 3600x
- **Rate Limiting**: Automatic rate limit management to stay within free tiers
- **Offline Mode**: Pre-cache weather data for reproducible offline simulations
- **Graceful Degradation**: Automatic fallback to synthetic on API errors

## Module Components

### WeatherConfig Dataclass

Configuration for weather API integration.

```python
@dataclass
class WeatherConfig:
    api_provider: str = "weatherapi"  # "openweathermap", "weatherapi", "visualcrossing"
    api_key: str = ""

    # Location specification (use EITHER location_name OR lat/lon)
    location_name: str = ""  # e.g., "Lagos", "Port Harcourt"
    country: str = ""         # Optional: "Nigeria", "KE", "NG"
    latitude: float = 0.0
    longitude: float = 0.0

    # Cache configuration
    cache_enabled: bool = True
    cache_db_path: str = "weather_cache.db"
    cache_ttl_hours: int = 24

    # Rate limiting
    rate_limit_calls_per_minute: int = 60
    timeout_seconds: int = 10
```

**Location Specification**: Two methods supported:
1. **Location name** (recommended): `location_name="Lagos"`, `country="Nigeria"`
2. **Coordinates**: `latitude=6.5244`, `longitude=3.3792`

**Method**: `get_location_query()` returns formatted query string for API calls.

### EnvironmentalDataSource (Abstract Base Class)

Abstract interface that both synthetic and real weather sources implement:

```python
class EnvironmentalDataSource(ABC):
    @abstractmethod
    def get_conditions(self, elapsed_hours: float,
                      timestamp: Optional[datetime] = None) -> Dict:
        pass
```

This abstraction allows equipment simulators to use either weather source without code changes:

```python
# Works with EnvironmentalConditions (synthetic)
env = EnvironmentalConditions(LocationType.SAHEL)
conditions = env.get_conditions(elapsed_hours=100)

# Works with WeatherAPIClient (real weather)
env = create_hybrid_environment(use_real_weather=True, ...)
conditions = env.get_conditions(elapsed_hours=0, timestamp=datetime.now())
```

### WeatherAPIClient

Real-time weather API client with multi-provider support and rate limiting.

**Initialization:**
```python
config = WeatherConfig(
    api_provider="weatherapi",
    api_key="your_key",
    location_name="Lagos",
    country="Nigeria"
)
client = WeatherAPIClient(config)
```

**Key Methods:**

#### get_conditions(elapsed_hours, timestamp)

Fetch current weather conditions from API.

**Parameters:**
- `elapsed_hours`: float - Ignored for real weather (uses timestamp instead)
- `timestamp`: Optional[datetime] - Timestamp for lookup (defaults to now)

**Returns:** Dictionary with:
- `ambient_temp_C`: Temperature (°C)
- `humidity_percent`: Relative humidity (%)
- `pressure_kPa`: Atmospheric pressure (kPa)
- `wind_speed_m_s`: Wind speed (m/s)
- `hour_of_day`: Hour (0-23)
- `day_of_year`: Day (1-365)
- `location`: "real_weather_{provider}"
- `temp_derating_factor`: Gas turbine derating
- `density_ratio`: Air density vs ISO
- `corrosion_factor`: 1.0 (would need additional data)
- `fouling_factor`: 1.0
- `ice_formation_risk`: 0.0

**Rate Limiting:**
- Automatically enforces `rate_limit_calls_per_minute` (default: 60)
- Sleeps when limit reached to avoid quota exhaustion
- Resets counter every minute

**Provider-Specific Implementations:**

1. **OpenWeatherMap** (`_fetch_openweathermap`):
   - Endpoint: `api.openweathermap.org/data/2.5/weather`
   - Supports city name (`q` parameter) or lat/lon
   - Units: metric

2. **WeatherAPI.com** (`_fetch_weatherapi`):
   - Endpoint: `api.weatherapi.com/v1/current.json`
   - Unified `q` parameter for both city and coordinates
   - Units: metric

3. **Visual Crossing** (`_fetch_visualcrossing`):
   - Endpoint: `weather.visualcrossing.com/.../timeline/{location}/{date}`
   - Supports city name or coordinates in URL path
   - Units: metric

### CachedWeatherEnvironment

Cached weather environment with SQLite persistence for cost optimization and offline operation.

**Initialization:**
```python
cached_env = CachedWeatherEnvironment(
    weather_client=api_client,          # WeatherAPIClient instance
    fallback_source=synthetic_env,      # EnvironmentalConditions instance
    config=weather_config
)
```

**Cache Database Schema:**
```sql
CREATE TABLE weather_cache (
    location_query TEXT,     -- e.g., "Lagos,Nigeria"
    timestamp TEXT,          -- ISO format timestamp (hour precision)
    ambient_temp_C REAL,
    humidity_percent REAL,
    pressure_kPa REAL,
    cached_at TEXT,          -- When cached (for TTL)
    PRIMARY KEY (location_query, timestamp)
)
```

**Key Methods:**

#### get_conditions(elapsed_hours, timestamp)

Get weather with caching hierarchy:

1. Check cache for matching (location, timestamp)
2. If cache miss or expired: fetch from API
3. If API fails: use fallback synthetic source
4. Store successful API results in cache

**Cache TTL**: Configurable (default: 24 hours)

**Timestamp Rounding**: Rounds to nearest hour for cache efficiency

#### preload_cache(start_date, end_date, interval_hours)

Pre-populate cache with historical weather for offline simulation.

**Parameters:**
- `start_date`: datetime - Start of simulation period
- `end_date`: datetime - End of simulation period
- `interval_hours`: int - Hours between data points (default: 1)

**Example:**
```python
# Pre-cache 180 days of hourly weather
cached_env.preload_cache(
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 6, 30),
    interval_hours=1
)
# Result: 180 * 24 = 4,320 API calls
# Simulation can now run offline using cached data
```

**Use Case**: Run-to-failure simulations requiring reproducible weather patterns.

### create_hybrid_environment (Factory Function)

Convenience function for creating hybrid environmental data sources.

```python
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
) -> EnvironmentalDataSource
```

**Parameters:**
- `use_real_weather`: If False, returns synthetic environment
- `api_provider`: "weatherapi", "openweathermap", or "visualcrossing"
- `api_key`: API key for weather service
- `location_name`: City name (e.g., "Lagos", "Addis Ababa")
- `country`: Optional country for disambiguation
- `latitude`/`longitude`: Alternative coordinate specification
- `fallback_source`: Synthetic environment for API failures (auto-created if None)
- `cache_enabled`: Enable SQLite caching

**Returns:** `EnvironmentalDataSource` instance (either synthetic or cached API client)

## Usage Examples

### Basic Synthetic Weather

```python
from environmental_conditions import EnvironmentalConditions, LocationType

# Create synthetic Sahel environment
env = EnvironmentalConditions(location_type=LocationType.SAHEL)
conditions = env.get_conditions(elapsed_hours=1000)

print(f"Temp: {conditions['ambient_temp_C']:.1f}°C")
print(f"Humidity: {conditions['humidity_percent']:.1f}%")
print(f"Dust: {conditions['fouling_factor']:.2f}x")
```

### Real Weather with Location Name

```python
from weather_api_client import create_hybrid_environment
from datetime import datetime

# Nigerian coastal installation
env = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key",
    location_name="Lagos",
    country="Nigeria",
    cache_enabled=True
)

# Get current weather
conditions = env.get_conditions(0, datetime.now())
print(f"Lagos: {conditions['ambient_temp_C']:.1f}°C, {conditions['humidity_percent']:.1f}%")

# Second call uses cache (no API call)
conditions = env.get_conditions(0, datetime.now())
```

### Real Weather with Coordinates

```python
# Ethiopian highlands installation (Addis Ababa)
env = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key",
    latitude=9.0320,
    longitude=38.7469,
    cache_enabled=True
)

conditions = env.get_conditions(0, datetime.now())
print(f"Addis Ababa: {conditions['ambient_temp_C']:.1f}°C")
print(f"Pressure: {conditions['pressure_kPa']:.1f} kPa (altitude effect)")
```

### Hybrid with Fallback

```python
from environmental_conditions import EnvironmentalConditions, LocationType

# Create synthetic fallback
fallback = EnvironmentalConditions(LocationType.SAVANNA)

# Create hybrid with automatic fallback
env = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key",
    location_name="Harare",
    country="Zimbabwe",
    fallback_source=fallback,
    cache_enabled=True
)

# If API fails, automatically uses synthetic Savanna weather
conditions = env.get_conditions(0, datetime.now())
```

### Pre-caching for Offline Simulation

```python
from datetime import datetime, timedelta

# Create environment
env = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key",
    location_name="Port Harcourt",
    country="Nigeria",
    cache_enabled=True
)

# Pre-load 90 days of hourly weather
start = datetime(2025, 1, 1)
end = start + timedelta(days=90)
env.preload_cache(start, end, interval_hours=1)

# Now run simulation offline using cached data
for day in range(90):
    for hour in range(24):
        timestamp = start + timedelta(days=day, hours=hour)
        conditions = env.get_conditions(0, timestamp)
        # Process simulation step with real weather
```

### Equipment Integration

Same interface for both synthetic and real weather:

```python
# Gas turbine simulation with real weather
from gas_turbine import GasTurbine

gt = GasTurbine(equipment_id=1)
env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Lagos",
    api_key=api_key
)

for hour in range(8760):  # One year
    timestamp = start_date + timedelta(hours=hour)
    conditions = env.get_conditions(0, timestamp)

    # Apply environmental impacts
    telemetry = gt.generate_telemetry(
        speed_target=gt.rated_speed,
        operating_mode='steady_state'
    )

    # Temperature derating
    power = telemetry['power_mw'] * conditions['temp_derating_factor']

    # Environmental degradation
    if conditions['fouling_factor'] > 1.5:
        # High dust - accelerate compressor fouling
        pass
```

## Weather API Providers

### WeatherAPI.com (Recommended)

**Pros:**
- Generous free tier: 1M calls/month
- Simple unified API for city names and coordinates
- Fast response times
- Good coverage in Africa

**Free Tier:**
- 1,000,000 calls/month
- Current weather, forecast, historical
- No credit card required

**Signup:** https://www.weatherapi.com/signup.aspx

**Example URL:**
```
https://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q=Lagos,Nigeria
```

### OpenWeatherMap

**Pros:**
- Well-established service
- Extensive documentation
- Multiple data products

**Free Tier:**
- 60 calls/minute
- 1,000,000 calls/month
- Current weather only (historical requires paid plan)

**Signup:** https://openweathermap.org/api

**Example URL:**
```
https://api.openweathermap.org/data/2.5/weather?q=Lagos,Nigeria&appid=YOUR_KEY&units=metric
```

### Visual Crossing

**Pros:**
- Historical weather data on free tier
- Good for pre-caching long simulations
- Timeline API for date ranges

**Free Tier:**
- 1,000 calls/day (limited but sufficient for pre-caching)
- Historical and forecast data

**Signup:** https://www.visualcrossing.com/weather-api

**Example URL:**
```
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Lagos,Nigeria/2025-01-01?key=YOUR_KEY&unitGroup=metric
```

## Cost Optimization

### Caching Strategy

**Without Caching:**
- 1 simulation second = 1 API call
- 180-day simulation @ 1 sample/sec = 15,552,000 calls (exceeds all free tiers)

**With Hourly Caching:**
- 1 hour = 1 API call (3600 seconds use same cached value)
- 180-day simulation = 180 × 24 = 4,320 calls (well within free tiers)
- **Reduction: 3600x fewer API calls**

**Cache TTL Considerations:**
- Short TTL (1-6 hours): More up-to-date but more API calls
- Medium TTL (24 hours, default): Good balance for historical simulations
- Long TTL (7+ days): Essentially permanent cache for reproducibility

### Rate Limiting

Built-in rate limiting prevents exceeding provider limits:

```python
config = WeatherConfig(
    api_provider="weatherapi",
    api_key=api_key,
    rate_limit_calls_per_minute=60  # Stays under limit
)
```

Automatic behavior:
- Tracks calls per minute
- Sleeps when limit reached
- Resets counter every minute

### Pre-caching for Cost Control

Pre-cache during development, reuse for production runs:

```python
# Development: Pre-cache once (uses API calls)
env.preload_cache(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    interval_hours=1
)
# Cost: 365 × 24 = 8,760 API calls

# Production: Run unlimited simulations (uses cache, no API calls)
for run in range(100):
    # Each run uses same cached weather - reproducible and free
    simulate_equipment_lifecycle(env, start_date, end_date)
```

## African Location Examples

### West African Sahel Region

```python
# Mali, Niger, Chad installations
mali_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Bamako",
    country="Mali",
    api_key=api_key
)

niger_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Niamey",
    country="Niger",
    api_key=api_key
)

# Captures Harmattan dust season effects (Nov-Mar)
```

### Ethiopian Highlands

```python
# Highland tropical climate
ethiopia_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Addis Ababa",
    country="Ethiopia",
    api_key=api_key
)

# Real altitude effects on pressure and temperature
```

### Southern African Savanna

```python
# Zimbabwe, Zambia installations
zimbabwe_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Harare",
    country="Zimbabwe",
    api_key=api_key
)

zambia_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Lusaka",
    country="Zambia",
    api_key=api_key
)

# Southern Hemisphere seasons (summer in Jan, winter in Jul)
```

### Nigerian Coastal/Inland

```python
# Coastal (Sahel-adjacent with marine influence)
lagos_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Lagos",
    country="Nigeria",
    api_key=api_key
)

# Inland (more Sahel characteristics)
abuja_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Abuja",
    country="Nigeria",
    api_key=api_key
)
```

## Performance Considerations

### API Call Overhead

**Typical Response Times:**
- WeatherAPI.com: 100-300ms
- OpenWeatherMap: 150-400ms
- Visual Crossing: 200-500ms

**Impact on Simulation:**
- Without caching: 1000 timesteps × 300ms = 5 minutes overhead
- With caching: 1000 timesteps ÷ 3600 × 300ms = ~0.08 seconds overhead

### Cache Performance

**SQLite Cache:**
- Read: <1ms per query
- Write: ~5ms per insert
- Index: location_query + timestamp for O(log n) lookups

**Memory Footprint:**
- Config: ~200 bytes
- Cache connection: ~50 KB
- No in-memory data besides current record

### Optimization Recommendations

1. **Use hourly caching** (default) for 3600x call reduction
2. **Pre-cache** during off-hours for production runs
3. **Batch multiple equipment** using same location (share cache)
4. **Monitor cache hit rate** to tune TTL

## Limitations and Future Enhancements

### Current Limitations

1. **Current Weather Only**: No historical lookup (except Visual Crossing)
2. **Simplified Wind**: Only wind speed, no direction or gusts
3. **No Precipitation**: Rain/snow not captured
4. **Basic Derived Impacts**: corrosion_factor, fouling_factor set to 1.0
5. **Cache Migration**: No automatic schema updates for cache DB

### Potential Enhancements

1. **Historical Weather API**: Use Visual Crossing for past weather
2. **Wind Direction**: Add wind effects on cooling and structure loading
3. **Precipitation Integration**: Rain/snow impact on fouling, icing, cooling
4. **Enhanced Corrosion Modeling**: Use humidity + temperature for real corrosion rates
5. **Dust API Integration**: AQI (Air Quality Index) for real-time fouling prediction
6. **Cache Analytics**: Dashboard for cache hit rates and API usage
7. **Multi-Location Simulation**: Grid-based weather for pipeline/transmission systems

## Error Handling

### API Failures

Automatic fallback cascade:
1. Try API fetch
2. On timeout/error: check fallback_source
3. If fallback available: use synthetic weather
4. If no fallback: raise RuntimeError

```python
try:
    conditions = env.get_conditions(0, timestamp)
except RuntimeError as e:
    print(f"Weather data unavailable: {e}")
    # Handle gracefully
```

### Common Error Scenarios

1. **Invalid API Key**: `ValueError` on initialization
2. **Network Timeout**: Automatic fallback to synthetic
3. **Rate Limit Exceeded**: Automatic sleep and retry
4. **Invalid Location**: API returns error, fallback to synthetic
5. **Cache Corruption**: Recreates cache table on next run

## Validation

### Data Format Consistency

Both synthetic and real weather return identical structure:

```python
{
    'ambient_temp_C': float,
    'humidity_percent': float,
    'pressure_kPa': float,
    'wind_speed_m_s': float,  # 0.0 for synthetic
    'hour_of_day': int,
    'day_of_year': int,
    'location': str,
    'temp_derating_factor': float,
    'density_ratio': float,
    'corrosion_factor': float,
    'fouling_factor': float,
    'ice_formation_risk': float
}
```

### Real Weather Validation

Spot-check real weather against expected ranges:
- Lagos: 25-32°C typical, 70-90% humidity
- Addis Ababa: 10-25°C, 30-70% humidity, ~98 kPa pressure
- Harare: 15-30°C, 40-70% humidity

## References

1. WeatherAPI.com Documentation - https://www.weatherapi.com/docs/
2. OpenWeatherMap API Documentation - https://openweathermap.org/api
3. Visual Crossing Weather API - https://www.visualcrossing.com/resources/documentation/weather-api/
4. SQLite Documentation - https://www.sqlite.org/docs.html

## See Also

- [environmental_conditions.md](environmental_conditions.md) - Synthetic environmental modeling
- [gas_turbine.md](gas_turbine.md) - Gas turbine integration with environmental conditions
- [centrifugal_compressor.md](centrifugal_compressor.md) - Compressor environmental impacts
- [centrifugal_pump.md](centrifugal_pump.md) - Pump cold weather considerations
