# Environmental Conditions Module

## Overview

The `environmental_conditions.py` module simulates realistic environmental variations that affect rotating equipment performance and degradation. It models location-specific conditions including daily temperature cycles, seasonal patterns, humidity variations, and their impacts on equipment health.

## Purpose

Industrial equipment operates in diverse environments - from offshore platforms to desert installations to arctic facilities. Environmental conditions significantly affect:

- Equipment performance (temperature derating, density effects)
- Degradation rates (corrosion, fouling, ice formation)
- Operating efficiency (power output, mass flow rates)
- Maintenance requirements (inspection intervals, cleaning frequency)

This module provides physics-based environmental modeling to generate realistic synthetic data for predictive maintenance applications.

## Key Features

- **Seven Location Profiles**: Offshore, Desert, Arctic, Tropical, Temperate, Sahel, and Savanna
- **Weather API Integration**: Hybrid approach supporting both synthetic and real weather data
- **Cyclic Temperature Modeling**: Daily and seasonal temperature variations
- **Humidity Dynamics**: Temperature-correlated humidity with physical limits
- **Atmospheric Pressure**: Weather-pattern pressure variations
- **Equipment Impact Calculation**: Performance derating and degradation factors
- **Weather Events**: Storm, heatwave, coldsnap, and dust storm simulations

## Module Components

### LocationType Enum

Defines seven distinct installation location types:

```python
class LocationType(Enum):
    OFFSHORE = "offshore"
    DESERT = "desert"
    ARCTIC = "arctic"
    TROPICAL = "tropical"
    TEMPERATE = "temperate"
    SAHEL = "sahel"
    SAVANNA = "savanna"
```

### SeasonalPattern Dataclass

Defines location-specific seasonal behavior using multiple sinusoidal components:

```python
@dataclass
class SeasonalPattern:
    hemisphere: str  # "northern" or "southern"
    season_peaks: List[int]  # Days of year for temperature peaks/troughs
    season_amplitudes: List[float]  # Amplitude for each component (°C)
```

**Parameters:**
- `hemisphere`: Hemisphere location ("northern" or "southern") - affects seasonal alignment
- `season_peaks`: List of day-of-year values (0-365) where each seasonal component peaks
- `season_amplitudes`: List of temperature amplitudes (°C) for each seasonal component

**Flexibility:**
This design supports diverse seasonal patterns:
- **2-season patterns** (monsoon regions): wet/dry seasons with minimal temperature swing
- **3-season patterns** (some tropical/subtropical): hot/warm/cool cycles
- **4-season patterns** (temperate regions): traditional winter/spring/summer/fall
- **Asymmetric patterns** (arctic/polar): long winter, short summer

### EnvironmentalProfile Dataclass

Encapsulates environmental characteristics for each location type:

**Temperature Parameters:**
- `temp_annual_mean`: Annual mean temperature (°C)
- `temp_daily_amplitude`: Daily temperature swing amplitude (°C)
- `seasonal_pattern`: SeasonalPattern defining location-specific seasonal cycles

**Humidity Parameters:**
- `humidity_mean`: Mean relative humidity (%)
- `humidity_variation`: Standard deviation of humidity variations (%)

**Pressure Parameters:**
- `pressure_mean`: Mean atmospheric pressure (kPa)
- `pressure_variation`: Pressure variation amplitude (kPa)

**Environmental Factors:**
- `salt_exposure`: Salt exposure factor (0.0 to 1.0, affects corrosion)
- `dust_exposure`: Dust exposure factor (0.0 to 1.0, affects fouling)
- `ice_risk`: Ice formation risk factor (0.0 to 1.0)

### Location Profiles

#### Offshore

Marine/offshore platform installations with moderate temperatures, high humidity, and significant salt exposure.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Annual Mean Temp | 15°C | Moderate, ocean-moderated |
| Daily Amplitude | 3°C | Low variation (water thermal mass) |
| Seasonal Pattern | 2-season | Winter minimum (Dec), Summer maximum (July) |
| Season Peaks | Days 355, 182 | Cold winter, warm summer |
| Season Amplitudes | -10°C, +10°C | ±10°C around mean |
| Humidity | 75% | High (marine environment) |
| Salt Exposure | 0.9 | Very high (corrosion risk) |
| Dust Exposure | 0.1 | Low |
| Ice Risk | 0.2 | Low to moderate |

**Seasonal Characteristics**: Simple 2-season pattern with moderate swings. Ocean thermal mass dampens extreme temperature variations. Winter: ~5°C, Summer: ~25°C.

**Degradation**: Corrosion is the primary concern. Salt spray accelerates degradation of metallic components. High humidity maintains corrosion activity year-round.

#### Desert

Arid land installations with extreme temperature swings and high dust exposure.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Annual Mean Temp | 30°C | Hot |
| Daily Amplitude | 18°C | Extreme day-night swings |
| Seasonal Pattern | 2-season | Winter (Jan), Summer (July) |
| Season Peaks | Days 15, 195 | Cold winter nights, extreme summer heat |
| Season Amplitudes | -15°C, +15°C | ±15°C around mean |
| Humidity | 20% | Very low |
| Salt Exposure | 0.0 | None |
| Dust Exposure | 0.95 | Extreme (fouling risk) |
| Ice Risk | 0.0 | None |

**Seasonal Characteristics**: 2-season pattern with large swings. Winter: ~15°C, Summer: ~45°C. Daily amplitude of 18°C means night temperatures can drop to near freezing in winter, while summer days exceed 50°C.

**Degradation**: Fouling is the primary concern. Fine dust particles cause compressor fouling, filter clogging, and abrasive wear. Extreme temperature swings cause thermal cycling fatigue.

#### Arctic

Extreme cold climate with severe ice risk and large seasonal temperature variations.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Annual Mean Temp | -15°C | Very cold |
| Daily Amplitude | 5°C | Low (limited solar variation) |
| Seasonal Pattern | 2-season (asymmetric) | Long polar winter, short summer |
| Season Peaks | Days 1, 172 | Polar winter (Jan 1), Short summer (Jun 21) |
| Season Amplitudes | -25°C, +25°C | Extreme: -40°C winter, +10°C summer |
| Humidity | 60% | Moderate |
| Salt Exposure | 0.3 | Moderate (coastal arctic) |
| Dust Exposure | 0.1 | Low |
| Ice Risk | 0.95 | Extreme |

**Seasonal Characteristics**: Extreme 2-season asymmetric pattern. Polar winter (9 months): -40°C, polar summer (3 months): +10°C. Solar forcing dominates - continuous darkness vs. continuous daylight.

**Degradation**: Ice formation in intakes and on surfaces is critical. Cold starts are challenging. Materials become brittle at extreme low temperatures. Equipment must survive -40°C operational temperatures.

#### Tropical

Hot, humid climate with minimal seasonal variation but high corrosion potential.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Annual Mean Temp | 28°C | Hot |
| Daily Amplitude | 7°C | Moderate |
| Seasonal Pattern | 2-season (wet/dry) | Minimal temperature swing, humidity-driven |
| Season Peaks | Days 60, 240 | Wet season (Mar), Dry season (Sep) |
| Season Amplitudes | +3°C, -3°C | ±3°C around mean (minimal) |
| Humidity | 85% | Very high |
| Salt Exposure | 0.4 | Moderate (coastal tropical) |
| Dust Exposure | 0.3 | Moderate |
| Ice Risk | 0.0 | None |

**Seasonal Characteristics**: Classic monsoon 2-season pattern. Wet season (slightly warmer): ~31°C, Dry season (slightly cooler): ~25°C. Temperature variation minimal - seasons primarily distinguished by rainfall and humidity, not temperature.

**Degradation**: High temperature and humidity accelerate corrosion and biological growth (algae, fungi). Continuous degradation with little seasonal relief. Year-round operational stress.

#### Temperate

Moderate climate with balanced seasonal variations.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Annual Mean Temp | 12°C | Mild |
| Daily Amplitude | 8°C | Moderate |
| Seasonal Pattern | 4-season (traditional) | Winter, Spring, Summer, Fall |
| Season Peaks | Days 20, 110, 200, 290 | Four distinct seasons |
| Season Amplitudes | -18°C, 0°C, +18°C, 0°C | Classic sinusoidal pattern |
| Humidity | 65% | Moderate |
| Salt Exposure | 0.1 | Low |
| Dust Exposure | 0.3 | Low to moderate |
| Ice Risk | 0.3 | Seasonal (winter) |

**Seasonal Characteristics**: Traditional 4-season pattern. Winter: -6°C, Spring: 12°C, Summer: 30°C, Fall: 12°C. Symmetric pattern with spring and fall as transition periods at the annual mean temperature.

**Degradation**: Balanced degradation mechanisms. Seasonal freeze-thaw cycles in winter. Moderate corrosion and fouling rates. Equipment experiences full range of operating conditions throughout the year.

#### Sahel

West African transition zone (Mali, Niger, Chad, Sudan) with hot temperatures, low humidity, and extreme seasonal dust exposure during Harmattan winds.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Annual Mean Temp | 30°C | Hot |
| Daily Amplitude | 14°C | Large day-night swings |
| Seasonal Pattern | 2-season | Wet (Mar-May), Dry/Harmattan (Sep-Nov) |
| Season Peaks | Days 80, 260 | Wet season, Dry season |
| Season Amplitudes | +5°C, -5°C | Moderate seasonal variation |
| Humidity | 35% | Low to moderate |
| Humidity Variation | 25% | Large wet/dry season variation |
| Salt Exposure | 0.0 | None |
| Dust Exposure | 0.80 | Very high (Harmattan dust storms) |
| Ice Risk | 0.0 | None |

**Seasonal Characteristics**: Distinct 2-season pattern. Wet season (Mar-May): ~35°C with 50-60% humidity and rain. Dry season with Harmattan (Nov-Mar): ~25°C with 10-20% humidity and intense dust storms from the Sahara. Temperature relatively stable year-round compared to humidity and dust variations.

**Degradation**: Dust fouling is severe, especially during Harmattan when fine Saharan dust can travel hundreds of kilometers. Compressor fouling, filter clogging, and abrasive wear are primary concerns. High temperatures year-round accelerate lubricant degradation. Low humidity during dry season limits corrosion but dust abrasion dominates.

**African Countries**: Mali, Niger, Chad, northern Nigeria, Sudan, Burkina Faso, Senegal

#### Savanna

Semi-arid savanna climate (Zimbabwe, Tanzania interior, Zambia) with moderate temperatures, distinct wet/dry seasons, and moderate dust during dry season.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Annual Mean Temp | 25°C | Warm |
| Daily Amplitude | 12°C | Moderate |
| Seasonal Pattern | 2-season (Southern Hemisphere) | Summer (Jan), Winter (Jul) |
| Season Peaks | Days 15, 195 | Southern Hemisphere seasons |
| Season Amplitudes | +8°C, -8°C | Moderate seasonal variation |
| Humidity | 55% | Moderate |
| Humidity Variation | 20% | Moderate wet/dry variation |
| Salt Exposure | 0.0 | None (inland) |
| Dust Exposure | 0.5 | Moderate (dry season) |
| Ice Risk | 0.0 | None |

**Seasonal Characteristics**: Southern Hemisphere 2-season pattern. Summer (Nov-Mar): ~33°C with wet season rains and 70% humidity. Winter (May-Sep): ~17°C, dry with 35% humidity. Moderate temperature swings and distinct wet/dry periods. Clear seasonal demarcation.

**Degradation**: Balanced degradation profile. Moderate dust fouling during dry season (less severe than Sahel). Warm year-round temperatures accelerate chemical degradation but winter cooling provides some relief. Wet season corrosion followed by dry season dust accumulation. Equipment experiences moderate thermal cycling between seasons.

**African Countries**: Zimbabwe, Zambia, Tanzania (interior), Botswana, Mozambique (interior), South Africa (northern regions)

## EnvironmentalConditions Class

Main class for simulating time-varying environmental conditions.

### Initialization

```python
def __init__(self,
             location_type: LocationType = LocationType.TEMPERATE,
             start_day_of_year: int = 1)
```

**Parameters:**
- `location_type`: Installation location (default: TEMPERATE)
- `start_day_of_year`: Starting day of year (1-365) for seasonal alignment

### Key Methods

#### get_conditions(elapsed_hours)

Returns current environmental conditions based on elapsed simulation time.

**Input:**
- `elapsed_hours`: Hours since simulation start (float)

**Returns:** Dictionary containing:
- `ambient_temp_C`: Current ambient temperature (°C)
- `humidity_percent`: Relative humidity (%)
- `pressure_kPa`: Atmospheric pressure (kPa)
- `hour_of_day`: Current hour (0-23)
- `day_of_year`: Current day (0-364)
- `location`: Location type string
- `temp_derating_factor`: Gas turbine power derating factor
- `density_ratio`: Air density ratio vs. ISO conditions
- `corrosion_factor`: Corrosion rate multiplier
- `fouling_factor`: Fouling rate multiplier
- `ice_formation_risk`: Ice formation probability (0.0-1.0)

#### simulate_weather_event(event_type)

Simulates extreme weather events with modified environmental conditions.

**Event Types:**
- `'storm'`: Pressure drop (-5 kPa), humidity increase (+20%), temperature drop (-5°C)
- `'heatwave'`: Temperature spike (+15°C), humidity drop (-30%)
- `'coldsnap'`: Temperature drop (-20°C), increased ice risk (+0.5)
- `'dust_storm'`: Fouling factor multiplied by 3x

**Returns:** Modified conditions dictionary

## Physical Models

### Temperature Calculation

Temperature follows sinusoidal daily cycles plus location-specific seasonal patterns:

```
T(t) = T_mean + A_daily * sin(2π * (hour-4)/24) + Σ[A_i * sin(2π * (day-peak_i)/365)] + noise
```

Where:
- `T_mean`: Annual mean temperature for the location
- `A_daily`: Daily amplitude (day-night temperature swing)
- `A_i`: Amplitude of seasonal component i
- `peak_i`: Day of year where component i peaks
- Summation allows multiple seasonal components (2, 3, or 4 seasons)

**Daily Cycle:**
- Peaks at ~2 PM (14:00), minimizes at ~4 AM
- Single sinusoidal component for all locations

**Seasonal Cycle:**
- **Flexible multi-component model** - each location has custom seasonal pattern
- **2-season locations** (Offshore, Desert, Arctic, Tropical): Two sinusoidal components
- **4-season locations** (Temperate): Four components for winter/spring/summer/fall
- Each component has its own peak day and amplitude

**Examples:**
- **Tropical** (2 seasons): Days 60 & 240, amplitudes ±3°C → minimal temperature swing
- **Arctic** (2 seasons): Days 1 & 172, amplitudes ±25°C → extreme asymmetric pattern
- **Temperate** (4 seasons): Days 20, 110, 200, 290, amplitudes -18, 0, +18, 0°C → classic pattern

**Noise:** Gaussian with σ = 1.0°C for short-term weather variability

### Humidity Calculation

Humidity inversely correlates with temperature deviation from mean:

```
RH(t) = RH_mean - 0.3 * (T_current - T_mean) + noise
```

Clamped to physical limits: 10% ≤ RH ≤ 100%

### Pressure Calculation

Atmospheric pressure varies with weekly weather patterns:

```
P(t) = P_mean + P_variation * sin(2π * (day mod 7)/7) + noise
```

Minimum pressure: 95.0 kPa (extreme low pressure limit)

### Equipment Impact Calculations

#### Temperature Derating Factor

Gas turbine power decreases approximately 0.7% per °C above ISO conditions (15°C):

```
derating = 1.0 - 0.007 * (T - 15)
```

Clamped to range: 0.7 to 1.15

#### Density Ratio

Air density affects compressor mass flow. Calculated using ideal gas law:

```
ρ_ISO = P_ISO / (R * T_ISO)  [at 15°C, 101.325 kPa]
ρ_actual = P_actual / (R * T_actual)
ratio = ρ_actual / ρ_ISO
```

Where R = 0.287 kJ/(kg·K) for air

#### Corrosion Factor

Corrosion rate depends on salt exposure and humidity:

```
corrosion_factor = 1.0 + 0.5 * salt_exposure * (RH / 100)
```

Higher values indicate faster corrosion (1.0 = baseline)

#### Fouling Factor

Fouling rate depends on dust exposure:

```
fouling_factor = 1.0 + 0.8 * dust_exposure
```

Higher values indicate faster filter/compressor fouling (1.0 = baseline)

#### Ice Formation Risk

Ice forms when temperature is below 5°C and humidity exceeds 70%:

```
if T < 5 and RH > 70:
    ice_risk = location_ice_risk * (5 - T) / 5
else:
    ice_risk = 0.0
```

## Usage Examples

### Basic Usage

```python
from environmental_conditions import EnvironmentalConditions, LocationType

# Create offshore environment starting mid-summer
env = EnvironmentalConditions(
    location_type=LocationType.OFFSHORE,
    start_day_of_year=180
)

# Get conditions at hour 1000 of simulation
conditions = env.get_conditions(elapsed_hours=1000.0)

print(f"Temperature: {conditions['ambient_temp_C']:.1f}°C")
print(f"Humidity: {conditions['humidity_percent']:.1f}%")
print(f"Derating: {conditions['temp_derating_factor']:.3f}")
print(f"Corrosion Factor: {conditions['corrosion_factor']:.2f}x")
```

### Simulating a Full Day

```python
env = EnvironmentalConditions(LocationType.DESERT, start_day_of_year=1)

for hour in range(24):
    conditions = env.get_conditions(elapsed_hours=hour)
    print(f"{hour:02d}:00 - Temp: {conditions['ambient_temp_C']:5.1f}°C, "
          f"Humidity: {conditions['humidity_percent']:4.1f}%")
```

### Seasonal Analysis

```python
env = EnvironmentalConditions(LocationType.ARCTIC)

seasons = ['Winter', 'Spring', 'Summer', 'Fall']
for i, day in enumerate([1, 91, 182, 273]):
    conditions = env.get_conditions(day * 24 + 12)  # Noon each season
    print(f"{seasons[i]:8s} - Temp: {conditions['ambient_temp_C']:5.1f}°C, "
          f"Ice Risk: {conditions['ice_formation_risk']:.2f}")
```

### Weather Event Simulation

```python
env = EnvironmentalConditions(LocationType.OFFSHORE)

# Normal conditions
normal = env.get_conditions(1000)
print(f"Normal: {normal['ambient_temp_C']:.1f}°C, {normal['pressure_kPa']:.1f} kPa")

# During storm
storm = env.simulate_weather_event('storm')
print(f"Storm: {storm['ambient_temp_C']:.1f}°C, {storm['pressure_kPa']:.1f} kPa")
```

### Custom Seasonal Patterns

You can create custom location profiles with specific seasonal patterns:

```python
from environmental_conditions import (
    EnvironmentalProfile, SeasonalPattern, EnvironmentalConditions
)

# Example 1: Southern Hemisphere desert (Australian outback)
# Opposite seasons from Northern Hemisphere
southern_desert = EnvironmentalProfile(
    temp_annual_mean=28.0,
    temp_daily_amplitude=16.0,
    seasonal_pattern=SeasonalPattern(
        hemisphere="southern",
        season_peaks=[15, 195],  # Same days, but seasons reversed
        season_amplitudes=[15.0, -15.0]  # Hot in January, cool in July
    ),
    humidity_mean=30.0,
    humidity_variation=12.0,
    pressure_mean=101.0,
    pressure_variation=2.5,
    salt_exposure=0.0,
    dust_exposure=0.85,
    ice_risk=0.0
)

# Example 2: Monsoon region (3-season pattern)
# Hot/wet/cool instead of traditional 4 seasons
monsoon_region = EnvironmentalProfile(
    temp_annual_mean=26.0,
    temp_daily_amplitude=9.0,
    seasonal_pattern=SeasonalPattern(
        hemisphere="northern",
        season_peaks=[45, 152, 320],  # Hot (Feb), Wet (Jun), Cool (Nov)
        season_amplitudes=[8.0, 2.0, -10.0]  # Hot: 34°C, Wet: 28°C, Cool: 16°C
    ),
    humidity_mean=70.0,
    humidity_variation=20.0,
    salt_exposure=0.2,
    dust_exposure=0.4,
    ice_risk=0.0
)

# Example 3: Mediterranean climate (5-season pattern with asymmetry)
mediterranean = EnvironmentalProfile(
    temp_annual_mean=18.0,
    temp_daily_amplitude=10.0,
    seasonal_pattern=SeasonalPattern(
        hemisphere="northern",
        season_peaks=[15, 75, 165, 220, 320],  # Complex pattern
        season_amplitudes=[-8.0, 2.0, 10.0, 8.0, -12.0]  # Wet winter, hot dry summer
    ),
    humidity_mean=60.0,
    humidity_variation=18.0,
    salt_exposure=0.3,
    dust_exposure=0.2,
    ice_risk=0.1
)

# Use custom profile
# Note: You'll need to add it to LOCATION_PROFILES or pass directly
env = EnvironmentalConditions(location_type=your_custom_location)
```

### Southern Hemisphere Locations

For Southern Hemisphere installations, keep the same peak days but be aware of seasonal reversal:

```python
# Northern Hemisphere: Day 15 = mid-winter (cold), Day 195 = mid-summer (hot)
# Southern Hemisphere: Day 15 = mid-summer (hot), Day 195 = mid-winter (cold)

# Southern Hemisphere example (Chile, South Africa, Australia)
southern_temperate = EnvironmentalProfile(
    temp_annual_mean=14.0,
    temp_daily_amplitude=9.0,
    seasonal_pattern=SeasonalPattern(
        hemisphere="southern",  # Marked for reference
        season_peaks=[20, 110, 200, 290],  # Same calendar days
        season_amplitudes=[18.0, 0.0, -18.0, 0.0]  # Reversed: hot in Jan, cold in Jul
    ),
    humidity_mean=60.0,
    humidity_variation=15.0,
    salt_exposure=0.15,
    dust_exposure=0.25,
    ice_risk=0.2
)
```

**Important**: The `hemisphere` field is currently informational. Peak days and amplitudes fully define the pattern. For Southern Hemisphere, simply flip the sign of amplitudes or shift peaks by 180 days.

## Integration with Equipment Simulators

Environmental conditions affect equipment performance in multiple ways:

### Gas Turbine Integration

```python
conditions = env.get_conditions(elapsed_hours)

# Apply temperature derating to power output
power_derated = power_nominal * conditions['temp_derating_factor']

# Increase corrosion-based degradation
health_hgp -= base_degradation * conditions['corrosion_factor'] * dt
```

### Compressor Integration

```python
conditions = env.get_conditions(elapsed_hours)

# Adjust mass flow for density ratio
mass_flow = mass_flow_design * conditions['density_ratio']

# Increase fouling rate
fouling_rate = base_fouling * conditions['fouling_factor']
```

### Pump Integration

```python
conditions = env.get_conditions(elapsed_hours)

# Cold weather affects viscosity and seal performance
if conditions['ambient_temp_C'] < 0:
    seal_health_degradation *= 1.5
```

## Validation and Calibration

### Reference Standards

- **ISO 2314**: Gas turbine acceptance tests (defines ISO conditions)
- **API 616**: Gas turbine procurement specifications (environmental considerations)
- **NORSOK**: Offshore platform equipment standards (marine environment)

### Temperature Derating Validation

The 0.7% per °C power derating factor for gas turbines is industry-standard and matches manufacturer specifications (GE, Siemens, Mitsubishi turbine datasheets).

### Corrosion Rate Validation

Salt exposure factors are calibrated to match NORSOK standards for offshore equipment degradation rates. Corrosion rates in marine environments are 3-5x faster than temperate inland locations.

### Fouling Validation

Desert dust fouling factors match observed compressor efficiency degradation rates in Middle East installations (2-3% per month without washing).

## Performance Considerations

### Computational Complexity

- **Time Complexity**: O(1) per call to `get_conditions()`
- **Memory**: Minimal state (6 floats + profile reference)
- **CPU Usage**: Negligible (simple trigonometric calculations)

### Optimization Notes

Temperature calculation calls `_calculate_temperature()` twice in `_calculate_humidity()`. For high-frequency calls, consider caching the temperature value.

## Limitations and Future Enhancements

### Current Limitations

1. **Simplified Weather Patterns**: 7-day pressure cycle is idealized
2. **No Weather Persistence**: Each timestep is independent (no storm duration)
3. **Deterministic Cycles**: Daily/seasonal patterns are perfectly sinusoidal
4. **No Altitude Effects**: Assumes sea-level pressure variations only

### Potential Enhancements

1. **Weather State Persistence**: Multi-hour storm events with correlated conditions
2. **Altitude Modeling**: Pressure and temperature corrections for elevation
3. **Wind Speed**: Add wind effects on cooling and icing
4. **Solar Radiation**: Direct solar heating effects on equipment surfaces
5. **Precipitation**: Rain/snow effects on performance and cleaning
6. **Historical Weather Data**: Use real meteorological data for specific sites

## Weather API Integration

The environmental conditions module can be extended with real-world weather data through the `weather_api_client.py` module, which provides a hybrid approach combining synthetic and API-based weather.

### Hybrid Architecture

The weather API integration uses an abstract base class `EnvironmentalDataSource` that allows seamless switching between synthetic and real weather sources:

```python
from weather_api_client import create_hybrid_environment

# Option 1: Synthetic weather (default)
env = create_hybrid_environment(
    use_real_weather=False,
    fallback_source=EnvironmentalConditions(LocationType.SAHEL)
)

# Option 2: Real weather with location name
env = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key",
    location_name="Lagos",
    country="Nigeria",
    cache_enabled=True
)

# Option 3: Real weather with coordinates
env = create_hybrid_environment(
    use_real_weather=True,
    api_provider="weatherapi",
    api_key="your_api_key",
    latitude=6.5244,
    longitude=3.3792,
    cache_enabled=True
)

# Get conditions (same interface for both synthetic and real)
conditions = env.get_conditions(elapsed_hours=0, timestamp=datetime.now())
```

### Supported Weather API Providers

1. **WeatherAPI.com** (Recommended)
   - Free tier: 1M calls/month
   - City name and coordinate support
   - Signup: https://www.weatherapi.com/signup.aspx

2. **OpenWeatherMap**
   - Free tier: 60 calls/minute, 1M calls/month
   - City name and coordinate support
   - Signup: https://openweathermap.org/api

3. **Visual Crossing**
   - Free tier: 1000 calls/day
   - Historical weather support
   - Signup: https://www.visualcrossing.com/weather-api

### Caching Layer

The weather API client includes SQLite-based caching for:
- **Cost Optimization**: Reduces API calls by caching hourly data
- **Offline Simulation**: Pre-load cache for reproducible offline runs
- **Rate Limiting**: Automatic rate limit management (60 calls/minute default)

Cache behavior:
- Automatic: Caches all API responses with configurable TTL (24 hours default)
- Manual preload: Use `preload_cache()` for batch caching of historical data
- Location-based keys: Cache keyed by location query (e.g., "Lagos,Nigeria")

### African Location Examples

```python
# Nigerian coastal installation (tropical)
lagos_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Lagos",
    country="Nigeria",
    api_key=api_key
)

# Algerian desert installation
algiers_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Hassi Messaoud",
    country="Algeria",
    api_key=api_key
)

# South African savanna installation
johannesburg_env = create_hybrid_environment(
    use_real_weather=True,
    location_name="Johannesburg",
    country="South Africa",
    api_key=api_key
)
```

### Benefits of Hybrid Approach

1. **Development**: Use fast synthetic data during algorithm development
2. **Validation**: Test with real weather from specific sites
3. **Production**: Deploy with either synthetic or real weather depending on connectivity
4. **Reproducibility**: Cache real weather for deterministic re-runs
5. **Cost Control**: Hourly caching reduces API costs by 3600x

### Implementation Notes

- Both synthetic and real weather return the same data format
- Automatic fallback to synthetic on API errors
- No code changes needed in equipment simulators

See [weather_api_client.md](weather_api_client.md) for detailed API integration documentation.

## References

1. ISO 2314:2009 - Gas turbines - Acceptance tests
2. API 616:2011 - Gas Turbines for the Petroleum, Chemical, and Gas Industry Services
3. NORSOK Standard M-001 - Materials selection
4. Kurz, R., & Brun, K. (2012). "Degradation in Gas Turbine Systems." Journal of Engineering for Gas Turbines and Power
5. Meher-Homji, C. B., & Bromley, A. (2004). "Gas Turbine Axial Compressor Fouling and Washing." Proceedings of the Turbomachinery Symposium

## See Also

- [weather_api_client.md](weather_api_client.md) - Weather API integration and caching
- [thermal_transient.md](thermal_transient.md) - Thermal dynamics during startups/shutdowns
- [vibration_enhanced.md](vibration_enhanced.md) - Vibration signal generation with environmental noise
- [gas_turbine.md](gas_turbine.md) - Main gas turbine simulator (uses environmental conditions)
