"""
Weather & Locations Router — Environmental profiles, synthetic/live weather, cache management.
"""
from typing import List
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from api.config import get_settings
from api.schemas.weather import (
    LocationTypeEnum,
    EnvironmentalProfileResponse,
    WeatherConditionsResponse,
    CachePreloadRequest,
    CacheStatsResponse
)
from data_simulation.physics.environmental_conditions import (
        LocationType, LOCATION_PROFILES, EnvironmentalConditions)
from data_simulation.physics.weather_api_client import (
    WeatherAPIClient, WeatherConfig, CachedWeatherEnvironment)
from datetime import datetime
import glob
import sqlite3
import os

router = APIRouter()
def _get_profile(location_type: LocationTypeEnum):
    """Resolve LocationType enum and fetch profile."""
    try:
        lt = LocationType(location_type.value)
    except ValueError:
        raise HTTPException(404, f"Unknown location type: {location_type}")
    return lt, LOCATION_PROFILES[lt]

# Environmental profiles
@router.get("/profiles", response_model=List[EnvironmentalProfileResponse])
def list_profiles():
    """List all 7 predefined environmental profiles."""
    results = []
    for lt, profile in LOCATION_PROFILES.items():
        results.append(EnvironmentalProfileResponse(
            location_type=lt.value,
            temp_annual_mean=profile.temp_annual_mean,
            temp_daily_amplitude=profile.temp_daily_amplitude,
            humidity_mean=profile.humidity_mean,
            humidity_variation=profile.humidity_variation,
            pressure_mean=profile.pressure_mean,
            pressure_variation=profile.pressure_variation,
            salt_exposure=profile.salt_exposure,
            dust_exposure=profile.dust_exposure,
            ice_risk=profile.ice_risk
        ))
    return results

@router.get("/profiles/{location_type}", response_model=EnvironmentalProfileResponse)
def get_profile(location_type: LocationTypeEnum):
    """Get environmental profile for a specific location type."""
    lt, profile = _get_profile(location_type)
    return EnvironmentalProfileResponse(
        location_type=lt.value,
        temp_annual_mean=profile.temp_annual_mean,
        temp_daily_amplitude=profile.temp_daily_amplitude,
        humidity_mean=profile.humidity_mean,
        humidity_variation=profile.humidity_variation,
        pressure_mean=profile.pressure_mean,
        pressure_variation=profile.pressure_variation,
        salt_exposure=profile.salt_exposure,
        dust_exposure=profile.dust_exposure,
        ice_risk=profile.ice_risk
    )

# Synthetic weather
@router.get("/synthetic", response_model=WeatherConditionsResponse)
def get_synthetic_conditions(
    location_type: LocationTypeEnum = Query(...),
    elapsed_hours: float = Query(0.0, ge=0.0, description="Hours since simulation start"),
    start_day_of_year: int = Query(1, ge=1, le=365),
):
    """Get synthetic environmental conditions for a location type and time offset."""
    try:
        lt = LocationType(location_type.value)
    except ValueError:
        raise HTTPException(404, f"Unknown location type: {location_type}")
    env = EnvironmentalConditions(location_type=lt, start_day_of_year=start_day_of_year)
    conditions = env.get_conditions(elapsed_hours)
    return WeatherConditionsResponse(**conditions)

# Live weather
@router.get("/current", response_model=WeatherConditionsResponse)
def get_current_weather(
    location_name: str = Query(None, description="Location name (e.g., 'Lagos, Nigeria')"),
    latitude: float = Query(None),
    longitude: float = Query(None),
):
    """Fetch current weather from external API.

    Returns 503 if no WEATHER_API_KEY is configured.
    """
    settings = get_settings()
    if not settings.weather_api_key:
        raise HTTPException(503, "WEATHER_API_KEY not configured. Use /synthetic endpoint instead.")

    if not location_name and (latitude is None or longitude is None):
        raise HTTPException(400, "Provide location_name or both latitude and longitude")

    config = WeatherConfig(
        api_key=settings.weather_api_key,
        api_provider=settings.weather_api_provider,
        location_name=location_name or "",
        latitude=latitude or 0.0,
        longitude=longitude or 0.0,
    )
    client = WeatherAPIClient(config)
    try:
        conditions = client.get_conditions(elapsed_hours=0.0, timestamp=datetime.now())
    except Exception as e:
        raise HTTPException(502, f"Weather API error: {e}")

    return WeatherConditionsResponse(**conditions)


# Cache management
@router.post("/cache/preload", status_code=202)
def preload_cache(
    request: CachePreloadRequest,
    background_tasks: BackgroundTasks,
):
    """Preload weather cache for a date range. Runs as background task."""
    settings = get_settings()
    if not settings.weather_api_key:
        raise HTTPException(503, "WEATHER_API_KEY not configured")

    # Validate date range against rate limits
    hours_to_cache = (request.end_date - request.start_date).total_seconds() / 3600
    if hours_to_cache <= 0:
        raise HTTPException(400, "end_date must be after start_date")
    if hours_to_cache > 8760:  # 1 year max
        raise HTTPException(400, "Date range exceeds 1 year maximum")

    def _preload():
        config = WeatherConfig(
            api_key=settings.weather_api_key,
            api_provider=request.api_provider,
            location_name=request.location_name,
            cache_enabled=True,
        )
        client = WeatherAPIClient(config)
        fallback = EnvironmentalConditions(location_type=LocationType.TEMPERATE)
        cached_env = CachedWeatherEnvironment(
            weather_client=client, fallback_source=fallback, config=config
        )
        cached_env.preload_cache(
            start_date=request.start_date,
            end_date=request.end_date,
            interval_hours=1,
        )

    background_tasks.add_task(_preload)
    return {
        "message": "Cache preload started",
        "location": request.location_name,
        "hours_to_cache": int(hours_to_cache),
    }


@router.get("/cache/stats", response_model=CacheStatsResponse)
def get_cache_stats():
    """Get weather cache statistics across all location-specific DB files."""
    cache_files = glob.glob("weather_cache*.db")
    if not cache_files:
        return CacheStatsResponse(total_cached_entries=0)

    total, oldest, newest = 0, None, None
    for path in cache_files:
        try:
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM weather_cache"
            ).fetchone()
            conn.close()
            total += row[0] or 0
            if row[1] and (oldest is None or row[1] < oldest):
                oldest = row[1]
            if row[2] and (newest is None or row[2] > newest):
                newest = row[2]
        except Exception:
            continue
    return CacheStatsResponse(
        total_cached_entries=total, oldest_entry=oldest, newest_entry=newest
    )

@router.delete("/cache")
def clear_cache():
    """Clear all weather cache files (including location-specific DBs)."""
    cache_files = glob.glob("weather_cache*.db")
    if not cache_files:
        return {"message": "No cache files found"}
    for path in cache_files:
        os.remove(path)
    return {"message": f"Cleared {len(cache_files)} cache file(s)"}