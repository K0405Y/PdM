"""
Pydantic schemas for weather and environmental endpoints.
"""
from typing import List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class LocationTypeEnum(str, Enum):
    offshore = "offshore"
    desert = "desert"
    arctic = "arctic"
    tropical = "tropical"
    temperate = "temperate"
    sahel = "sahel"
    savanna = "savanna"

class EnvironmentalProfileResponse(BaseModel):
    location_type: str
    temp_annual_mean: float
    temp_daily_amplitude: float
    humidity_mean: float
    humidity_variation: float
    pressure_mean: float
    pressure_variation: float
    salt_exposure: float
    dust_exposure: float
    ice_risk: float

class WeatherConditionsResponse(BaseModel):
    """Current or synthetic weather conditions."""
    ambient_temp_C: float
    humidity_percent: float
    pressure_kPa: float
    hour_of_day: float
    day_of_year: int
    location: str
    temp_derating_factor: Optional[float] = None
    density_ratio: Optional[float] = None
    corrosion_factor: Optional[float] = None
    fouling_factor: Optional[float] = None
    ice_formation_risk: Optional[float] = None

class CachePreloadRequest(BaseModel):
    """Preload weather cache for a location and date range."""
    location_name: str = Field(..., description="Location name for weather API (e.g., 'Lagos, Nigeria')")
    start_date: datetime
    end_date: datetime
    api_provider: str = "weatherapi"

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "location_name": "Lagos, Nigeria",
                "start_date": "2025-01-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "api_provider": "weatherapi",
            }]
        }
    }

class CacheStatsResponse(BaseModel):
    total_cached_entries: int
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None

class HistoricalWeatherEntry(BaseModel):
    timestamp: datetime
    ambient_temp_C: float
    humidity_percent: float
    pressure_kPa: float
    cached_at: datetime

class HistoricalWeatherResponse(BaseModel):
    location_name: str
    start_date: datetime
    end_date: datetime
    entries: List[HistoricalWeatherEntry]
    total_entries: int
