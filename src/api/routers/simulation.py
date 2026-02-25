"""
Simulation Router — Trigger simulation runs and poll job status.
Placeholder that expands as the system matures.
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from api.dependencies import get_db, get_master_data
from api.config import get_settings
from ingestion.db_setup import Database, MasterData
from ingestion.equipment_sim import simulate_equipment as sim_equip
from ingestion.bulk_insert import bulk_insert_telemetry, insert_failures, insert_maintenance
from data_simulation.gas_turbine import GasTurbine
from data_simulation.compressor import Compressor
from data_simulation.pump import Pump
from data_simulation.physics.weather_api_client import (
    WeatherConfig, WeatherAPIClient, CachedWeatherEnvironment
)
from data_simulation.physics.environmental_conditions import (
    EnvironmentalConditions, LocationType
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job tracker
_jobs: Dict[str, dict] = {}

class SimulationRequest(BaseModel):
    equipment_ids: List[int]
    equipment_type: str  # turbine, compressor, pump
    duration_days: int = Field(180, ge=1, le=730)
    sample_interval_min: int = Field(5, ge=1, le=60)
    degradation_multiplier: float = Field(1.0, ge=0.1, le=10.0)
    auto_ingest: bool = True

    weather_location_name: Optional[str] = Field(
        None, description="Location name for weather API (e.g., 'Lagos, Nigeria')")
    weather_latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Latitude for weather lookup")
    weather_longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Longitude for weather lookup")
    weather_fallback_location_type: Optional[str] = Field(
        None, description="Synthetic fallback profile if API fails "
        "(offshore, desert, arctic, tropical, temperate, sahel, savanna)")

    @model_validator(mode="after")
    def validate_weather_coords(self):
        if (self.weather_latitude is None) != (self.weather_longitude is None):
            raise ValueError(
                "weather_latitude and weather_longitude must both be provided or both omitted")
        return self

class SimulationJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class SimulationStatusResponse(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_generated: int = 0
    failures_generated: int = 0
    error: Optional[str] = None

def _build_env_model(request: SimulationRequest):
    """Build an environmental data source from the simulation request.

    Returns None if no weather parameters were provided (preserves existing behavior).
    Falls back to synthetic EnvironmentalConditions when the API key is missing or
    the weather client cannot be created.
    """
    has_location = request.weather_location_name is not None
    has_coords = (request.weather_latitude is not None
                  and request.weather_longitude is not None)

    if not has_location and not has_coords:
        return None

    # Build synthetic fallback
    fallback_type = LocationType.TEMPERATE
    if request.weather_fallback_location_type:
        try:
            fallback_type = LocationType(request.weather_fallback_location_type)
        except ValueError:
            pass
    fallback = EnvironmentalConditions(location_type=fallback_type)

    # Check for API key
    settings = get_settings()
    if not settings.weather_api_key:
        logger.warning("WEATHER_API_KEY not configured; using synthetic fallback")
        return fallback

    config = WeatherConfig(
        api_provider=settings.weather_api_provider,
        api_key=settings.weather_api_key,
        location_name=request.weather_location_name or "",
        latitude=request.weather_latitude or 0.0,
        longitude=request.weather_longitude or 0.0,
        cache_enabled=True,
    )

    try:
        client = WeatherAPIClient(config)
    except Exception:
        logger.warning("Failed to create WeatherAPIClient; using synthetic fallback")
        return fallback

    return CachedWeatherEnvironment(
        weather_client=client,
        fallback_source=fallback,
        config=config,
    )


@router.post("/run", response_model=SimulationJobResponse, status_code=202)
def trigger_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
    master: MasterData = Depends(get_master_data),
):
    """Trigger a simulation run as a background task.

    Returns a job_id to poll for status via GET /{job_id}/status.
    If auto_ingest is true, simulation output is bulk-inserted into the database.
    """
    if request.equipment_type not in ("turbine", "compressor", "pump"):
        raise HTTPException(400, f"Unknown equipment type: {request.equipment_type}")

    # Validate equipment IDs exist
    configs = master.get_configs(request.equipment_ids, request.equipment_type)
    if len(configs) != len(request.equipment_ids):
        found_ids = {c.get(f"{request.equipment_type}_id") or c.get("turbine_id") or c.get("compressor_id") or c.get("pump_id") for c in configs}
        missing = set(request.equipment_ids) - found_ids
        raise HTTPException(404, f"Equipment IDs not found: {missing}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "started_at": None,
        "completed_at": None,
        "records_generated": 0,
        "failures_generated": 0,
        "error": None,
    }

    def _run_simulation():
        equipment_classes = {
            "turbine": GasTurbine,
            "compressor": Compressor,
            "pump": Pump,
        }

        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.now()

        total_telemetry = []
        total_failures = []
        total_maintenance = []

        try:
            # Build weather environment (shared across all equipment in this run)
            env_model = _build_env_model(request)

            for config in configs:
                eq_type = request.equipment_type
                EquipmentClass = equipment_classes[eq_type]

                # Determine equipment ID from config
                id_keys = {"turbine": "turbine_id", "compressor": "compressor_id", "pump": "pump_id"}
                eq_id = config[id_keys[eq_type]]

                # Build equipment instance from config
                constructor_kwargs = {
                    k: v for k, v in config.items()
                    if k not in (id_keys[eq_type], "name", "serial_number", "location",
                                 "installed_date", "status")
                }
                if env_model is not None:
                    constructor_kwargs["env_model"] = env_model
                    constructor_kwargs["enable_environmental"] = True

                equipment = EquipmentClass(**constructor_kwargs)

                num_samples = int(request.duration_days * 24 * 60 / request.sample_interval_min)
                start_time = datetime(2025, 7, 1)
                interval = timedelta(minutes=request.sample_interval_min)

                telemetry_batch = []
                for record in sim_equip(
                    equipment, eq_id, num_samples, start_time, interval,
                    degradation_multiplier=request.degradation_multiplier,
                ):
                    if record["type"] == "telemetry":
                        telemetry_batch.append(record)
                    elif record["type"] == "failure":
                        total_failures.append(record)
                    elif record["type"] in ("maintenance_start", "maintenance_complete"):
                        total_maintenance.append(record)

                total_telemetry.extend(telemetry_batch)
                _jobs[job_id]["records_generated"] = len(total_telemetry)
                _jobs[job_id]["failures_generated"] = len(total_failures)

            # Auto-ingest if requested
            if request.auto_ingest and total_telemetry:
                bulk_insert_telemetry(db, total_telemetry, request.equipment_type)
            if request.auto_ingest and total_failures:
                insert_failures(db, total_failures, request.equipment_type)
            if request.auto_ingest and total_maintenance:
                insert_maintenance(db, total_maintenance, request.equipment_type)

            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = datetime.now()

        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed_at"] = datetime.now()
            _jobs[job_id]["error"] = str(e)

    background_tasks.add_task(_run_simulation)

    return SimulationJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Simulation queued for {len(request.equipment_ids)} {request.equipment_type}(s)",
    )


@router.get("/{job_id}/status", response_model=SimulationStatusResponse)
def get_job_status(job_id: str):
    """Poll the status of a simulation job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    job = _jobs[job_id]
    return SimulationStatusResponse(job_id=job_id, **job)