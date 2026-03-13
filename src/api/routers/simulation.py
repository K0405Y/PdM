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
from api.config import get_settings, load_table_config
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
from data_simulation.ml_utils import OutputMode

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job tracker
_jobs: Dict[str, dict] = {}

class SimulationRequest(BaseModel):
    """Trigger a physics-based simulation for one or more equipment units."""
    equipment_ids: List[int]
    equipment_type: str  # turbine, compressor, pump
    duration_days: int = Field(180, ge=1, le=730)
    sample_interval_min: int = Field(5, ge=1, le=60)
    degradation_multiplier: float = Field(1.0, ge=0.1, le=10.0)
    auto_ingest: bool = True
    output_mode: str = Field(
        "ground_truth",
        description="Output mode: ground_truth (training with health indicators), "
                    "sensor_only (inference testing, no health)"
    )
    start_time: Optional[datetime] = Field(
        None, description="Simulation start time (ISO 8601). Defaults to now - duration_days.")

    weather_location_name: Optional[str] = Field(
        None, description="Location name for weather API (e.g., 'Lagos, Nigeria')")
    weather_latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Latitude for weather lookup")
    weather_longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Longitude for weather lookup")
    weather_fallback_location_type: Optional[str] = Field(
        None, description="Synthetic fallback profile if API fails "
        "(offshore, desert, arctic, tropical, temperate, sahel, savanna)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "equipment_ids": [1, 2, 3],
                "equipment_type": "turbine",
                "duration_days": 90,
                "sample_interval_min": 5,
                "degradation_multiplier": 1.5,
                "auto_ingest": True,
                "output_mode": "ground_truth",
                "start_time": "2025-01-01T00:00:00",
                "weather_location_name": "Lagos, Nigeria",
                "weather_fallback_location_type": "tropical",
            }]
        }
    }

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
        cache_ttl_hours=87600
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
    # Validate output_mode
    try:
        output_mode = OutputMode(request.output_mode)
    except ValueError:
        raise HTTPException(400, f"Invalid output_mode: {request.output_mode}. "
                                 f"Valid: {[m.value for m in OutputMode]}")

    yaml_cfg = load_table_config()
    if request.equipment_type not in yaml_cfg["equipment_types"]:
        raise HTTPException(400, f"Unknown equipment type: {request.equipment_type}")

    # Validate equipment IDs exist
    configs = master.get_configs(request.equipment_ids, request.equipment_type)
    if len(configs) != len(request.equipment_ids):
        yaml_cfg = load_table_config()
        id_col = yaml_cfg["equipment_types"][request.equipment_type]["master"]["id_column"]
        found_ids = {c.get(id_col) for c in configs}
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

    def _build_constructor_kwargs(eq_type: str, config: dict) -> dict:
        """Map DB config rows to equipment constructor parameters.

        Infers initial_health keys from insert_columns with 'initial_health_'
        prefix, and maps design params via constructor_params in the YAML config.
        """
        mcfg = yaml_cfg["equipment_types"][eq_type]["master"]
        kwargs = {"name": config.get("name", "")}

        # Design/operational params from constructor_params mapping
        for db_col, ctor_param in mcfg.get("constructor_params", {}).items():
            if db_col in config:
                kwargs[ctor_param] = config[db_col]

        # Build initial_health dict from insert_columns with "initial_health_" prefix
        health = {}
        for col in mcfg["insert_columns"]:
            if col.startswith("initial_health_"):
                key = col[len("initial_health_"):]
                health[key] = config.get(col, 0.9)
        if health:
            kwargs["initial_health"] = health

        return kwargs

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

            # Resolve simulation start time
            start_time = request.start_time or (
                datetime.now() - timedelta(days=request.duration_days)
            )

            # Pre-populate weather cache
            if isinstance(env_model, CachedWeatherEnvironment) and env_model.weather_client:
                try:
                    end_time = start_time + timedelta(days=request.duration_days)
                    logger.info(
                        "Preloading weather cache from %s to %s",
                        start_time.date(), end_time.date(),
                    )
                    env_model.preload_cache(start_time, end_time, interval_hours=1)
                except Exception as e:
                    logger.warning("Cache preload failed, will fetch per tick: %s", e)

            logger.info(
                "Starting simulation: %d %s(s), %d days, %d-min intervals",
                len(configs), request.equipment_type,
                request.duration_days, request.sample_interval_min,
            )

            for idx, config in enumerate(configs, 1):
                eq_type = request.equipment_type
                EquipmentClass = equipment_classes[eq_type]

                # Determine equipment ID from config
                yaml_cfg = load_table_config()
                id_col = yaml_cfg["equipment_types"][eq_type]["master"]["id_column"]
                eq_id = config[id_col]
                logger.info(
                    "Simulating %s %d/%d (ID=%s)",
                    eq_type, idx, len(configs), eq_id,
                )

                # Build equipment instance from config
                constructor_kwargs = _build_constructor_kwargs(eq_type, config)
                constructor_kwargs["output_mode"] = output_mode
                if env_model is not None:
                    constructor_kwargs["env_model"] = env_model
                    constructor_kwargs["enable_environmental"] = True

                equipment = EquipmentClass(**constructor_kwargs)

                telemetry_batch = []
                for record in sim_equip(
                    equipment, eq_id, eq_type,
                    request.duration_days, request.sample_interval_min,
                    start_time=start_time,
                    degradation_multiplier=request.degradation_multiplier,
                ):
                    if record["type"] == "telemetry":
                        telemetry_batch.append(record)
                    elif record["type"] == "failure":
                        total_failures.append(record)
                    elif record["type"] == "maintenance_start":
                        total_maintenance.append(record)

                total_telemetry.extend(telemetry_batch)
                _jobs[job_id]["records_generated"] = len(total_telemetry)
                _jobs[job_id]["failures_generated"] = len(total_failures)
                logger.info(
                    "  %s ID=%s done: %d records, %d failures",
                    eq_type, eq_id, len(telemetry_batch), len(total_failures),
                )

            logger.info(
                "Simulation complete: %d total records, %d failures, %d maintenance events",
                len(total_telemetry), len(total_failures), len(total_maintenance),
            )

            # Auto-ingest if requested
            use_test = (output_mode == OutputMode.SENSOR_ONLY)
            if request.auto_ingest and total_telemetry:
                bulk_insert_telemetry(db, total_telemetry, request.equipment_type,
                                      use_test_schema=use_test)
            if request.auto_ingest and total_failures:
                insert_failures(db, total_failures, use_test_schema=use_test)
            if request.auto_ingest and total_maintenance:
                insert_maintenance(db, total_maintenance, use_test_schema=use_test)

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