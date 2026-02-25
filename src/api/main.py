"""
PdM API — FastAPI app.
Provides REST endpoints for master data, weather, telemetry, ML pipelines,
and simulation management.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from api.config import get_settings
from api.dependencies import init_database, shutdown_database, get_db
from api.routers import master_data, weather, telemetry, ml, simulation
from api.utils import TABLE_CONFIG

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    yield
    shutdown_database()


app = FastAPI(
    title="PdM — Predictive Maintenance API",
    description=(
        "REST API for equipment master data, location weather, telemetry "
        "ingestion/query, ML training pipelines, and simulation management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(master_data.router, prefix="/api/v1/master-data", tags=["Master Data"])
app.include_router(weather.router, prefix="/api/v1/weather", tags=["Weather & Locations"])
app.include_router(telemetry.router, prefix="/api/v1/telemetry", tags=["Telemetry"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["ML Pipelines"])
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])


@app.get("/health", tags=["System"])
def health_check():
    """Probes database connectivity. Returns 503 if DB is unreachable."""
    try:
        db = get_db()
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": "disconnected", "detail": str(e)},
        )


@app.get("/api/v1/meta/schema/{equipment_type}", tags=["System"])
def get_schema_contract(equipment_type: str):
    """Returns column schema contract for an equipment type.

    Identifies stable vs. derived columns so ML pipelines can validate
    feature expectations at startup.
    """
    if equipment_type not in TABLE_CONFIG:
        from fastapi import HTTPException
        raise HTTPException(404, f"Unknown equipment type: {equipment_type}")

    config = TABLE_CONFIG[equipment_type]
    stable_cols = (
        ["telemetry_id", "sample_time", "operating_hours"]
        + [config["id_col"]]
        + config["health_cols"]
        + config["key_numeric_cols"]
    )
    derived_cols = [
        "operating_state",
        "vibration_trend_7d",
        "temp_variation_24h",
        "speed_stability",
        "efficiency_degradation_rate",
        "pressure_ratio",
        "load_factor",
    ]
    return {
        "equipment_type": equipment_type,
        "stable_columns": stable_cols,
        "derived_columns": derived_cols,
        "note": "Stable columns are guaranteed not to rename or change units. "
                "Derived columns are computed server-side and may evolve.",
    }