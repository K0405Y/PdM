"""
Master Data Router — Equipment CRUD, batch seeding, schema init, failure modes.
"""
import math
import os
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
from api.dependencies import get_db, get_db_session, get_master_data
from api.config import get_settings, load_table_config
from api.utils import rows_to_dicts, MASTER_TABLES
from api.schemas.common import PaginatedResponse, MessageResponse
from api.schemas.master_data import (
    EquipmentType,
    EquipmentStatus,
    GasTurbineCreate, GasTurbineUpdate, GasTurbineResponse,
    CompressorCreate, CompressorUpdate, CompressorResponse,
    PumpCreate, PumpUpdate, PumpResponse,
    SeedRequest, SeedResponse,
    FailureModeDetail,
)
from ingestion.db_setup import Database, MasterData

router = APIRouter()

# Schema initialisation
@router.post("/schemas/init", response_model=MessageResponse)
def init_schemas(db: Database = Depends(get_db)):
    """Execute all SQL schema scripts to initialise the database."""
    settings = get_settings()
    schemas_dir = settings.db_schemas_dir
    if not os.path.isdir(schemas_dir):
        raise HTTPException(500, f"Schemas directory not found: {schemas_dir}")

    scripts = sorted(f for f in os.listdir(schemas_dir) if f.endswith(".sql"))
    for script in scripts:
        db.execute_script(os.path.join(schemas_dir, script))

    return MessageResponse(message=f"Executed {len(scripts)} schema scripts", count=len(scripts))


# Batch seeding
@router.post("/seed", response_model=SeedResponse, status_code=201)
def seed_equipment(request: SeedRequest, master: MasterData = Depends(get_master_data)):
    """Batch-seed equipment master data (turbines, compressors, pumps)."""
    turbine_ids = master.seed_turbines(request.turbine_count) if request.turbine_count else []
    compressor_ids = master.seed_compressors(request.compressor_count) if request.compressor_count else []
    pump_ids = master.seed_pumps(request.pump_count) if request.pump_count else []
    return SeedResponse(
        turbine_ids=turbine_ids,
        compressor_ids=compressor_ids,
        pump_ids=pump_ids,
        message=(
            f"Seeded {len(turbine_ids)} turbines, "
            f"{len(compressor_ids)} compressors, "
            f"{len(pump_ids)} pumps"
        ),
    )

# Helper to get master config for an equipment type
def _master_cfg(eq_type: str) -> dict:
    """Return the master table config for an equipment type from YAML."""
    cfg = load_table_config()
    return cfg["equipment_types"][eq_type]["master"]


# Gas Turbines CRUD
@router.get("/turbines", response_model=PaginatedResponse)
def list_turbines(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: Optional[str] = None,
    location: Optional[str] = None,
    session: Session = Depends(get_db_session),
):
    """List gas turbines with offset pagination and optional filters."""
    mcfg = _master_cfg("turbine")
    table, id_col = mcfg["table"], mcfg["id_column"]
    where, params = _build_where({"status": status, "location": location})
    total = session.execute(
        text(f"SELECT COUNT(*) FROM {table}{where}"), params
    ).scalar()
    rows = session.execute(
        text(f"SELECT * FROM {table}{where} ORDER BY {id_col} LIMIT :lim OFFSET :off"),
        {**params, "lim": page_size, "off": (page - 1) * page_size},
    )
    columns = list(rows.keys())
    items = rows_to_dicts(rows.fetchall(), columns)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        total_pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/turbines/{turbine_id}", response_model=GasTurbineResponse)
def get_turbine(turbine_id: int, master: MasterData = Depends(get_master_data)):
    """Retrieve a single gas turbine by ID."""
    configs = master.get_configs([turbine_id], "turbine")
    if not configs:
        raise HTTPException(404, "Turbine not found")
    return GasTurbineResponse(**configs[0])


@router.post("/turbines", response_model=GasTurbineResponse, status_code=201)
def create_turbine(body: GasTurbineCreate, session: Session = Depends(get_db_session)):
    """Create a new gas turbine. Only 'name' is required; all other fields have defaults."""
    mcfg = _master_cfg("turbine")
    table = mcfg["table"]
    cols = mcfg["insert_columns"]
    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    result = session.execute(
        text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) RETURNING *"),
        {c: getattr(body, c) for c in cols},
    )
    columns = list(result.keys())
    row = result.fetchone()
    session.commit()
    return GasTurbineResponse(**dict(zip(columns, row)))


@router.patch("/turbines/{turbine_id}", response_model=GasTurbineResponse)
def update_turbine(turbine_id: int, body: GasTurbineUpdate, session: Session = Depends(get_db_session)):
    """Partially update a gas turbine. Supply only the fields to change."""
    mcfg = _master_cfg("turbine")
    table, id_col = mcfg["table"], mcfg["id_column"]
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = turbine_id
    result = session.execute(
        text(f"UPDATE {table} SET {set_clause} WHERE {id_col} = :id RETURNING *"),
        updates,
    )
    columns = list(result.keys())
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "Turbine not found")
    session.commit()
    return GasTurbineResponse(**dict(zip(columns, row)))


@router.delete("/turbines/{turbine_id}", response_model=MessageResponse)
def delete_turbine(turbine_id: int, session: Session = Depends(get_db_session)):
    """Delete a gas turbine by ID. Returns 404 if not found."""
    mcfg = _master_cfg("turbine")
    table, id_col = mcfg["table"], mcfg["id_column"]
    result = session.execute(
        text(f"DELETE FROM {table} WHERE {id_col} = :id"),
        {"id": turbine_id},
    )
    session.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "Turbine not found")
    return MessageResponse(message="Turbine deleted", count=1)


# Compressors CRUD
@router.get("/compressors", response_model=PaginatedResponse)
def list_compressors(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: Optional[str] = None,
    location: Optional[str] = None,
    session: Session = Depends(get_db_session),
):
    """List compressors with offset pagination and optional filters."""
    mcfg = _master_cfg("compressor")
    table, id_col = mcfg["table"], mcfg["id_column"]
    where, params = _build_where({"status": status, "location": location})
    total = session.execute(
        text(f"SELECT COUNT(*) FROM {table}{where}"), params
    ).scalar()
    rows = session.execute(
        text(f"SELECT * FROM {table}{where} ORDER BY {id_col} LIMIT :lim OFFSET :off"),
        {**params, "lim": page_size, "off": (page - 1) * page_size},
    )
    columns = list(rows.keys())
    items = rows_to_dicts(rows.fetchall(), columns)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        total_pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/compressors/{compressor_id}", response_model=CompressorResponse)
def get_compressor(compressor_id: int, master: MasterData = Depends(get_master_data)):
    """Retrieve a single compressor by ID."""
    configs = master.get_configs([compressor_id], "compressor")
    if not configs:
        raise HTTPException(404, "Compressor not found")
    return CompressorResponse(**configs[0])


@router.post("/compressors", response_model=CompressorResponse, status_code=201)
def create_compressor(body: CompressorCreate, session: Session = Depends(get_db_session)):
    """Create a new compressor. Only 'name' is required; all other fields have defaults."""
    mcfg = _master_cfg("compressor")
    table = mcfg["table"]
    cols = mcfg["insert_columns"]
    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    result = session.execute(
        text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) RETURNING *"),
        {c: getattr(body, c) for c in cols},
    )
    columns = list(result.keys())
    row = result.fetchone()
    session.commit()
    return CompressorResponse(**dict(zip(columns, row)))


@router.patch("/compressors/{compressor_id}", response_model=CompressorResponse)
def update_compressor(compressor_id: int, body: CompressorUpdate, session: Session = Depends(get_db_session)):
    """Partially update a compressor. Supply only the fields to change."""
    mcfg = _master_cfg("compressor")
    table, id_col = mcfg["table"], mcfg["id_column"]
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = compressor_id
    result = session.execute(
        text(f"UPDATE {table} SET {set_clause} WHERE {id_col} = :id RETURNING *"),
        updates,
    )
    columns = list(result.keys())
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "Compressor not found")
    session.commit()
    return CompressorResponse(**dict(zip(columns, row)))


@router.delete("/compressors/{compressor_id}", response_model=MessageResponse)
def delete_compressor(compressor_id: int, session: Session = Depends(get_db_session)):
    """Delete a compressor by ID. Returns 404 if not found."""
    mcfg = _master_cfg("compressor")
    table, id_col = mcfg["table"], mcfg["id_column"]
    result = session.execute(
        text(f"DELETE FROM {table} WHERE {id_col} = :id"),
        {"id": compressor_id},
    )
    session.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "Compressor not found")
    return MessageResponse(message="Compressor deleted", count=1)


# Pumps CRUD
@router.get("/pumps", response_model=PaginatedResponse)
def list_pumps(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: Optional[str] = None,
    location: Optional[str] = None,
    service_type: Optional[str] = None,
    session: Session = Depends(get_db_session),
):
    """List pumps with offset pagination and optional filters (including service_type)."""
    mcfg = _master_cfg("pump")
    table, id_col = mcfg["table"], mcfg["id_column"]
    where, params = _build_where({"status": status, "location": location, "service_type": service_type})
    total = session.execute(
        text(f"SELECT COUNT(*) FROM {table}{where}"), params
    ).scalar()
    rows = session.execute(
        text(f"SELECT * FROM {table}{where} ORDER BY {id_col} LIMIT :lim OFFSET :off"),
        {**params, "lim": page_size, "off": (page - 1) * page_size},
    )
    columns = list(rows.keys())
    items = rows_to_dicts(rows.fetchall(), columns)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        total_pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/pumps/{pump_id}", response_model=PumpResponse)
def get_pump(pump_id: int, master: MasterData = Depends(get_master_data)):
    """Retrieve a single pump by ID."""
    configs = master.get_configs([pump_id], "pump")
    if not configs:
        raise HTTPException(404, "Pump not found")
    return PumpResponse(**configs[0])


@router.post("/pumps", response_model=PumpResponse, status_code=201)
def create_pump(body: PumpCreate, session: Session = Depends(get_db_session)):
    """Create a new pump. Only 'name' is required; all other fields have defaults."""
    mcfg = _master_cfg("pump")
    table = mcfg["table"]
    cols = mcfg["insert_columns"]
    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    result = session.execute(
        text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) RETURNING *"),
        {c: getattr(body, c) for c in cols},
    )
    columns = list(result.keys())
    row = result.fetchone()
    session.commit()
    return PumpResponse(**dict(zip(columns, row)))


@router.patch("/pumps/{pump_id}", response_model=PumpResponse)
def update_pump(pump_id: int, body: PumpUpdate, session: Session = Depends(get_db_session)):
    """Partially update a pump. Supply only the fields to change."""
    mcfg = _master_cfg("pump")
    table, id_col = mcfg["table"], mcfg["id_column"]
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = pump_id
    result = session.execute(
        text(f"UPDATE {table} SET {set_clause} WHERE {id_col} = :id RETURNING *"),
        updates,
    )
    columns = list(result.keys())
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "Pump not found")
    session.commit()
    return PumpResponse(**dict(zip(columns, row)))


@router.delete("/pumps/{pump_id}", response_model=MessageResponse)
def delete_pump(pump_id: int, session: Session = Depends(get_db_session)):
    """Delete a pump by ID. Returns 404 if not found."""
    mcfg = _master_cfg("pump")
    table, id_col = mcfg["table"], mcfg["id_column"]
    result = session.execute(
        text(f"DELETE FROM {table} WHERE {id_col} = :id"),
        {"id": pump_id},
    )
    session.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "Pump not found")
    return MessageResponse(message="Pump deleted", count=1)


# Equipment status change
@router.patch("/{equipment_type}/{equipment_id}/status", response_model=MessageResponse)
def update_equipment_status(
    equipment_type: EquipmentType,
    equipment_id: int,
    status: EquipmentStatus = Query(..., description="New status"),
    session: Session = Depends(get_db_session),
):
    """Change the status of any equipment (active, inactive, maintenance)."""
    table, id_col = MASTER_TABLES[equipment_type.value]
    result = session.execute(
        text(f"UPDATE {table} SET status = :status WHERE {id_col} = :id"),
        {"status": status.value, "id": equipment_id},
    )
    session.commit()
    if result.rowcount == 0:
        raise HTTPException(404, f"{equipment_type.value.title()} {equipment_id} not found")
    return MessageResponse(
        message=f"{equipment_type.value.title()} {equipment_id} status set to {status.value}",
        count=1,
    )


# Failure Modes — loaded from YAML config
def _build_failure_mode_metadata() -> Dict[str, List[FailureModeDetail]]:
    """Build FAILURE_MODE_METADATA from YAML config."""
    cfg = load_table_config()
    result = {}
    for eq_type, modes in cfg.get("failure_modes", {}).items():
        result[eq_type] = [
            FailureModeDetail(
                mode_code=m["mode_code"],
                equipment_type=eq_type,
                description=m["description"],
                failure_threshold=m["failure_threshold"],
                severity=m["severity"],
                primary_indicators=m.get("primary_indicators", []),
                lagging_indicators=m.get("lagging_indicators", []),
                typical_lead_time_hours=m.get("typical_lead_time_hours", ""),
            )
            for m in modes
        ]
    return result


FAILURE_MODE_METADATA = _build_failure_mode_metadata()


@router.get("/failure-modes", response_model=List[FailureModeDetail])
def list_failure_modes():
    """List all failure modes with ML-relevant metadata."""
    all_modes = []
    for modes in FAILURE_MODE_METADATA.values():
        all_modes.extend(modes)
    return all_modes


@router.get("/failure-modes/{equipment_type}", response_model=List[FailureModeDetail])
def list_failure_modes_by_type(equipment_type: str):
    """List failure modes for a specific equipment type with ML metadata."""
    if equipment_type not in FAILURE_MODE_METADATA:
        raise HTTPException(404, f"Unknown equipment type: {equipment_type}")
    return FAILURE_MODE_METADATA[equipment_type]

# Internal helpers
def _build_where(filters: dict) -> tuple:
    """Build WHERE clause from non-None filter values."""
    clauses = []
    params = {}
    for col, val in filters.items():
        if val is not None:
            clauses.append(f"{col} = :{col}")
            params[col] = val
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params