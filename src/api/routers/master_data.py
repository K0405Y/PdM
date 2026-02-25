"""
Master Data Router — Equipment CRUD, batch seeding, schema init, failure modes.
"""
import math
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
from api.dependencies import get_db, get_db_session, get_master_data
from api.config import get_settings
from api.utils import rows_to_dicts
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

# Gas Turbines CRUD
@router.get("/turbines", response_model=PaginatedResponse)
def list_turbines(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    status: Optional[str] = None,
    location: Optional[str] = None,
    session: Session = Depends(get_db_session),
):
    where, params = _build_where({"status": status, "location": location})
    total = session.execute(
        text(f"SELECT COUNT(*) FROM master_data.gas_turbines{where}"), params
    ).scalar()
    rows = session.execute(
        text(f"SELECT * FROM master_data.gas_turbines{where} ORDER BY turbine_id LIMIT :lim OFFSET :off"),
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
    configs = master.get_configs([turbine_id], "turbine")
    if not configs:
        raise HTTPException(404, "Turbine not found")
    return GasTurbineResponse(**configs[0])


@router.post("/turbines", response_model=GasTurbineResponse, status_code=201)
def create_turbine(body: GasTurbineCreate, session: Session = Depends(get_db_session)):
    result = session.execute(text("""
        INSERT INTO master_data.gas_turbines
        (name, serial_number, location, installed_date,
         initial_health_hgp, initial_health_blade,
         initial_health_bearing, initial_health_fuel)
        VALUES (:name, :sn, :loc, :date, :hgp, :blade, :bearing, :fuel)
        RETURNING *
    """), {
        "name": body.name, "sn": body.serial_number,
        "loc": body.location, "date": body.installed_date,
        "hgp": body.initial_health_hgp, "blade": body.initial_health_blade,
        "bearing": body.initial_health_bearing, "fuel": body.initial_health_fuel,
    })
    columns = list(result.keys())
    row = result.fetchone()
    session.commit()
    return GasTurbineResponse(**dict(zip(columns, row)))


@router.patch("/turbines/{turbine_id}", response_model=GasTurbineResponse)
def update_turbine(turbine_id: int, body: GasTurbineUpdate, session: Session = Depends(get_db_session)):
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = turbine_id
    result = session.execute(
        text(f"UPDATE master_data.gas_turbines SET {set_clause} WHERE turbine_id = :id RETURNING *"),
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
    result = session.execute(
        text("DELETE FROM master_data.gas_turbines WHERE turbine_id = :id"),
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
    where, params = _build_where({"status": status, "location": location})
    total = session.execute(
        text(f"SELECT COUNT(*) FROM master_data.compressors{where}"), params
    ).scalar()
    rows = session.execute(
        text(f"SELECT * FROM master_data.compressors{where} ORDER BY compressor_id LIMIT :lim OFFSET :off"),
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
    configs = master.get_configs([compressor_id], "compressor")
    if not configs:
        raise HTTPException(404, "Compressor not found")
    return CompressorResponse(**configs[0])


@router.post("/compressors", response_model=CompressorResponse, status_code=201)
def create_compressor(body: CompressorCreate, session: Session = Depends(get_db_session)):
    result = session.execute(text("""
        INSERT INTO master_data.compressors
        (name, serial_number, location, installed_date,
         design_flow_m3h, design_head_kj_kg,
         initial_health_impeller, initial_health_bearing)
        VALUES (:name, :sn, :loc, :date, :flow, :head, :imp, :bear)
        RETURNING *
    """), {
        "name": body.name, "sn": body.serial_number,
        "loc": body.location, "date": body.installed_date,
        "flow": body.design_flow_m3h, "head": body.design_head_kj_kg,
        "imp": body.initial_health_impeller, "bear": body.initial_health_bearing,
    })
    columns = list(result.keys())
    row = result.fetchone()
    session.commit()
    return CompressorResponse(**dict(zip(columns, row)))


@router.patch("/compressors/{compressor_id}", response_model=CompressorResponse)
def update_compressor(compressor_id: int, body: CompressorUpdate, session: Session = Depends(get_db_session)):
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = compressor_id
    result = session.execute(
        text(f"UPDATE master_data.compressors SET {set_clause} WHERE compressor_id = :id RETURNING *"),
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
    result = session.execute(
        text("DELETE FROM master_data.compressors WHERE compressor_id = :id"),
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
    where, params = _build_where({"status": status, "location": location, "service_type": service_type})
    total = session.execute(
        text(f"SELECT COUNT(*) FROM master_data.pumps{where}"), params
    ).scalar()
    rows = session.execute(
        text(f"SELECT * FROM master_data.pumps{where} ORDER BY pump_id LIMIT :lim OFFSET :off"),
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
    configs = master.get_configs([pump_id], "pump")
    if not configs:
        raise HTTPException(404, "Pump not found")
    return PumpResponse(**configs[0])


@router.post("/pumps", response_model=PumpResponse, status_code=201)
def create_pump(body: PumpCreate, session: Session = Depends(get_db_session)):
    result = session.execute(text("""
        INSERT INTO master_data.pumps
        (name, serial_number, service_type, location, installed_date,
         design_flow_m3h, design_head_m, design_speed_rpm, fluid_density_kg_m3,
         npsh_available_m, initial_health_impeller, initial_health_seal,
         initial_health_bearing_de, initial_health_bearing_nde)
        VALUES (:name, :sn, :svc, :loc, :date, :flow, :head, :speed, :dens,
                :npsh, :imp, :seal, :bde, :bnde)
        RETURNING *
    """), {
        "name": body.name, "sn": body.serial_number,
        "svc": body.service_type, "loc": body.location,
        "date": body.installed_date, "flow": body.design_flow_m3h,
        "head": body.design_head_m, "speed": body.design_speed_rpm,
        "dens": body.fluid_density_kg_m3, "npsh": body.npsh_available_m,
        "imp": body.initial_health_impeller, "seal": body.initial_health_seal,
        "bde": body.initial_health_bearing_de, "bnde": body.initial_health_bearing_nde,
    })
    columns = list(result.keys())
    row = result.fetchone()
    session.commit()
    return PumpResponse(**dict(zip(columns, row)))


@router.patch("/pumps/{pump_id}", response_model=PumpResponse)
def update_pump(pump_id: int, body: PumpUpdate, session: Session = Depends(get_db_session)):
    updates = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = pump_id
    result = session.execute(
        text(f"UPDATE master_data.pumps SET {set_clause} WHERE pump_id = :id RETURNING *"),
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
    result = session.execute(
        text("DELETE FROM master_data.pumps WHERE pump_id = :id"),
        {"id": pump_id},
    )
    session.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "Pump not found")
    return MessageResponse(message="Pump deleted", count=1)


# Equipment status change
_STATUS_TABLE_MAP = {
    "turbine": ("master_data.gas_turbines", "turbine_id"),
    "compressor": ("master_data.compressors", "compressor_id"),
    "pump": ("master_data.pumps", "pump_id"),
}


@router.patch("/{equipment_type}/{equipment_id}/status", response_model=MessageResponse)
def update_equipment_status(
    equipment_type: EquipmentType,
    equipment_id: int,
    status: EquipmentStatus = Query(..., description="New status"),
    session: Session = Depends(get_db_session),
):
    """Change the status of any equipment (active, inactive, maintenance)."""
    table, id_col = _STATUS_TABLE_MAP[equipment_type.value]
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


# Failure Modes
FAILURE_MODE_METADATA = {
    "gas_turbine": [
        FailureModeDetail(
            mode_code="F_HGP", equipment_type="gas_turbine",
            description="Hot Gas Path Degradation - Combustion liner cracking",
            failure_threshold=0.45, severity="safety_critical",
            primary_indicators=["egt_celsius", "efficiency_fraction"],
            lagging_indicators=["vibration_rms_mm_s"],
            typical_lead_time_hours="2000-6000",
        ),
        FailureModeDetail(
            mode_code="F_BLADE", equipment_type="gas_turbine",
            description="Blade Erosion - Leading edge degradation",
            failure_threshold=0.40, severity="safety_critical",
            primary_indicators=["vibration_rms_mm_s", "vibration_peak_mm_s"],
            lagging_indicators=["efficiency_fraction"],
            typical_lead_time_hours="3000-8000",
        ),
        FailureModeDetail(
            mode_code="F_BEARING", equipment_type="gas_turbine",
            description="Bearing Failure - Lubrication/mechanical degradation",
            failure_threshold=0.35, severity="safety_critical",
            primary_indicators=["vibration_rms_mm_s", "oil_temp_celsius"],
            lagging_indicators=["speed_rpm"],
            typical_lead_time_hours="1000-4000",
        ),
        FailureModeDetail(
            mode_code="F_FUEL", equipment_type="gas_turbine",
            description="Fuel System Fouling - Nozzle blockage",
            failure_threshold=0.55, severity="performance",
            primary_indicators=["fuel_flow_kg_s", "egt_celsius"],
            lagging_indicators=["efficiency_fraction"],
            typical_lead_time_hours="2000-5000",
        ),
    ],
    "compressor": [
        FailureModeDetail(
            mode_code="F_IMPELLER", equipment_type="compressor",
            description="Impeller Degradation - Erosion or fouling",
            failure_threshold=0.42, severity="availability",
            primary_indicators=["efficiency_fraction", "head_kj_kg"],
            lagging_indicators=["vibration_amplitude_mm", "power_kw"],
            typical_lead_time_hours="2000-6000",
        ),
        FailureModeDetail(
            mode_code="F_BEARING", equipment_type="compressor",
            description="Bearing Failure - Journal or thrust bearing damage",
            failure_threshold=0.38, severity="safety_critical",
            primary_indicators=["bearing_temp_de_celsius", "bearing_temp_nde_celsius", "orbit_amplitude_mm"],
            lagging_indicators=["thrust_bearing_temp_celsius"],
            typical_lead_time_hours="1000-4000",
        ),
        FailureModeDetail(
            mode_code="F_SEAL_PRIMARY", equipment_type="compressor",
            description="Primary Dry Gas Seal Failure",
            failure_threshold=0.25, severity="safety_critical",
            primary_indicators=["primary_seal_leakage_kg_s", "seal_health_primary"],
            lagging_indicators=["discharge_pressure_kpa"],
            typical_lead_time_hours="1500-5000",
        ),
        FailureModeDetail(
            mode_code="F_SEAL_SECONDARY", equipment_type="compressor",
            description="Secondary Dry Gas Seal Failure",
            failure_threshold=0.25, severity="availability",
            primary_indicators=["secondary_seal_leakage_kg_s", "seal_health_secondary"],
            lagging_indicators=[],
            typical_lead_time_hours="2000-6000",
        ),
        FailureModeDetail(
            mode_code="F_HIGH_VIBRATION", equipment_type="compressor",
            description="High Vibration Trip - Shaft orbit amplitude exceeded safety limits",
            failure_threshold=0.0, severity="safety_critical",
            primary_indicators=["orbit_amplitude_mm", "sync_amplitude_mm"],
            lagging_indicators=["bearing_temp_de_celsius"],
            typical_lead_time_hours="100-500",
        ),
        FailureModeDetail(
            mode_code="F_SURGE", equipment_type="compressor",
            description="Compressor Surge - Anti-surge protection failure, flow reversal damage",
            failure_threshold=0.0, severity="safety_critical",
            primary_indicators=["surge_margin_percent", "flow_m3h"],
            lagging_indicators=["discharge_pressure_kpa", "suction_pressure_kpa"],
            typical_lead_time_hours="10-100",
        ),
    ],
    "pump": [
        FailureModeDetail(
            mode_code="F_IMPELLER", equipment_type="pump",
            description="Impeller Degradation - Erosion, corrosion, or damage",
            failure_threshold=0.35, severity="availability",
            primary_indicators=["efficiency_fraction", "head_m", "bep_deviation_percent"],
            lagging_indicators=["vibration_rms_mm_s", "power_kw"],
            typical_lead_time_hours="1500-5000",
        ),
        FailureModeDetail(
            mode_code="F_SEAL", equipment_type="pump",
            description="Mechanical Seal Failure - Wear, thermal damage, or contamination",
            failure_threshold=0.40, severity="availability",
            primary_indicators=["seal_leakage_rate", "fluid_temp_celsius"],
            lagging_indicators=["vibration_rms_mm_s"],
            typical_lead_time_hours="1000-3000",
        ),
        FailureModeDetail(
            mode_code="F_BEARING_DRIVE_END", equipment_type="pump",
            description="Drive End Bearing Failure - Fatigue, lubrication, or contamination",
            failure_threshold=0.28, severity="safety_critical",
            primary_indicators=["bearing_temp_de_celsius", "vibration_rms_mm_s"],
            lagging_indicators=["motor_current_amps"],
            typical_lead_time_hours="800-3000",
        ),
        FailureModeDetail(
            mode_code="F_BEARING_NON_DRIVE_END", equipment_type="pump",
            description="Non-Drive End Bearing Failure",
            failure_threshold=0.28, severity="availability",
            primary_indicators=["bearing_temp_nde_celsius", "vibration_rms_mm_s"],
            lagging_indicators=[],
            typical_lead_time_hours="1000-4000",
        ),
        FailureModeDetail(
            mode_code="F_BEARING_OVERTEMP", equipment_type="pump",
            description="Bearing Overtemperature - Excessive friction or cooling failure",
            failure_threshold=0.0, severity="safety_critical",
            primary_indicators=["bearing_temp_de_celsius", "bearing_temp_nde_celsius"],
            lagging_indicators=["vibration_rms_mm_s"],
            typical_lead_time_hours="50-200",
        ),
        FailureModeDetail(
            mode_code="F_HIGH_VIBRATION", equipment_type="pump",
            description="High Vibration Trip - Mechanical instability",
            failure_threshold=0.0, severity="safety_critical",
            primary_indicators=["vibration_rms_mm_s", "vibration_peak_mm_s"],
            lagging_indicators=["bearing_temp_de_celsius"],
            typical_lead_time_hours="50-300",
        ),
        FailureModeDetail(
            mode_code="F_CAVITATION", equipment_type="pump",
            description="Severe Cavitation - NPSH margin critical",
            failure_threshold=0.0, severity="availability",
            primary_indicators=["cavitation_margin_m", "npsh_available_m", "npsh_required_m"],
            lagging_indicators=["vibration_rms_mm_s", "efficiency_fraction"],
            typical_lead_time_hours="100-500",
        ),
        FailureModeDetail(
            mode_code="F_MOTOR_OVERLOAD", equipment_type="pump",
            description="Motor Overload - Excessive current draw",
            failure_threshold=0.0, severity="availability",
            primary_indicators=["motor_current_amps", "motor_current_ratio"],
            lagging_indicators=["power_kw"],
            typical_lead_time_hours="10-100",
        ),
    ],
}


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