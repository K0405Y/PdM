"""
Telemetry Router — Ingest, query (cursor-paginated), health, summary, failures, maintenance.
"""
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session
from api.dependencies import get_db, get_db_session
from api.utils import (
    TABLE_CONFIG, classify_operating_state, row_to_dict,
    rows_to_dicts, validate_equipment_exists,
)
from api.schemas.telemetry import (
    EquipmentTypeEnum, BinInterval, OperatingState,
    TelemetryIngestRequest, TelemetryIngestResponse,
    TelemetryRow, CursorPaginatedTelemetry,
    HealthIndicators,
    BinnedSummaryBin, BinnedSummaryResponse,
    FailureEventResponse, MaintenanceEventResponse,
)
from ingestion.db_setup import Database
from data_simulation.physics.environmental_conditions import (
    LocationType, EnvironmentalConditions,
)
from datetime import datetime


router = APIRouter()

# Helpers
def _get_config(equipment_type: str) -> dict:
    if equipment_type not in TABLE_CONFIG:
        raise HTTPException(400, f"Unknown equipment type: {equipment_type}")
    return TABLE_CONFIG[equipment_type]

def _enrich_row(row_dict: dict) -> TelemetryRow:
    """Add operating_state to a telemetry row dict."""
    speed = row_dict.get("speed_rpm", 0) or 0
    speed_target = row_dict.get("speed_target_rpm", 0) or 0
    op_state = classify_operating_state(speed, speed_target)
    tid = row_dict.pop("telemetry_id", None)
    st = row_dict.pop("sample_time", None)
    oh = row_dict.pop("operating_hours", None)
    return TelemetryRow(
        telemetry_id=tid,
        sample_time=st,
        operating_hours=oh,
        operating_state=op_state,
        data=row_dict,
    )


def _bin_interval_sql(bin_interval: BinInterval) -> str:
    """Convert BinInterval enum to PostgreSQL date_trunc / interval expression."""
    mapping = {
        BinInterval.five_min: "date_trunc('hour', sample_time) + INTERVAL '5 min' * FLOOR(EXTRACT(MINUTE FROM sample_time) / 5)",
        BinInterval.one_hour: "date_trunc('hour', sample_time)",
        BinInterval.eight_hour: "date_trunc('day', sample_time) + INTERVAL '8 hours' * FLOOR(EXTRACT(HOUR FROM sample_time) / 8)",
        BinInterval.twenty_four_hour: "date_trunc('day', sample_time)",
        BinInterval.shift: "date_trunc('day', sample_time) + CASE WHEN EXTRACT(HOUR FROM sample_time) < 18 THEN INTERVAL '6 hours' ELSE INTERVAL '18 hours' END",
    }
    return mapping[bin_interval]


def _bin_duration(bin_interval: BinInterval) -> str:
    mapping = {
        BinInterval.five_min: "INTERVAL '5 minutes'",
        BinInterval.one_hour: "INTERVAL '1 hour'",
        BinInterval.eight_hour: "INTERVAL '8 hours'",
        BinInterval.twenty_four_hour: "INTERVAL '1 day'",
        BinInterval.shift: "INTERVAL '12 hours'",
    }
    return mapping[bin_interval]


# Ingestion
@router.post("/ingest", response_model=TelemetryIngestResponse, status_code=201)
def ingest_telemetry(
    request: TelemetryIngestRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
):
    """Bulk ingest telemetry records. Uses PostgreSQL COPY for performance.

    For payloads > 10,000 records, runs as background task and returns 202.
    """
    from ingestion.bulk_insert import bulk_insert_telemetry

    records = [r.model_dump() for r in request.records]
    equipment_type = request.equipment_type.value

    if len(records) > 10000:
        def _bg_ingest():
            bulk_insert_telemetry(db, records, equipment_type)
        background_tasks.add_task(_bg_ingest)
        return TelemetryIngestResponse(
            inserted_count=len(records),
            equipment_type=equipment_type,
            message=f"Background ingestion started for {len(records)} records",
        )

    count = bulk_insert_telemetry(db, records, equipment_type)
    return TelemetryIngestResponse(
        inserted_count=count,
        equipment_type=equipment_type,
        message=f"Ingested {count} {equipment_type} telemetry records",
    )



# Query — cursor-paginated
@router.get("/{equipment_type}/{equipment_id}", response_model=CursorPaginatedTelemetry)
def query_telemetry(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    start_time: Optional[str] = Query(None, description="Start time (YYYY-MM-DD HH:MM:SS)"),
    end_time: Optional[str] = Query(None, description="End time (YYYY-MM-DD HH:MM:SS)"),
    operating_state: Optional[OperatingState] = Query(None, description="Filter by operating state"),
    after_id: Optional[int] = Query(None, description="Cursor: telemetry_id to paginate after"),
    limit: int = Query(1000, ge=1, le=5000),
    session: Session = Depends(get_db_session),
):
    """Query telemetry with cursor-based pagination.

    Pass `after_id` from `next_cursor` in previous response to get next page.
    """
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    where_clauses = [f"{config['id_col']} = :eq_id"]
    params = {"eq_id": equipment_id}

    if start_time:
        where_clauses.append("sample_time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("sample_time <= :end_time")
        params["end_time"] = end_time
    if after_id is not None:
        where_clauses.append(f"{config['telemetry_id_col']} > :after_id")
        params["after_id"] = after_id

    where = " AND ".join(where_clauses)
    # Fetch limit + 1 to detect has_more
    sql = f"SELECT * FROM {config['telemetry_table']} WHERE {where} ORDER BY {config['telemetry_id_col']} ASC LIMIT :lim"
    params["lim"] = limit + 1

    result = session.execute(text(sql), params)
    columns = list(result.keys())
    rows = result.fetchall()

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    items = []
    for row in rows:
        d = dict(zip(columns, row))
        item = _enrich_row(d)
        # Filter by operating state if requested (post-query since it's computed)
        if operating_state and item.operating_state != operating_state.value:
            continue
        items.append(item)

    next_cursor = items[-1].telemetry_id if items and has_more else None

    return CursorPaginatedTelemetry(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
        limit=limit,
    )

# Latest
@router.get("/{equipment_type}/{equipment_id}/latest", response_model=TelemetryRow)
def get_latest_telemetry(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    session: Session = Depends(get_db_session),
):
    """Get the most recent telemetry row for an equipment unit."""
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    sql = f"""
        SELECT * FROM {config['telemetry_table']}
        WHERE {config['id_col']} = :eq_id
        ORDER BY sample_time DESC LIMIT 1
    """
    result = session.execute(text(sql), {"eq_id": equipment_id})
    columns = list(result.keys())
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "No telemetry data found for this equipment")
    return _enrich_row(dict(zip(columns, row)))


# Health indicators
@router.get("/{equipment_type}/{equipment_id}/health", response_model=HealthIndicators)
def get_health(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    session: Session = Depends(get_db_session),
):
    """Get latest health indicators for an equipment unit."""
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    health_cols = ", ".join(config["health_cols"])
    sql = f"""
        SELECT sample_time, speed_rpm, speed_target_rpm, {health_cols}
        FROM {config['telemetry_table']}
        WHERE {config['id_col']} = :eq_id
        ORDER BY sample_time DESC LIMIT 1
    """
    result = session.execute(text(sql), {"eq_id": equipment_id})
    row = result.fetchone()
    if not row:
        raise HTTPException(404, "No telemetry data found")

    sample_time = row[0]
    speed = row[1] or 0
    speed_target = row[2] or 0
    health_values = row[3:]

    return HealthIndicators(
        equipment_id=equipment_id,
        equipment_type=equipment_type.value,
        sample_time=sample_time,
        operating_state=classify_operating_state(speed, speed_target),
        health_components=dict(zip(config["health_cols"], health_values)),
    )

# Binned summary
@router.get("/{equipment_type}/{equipment_id}/summary", response_model=BinnedSummaryResponse)
def get_summary(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    start_time: str = Query(..., description="Start time (YYYY-MM-DD HH:MM:SS)"),
    end_time: str = Query(..., description="End time (YYYY-MM-DD HH:MM:SS)"),
    bin_interval: BinInterval = Query(BinInterval.one_hour),
    session: Session = Depends(get_db_session),
):
    """Get aggregated statistics in configurable time bins.

    Directly usable for feature engineering pipelines.
    """
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    bin_expr = _bin_interval_sql(bin_interval)
    bin_dur = _bin_duration(bin_interval)

    # Build aggregate expressions for key numeric columns
    agg_parts = []
    for col in config["key_numeric_cols"]:
        agg_parts.append(f"AVG({col}) AS avg_{col}")
        agg_parts.append(f"MIN({col}) AS min_{col}")
        agg_parts.append(f"MAX({col}) AS max_{col}")
        agg_parts.append(f"STDDEV({col}) AS std_{col}")
    agg_sql = ", ".join(agg_parts)

    sql = f"""
        SELECT
            {bin_expr} AS bin_start,
            {bin_expr} + {bin_dur} AS bin_end,
            COUNT(*) AS record_count,
            {agg_sql}
        FROM {config['telemetry_table']}
        WHERE {config['id_col']} = :eq_id
          AND sample_time >= :start_time
          AND sample_time <= :end_time
        GROUP BY bin_start, bin_end
        ORDER BY bin_start
    """
    result = session.execute(text(sql), {
        "eq_id": equipment_id,
        "start_time": start_time,
        "end_time": end_time,
    })
    columns = list(result.keys())
    rows = result.fetchall()

    bins = []
    for row in rows:
        d = dict(zip(columns, row))
        stats = {}
        for col in config["key_numeric_cols"]:
            stats[col] = {
                "avg": d.get(f"avg_{col}"),
                "min": d.get(f"min_{col}"),
                "max": d.get(f"max_{col}"),
                "stddev": d.get(f"std_{col}"),
            }
        bins.append(BinnedSummaryBin(
            bin_start=d["bin_start"],
            bin_end=d["bin_end"],
            record_count=d["record_count"],
            stats=stats,
        ))

    return BinnedSummaryResponse(
        equipment_id=equipment_id,
        equipment_type=equipment_type.value,
        bin_interval=bin_interval.value,
        bins=bins,
    )

# Enriched telemetry (with environmental conditions)
@router.get("/{equipment_type}/{equipment_id}/enriched", response_model=CursorPaginatedTelemetry)
def get_enriched_telemetry(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    start_time: str = Query(..., description="Start time (YYYY-MM-DD HH:MM:SS)"),
    end_time: str = Query(..., description="End time (YYYY-MM-DD HH:MM:SS)"),
    location_type: str = Query("tropical", description="Location type for environmental model"),
    use_real_weather: bool = Query(False, description="Use cached real weather if available"),
    after_id: Optional[int] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    session: Session = Depends(get_db_session),
):
    """Return telemetry enriched with environmental conditions.

    Joins weather data with each telemetry row — useful for gas turbine
    analysis where ambient conditions drive EGT margin and power output.
    """
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    try:
        lt = LocationType(location_type)
    except ValueError:
        raise HTTPException(400, f"Unknown location type: {location_type}")

    env = EnvironmentalConditions(location_type=lt)

    where_clauses = [f"{config['id_col']} = :eq_id", "sample_time >= :start_time", "sample_time <= :end_time"]
    params = {"eq_id": equipment_id, "start_time": start_time, "end_time": end_time}
    if after_id is not None:
        where_clauses.append(f"{config['telemetry_id_col']} > :after_id")
        params["after_id"] = after_id

    where = " AND ".join(where_clauses)
    sql = f"SELECT * FROM {config['telemetry_table']} WHERE {where} ORDER BY {config['telemetry_id_col']} ASC LIMIT :lim"
    params["lim"] = limit + 1

    result = session.execute(text(sql), params)
    columns = list(result.keys())
    rows = result.fetchall()

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    # Parse the first row's sample_time as reference for elapsed hours
    ref_time = None
    items = []
    for row in rows:
        d = dict(zip(columns, row))
        sample_time = d.get("sample_time")
        if ref_time is None and sample_time:
            ref_time = sample_time
        # Compute elapsed hours from reference
        elapsed = 0.0
        if ref_time and sample_time:
            elapsed = (sample_time - ref_time).total_seconds() / 3600

        conditions = env.get_conditions(elapsed)
        # Merge environmental fields into data
        for key in ["ambient_temp_C", "humidity_percent", "pressure_kPa", "corrosion_factor", "fouling_factor"]:
            d[key] = conditions.get(key)

        items.append(_enrich_row(d))

    next_cursor = items[-1].telemetry_id if items and has_more else None
    return CursorPaginatedTelemetry(items=items, next_cursor=next_cursor, has_more=has_more, limit=limit)

# Failure history
@router.get("/failures/{equipment_type}/{equipment_id}", response_model=List[FailureEventResponse])
def get_failures(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    start_time: Optional[str] = Query(None, description="Start time (YYYY-MM-DD HH:MM:SS)"),
    end_time: Optional[str] = Query(None, description="End time (YYYY-MM-DD HH:MM:SS)"),
    session: Session = Depends(get_db_session),
):
    """Get failure event history for an equipment unit."""
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    where_clauses = [f"{config['failure_id_col']} = :eq_id"]
    params = {"eq_id": equipment_id}
    if start_time:
        where_clauses.append("failure_time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("failure_time <= :end_time")
        params["end_time"] = end_time

    where = " AND ".join(where_clauses)
    sql = f"SELECT * FROM {config['failure_table']} WHERE {where} ORDER BY failure_time DESC"
    result = session.execute(text(sql), params)
    columns = list(result.keys())
    rows = result.fetchall()

    items = []
    for row in rows:
        d = dict(zip(columns, row))
        items.append(FailureEventResponse(
            failure_id=d.get("failure_id"),
            equipment_id=equipment_id,
            failure_time=d.get("failure_time"),
            operating_hours_at_failure=d.get("operating_hours_at_failure"),
            failure_mode_code=d.get("failure_mode_code"),
            failure_description=d.get("failure_description"),
        ))
    return items

# Maintenance history
@router.get("/maintenance/{equipment_type}/{equipment_id}", response_model=List[MaintenanceEventResponse])
def get_maintenance(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    start_time: Optional[str] = Query(None, description="Start time (YYYY-MM-DD HH:MM:SS)"),
    end_time: Optional[str] = Query(None, description="End time (YYYY-MM-DD HH:MM:SS)"),
    session: Session = Depends(get_db_session),
):
    """Get maintenance event history for an equipment unit."""
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    where_clauses = [f"{config['maintenance_id_col']} = :eq_id"]
    params = {"eq_id": equipment_id}
    if start_time:
        where_clauses.append("start_time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("start_time <= :end_time")
        params["end_time"] = end_time

    where = " AND ".join(where_clauses)
    sql = f"SELECT * FROM {config['maintenance_table']} WHERE {where} ORDER BY start_time DESC"
    result = session.execute(text(sql), params)
    columns = list(result.keys())
    rows = result.fetchall()

    items = []
    for row in rows:
        d = dict(zip(columns, row))
        items.append(MaintenanceEventResponse(
            maintenance_id=d.get("maintenance_id"),
            equipment_id=equipment_id,
            start_time=d.get("start_time"),
            end_time=d.get("end_time"),
            failure_code=d.get("failure_code"),
            downtime_hours=d.get("downtime_hours"),
            repaired_components=d.get("repaired_components"),
        ))
    return items