"""
ML Pipelines Router — Training export, feature windows, label vectors, dataset stats,
and training job submission/management.

Purpose-built for ML training workflows. These endpoints enforce consistent
feature engineering, labeling, and data access patterns across all model experiments.
"""
import hashlib
import json as _json
import os
import time
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import (
    APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query,
    Request, Response, status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
import pandas as pd
import io
from api.dependencies import get_db_session
from api.utils import TABLE_CONFIG, classify_operating_state, validate_equipment_exists
from api.schemas.telemetry import EquipmentTypeEnum, OperatingState
from api.schemas.ml import (
    LabelStrategy, ExportFormat,
    FeatureWindow, FeatureWindowsResponse,
    LabelEntry, LabelVectorResponse,
    FeatureStat, HealthDistribution, ClassBalance, TimeCoverage,
    DatasetStatsResponse,
    TrainingTask, JobStatus,
    TrainingJobRequest, TrainingJobResponse, TrainingJobListResponse, MlflowLink,
)
from api.workers.ml_training import run_training_job
from ml.feature_prep import FeatureEngineer

router = APIRouter()


def _get_config(equipment_type: str) -> dict:
    if equipment_type not in TABLE_CONFIG:
        raise HTTPException(400, f"Unknown equipment type: {equipment_type}")
    return TABLE_CONFIG[equipment_type]



# 1. Training Dataset Export
@router.get("/{equipment_type}/export")
def export_training_data(
    equipment_type: EquipmentTypeEnum,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    equipment_ids: Optional[str] = Query(None, description="Comma-separated equipment IDs"),
    health_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Min health on any component"),
    health_max: Optional[float] = Query(None, ge=0.0, le=1.0, description="Max health (e.g., 0.4 for failure-proximate)"),
    failure_mode: Optional[str] = Query(None, description="Filter records near a specific failure mode"),
    operating_state: Optional[OperatingState] = Query(None),
    include_derived_features: bool = Query(False, description="Compute derived features on-the-fly via FeatureEngineer"),
    format: ExportFormat = Query(ExportFormat.json),
    after_id: Optional[int] = Query(None, description="Cursor for pagination"),
    limit: int = Query(5000, ge=1, le=50000),
    session: Session = Depends(get_db_session),
):
    """Export a training-ready dataset with filtering for ML workflows.

    Supports health range filtering, failure mode proximity, operating state,
    and cursor pagination. CSV format streams for large datasets.
    """
    config = _get_config(equipment_type.value)

    where_clauses = ["1=1"]
    params = {}

    if equipment_ids:
        id_list = [int(x.strip()) for x in equipment_ids.split(",")]
        where_clauses.append(f"{config['id_col']} = ANY(:eq_ids)")
        params["eq_ids"] = id_list

    if start_time:
        where_clauses.append("sample_time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("sample_time <= :end_time")
        params["end_time"] = end_time

    # Health range filter — check if ANY health component is in range
    if health_min is not None or health_max is not None:
        health_conditions = []
        for hcol in config["health_cols"]:
            parts = []
            if health_min is not None:
                parts.append(f"{hcol} >= :health_min")
            if health_max is not None:
                parts.append(f"{hcol} <= :health_max")
            health_conditions.append("(" + " AND ".join(parts) + ")")
        where_clauses.append("(" + " OR ".join(health_conditions) + ")")
        if health_min is not None:
            params["health_min"] = health_min
        if health_max is not None:
            params["health_max"] = health_max

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

    items = []
    feature_engineer = FeatureEngineer() if include_derived_features else None

    for row in rows:
        d = dict(zip(columns, row))
        speed = d.get("speed_rpm", 0) or 0
        speed_target = d.get("speed_target_rpm", 0) or 0
        d["operating_state"] = classify_operating_state(speed, speed_target)

        # Filter by operating state if requested
        if operating_state and d["operating_state"] != operating_state.value:
            continue

        # Compute derived features on-the-fly if requested
        if feature_engineer is not None:
            derived = feature_engineer.compute(d)
            d.update(derived)

        items.append(d)

    if format == ExportFormat.csv:
        import csv
        import io

        def generate_csv():
            if not items:
                return
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=items[0].keys())
            writer.writeheader()
            for item in items:
                writer.writerow({k: (v if v is not None else "") for k, v in item.items()})
            yield output.getvalue()

        return StreamingResponse(
            generate_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={equipment_type.value}_export.csv"},
        )

    next_cursor = items[-1].get("telemetry_id") if items and has_more else None
    return {
        "items": items,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "count": len(items),
    }

# 1b. Bulk Export as Parquet
@router.get("/{equipment_type}/bulk-export")
def bulk_export_training_data(
    equipment_type: EquipmentTypeEnum,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    equipment_ids: Optional[str] = Query(None, description="Comma-separated equipment IDs"),
    session: Session = Depends(get_db_session),
):
    """Export telemetry as a single compressed Parquet file.

    Designed for bulk data transfer to remote environments (Colab, etc.).
    Parquet is ~5-10x smaller than CSV and preserves column types.
    Queries in 50k-row chunks to limit server memory usage.
    """

    config = _get_config(equipment_type.value)

    where_clauses = ["1=1"]
    params = {}

    if equipment_ids:
        id_list = [int(x.strip()) for x in equipment_ids.split(",")]
        where_clauses.append(f"{config['id_col']} = ANY(:eq_ids)")
        params["eq_ids"] = id_list
    if start_time:
        where_clauses.append("sample_time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("sample_time <= :end_time")
        params["end_time"] = end_time

    where = " AND ".join(where_clauses)
    chunk_size = 50000
    all_chunks = []
    after_id = 0

    while True:
        chunk_params = {**params, "after_id": after_id, "lim": chunk_size}
        sql = f"""
            SELECT * FROM {config['telemetry_table']}
            WHERE {where} AND {config['telemetry_id_col']} > :after_id
            ORDER BY {config['telemetry_id_col']} ASC
            LIMIT :lim
        """
        result = session.execute(text(sql), chunk_params)
        columns = list(result.keys())
        rows = result.fetchall()

        if not rows:
            break

        all_chunks.append(pd.DataFrame(rows, columns=columns))
        after_id = rows[-1][columns.index(config['telemetry_id_col'])]

        if len(rows) < chunk_size:
            break

    if not all_chunks:
        df = pd.DataFrame()
    else:
        df = pd.concat(all_chunks, ignore_index=True)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression="snappy")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={equipment_type.value}_bulk.parquet"},
    )


# 2. Feature Window Queries
@router.get("/{equipment_type}/{equipment_id}/windows", response_model=FeatureWindowsResponse)
def get_feature_windows(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    window_size: int = Query(288, ge=1, le=10000, description="Samples per window (288 = 24h @ 5min)"),
    failure_modes: Optional[str] = Query(None, description="Comma-separated failure mode codes"),
    include_normal: bool = Query(False, description="Include equal-count normal windows"),
    stride: Optional[int] = Query(None, description="Step size for normal windows (default=window_size)"),
    session: Session = Depends(get_db_session),
):
    """Get rolling time windows preceding failure events.

    Core primitive for supervised sequence learning (LSTM, Transformer, etc.).
    Each window contains `window_size` consecutive telemetry rows ending at
    (or just before) a failure event.
    """
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    # 1. Get failure events for this equipment
    failure_where = [f"{config['failure_id_col']} = :eq_id"]
    params = {"eq_id": equipment_id}
    if failure_modes:
        mode_list = [m.strip() for m in failure_modes.split(",")]
        failure_where.append("failure_mode_code = ANY(:modes)")
        params["modes"] = mode_list

    failure_sql = f"""
        SELECT failure_id, failure_time, failure_mode_code
        FROM {config['failure_table']}
        WHERE {' AND '.join(failure_where)}
        ORDER BY failure_time
    """
    failure_result = session.execute(text(failure_sql), params)
    failures = failure_result.fetchall()

    windows = []

    # 2. For each failure, fetch preceding window_size telemetry rows
    for failure in failures:
        fid, ftime, fmode = failure[0], failure[1], failure[2]
        tel_sql = f"""
            SELECT * FROM {config['telemetry_table']}
            WHERE {config['id_col']} = :eq_id
              AND sample_time < :ftime
            ORDER BY sample_time DESC
            LIMIT :wsize
        """
        tel_result = session.execute(text(tel_sql), {"eq_id": equipment_id, "ftime": ftime, "wsize": window_size})
        tel_columns = list(tel_result.keys())
        tel_rows = tel_result.fetchall()

        if len(tel_rows) < window_size:
            continue  # Skip if not enough preceding data

        records = []
        for r in reversed(tel_rows):  # Reverse to chronological order
            d = dict(zip(tel_columns, r))
            speed = d.get("speed_rpm", 0) or 0
            speed_target = d.get("speed_target_rpm", 0) or 0
            d["operating_state"] = classify_operating_state(speed, speed_target)
            records.append(d)

        windows.append(FeatureWindow(
            failure_id=fid,
            failure_mode=fmode,
            failure_time=ftime,
            label="failure",
            window_start=records[0].get("sample_time") if records else None,
            records=records,
        ))

    failure_count = len(windows)

    # 3. Optionally add normal operation windows
    normal_count = 0
    if include_normal and failure_count > 0:
        actual_stride = stride or window_size
        # Get time ranges that are NOT near any failure
        # Simple approach: sample from the middle of the operational period
        time_sql = f"""
            SELECT MIN(sample_time), MAX(sample_time), COUNT(*)
            FROM {config['telemetry_table']}
            WHERE {config['id_col']} = :eq_id
        """
        time_result = session.execute(text(time_sql), {"eq_id": equipment_id}).fetchone()
        total_records = time_result[2]

        if total_records > window_size * 2:
            # Fetch normal windows starting from the beginning, striding by stride
            normal_sql = f"""
                SELECT * FROM {config['telemetry_table']}
                WHERE {config['id_col']} = :eq_id
                ORDER BY sample_time ASC
                LIMIT :total_limit
            """
            # Fetch enough for failure_count normal windows
            needed = failure_count * window_size + (failure_count - 1) * actual_stride
            normal_result = session.execute(text(normal_sql), {"eq_id": equipment_id, "total_limit": needed})
            normal_columns = list(normal_result.keys())
            normal_rows = normal_result.fetchall()

            offset = 0
            while normal_count < failure_count and offset + window_size <= len(normal_rows):
                slice_rows = normal_rows[offset:offset + window_size]
                records = []
                is_near_failure = False
                for r in slice_rows:
                    d = dict(zip(normal_columns, r))
                    # Check if this window overlaps with any failure event
                    for failure in failures:
                        ftime = failure[1]
                        sample_t = d.get("sample_time")
                        if sample_t and ftime:
                            hours_diff = abs((ftime - sample_t).total_seconds() / 3600)
                            if hours_diff < 168:  # Within 7 days of failure
                                is_near_failure = True
                                break
                    if is_near_failure:
                        break
                    speed = d.get("speed_rpm", 0) or 0
                    speed_target = d.get("speed_target_rpm", 0) or 0
                    d["operating_state"] = classify_operating_state(speed, speed_target)
                    records.append(d)

                if not is_near_failure and len(records) == window_size:
                    windows.append(FeatureWindow(
                        label="normal",
                        window_start=records[0].get("sample_time"),
                        records=records,
                    ))
                    normal_count += 1

                offset += actual_stride

    return FeatureWindowsResponse(
        equipment_id=equipment_id,
        equipment_type=equipment_type.value,
        window_size=window_size,
        windows=windows,
        total_failure_windows=failure_count,
        total_normal_windows=normal_count,
    )


# 3. Label Vector
@router.get("/{equipment_type}/{equipment_id}/labels", response_model=LabelVectorResponse)
def get_label_vector(
    equipment_type: EquipmentTypeEnum,
    equipment_id: int,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    prediction_horizon: float = Query(168, ge=1, description="Hours before failure to label positive (default 7 days)"),
    label_strategy: LabelStrategy = Query(LabelStrategy.binary),
    session: Session = Depends(get_db_session),
):
    """Generate a time-aligned label vector for supervised training.

    One label per telemetry sample:
    - binary: 0=normal, 1=within prediction_horizon of failure
    - rul: remaining useful life in hours until next failure
    - multiclass: 0=normal, 1=degrading, 2=imminent, 3=failed
    """
    config = _get_config(equipment_type.value)
    validate_equipment_exists(session, equipment_type.value, equipment_id)

    # Get sample times
    where_clauses = [f"{config['id_col']} = :eq_id"]
    params = {"eq_id": equipment_id}
    if start_time:
        where_clauses.append("sample_time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("sample_time <= :end_time")
        params["end_time"] = end_time

    where = " AND ".join(where_clauses)
    sample_sql = f"SELECT sample_time FROM {config['telemetry_table']} WHERE {where} ORDER BY sample_time ASC"
    sample_result = session.execute(text(sample_sql), params)
    sample_times = [row[0] for row in sample_result.fetchall()]

    if not sample_times:
        raise HTTPException(404, "No telemetry data found for this equipment and time range")

    # Get failure times
    failure_sql = f"""
        SELECT failure_time FROM {config['failure_table']}
        WHERE {config['failure_id_col']} = :eq_id
        ORDER BY failure_time ASC
    """
    failure_result = session.execute(text(failure_sql), {"eq_id": equipment_id})
    failure_times = [row[0] for row in failure_result.fetchall()]

    # Generate labels
    labels = []
    positive_count = 0

    for st in sample_times:
        if label_strategy == LabelStrategy.binary:
            label = 0
            for ft in failure_times:
                hours_before = (ft - st).total_seconds() / 3600
                if 0 <= hours_before <= prediction_horizon:
                    label = 1
                    break
            if label == 1:
                positive_count += 1
            labels.append(LabelEntry(sample_time=st, label=label))

        elif label_strategy == LabelStrategy.rul:
            # RUL = hours until next failure (None if no future failure)
            rul_val = None
            for ft in failure_times:
                hours_until = (ft - st).total_seconds() / 3600
                if hours_until >= 0:
                    rul_val = round(hours_until, 2)
                    break
            if rul_val is not None and rul_val <= prediction_horizon:
                positive_count += 1
            labels.append(LabelEntry(sample_time=st, label=rul_val))

        elif label_strategy == LabelStrategy.multiclass:
            # 0=normal, 1=degrading (>48h to failure), 2=imminent (<=48h), 3=at failure
            label = 0
            for ft in failure_times:
                hours_before = (ft - st).total_seconds() / 3600
                if hours_before < 0:
                    continue
                if hours_before <= 1:
                    label = 3
                elif hours_before <= 48:
                    label = 2
                elif hours_before <= prediction_horizon:
                    label = 1
                break
            if label > 0:
                positive_count += 1
            labels.append(LabelEntry(sample_time=st, label=label))

    total = len(labels)
    return LabelVectorResponse(
        equipment_id=equipment_id,
        equipment_type=equipment_type.value,
        label_strategy=label_strategy.value,
        prediction_horizon_hours=prediction_horizon,
        labels=labels,
        total_samples=total,
        positive_samples=positive_count,
        class_ratio=round(positive_count / total, 4) if total else 0.0,
    )


# 4. Dataset Statistics
@router.get("/{equipment_type}/dataset-stats", response_model=DatasetStatsResponse)
def get_dataset_stats(
    equipment_type: EquipmentTypeEnum,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    equipment_ids: Optional[str] = Query(None, description="Comma-separated IDs"),
    session: Session = Depends(get_db_session),
):
    """Get dataset statistics for ML diagnostics.

    Returns class balance, per-feature mean/std/skew, health distributions,
    and time coverage. Essential for diagnosing class imbalance before training.
    """
    config = _get_config(equipment_type.value)

    where_clauses = ["1=1"]
    params = {}
    if equipment_ids:
        id_list = [int(x.strip()) for x in equipment_ids.split(",")]
        where_clauses.append(f"{config['id_col']} = ANY(:eq_ids)")
        params["eq_ids"] = id_list
    if start_time:
        where_clauses.append("sample_time >= :start_time")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("sample_time <= :end_time")
        params["end_time"] = end_time

    where = " AND ".join(where_clauses)

    # Total records and equipment count
    count_sql = f"""
        SELECT COUNT(*), COUNT(DISTINCT {config['id_col']}),
               MIN(sample_time), MAX(sample_time),
               SUM(operating_hours)
        FROM {config['telemetry_table']}
        WHERE {where}
    """
    count_result = session.execute(text(count_sql), params).fetchone()
    total_records = count_result[0]
    total_equipment = count_result[1]
    time_start = count_result[2]
    time_end = count_result[3]
    total_operating_hours = count_result[4] or 0

    # Feature statistics
    feature_stats = {}
    all_cols = config["key_numeric_cols"] + config["health_cols"]
    for col in all_cols:
        stat_sql = f"""
            SELECT
                AVG({col}),
                STDDEV({col}),
                MIN({col}),
                MAX({col}),
                COUNT(*) - COUNT({col}) AS null_count,
                COUNT(*) AS total_count,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}) AS median
            FROM {config['telemetry_table']}
            WHERE {where}
        """
        stat_result = session.execute(text(stat_sql), params).fetchone()
        avg_val = stat_result[0]
        std_val = stat_result[1]
        median_val = stat_result[6]
        total = stat_result[5] or 1

        # Approximate skewness: Pearson's second coefficient
        skew = None
        if std_val and std_val > 0 and avg_val is not None and median_val is not None:
            skew = round(3 * (float(avg_val) - float(median_val)) / float(std_val), 3)

        null_pct = round((stat_result[4] or 0) / total * 100, 2)
        feature_stats[col] = FeatureStat(
            mean=round(float(avg_val), 4) if avg_val is not None else None,
            std=round(float(std_val), 4) if std_val is not None else None,
            min=round(float(stat_result[2]), 4) if stat_result[2] is not None else None,
            max=round(float(stat_result[3]), 4) if stat_result[3] is not None else None,
            skew=skew,
            null_pct=null_pct,
        )

    # Health distribution
    health_dist = {}
    for hcol in config["health_cols"]:
        hdist_sql = f"""
            SELECT AVG({hcol}), STDDEV({hcol}),
                   COUNT(CASE WHEN {hcol} < 0.5 THEN 1 END)::FLOAT / NULLIF(COUNT({hcol}), 0)
            FROM {config['telemetry_table']}
            WHERE {where}
        """
        hdist_result = session.execute(text(hdist_sql), params).fetchone()
        health_dist[hcol] = HealthDistribution(
            mean=round(float(hdist_result[0]), 4) if hdist_result[0] is not None else None,
            std=round(float(hdist_result[1]), 4) if hdist_result[1] is not None else None,
            pct_below_0_5=round(float(hdist_result[2]) * 100, 2) if hdist_result[2] is not None else 0.0,
        )

    # Class balance (failure counts)
    failure_where = ["1=1"]
    f_params = {}
    if equipment_ids:
        failure_where.append(f"{config['failure_id_col']} = ANY(:eq_ids)")
        f_params["eq_ids"] = id_list
    if start_time:
        failure_where.append("failure_time >= :start_time")
        f_params["start_time"] = start_time
    if end_time:
        failure_where.append("failure_time <= :end_time")
        f_params["end_time"] = end_time

    f_where = " AND ".join(failure_where)
    failure_count_sql = f"""
        SELECT failure_mode_code, COUNT(*)
        FROM {config['failure_table']}
        WHERE {f_where}
        GROUP BY failure_mode_code
    """
    failure_counts = session.execute(text(failure_count_sql), f_params).fetchall()
    by_mode = {row[0]: row[1] for row in failure_counts if row[0]}
    total_failures = sum(by_mode.values())

    failure_rate = None
    if total_operating_hours and total_operating_hours > 0:
        failure_rate = round(total_failures / (float(total_operating_hours) / 1000), 3)

    return DatasetStatsResponse(
        equipment_type=equipment_type.value,
        total_records=total_records,
        total_equipment=total_equipment,
        time_coverage=TimeCoverage(start=time_start, end=time_end),
        class_balance=ClassBalance(
            total_failures=total_failures,
            by_mode=by_mode,
            failure_rate_per_1000h=failure_rate,
        ),
        feature_stats=feature_stats,
        health_distribution=health_dist,
    )


# 5. Training Jobs
#
# POST /{equipment_type}/train     — submit a job (async by default; Prefer: wait=N for sync)
# GET  /jobs                       — list with filters + cursor
# GET  /jobs/{job_id}              — single-job state (Prefer: wait=N to block until terminal)
# GET  /jobs/{job_id}/mlflow       — proxy MLflow run details on demand
# POST /jobs/{job_id}/cancel       — cooperative cancel
# POST /jobs/{job_id}/archive      — soft-hide (rows are never deleted)


_MIN_WAIT_S = 1
_MAX_WAIT_S = 300
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _parse_prefer_header(prefer: Optional[str]) -> dict:
    """Parse RFC 7240 Prefer header — only respond-async / wait=N concern us."""
    out = {"respond_async": False, "wait": None}
    if not prefer:
        return out
    for token in (t.strip() for t in prefer.split(",")):
        if token.lower() == "respond-async":
            out["respond_async"] = True
        elif token.lower().startswith("wait="):
            try:
                w = int(token.split("=", 1)[1])
                out["wait"] = max(_MIN_WAIT_S, min(_MAX_WAIT_S, w))
            except ValueError:
                pass
    return out


def _resolve_request_id(x_request_id: Optional[str]) -> UUID:
    if not x_request_id:
        return uuid4()
    try:
        return UUID(x_request_id)
    except ValueError:
        return uuid4()


def _hash_request_body(body: dict) -> str:
    """Stable hash of the canonical request body for idempotency comparison."""
    return hashlib.sha256(_json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def _row_to_response(row: dict) -> TrainingJobResponse:
    mlflow = MlflowLink(
        run_id=row.get("mlflow_run_id"),
        experiment_id=row.get("mlflow_experiment_id"),
        ui_url=_mlflow_ui_url(row.get("mlflow_experiment_id"), row.get("mlflow_run_id")),
    )
    return TrainingJobResponse(
        job_id=row["job_id"],
        task=row["task"],
        equipment_type=row["equipment_type"],
        status=row["status"],
        progress_message=row.get("progress_message"),
        error=row.get("error"),
        submitted_at=row["submitted_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        submitted_by=row.get("submitted_by"),
        request_id=row.get("request_id"),
        idempotency_key=row.get("idempotency_key"),
        cache_key=row.get("cache_key"),
        config=row["config"],
        mlflow=mlflow,
        metrics=row.get("metrics"),
        archived=row.get("archived", False),
    )


def _mlflow_ui_url(experiment_id: Optional[str], run_id: Optional[str]) -> Optional[str]:
    base = os.environ.get("MLFLOW_UI_URL", "").rstrip("/")
    if not base or not run_id or not experiment_id:
        return None
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def _set_common_headers(
    response: Response,
    *,
    request_id: UUID,
    job_id: Optional[UUID] = None,
    mlflow_run_id: Optional[str] = None,
    preference_applied: Optional[str] = None,
    suggest_retry: bool = False,
) -> None:
    response.headers["X-Request-Id"] = str(request_id)
    response.headers["Cache-Control"] = "no-store"
    if job_id is not None:
        response.headers["X-Job-Id"] = str(job_id)
    if mlflow_run_id:
        response.headers["X-MLflow-Run-Id"] = mlflow_run_id
    if preference_applied:
        response.headers["Preference-Applied"] = preference_applied
    if suggest_retry:
        response.headers["Retry-After"] = "5"


def _fetch_job(session: Session, job_id: UUID) -> Optional[dict]:
    row = session.execute(
        text("""
            SELECT job_id, task, equipment_type, config, status, progress_message,
                   error, submitted_at, started_at, finished_at,
                   submitted_by, request_id, idempotency_key,
                   cache_key, mlflow_run_id, mlflow_experiment_id,
                   metrics, archived
            FROM ml_jobs.training_jobs
            WHERE job_id = :id
        """),
        {"id": str(job_id)},
    ).mappings().fetchone()
    return dict(row) if row else None


def _wait_for_terminal(session: Session, job_id: UUID, max_wait: int) -> dict:
    """Poll the row until terminal or until max_wait seconds elapse."""
    deadline = time.monotonic() + max_wait
    poll_s = 0.5
    while True:
        row = _fetch_job(session, job_id)
        if not row:
            raise HTTPException(404, f"Job {job_id} not found")
        if row["status"] in _TERMINAL_STATUSES:
            return row
        if time.monotonic() >= deadline:
            return row
        session.commit()  # release any held snapshot before sleeping
        time.sleep(min(poll_s, deadline - time.monotonic()))
        poll_s = min(poll_s * 1.5, 5.0)


@router.post(
    "/{equipment_type}/train",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a training job",
    description=(
        "Queue a training job. Async by default (202 Accepted). "
        "Send `Prefer: wait=<seconds>` (1–300) to block until terminal state, or "
        "`Prefer: respond-async` to force async even with a wait header."
    ),
)
def submit_training_job(
    equipment_type: EquipmentTypeEnum,
    body: TrainingJobRequest,
    request: Request,
    response: Response,
    background: BackgroundTasks,
    session: Session = Depends(get_db_session),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key", max_length=128),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id", max_length=128),
    prefer: Optional[str] = Header(None, alias="Prefer"),
):
    """Submit one training job. See module docstring for header semantics."""
    request_id = _resolve_request_id(x_request_id)
    prefer_opts = _parse_prefer_header(prefer)
    body_dict = body.model_dump(mode="json")

    # Idempotency: short-circuit if (key, submitted_by) already exists.
    if idempotency_key:
        existing = session.execute(
            text("""
                SELECT * FROM ml_jobs.training_jobs
                WHERE idempotency_key = :k
                  AND COALESCE(submitted_by, '') = COALESCE(:u, '')
            """),
            {"k": idempotency_key, "u": x_user_id or ""},
        ).mappings().fetchone()
        if existing:
            existing = dict(existing)
            stored = {
                "task": existing["task"],
                "equipment_type": existing["equipment_type"],
                "config": existing["config"],
                "use_cache": existing["config"].get("_use_cache", True),
                "force_recompute": existing["config"].get("_force_recompute", False),
            }
            incoming = {
                "task": body.task.value,
                "equipment_type": equipment_type.value,
                "config": body_dict["config"],
                "use_cache": body.use_cache,
                "force_recompute": body.force_recompute,
            }
            if _hash_request_body(stored) != _hash_request_body(incoming):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Idempotency-Key '{idempotency_key}' already used with a different "
                        f"request body. Existing job: {existing['job_id']}"
                    ),
                )
            response.status_code = status.HTTP_303_SEE_OTHER
            response.headers["Location"] = f"/api/v1/ml/jobs/{existing['job_id']}"
            _set_common_headers(
                response,
                request_id=request_id,
                job_id=existing["job_id"],
                mlflow_run_id=existing.get("mlflow_run_id"),
            )
            return _row_to_response(existing)

    # Insert the new job. Stash use_cache/force_recompute inside config so the
    # worker can read them without an extra column.
    persisted_config = dict(body_dict["config"])
    persisted_config["_use_cache"] = body.use_cache
    persisted_config["_force_recompute"] = body.force_recompute

    job_id = uuid4()
    session.execute(
        text("""
            INSERT INTO ml_jobs.training_jobs
                (job_id, task, equipment_type, config, status,
                 submitted_by, request_id, idempotency_key,
                 client_ip, user_agent)
            VALUES
                (:job_id, :task, :equipment_type, CAST(:config AS JSONB), 'queued',
                 :submitted_by, :request_id, :idem,
                 CAST(NULLIF(:client_ip, '') AS INET), :ua)
        """),
        {
            "job_id": str(job_id),
            "task": body.task.value,
            "equipment_type": equipment_type.value,
            "config": _json.dumps(persisted_config),
            "submitted_by": x_user_id,
            "request_id": str(request_id),
            "idem": idempotency_key,
            "client_ip": _client_ip(request) or "",
            "ua": request.headers.get("user-agent"),
        },
    )
    session.commit()

    # Schedule the worker. Pass the validated body + tracing fields so the
    # worker can stamp MLflow tags without re-reading the row.
    worker_payload = {
        "config": body_dict["config"],
        "use_cache": body.use_cache,
        "force_recompute": body.force_recompute,
        "_submitted_by": x_user_id,
        "_request_id": str(request_id),
    }
    background.add_task(
        run_training_job, job_id, equipment_type.value, body.task.value, worker_payload,
    )

    # Sync mode? Block until terminal (or budget exhausted).
    pref_applied = None
    if not prefer_opts["respond_async"] and prefer_opts["wait"]:
        row = _wait_for_terminal(session, job_id, prefer_opts["wait"])
        pref_applied = f"wait={prefer_opts['wait']}"
        if row["status"] in _TERMINAL_STATUSES:
            response.status_code = status.HTTP_200_OK
        # else: still running — fall through with 200 + Retry-After
        response.headers["Location"] = f"/api/v1/ml/jobs/{job_id}"
        _set_common_headers(
            response,
            request_id=request_id,
            job_id=job_id,
            mlflow_run_id=row.get("mlflow_run_id"),
            preference_applied=pref_applied,
            suggest_retry=row["status"] not in _TERMINAL_STATUSES,
        )
        return _row_to_response(row)

    # Async path — return 202 with Location.
    row = _fetch_job(session, job_id)
    response.headers["Location"] = f"/api/v1/ml/jobs/{job_id}"
    _set_common_headers(
        response,
        request_id=request_id,
        job_id=job_id,
        suggest_retry=True,
    )
    return _row_to_response(row)


@router.get(
    "/jobs",
    response_model=TrainingJobListResponse,
    summary="List training jobs",
)
def list_training_jobs(
    response: Response,
    statuses: Optional[str] = Query(None, alias="status", description="CSV of statuses to include"),
    task: Optional[TrainingTask] = Query(None),
    equipment_type: Optional[EquipmentTypeEnum] = Query(None),
    submitted_by: Optional[str] = Query(None),
    include_archived: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    after: Optional[UUID] = Query(None, description="Cursor (job_id from previous page)"),
    session: Session = Depends(get_db_session),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    request_id = _resolve_request_id(x_request_id)

    where, params = ["1=1"], {}
    if not include_archived:
        where.append("archived = FALSE")
    if statuses:
        params["statuses"] = [s.strip() for s in statuses.split(",")]
        where.append("status = ANY(CAST(:statuses AS ml_jobs.job_status[]))")
    if task:
        where.append("task = CAST(:task AS ml_jobs.job_task)")
        params["task"] = task.value
    if equipment_type:
        where.append("equipment_type = :equipment_type")
        params["equipment_type"] = equipment_type.value
    if submitted_by:
        where.append("submitted_by = :submitted_by")
        params["submitted_by"] = submitted_by
    if after is not None:
        where.append("submitted_at < (SELECT submitted_at FROM ml_jobs.training_jobs WHERE job_id = :after)")
        params["after"] = str(after)

    sql = f"""
        SELECT job_id, task, equipment_type, config, status, progress_message,
               error, submitted_at, started_at, finished_at,
               submitted_by, request_id, idempotency_key,
               cache_key, mlflow_run_id, mlflow_experiment_id,
               metrics, archived
        FROM ml_jobs.training_jobs
        WHERE {' AND '.join(where)}
        ORDER BY submitted_at DESC
        LIMIT :lim
    """
    params["lim"] = limit + 1
    rows = [dict(r) for r in session.execute(text(sql), params).mappings().fetchall()]

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    items = [_row_to_response(r) for r in rows]
    next_cursor = items[-1].job_id if items and has_more else None

    _set_common_headers(response, request_id=request_id)
    return TrainingJobListResponse(
        items=items, next_cursor=next_cursor, has_more=has_more, count=len(items),
    )


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingJobResponse,
    summary="Get training job state",
    description="Returns the current job state. `Prefer: wait=<seconds>` blocks until terminal.",
)
def get_training_job(
    job_id: UUID,
    response: Response,
    session: Session = Depends(get_db_session),
    prefer: Optional[str] = Header(None, alias="Prefer"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    request_id = _resolve_request_id(x_request_id)
    prefer_opts = _parse_prefer_header(prefer)

    if prefer_opts["wait"]:
        row = _wait_for_terminal(session, job_id, prefer_opts["wait"])
    else:
        row = _fetch_job(session, job_id)
        if not row:
            raise HTTPException(404, f"Job {job_id} not found")

    pref_applied = f"wait={prefer_opts['wait']}" if prefer_opts["wait"] else None
    is_terminal = row["status"] in _TERMINAL_STATUSES
    _set_common_headers(
        response,
        request_id=request_id,
        job_id=job_id,
        mlflow_run_id=row.get("mlflow_run_id"),
        preference_applied=pref_applied,
        suggest_retry=not is_terminal,
    )
    return _row_to_response(row)


@router.get(
    "/jobs/{job_id}/mlflow",
    summary="Proxy MLflow run details for a job",
    description="Lazily fetches the full MLflow run (params, metrics, tags, artifacts list).",
)
def get_training_job_mlflow(
    job_id: UUID,
    response: Response,
    session: Session = Depends(get_db_session),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    request_id = _resolve_request_id(x_request_id)
    row = _fetch_job(session, job_id)
    if not row:
        raise HTTPException(404, f"Job {job_id} not found")
    if not row.get("mlflow_run_id"):
        raise HTTPException(409, "Job has no MLflow run yet")

    try:
        from mlflow import MlflowClient
        client = MlflowClient()
        run = client.get_run(row["mlflow_run_id"])
    except Exception as exc:
        raise HTTPException(503, f"MLflow unavailable: {exc}")

    payload = {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
        "artifact_uri": run.info.artifact_uri,
    }
    _set_common_headers(
        response, request_id=request_id, job_id=job_id, mlflow_run_id=row["mlflow_run_id"],
    )
    return payload


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=TrainingJobResponse,
    summary="Request cooperative cancellation",
)
def cancel_training_job(
    job_id: UUID,
    response: Response,
    session: Session = Depends(get_db_session),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    request_id = _resolve_request_id(x_request_id)
    row = _fetch_job(session, job_id)
    if not row:
        raise HTTPException(404, f"Job {job_id} not found")
    if row["status"] in _TERMINAL_STATUSES:
        raise HTTPException(409, f"Job is already {row['status']}")

    # Worker checks status at phase boundaries; flag flips state authoritatively
    # only if the job hasn't started (queued → cancelled is final immediately).
    new_status = "cancelled"
    finished_at = datetime.now(timezone.utc) if row["status"] == "queued" else None
    session.execute(
        text("""
            UPDATE ml_jobs.training_jobs
               SET status = CAST(:s AS ml_jobs.job_status),
                   finished_at = COALESCE(:f, finished_at),
                   progress_message = 'cancellation requested'
             WHERE job_id = :id
        """),
        {"s": new_status, "f": finished_at, "id": str(job_id)},
    )
    session.commit()

    row = _fetch_job(session, job_id)
    _set_common_headers(response, request_id=request_id, job_id=job_id)
    return _row_to_response(row)


@router.post(
    "/jobs/{job_id}/archive",
    response_model=TrainingJobResponse,
    summary="Soft-archive a terminal job",
    description="Hides the job from default listings. Rows are never deleted (audit retention).",
)
def archive_training_job(
    job_id: UUID,
    response: Response,
    session: Session = Depends(get_db_session),
    x_request_id: Optional[str] = Header(None, alias="X-Request-Id"),
):
    request_id = _resolve_request_id(x_request_id)
    row = _fetch_job(session, job_id)
    if not row:
        raise HTTPException(404, f"Job {job_id} not found")
    if row["status"] not in _TERMINAL_STATUSES:
        raise HTTPException(409, f"Cannot archive a non-terminal job (status={row['status']})")

    session.execute(
        text("UPDATE ml_jobs.training_jobs SET archived = TRUE WHERE job_id = :id"),
        {"id": str(job_id)},
    )
    session.commit()

    row = _fetch_job(session, job_id)
    _set_common_headers(response, request_id=request_id, job_id=job_id)
    return _row_to_response(row)
