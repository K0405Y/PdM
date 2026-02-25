"""
ML Pipelines Router — Training export, feature windows, label vectors, dataset stats.

Purpose-built for ML training workflows. These endpoints enforce consistent
feature engineering, labeling, and data access patterns across all model experiments.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.dependencies import get_db_session
from api.utils import TABLE_CONFIG, classify_operating_state, validate_equipment_exists
from api.schemas.telemetry import EquipmentTypeEnum, OperatingState
from api.schemas.ml import (
    LabelStrategy, ExportFormat,
    FeatureWindow, FeatureWindowsResponse,
    LabelEntry, LabelVectorResponse,
    FeatureStat, HealthDistribution, ClassBalance, TimeCoverage,
    DatasetStatsResponse,
)

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
    include_derived_features: bool = Query(True),
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
    for row in rows:
        d = dict(zip(columns, row))
        speed = d.get("speed_rpm", 0) or 0
        speed_target = d.get("speed_target_rpm", 0) or 0
        d["operating_state"] = classify_operating_state(speed, speed_target)

        # Filter by operating state if requested
        if operating_state and d["operating_state"] != operating_state.value:
            continue

        # Strip derived features if not requested
        if not include_derived_features:
            for feat in ["vibration_trend_7d", "temp_variation_24h", "speed_stability",
                         "efficiency_degradation_rate", "pressure_ratio", "load_factor"]:
                d.pop(feat, None)

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
