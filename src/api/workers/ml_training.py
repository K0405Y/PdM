"""
ML Training Worker — runs in FastAPI BackgroundTasks.

Calls into train.py's shared pipeline (no logic duplication), updates the
ml_jobs.training_jobs row through the lifecycle, and stamps the MLflow run
with cross-link tags for bidirectional navigation.

Cancellation is cooperative: the worker re-checks the row's status at safe
checkpoints (between phases) and aborts if it's been flipped to 'cancelled'.
"""

from __future__ import annotations
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID
from sqlalchemy import text
from mlflow import MlflowClient
import ml.train as train  

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    
from api.dependencies import get_db  

logger = logging.getLogger("api.workers.ml_training")


class JobCancelled(Exception):
    """Raised at a checkpoint when the row's status has been flipped to cancelled."""


# Row IO — small helpers, kept here so the router doesn't grow a DB layer.
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _update_status(
    job_id: UUID,
    *,
    status: Optional[str] = None,
    progress_message: Optional[str] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
    error: Optional[str] = None,
    cache_key: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
    mlflow_experiment_id: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Apply a partial update to the job row in its own short transaction."""
    sets, params = [], {"job_id": str(job_id)}
    for col, val in [
        ("status", status),
        ("progress_message", progress_message),
        ("started_at", started_at),
        ("finished_at", finished_at),
        ("error", error),
        ("cache_key", cache_key),
        ("mlflow_run_id", mlflow_run_id),
        ("mlflow_experiment_id", mlflow_experiment_id),
    ]:
        if val is not None:
            sets.append(f"{col} = :{col}")
            params[col] = val

    if metrics is not None:
        import json as _json
        sets.append("metrics = CAST(:metrics AS JSONB)")
        params["metrics"] = _json.dumps(metrics, default=str)

    if not sets:
        return

    sql = f"UPDATE ml_jobs.training_jobs SET {', '.join(sets)} WHERE job_id = :job_id"
    db = get_db()
    with db.engine.begin() as conn:
        conn.execute(text(sql), params)


def _read_status(job_id: UUID) -> Optional[str]:
    db = get_db()
    with db.engine.connect() as conn:
        row = conn.execute(
            text("SELECT status FROM ml_jobs.training_jobs WHERE job_id = :id"),
            {"id": str(job_id)},
        ).fetchone()
    return row[0] if row else None


def _check_not_cancelled(job_id: UUID) -> None:
    if _read_status(job_id) == "cancelled":
        raise JobCancelled()



# MLflow tag stamping
def _stamp_mlflow_tags(run_id: Optional[str], tags: Dict[str, str]) -> None:
    """Add cross-link tags to an MLflow run after the training script finishes.

    Failure to stamp is non-fatal — tags are nice-to-have for navigation.
    """
    if not run_id:
        return
    try:
        client = MlflowClient()
        for k, v in tags.items():
            if v is not None:
                client.set_tag(run_id, k, str(v))
    except Exception as exc:
        logger.warning("Failed to stamp MLflow tags on run %s: %s", run_id, exc)



# Headline metrics extraction (denormalized snapshot for fast listing)
def _headline_classification(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the few classifier metrics worth caching on the job row."""
    keep = ("accuracy", "macro_f1", "weighted_f1", "roc_auc", "roc_auc_macro")
    return {k: metrics[k] for k in keep if k in metrics}


def _headline_regression(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Per-target {r2, mae, rmse} — drop everything else."""
    out = {}
    for col, m in metrics.items():
        if isinstance(m, dict):
            out[col] = {k: m[k] for k in ("r2", "mae", "rmse") if k in m}
    return out



# Worker entry point
def run_training_job(job_id: UUID, equipment_type: str, task: str, request: Dict[str, Any]) -> None:
    """BackgroundTasks entry point. Runs the full pipeline for one job.

    `request` is the validated TrainingJobRequest body (dict form). The DB
    row was inserted by the router before this was scheduled, so we only
    transition state and stamp MLflow tags from here.
    """
    cfg = train.PipelineConfig(
        equipment_type=equipment_type,
        **request["config"],
    )
    use_cache = request.get("use_cache", True)
    force_recompute = request.get("force_recompute", False)

    started = _now()
    _update_status(
        job_id,
        status="running",
        started_at=started,
        progress_message="loading raw data",
    )

    try:
        _check_not_cancelled(job_id)

        # Phase 1: shared pipeline (load → label → split → engineer → cache)
        _update_status(job_id, progress_message="engineering features")
        train_df, val_df, test_df, cache_key = train.get_engineered_splits(
            cfg, use_cache=use_cache, force_recompute=force_recompute,
        )
        _update_status(job_id, cache_key=cache_key)
        _check_not_cancelled(job_id)

        if task == "features_precompute":
            # Pre-warm the cache and exit; no MLflow run for this task.
            _update_status(
                job_id,
                status="succeeded",
                finished_at=_now(),
                progress_message="cache populated",
                metrics={"rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}},
            )
            return

        # Phase 2: dispatch to the task-specific training run
        _update_status(job_id, progress_message=f"training {task}")
        if task == "classification":
            metrics = train.run_classification(train_df, val_df, test_df, cfg)
            headline = _headline_classification(metrics)
        elif task == "regression":
            metrics = train.run_regression(train_df, val_df, test_df, cfg)
            headline = _headline_regression(metrics)
        else:
            raise ValueError(f"Unknown task: {task}")

        # Phase 3: stamp MLflow run with cross-link tags
        run_id, exp_id = _latest_mlflow_run_for(cfg.experiment_name)
        _stamp_mlflow_tags(run_id, {
            "pdm.job_id": str(job_id),
            "pdm.cache_key": cache_key,
            "pdm.task": task,
            "pdm.equipment_type": equipment_type,
            "pdm.submitted_by": request.get("_submitted_by") or "",
            "pdm.request_id": request.get("_request_id") or "",
        })

        _update_status(
            job_id,
            status="succeeded",
            finished_at=_now(),
            progress_message="complete",
            mlflow_run_id=run_id,
            mlflow_experiment_id=exp_id,
            metrics=headline,
        )

    except JobCancelled:
        _update_status(
            job_id,
            status="cancelled",
            finished_at=_now(),
            progress_message="cancelled by request",
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        tb = traceback.format_exc(limit=10)
        _update_status(
            job_id,
            status="failed",
            finished_at=_now(),
            error=f"{type(exc).__name__}: {exc}\n{tb}",
        )



# Best-effort lookup of the run the training script just produced.
def _latest_mlflow_run_for(experiment_name: str):
    """Return (run_id, experiment_id) of the most recent run in the experiment, if any."""
    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None, None
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None, exp.experiment_id
        return runs[0].info.run_id, exp.experiment_id
    except Exception as exc:
        logger.warning("MLflow lookup failed for experiment %s: %s", experiment_name, exc)
        return None, None