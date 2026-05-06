"""
Pydantic schemas for ML pipeline endpoints.
Training export, feature windows, label vectors, dataset statistics, training jobs.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field

class LabelStrategy(str, Enum):
    binary = "binary"          # 0 = normal, 1 = within prediction horizon of failure
    rul = "rul"                # Remaining useful life in hours
    multiclass = "multiclass"  # normal / degrading / imminent / failed

class ExportFormat(str, Enum):
    json = "json"
    csv = "csv"

# Feature window
class FeatureWindow(BaseModel):
    failure_id: Optional[int] = None
    failure_mode: Optional[str] = None
    failure_time: Optional[datetime] = None
    label: str  # "failure" or "normal"
    window_start: Optional[datetime] = None
    records: List[Dict[str, Any]]

class FeatureWindowsResponse(BaseModel):
    equipment_id: int
    equipment_type: str
    window_size: int
    windows: List[FeatureWindow]
    total_failure_windows: int
    total_normal_windows: int

# Label vector
class LabelEntry(BaseModel):
    sample_time: datetime
    label: Any  # int for binary/multiclass, float for RUL

class LabelVectorResponse(BaseModel):
    equipment_id: int
    equipment_type: str
    label_strategy: str
    prediction_horizon_hours: Optional[float] = None
    labels: List[LabelEntry]
    total_samples: int
    positive_samples: int
    class_ratio: float

# Dataset statistics
class FeatureStat(BaseModel):
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skew: Optional[float] = None
    null_pct: float = 0.0

class HealthDistribution(BaseModel):
    mean: Optional[float] = None
    std: Optional[float] = None
    pct_below_0_5: float = 0.0

class ClassBalance(BaseModel):
    total_failures: int
    by_mode: Dict[str, int]
    failure_rate_per_1000h: Optional[float] = None

class TimeCoverage(BaseModel):
    start: Optional[datetime] = None
    end: Optional[datetime] = None

class DatasetStatsResponse(BaseModel):
    equipment_type: str
    total_records: int
    total_equipment: int
    time_coverage: TimeCoverage
    class_balance: ClassBalance
    feature_stats: Dict[str, FeatureStat]
    health_distribution: Dict[str, HealthDistribution]


# Training Jobs
class TrainingTask(str, Enum):
    classification = "classification"
    regression = "regression"
    features_precompute = "features_precompute"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainingPipelineConfig(BaseModel):
    """Mirrors PipelineConfig in train.py — knobs that affect the training run.

    The first four influence the engineered-feature cache key; the rest are
    training-only and don't invalidate the cache.
    """
    prediction_horizon_hours: float = Field(168.0, gt=0, le=10000)
    test_fraction: float = Field(0.25, gt=0, lt=1)
    val_fraction: float = Field(0.10, gt=0, lt=1)

    experiment_name: str = "pdm_end_to_end"
    n_cv_folds: int = Field(5, ge=2, le=20)
    n_iter: int = Field(10, ge=1, le=200)
    model_type: str = Field("xgboost", pattern="^(xgboost)$")
    log_to_mlflow: bool = True


class TrainingJobRequest(BaseModel):
    """Body for POST /ml/{equipment_type}/train.

    `equipment_type` comes from the path; the body specifies task + config + cache flags.
    """
    task: TrainingTask
    config: TrainingPipelineConfig = Field(default_factory=TrainingPipelineConfig)
    use_cache: bool = True
    force_recompute: bool = False


class MlflowLink(BaseModel):
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    ui_url: Optional[str] = None


class TrainingJobResponse(BaseModel):
    """Job state surface. `metrics` is a denormalized snapshot; full data lives in MLflow."""
    job_id: UUID
    task: TrainingTask
    equipment_type: str
    status: JobStatus

    progress_message: Optional[str] = None
    error: Optional[str] = None

    submitted_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    submitted_by: Optional[str] = None
    request_id: Optional[UUID] = None
    idempotency_key: Optional[str] = None

    cache_key: Optional[str] = None
    config: Dict[str, Any]
    mlflow: MlflowLink = Field(default_factory=MlflowLink)
    metrics: Optional[Dict[str, Any]] = None

    archived: bool = False


class TrainingJobListResponse(BaseModel):
    items: List[TrainingJobResponse]
    next_cursor: Optional[UUID] = None
    has_more: bool
    count: int