"""
Pydantic schemas for ML pipeline endpoints.
Training export, feature windows, label vectors, dataset statistics.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
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