"""
Pydantic schemas for telemetry endpoints.
Ingestion, query, health indicators, failures, and maintenance.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class EquipmentTypeEnum(str, Enum):
    turbine = "turbine"
    compressor = "compressor"
    pump = "pump"

class BinInterval(str, Enum):
    five_min = "5m"
    one_hour = "1h"
    eight_hour = "8h"
    twenty_four_hour = "24h"
    shift = "shift"  # 12h aligned to 06:00/18:00

class OperatingState(str, Enum):
    running = "running"
    idle = "idle"
    startup = "startup"
    shutdown = "shutdown"

# Ingestion
class TelemetryRecord(BaseModel):
    """Single telemetry record for ingestion — matches bulk_insert format."""
    equipment_id: int
    sample_time: datetime
    operating_hours: float
    state: Dict[str, Any]

class TelemetryIngestRequest(BaseModel):
    equipment_type: EquipmentTypeEnum
    records: List[TelemetryRecord] = Field(..., max_length=50000)

class TelemetryIngestResponse(BaseModel):
    inserted_count: int
    equipment_type: str
    message: str


# Query responses
class TelemetryRow(BaseModel):
    """Generic telemetry row with computed operating_state."""
    telemetry_id: int
    sample_time: datetime
    operating_hours: Optional[float] = None
    operating_state: str = "unknown"
    data: Dict[str, Any]

class CursorPaginatedTelemetry(BaseModel):
    items: List[TelemetryRow]
    next_cursor: Optional[int] = None
    has_more: bool = False
    limit: int

class HealthIndicators(BaseModel):
    equipment_id: int
    equipment_type: str
    sample_time: datetime
    operating_state: str
    health_components: Dict[str, float]

class BinnedSummaryBin(BaseModel):
    bin_start: datetime
    bin_end: datetime
    record_count: int
    stats: Dict[str, Dict[str, Optional[float]]]  # col -> {avg, min, max, stddev}

class BinnedSummaryResponse(BaseModel):
    equipment_id: int
    equipment_type: str
    bin_interval: str
    bins: List[BinnedSummaryBin]

class FailureEventResponse(BaseModel):
    failure_id: int
    equipment_id: int
    failure_time: datetime
    operating_hours_at_failure: Optional[float] = None
    failure_mode_code: Optional[str] = None
    failure_description: Optional[str] = None

class MaintenanceEventResponse(BaseModel):
    maintenance_id: int
    equipment_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    failure_code: Optional[str] = None
    downtime_hours: Optional[float] = None
    repaired_components: Optional[Dict[str, Any]] = None