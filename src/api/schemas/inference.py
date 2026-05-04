"""Pydantic schemas for the inference router.

Inference endpoints accept raw sensor readings (DB column names) and return
predictions from the chained pipeline: health regressors --> classifier.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Single-record inference request.
    `sensor_data` keys match DB telemetry column names (e.g. `speed_rpm`,
    `vibration_rms_mm_s`). Missing fields are imputed using the model's
    training-set medians; unknown fields are ignored.
    """

    equipment_id: int = Field(..., description="Equipment unit identifier")
    sensor_data: Dict[str, Any] = Field(
        ..., description="Raw sensor readings keyed by DB column name"
    )
    sample_time: Optional[datetime] = Field(
        None, description="Record timestamp; defaults to server clock if absent"
    )


class BatchInferenceRequest(BaseModel):
    """Batch variant: multiple records for one equipment_id, time-ordered."""
    
    equipment_id: int
    records: List[Dict[str, Any]] = Field(..., min_length=1)


class HealthEstimationResult(BaseModel):
    """Health scores predicted by the regressor stage."""

    equipment_type: str
    equipment_id: int
    health_scores: Dict[str, float] = Field(
        ..., description="Component name -> predicted health"
    )


class ClassificationResult(BaseModel):
    """Failure-mode prediction from the classifier stage."""

    equipment_type: str
    equipment_id: int
    predicted_class: str = Field(..., description="Predicted failure mode label")
    predicted_class_index: int
    probabilities: Dict[str, float] = Field(
        ..., description="Probability per class label"
    )
    confidence: float = Field(..., description="Probability of the predicted class")
    health_scores: Optional[Dict[str, float]] = Field(
        None, description="Predicted health scores used as classifier inputs (if requested)"
    )


class FullAssessmentResult(BaseModel):
    """Combined output: classifier prediction + all health scores."""

    equipment_type: str
    equipment_id: int
    predicted_class: str
    predicted_class_index: int
    probabilities: Dict[str, float]
    confidence: float
    health_scores: Dict[str, float]


class TritonModelStatus(BaseModel):
    name: str
    ready: bool
    error: Optional[str] = None


class TritonStatusResponse(BaseModel):
    triton_url: str
    server_ready: bool
    models: List[TritonModelStatus]
