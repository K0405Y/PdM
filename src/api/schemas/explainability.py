"""Pydantic schemas for SHAP explainability responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Contributor(BaseModel):
    """One feature's contribution to a prediction."""

    feature_name: str
    shap_value: float
    feature_value: float
    is_predicted_health: bool = False


class ShapExplanation(BaseModel):
    """SHAP explanation for a single prediction."""

    model_name: str
    equipment_type: str
    equipment_id: int
    feature_names: List[str]
    base_value: float = Field(..., description="Expected model output over training data")
    predicted_value: float = Field(..., description="Model output for the input being explained")
    top_contributors: List[Contributor]
    shap_values: Optional[List[float]] = Field(
        None,
        description="Per-feature SHAP values, returned only when ?full=true",
    )
    health_input_columns: List[str] = Field(
        default_factory=list,
        description="Names of features that are predicted health inputs (only for classifier)",
    )


class FullAssessmentShapResponse(BaseModel):
    """Two-level explanation: classifier + each health regressor."""

    equipment_type: str
    equipment_id: int
    classifier_explanation: ShapExplanation
    health_explanations: Dict[str, ShapExplanation]


class ShapSummaryResponse(BaseModel):
    model_name: str
    feature_names: List[str]
    mean_abs_shap: List[float]
    n_samples: int
