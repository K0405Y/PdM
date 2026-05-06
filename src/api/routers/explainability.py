"""SHAP explainability endpoints.

Endpoints reuse the inference router's preprocessing logic so explanations
operate on the exact feature vectors the models would see in production.
"""

import logging
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import (
    get_explainer_manager,
    get_feature_engineer_cache,
    get_triton_client,
)
from api.routers.inference import (
    SUPPORTED_EQUIPMENT_TYPES,
    _build_classifier_input,
    _build_feature_vector,
    _check_equipment_type,
    _clip_health,
    _compute_features_for_record,
)
from api.schemas.explainability import (
    Contributor,
    FullAssessmentShapResponse,
    ShapExplanation,
    ShapSummaryResponse,
)
from api.schemas.inference import InferenceRequest
from src.ml.data_loader import get_health_columns

logger = logging.getLogger(__name__)
router = APIRouter()

def _serialize_explanation(
    model_name: str,
    equipment_type: str,
    equipment_id: int,
    result,
    include_full: bool,
) -> ShapExplanation:
    """Convert a ShapResult dataclass into the API response model."""
    return ShapExplanation(
        model_name=model_name,
        equipment_type=equipment_type,
        equipment_id=equipment_id,
        feature_names=result.feature_names,
        base_value=result.base_value,
        predicted_value=result.predicted_value,
        top_contributors=[Contributor(**c) for c in result.top_contributors],
        shap_values=(
            [float(x) for x in result.shap_values] if include_full else None
        ),
        health_input_columns=result.health_input_columns,
    )


def _prepare_inputs(
    equipment_type: str,
    request: InferenceRequest,
    triton,
    fe_cache,
):
    """Run preprocessing + health stage.
    Returns (health_features, classifier_input, health_scores, health_input_columns).
      - health_features: vector in the health regressors' training column order
        (also reused as the SHAP input for per-health explanations).
      - classifier_input: vector in the CLASSIFIER's training column order, with
        predicted health values placed at their original positions.

    Shared by classify/full-assessment SHAP endpoints; the health-only endpoint
    builds its own vector since it doesn't need the classifier input.
    """
    derived = _compute_features_for_record(
        equipment_type, request.equipment_id, request.sensor_data, fe_cache
    )
    classifier_meta = triton.get_metadata(f"{equipment_type}_classifier")
    classifier_feature_columns = classifier_meta.get("feature_columns") or []
    health_input_columns = classifier_meta.get("health_input_columns") or []
    if not classifier_feature_columns or not health_input_columns:
        raise HTTPException(
            500,
            f"Classifier metadata for {equipment_type} is incomplete; re-run export",
        )

    # Health regressors have their own (smaller) input schema.
    health_meta = triton.get_metadata(f"{equipment_type}_{health_input_columns[0]}")
    health_features = _build_feature_vector(
        health_meta["feature_columns"], request.sensor_data, derived
    )
    raw_health = triton.predict_all_health(
        equipment_type, health_features, health_columns=health_input_columns
    )
    health_scores = {c: _clip_health(v[0]) for c, v in raw_health.items()}

    # Classifier input in TRAINING column order - health values placed at their
    # original positions (which may be interleaved with raw features).
    classifier_input = _build_classifier_input(
        feature_columns=classifier_feature_columns,
        raw_record=request.sensor_data,
        derived=derived,
        health_scores=health_scores,
    )
    return health_features, classifier_input, health_scores, health_input_columns


# endpoints 

@router.post(
    "/{equipment_type}/health/{health_column}",
    response_model=ShapExplanation,
)
def shap_health(
    equipment_type: str,
    health_column: str,
    request: InferenceRequest,
    full: bool = Query(False, description="Include full per-feature SHAP vector"),
    top_n: int = Query(10, ge=1, le=200),
    triton=Depends(get_triton_client),
    fe_cache=Depends(get_feature_engineer_cache),
    explainer=Depends(get_explainer_manager),
):
    """SHAP explanation for one health regressor."""
    _check_equipment_type(equipment_type)
    if health_column not in get_health_columns(equipment_type):
        raise HTTPException(
            404,
            f"{equipment_type} does not have health column '{health_column}'",
        )

    derived = _compute_features_for_record(
        equipment_type, request.equipment_id, request.sensor_data, fe_cache
    )
    model_name = f"{equipment_type}_{health_column}"
    meta = triton.get_metadata(model_name)
    feature_columns: List[str] = meta["feature_columns"]
    features = _build_feature_vector(feature_columns, request.sensor_data, derived)

    result = explainer.explain_health(
        equipment_type=equipment_type,
        health_column=health_column,
        raw_features=features,
        top_n=top_n,
    )
    return _serialize_explanation(
        model_name=model_name,
        equipment_type=equipment_type,
        equipment_id=request.equipment_id,
        result=result,
        include_full=full,
    )


@router.post(
    "/{equipment_type}/classify",
    response_model=ShapExplanation,
)
def shap_classifier(
    equipment_type: str,
    request: InferenceRequest,
    full: bool = Query(False),
    top_n: int = Query(10, ge=1, le=200),
    class_index: Optional[int] = Query(
        None, description="Explain a specific class; default is the predicted class"
    ),
    triton=Depends(get_triton_client),
    fe_cache=Depends(get_feature_engineer_cache),
    explainer=Depends(get_explainer_manager),
):
    """SHAP explanation for the classifier on the augmented input.

    Predicted health features show up in `top_contributors` prefixed with
    `predicted_` so callers can distinguish raw sensors from predicted health.
    """
    _check_equipment_type(equipment_type)
    _, classifier_input, _, _ = _prepare_inputs(
        equipment_type, request, triton, fe_cache
    )
    result = explainer.explain_classifier(
        equipment_type=equipment_type,
        classifier_input=classifier_input,
        top_n=top_n,
        class_index=class_index,
    )
    return _serialize_explanation(
        model_name=f"{equipment_type}_classifier",
        equipment_type=equipment_type,
        equipment_id=request.equipment_id,
        result=result,
        include_full=full,
    )


@router.post(
    "/{equipment_type}/full-assessment",
    response_model=FullAssessmentShapResponse,
)
def shap_full_assessment(
    equipment_type: str,
    request: InferenceRequest,
    full: bool = Query(False),
    top_n: int = Query(10, ge=1, le=200),
    triton=Depends(get_triton_client),
    fe_cache=Depends(get_feature_engineer_cache),
    explainer=Depends(get_explainer_manager),
):
    """Two-level explanation: classifier-over-augmented + every health regressor."""
    _check_equipment_type(equipment_type)
    health_features, classifier_input, _, health_input_columns = _prepare_inputs(
        equipment_type, request, triton, fe_cache
    )
    bundle = explainer.explain_full_assessment(
        equipment_type=equipment_type,
        health_features=health_features,
        classifier_input=classifier_input,
        health_columns=health_input_columns,
        top_n=top_n,
    )
    classifier_payload = _serialize_explanation(
        model_name=f"{equipment_type}_classifier",
        equipment_type=equipment_type,
        equipment_id=request.equipment_id,
        result=bundle["classifier_explanation"],
        include_full=full,
    )
    health_payloads = {
        col: _serialize_explanation(
            model_name=f"{equipment_type}_{col}",
            equipment_type=equipment_type,
            equipment_id=request.equipment_id,
            result=res,
            include_full=full,
        )
        for col, res in bundle["health_explanations"].items()
    }
    return FullAssessmentShapResponse(
        equipment_type=equipment_type,
        equipment_id=request.equipment_id,
        classifier_explanation=classifier_payload,
        health_explanations=health_payloads,
    )


@router.post(
    "/{equipment_type}/batch",
    response_model=ShapSummaryResponse,
)
def shap_batch_summary(
    equipment_type: str,
    requests: List[InferenceRequest],
    model: str = Query(
        "classifier",
        description="Model purpose: 'classifier' or a health column name",
    ),
    triton=Depends(get_triton_client),
    fe_cache=Depends(get_feature_engineer_cache),
    explainer=Depends(get_explainer_manager),
):
    """Summary SHAP for a batch (mean |SHAP| per feature) -- input for beeswarm/summary plots."""
    _check_equipment_type(equipment_type)
    if not requests:
        raise HTTPException(400, "At least one request required")

    is_classifier = model == "classifier"
    if not is_classifier and model not in get_health_columns(equipment_type):
        raise HTTPException(404, f"Unknown model '{model}' for {equipment_type}")

    feature_matrix_rows = []
    for req in requests:
        if is_classifier:
            _, classifier_input, _, _ = _prepare_inputs(equipment_type, req, triton, fe_cache)
            feature_matrix_rows.append(classifier_input[0])
        else:
            derived = _compute_features_for_record(
                equipment_type, req.equipment_id, req.sensor_data, fe_cache
            )
            meta = triton.get_metadata(f"{equipment_type}_{model}")
            features = _build_feature_vector(
                meta["feature_columns"], req.sensor_data, derived
            )
            feature_matrix_rows.append(features[0])
    feature_matrix = np.vstack(feature_matrix_rows).astype(np.float32)

    model_name = (
        f"{equipment_type}_classifier" if is_classifier else f"{equipment_type}_{model}"
    )
    summary = explainer.explain_batch(model_name, feature_matrix)
    return ShapSummaryResponse(
        model_name=model_name,
        feature_names=summary.feature_names,
        mean_abs_shap=summary.mean_abs_shap,
        n_samples=feature_matrix.shape[0],
    )
