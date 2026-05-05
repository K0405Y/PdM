"""Real-time inference endpoints backed by Triton.

Pipeline:
    raw telemetry --> FeatureEngineer (stateful) --> select_features
        --> health regressors (Triton)
        --> concat raw_features + predicted_health
        --> classifier (Triton)
        --> response

The classifier sees an *augmented* input: raw features + predicted health
scores in the order recorded in its metadata.json (`health_input_columns`).
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import (
    get_feature_engineer_cache,
    get_triton_client,
)
from api.schemas.inference import (
    BatchInferenceRequest,
    ClassificationResult,
    FullAssessmentResult,
    HealthEstimationResult,
    InferenceRequest,
    TritonModelStatus,
    TritonStatusResponse,
)
from src.ml.data_loader import get_health_columns

logger = logging.getLogger(__name__)
router = APIRouter()


SUPPORTED_EQUIPMENT_TYPES = {"turbine", "compressor", "pump"}


def _check_equipment_type(equipment_type: str) -> None:
    if equipment_type not in SUPPORTED_EQUIPMENT_TYPES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown equipment_type '{equipment_type}'. "
            f"Supported: {sorted(SUPPORTED_EQUIPMENT_TYPES)}",
        )


def _build_feature_vector(
    feature_columns: List[str],
    raw_record: Dict[str, Any],
    derived: Dict[str, Any],
) -> np.ndarray:
    """Assemble the model input vector in the exact column order the model expects.
    `raw_record` is the user-provided sensor reading; `derived` is whatever the
    FeatureEngineer computed. Missing columns get 0.0 (the model's median-
    imputation happened at training time; runtime missing values are rare and
    a zero is the safe fallback). Returns shape (1, n_features).
    """
    merged = {**raw_record, **derived}
    row = np.zeros((1, len(feature_columns)), dtype=np.float32)
    for i, col in enumerate(feature_columns):
        v = merged.get(col)
        if v is None:
            continue
        try:
            row[0, i] = float(v)
        except (TypeError, ValueError):
            # Booleans / ints from the DB sometimes; cast via numpy
            row[0, i] = float(np.asarray(v, dtype=np.float32).item())
    return row


def _compute_features_for_record(
    equipment_type: str,
    equipment_id: int,
    record: Dict[str, Any],
    fe_cache,
) -> Dict[str, Any]:
    """Run FeatureEngineer.compute() with the per-equipment stateful instance."""
    fe = fe_cache.get(equipment_type, equipment_id)
    return fe.compute(record)


def _augment_with_health(
    raw_features: np.ndarray,
    raw_columns: List[str],
    health_scores: Dict[str, float],
    health_input_columns: List[str],
) -> np.ndarray:
    """Concat predicted health onto the raw feature vector in the expected order.

    `raw_features` shape: (batch, n_raw). Returns shape (batch, n_raw + n_health).
    """
    batch = raw_features.shape[0]
    health_block = np.zeros((batch, len(health_input_columns)), dtype=np.float32)
    for i, col in enumerate(health_input_columns):
        v = health_scores.get(col, 0.0)
        health_block[:, i] = float(v)
    return np.hstack([raw_features, health_block]).astype(np.float32)


def _clip_health(v: float) -> float:
    """Clamp predicted health to [0, 1] - regressor output can drift slightly out."""
    return float(min(max(v, 0.0), 1.0))


def _decode_class(class_index: int, class_names: Optional[List[str]]) -> str:
    if not class_names:
        return str(class_index)
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return str(class_index)


# endpoints


@router.get("/status", response_model=TritonStatusResponse)
def inference_status(triton=Depends(get_triton_client)):
    """Triton server reachability + per-model readiness."""
    server_ready = triton.is_healthy()
    model_names: List[str] = []
    for eq in sorted(SUPPORTED_EQUIPMENT_TYPES):
        model_names.append(f"{eq}_classifier")
        for col in get_health_columns(eq):
            model_names.append(f"{eq}_{col}")
    statuses = [TritonModelStatus(**triton.model_status(name)) for name in model_names]
    return TritonStatusResponse(
        triton_url=getattr(triton, "_url", "unknown"),
        server_ready=server_ready,
        models=statuses,
    )


@router.post(
    "/{equipment_type}/health",
    response_model=HealthEstimationResult,
)
def predict_health(
    equipment_type: str,
    request: InferenceRequest,
    triton=Depends(get_triton_client),
    fe_cache=Depends(get_feature_engineer_cache),
):
    """Run only the health regressor stage for one equipment unit."""
    _check_equipment_type(equipment_type)
    if not triton.is_healthy():
        raise HTTPException(503, "Triton inference server is not ready")

    health_cols = get_health_columns(equipment_type)
    if not health_cols:
        raise HTTPException(500, f"No health columns configured for {equipment_type}")

    # Run FeatureEngineer to populate trend/stability features
    derived = _compute_features_for_record(
        equipment_type, request.equipment_id, request.sensor_data, fe_cache
    )

    # Use the first regressor's metadata for column ordering. All regressors
    # for one equipment type share the same input schema.
    sample_meta = triton.get_metadata(f"{equipment_type}_{health_cols[0]}")
    feature_columns = sample_meta["feature_columns"]
    features = _build_feature_vector(feature_columns, request.sensor_data, derived)

    health_predictions = triton.predict_all_health(
        equipment_type, features, health_columns=health_cols
    )

    return HealthEstimationResult(
        equipment_type=equipment_type,
        equipment_id=request.equipment_id,
        health_scores={c: _clip_health(v[0]) for c, v in health_predictions.items()},
    )


@router.post(
    "/{equipment_type}/classify",
    response_model=ClassificationResult,
)
def predict_classifier(
    equipment_type: str,
    request: InferenceRequest,
    include_health: bool = Query(False, description="Include intermediate health scores in response"),
    triton=Depends(get_triton_client),
    fe_cache=Depends(get_feature_engineer_cache),
):
    """Run the chained pipeline (health regressors --> classifier)."""
    _check_equipment_type(equipment_type)
    if not triton.is_healthy():
        raise HTTPException(503, "Triton inference server is not ready")

    derived = _compute_features_for_record(
        equipment_type, request.equipment_id, request.sensor_data, fe_cache
    )

    classifier_meta = triton.get_metadata(f"{equipment_type}_classifier")
    raw_feature_columns = classifier_meta.get("raw_feature_columns") or []
    health_input_columns = classifier_meta.get("health_input_columns") or []
    class_names = classifier_meta.get("class_names")

    if not raw_feature_columns or not health_input_columns:
        raise HTTPException(
            500,
            f"Classifier metadata for {equipment_type} is missing "
            "raw_feature_columns or health_input_columns. Re-run the export.",
        )

    raw_features = _build_feature_vector(raw_feature_columns, request.sensor_data, derived)

    # Stage 1: health regressors
    health_predictions_raw = triton.predict_all_health(
        equipment_type, raw_features, health_columns=health_input_columns
    )
    health_scores = {c: _clip_health(v[0]) for c, v in health_predictions_raw.items()}

    # Stage 2: augment + classify
    augmented = _augment_with_health(
        raw_features=raw_features,
        raw_columns=raw_feature_columns,
        health_scores=health_scores,
        health_input_columns=health_input_columns,
    )
    class_indices, probabilities = triton.predict_classifier(equipment_type, augmented)
    class_idx = int(class_indices[0])
    class_label = _decode_class(class_idx, class_names)
    probs = probabilities[0]
    prob_map = {
        _decode_class(i, class_names): float(p) for i, p in enumerate(probs)
    }

    return ClassificationResult(
        equipment_type=equipment_type,
        equipment_id=request.equipment_id,
        predicted_class=class_label,
        predicted_class_index=class_idx,
        probabilities=prob_map,
        confidence=float(probs[class_idx]),
        health_scores=health_scores if include_health else None,
    )


@router.post(
    "/{equipment_type}/full-assessment",
    response_model=FullAssessmentResult,
)
def full_assessment(
    equipment_type: str,
    request: InferenceRequest,
    triton=Depends(get_triton_client),
    fe_cache=Depends(get_feature_engineer_cache),
):
    """Convenience endpoint: returns classifier prediction AND all health scores.

    Mechanically the same chained call as `/classify?include_health=true`,
    surfaced as a separate endpoint so consumers can document/intend it.
    """
    result = predict_classifier(
        equipment_type=equipment_type,
        request=request,
        include_health=True,
        triton=triton,
        fe_cache=fe_cache,
    )
    return FullAssessmentResult(
        equipment_type=result.equipment_type,
        equipment_id=result.equipment_id,
        predicted_class=result.predicted_class,
        predicted_class_index=result.predicted_class_index,
        probabilities=result.probabilities,
        confidence=result.confidence,
        health_scores=result.health_scores or {},
    )
