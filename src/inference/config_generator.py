"""Generate Triton FIL backend config.pbtxt files for tree models.

Produces protobuf text configurations that match the FIL backend's expected
schema for XGBoost JSON models. Classifier configs request probability output
across N classes; regressor configs return a single scalar.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default dynamic batching parameters - amortize inference cost across requests
DEFAULT_PREFERRED_BATCH_SIZES = (32, 64, 128)
DEFAULT_MAX_QUEUE_DELAY_US = 5000
DEFAULT_INSTANCE_COUNT = 2


def _build_dynamic_batching_block(
    preferred: tuple = DEFAULT_PREFERRED_BATCH_SIZES,
    max_queue_delay_us: int = DEFAULT_MAX_QUEUE_DELAY_US,
) -> str:
    sizes = ", ".join(str(s) for s in preferred)
    return (
        "dynamic_batching {\n"
        f"  preferred_batch_size: [ {sizes} ]\n"
        f"  max_queue_delay_microseconds: {max_queue_delay_us}\n"
        "}\n"
    )


def _build_instance_group_block(
    count: int = DEFAULT_INSTANCE_COUNT,
    use_gpu: bool = False,
) -> str:
    kind = "KIND_GPU" if use_gpu else "KIND_CPU"
    return (
        "instance_group [\n"
        f"  {{ count: {count}, kind: {kind} }}\n"
        "]\n"
    )

def _build_runtime_param(use_gpu: bool) -> str:
    runtime = "gpu" if use_gpu else "cpu"
    return f'parameters: {{ key: "runtime" value: {{ string_value: "{runtime}" }} }}\n'

def generate_classifier_config(
    name: str,
    n_features: int,
    n_classes: int,
    max_batch_size: int = 1024,
    use_gpu: bool = False,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
) -> str:
    """config.pbtxt for an XGBoost multi-class classifier served via FIL.

    FIL outputs class probabilities when `predict_proba` and `output_class`
    parameters are both set. Output dimension is n_classes (probability vector).
    """
    return (
        f'name: "{name}"\n'
        'backend: "fil"\n'
        f"max_batch_size: {max_batch_size}\n"
        "input [\n"
        "  {\n"
        '    name: "input__0"\n'
        "    data_type: TYPE_FP32\n"
        f"    dims: [ {n_features} ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "output__0"\n'
        "    data_type: TYPE_FP32\n"
        f"    dims: [ {n_classes} ]\n"
        "  }\n"
        "]\n"
        "parameters: { key: \"model_type\" value: { string_value: \"xgboost_json\" } }\n"
        "parameters: { key: \"predict_proba\" value: { string_value: \"true\" } }\n"
        "parameters: { key: \"output_class\" value: { string_value: \"true\" } }\n"
        "parameters: { key: \"threshold\" value: { string_value: \"0.5\" } }\n"
        f"parameters: {{ key: \"num_class\" value: {{ string_value: \"{n_classes}\" }} }}\n"
        + _build_runtime_param(use_gpu)
        + _build_dynamic_batching_block()
        + _build_instance_group_block(count=instance_count, use_gpu=use_gpu)
        + 'version_policy { latest { num_versions: 2 } }\n'
        + 'default_model_filename: "model.json"\n'
    )


def generate_regressor_config(
    name: str,
    n_features: int,
    max_batch_size: int = 1024,
    use_gpu: bool = False,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    model_type: str = "xgboost_json",
) -> str:
    """config.pbtxt for a tree regressor (XGBoost or RandomForest via treelite).

    Output is a single scalar per sample. RandomForest models converted via
    treelite use model_type 'treelite_checkpoint' instead of 'xgboost_json'.
    """
    return (
        f'name: "{name}"\n'
        'backend: "fil"\n'
        f"max_batch_size: {max_batch_size}\n"
        "input [\n"
        "  {\n"
        '    name: "input__0"\n'
        "    data_type: TYPE_FP32\n"
        f"    dims: [ {n_features} ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "output__0"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1 ]\n"
        "  }\n"
        "]\n"
        f'parameters: {{ key: "model_type" value: {{ string_value: "{model_type}" }} }}\n'
        'parameters: { key: "predict_proba" value: { string_value: "false" } }\n'
        'parameters: { key: "output_class" value: { string_value: "false" } }\n'
        'parameters: { key: "threshold" value: { string_value: "0.5" } }\n'
        + _build_runtime_param(use_gpu)
        + _build_dynamic_batching_block()
        + _build_instance_group_block(count=instance_count, use_gpu=use_gpu)
        + 'version_policy { latest { num_versions: 2 } }\n'
        + 'default_model_filename: "model.json"\n'
    )


def generate_config(
    name: str,
    purpose: str,
    n_features: int,
    n_classes: Optional[int] = None,
    use_gpu: bool = False,
    model_type: str = "xgboost_json",
) -> str:
    """Dispatch to classifier or regressor config generator based on purpose."""
    if purpose == "classifier":
        if n_classes is None or n_classes < 2:
            raise ValueError(f"Classifier config requires n_classes >= 2, got {n_classes}")
        return generate_classifier_config(
            name=name,
            n_features=n_features,
            n_classes=n_classes,
            use_gpu=use_gpu,
        )
    return generate_regressor_config(
        name=name,
        n_features=n_features,
        use_gpu=use_gpu,
        model_type=model_type,
    )
