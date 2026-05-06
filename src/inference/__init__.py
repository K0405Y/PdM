"""Triton Inference Server integration for the PdM platform.

Modules:
- export_models: Convert MLflow-tracked XGBoost models to Triton FIL format
- config_generator: Generate Triton config.pbtxt files
- model_registry: Track MLflow run_id <-> Triton version mapping
- triton_client: gRPC client wrapper for FastAPI
- explainer: SHAP TreeExplainer manager for the chained pipeline
"""
import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
sys.path.append(str(repo_root))