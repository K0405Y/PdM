"""gRPC client wrapper for Triton Inference Server.

Wraps tritonclient.grpc with PdM-aware methods so the FastAPI inference router
can call models by their Triton names without dealing with tensor plumbing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _load_metadata(model_repo_root: Path, model_name: str) -> dict:
    """Read metadata.json sidecar produced by export_models.py."""
    path = model_repo_root / model_name / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found for {model_name} at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TritonInferenceClient:
    """Thin wrapper over tritonclient.grpc.InferenceServerClient.

    Uses metadata.json sidecars to map equipment_type/health_column to Triton
    model names and to know each model's input/output dimensions.
    """

    INPUT_NAME = "input__0"
    OUTPUT_NAME = "output__0"

    def __init__(
        self,
        url: str = "localhost:8001",
        model_repo_root: Optional[Path] = None,
    ):
        # Lazy import: tritonclient is a heavy dep, only required at runtime
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as e:
            raise RuntimeError(
                "tritonclient[grpc] is required"
            ) from e
        self._grpcclient = grpcclient
        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        self._url = url

        if model_repo_root is None:
            model_repo_root = Path(__file__).resolve().parents[2] / "triton" / "model_repository"
        self.model_repo_root = Path(model_repo_root)

        # Cache: model_name -> metadata dict
        self._metadata_cache: Dict[str, dict] = {}

    # introspection

    def is_healthy(self) -> bool:
        """Return True if Triton server is reachable and ready."""
        try:
            return self._client.is_server_ready()
        except Exception:
            return False

    def model_status(self, model_name: str) -> Dict:
        """Return model readiness + version info from Triton."""
        try:
            ready = self._client.is_model_ready(model_name)
        except Exception as e:
            return {"name": model_name, "ready": False, "error": str(e)}
        return {"name": model_name, "ready": ready}

    def get_metadata(self, model_name: str) -> dict:
        """Load and cache metadata.json for a Triton model."""
        if model_name not in self._metadata_cache:
            self._metadata_cache[model_name] = _load_metadata(
                self.model_repo_root, model_name
            )
        return self._metadata_cache[model_name]

    # low-level inference

    def _infer(
        self,
        model_name: str,
        features: np.ndarray,
        n_outputs: int,
    ) -> np.ndarray:
        """Run inference. `features` is shape (batch, n_features).

        Returns array of shape (batch, n_outputs).
        """
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape {features.shape}")
        if features.dtype != np.float32:
            features = features.astype(np.float32, copy=False)

        infer_input = self._grpcclient.InferInput(
            self.INPUT_NAME,
            list(features.shape),
            "FP32",
        )
        infer_input.set_data_from_numpy(features)

        infer_output = self._grpcclient.InferRequestedOutput(self.OUTPUT_NAME)

        response = self._client.infer(
            model_name=model_name,
            inputs=[infer_input],
            outputs=[infer_output],
        )
        result = response.as_numpy(self.OUTPUT_NAME)
        if result is None:
            raise RuntimeError(f"Triton returned no output for {model_name}")

        # Some FIL configs return shape (batch,) for scalar regressors; normalize to 2D
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        return result

    # high-level methods

    def predict_classifier(
        self,
        equipment_type: str,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the classifier and return (class_indices, class_probabilities).

        `features` must already be the **augmented** input for the classifier
        (raw features + predicted health scores in metadata.health_input_columns
        order). The router builds this; the client just forwards.
        """
        model_name = f"{equipment_type}_classifier"
        meta = self.get_metadata(model_name)
        n_classes = meta.get("n_classes")
        if n_classes is None:
            raise RuntimeError(f"Classifier {model_name} metadata missing n_classes")

        probs = self._infer(model_name, features, n_outputs=n_classes)
        class_indices = np.argmax(probs, axis=1)
        return class_indices, probs

    def predict_health(
        self,
        equipment_type: str,
        health_column: str,
        features: np.ndarray,
    ) -> np.ndarray:
        """Run a single health regressor. Returns shape (batch,) of raw model output.

        Treelite-in-FIL-24.08 does not apply XGBoost's `base_score` for
        regressors, so we add it back from metadata when present.
        """
        model_name = f"{equipment_type}_{health_column}"
        out = self._infer(model_name, features, n_outputs=1).ravel()
        base_score = self.get_metadata(model_name).get("base_score")
        if base_score is not None:
            out = out + float(base_score)
        return out

    def predict_all_health(
        self,
        equipment_type: str,
        features: np.ndarray,
        health_columns: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run every health regressor for an equipment type.

        Each call is a separate gRPC request. For the typical case
        this is well within Triton's serving capacity and avoids the complexity
        of batched gRPC calls. If `health_columns` is None, the list is read
        from the classifier metadata (which records the health input order).
        """
        if health_columns is None:
            classifier_meta = self.get_metadata(f"{equipment_type}_classifier")
            health_columns = classifier_meta.get("health_input_columns") or []
        return {
            col: self.predict_health(equipment_type, col, features)
            for col in health_columns
        }

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
