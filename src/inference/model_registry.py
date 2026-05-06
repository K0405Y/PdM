"""Model version registry for Triton.

Tracks the mapping between MLflow model IDs and Triton version numbers
so we know which MLflow run produced each Triton version, and so we can
detect when a re-export should bump the version.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"


class ModelRegistry:
    """Persists MLflow <-> Triton version mappings as a JSON file in the repo root."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.path = self.repo_root / REGISTRY_FILENAME
        self._data: Dict[str, Dict[str, dict]] = self._load()

    def _load(self) -> Dict[str, Dict[str, dict]]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load registry at {self.path}: {e}. Starting fresh.")
            return {}

    def save(self) -> None:
        self.repo_root.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)

    def latest_version(self, model_name: str) -> int:
        """Return the highest Triton version number for a model, or 0 if absent."""
        versions = self._data.get(model_name, {})
        if not versions:
            return 0
        return max(int(v) for v in versions.keys())

    def latest_mlflow_id(self, model_name: str) -> Optional[str]:
        """Return the MLflow model_id stored for the latest Triton version, or None."""
        latest = self.latest_version(model_name)
        if latest == 0:
            return None
        return self._data[model_name][str(latest)].get("mlflow_model_id")

    def needs_new_version(self, model_name: str, mlflow_model_id: str) -> bool:
        """True if this MLflow model_id is not the latest for the given Triton model."""
        return self.latest_mlflow_id(model_name) != mlflow_model_id

    def add_version(
        self,
        model_name: str,
        mlflow_model_id: str,
        mlflow_run_id: str,
        extra: Optional[dict] = None,
    ) -> int:
        """Record a new Triton version. Returns the assigned version number."""
        next_version = self.latest_version(model_name) + 1
        entry = {
            "mlflow_model_id": mlflow_model_id,
            "mlflow_run_id": mlflow_run_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)
        self._data.setdefault(model_name, {})[str(next_version)] = entry
        return next_version

    def list_models(self) -> List[str]:
        return sorted(self._data.keys())

    def all_entries(self) -> Dict[str, Dict[str, dict]]:
        return dict(self._data)
