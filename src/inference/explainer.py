"""SHAP TreeExplainer manager for the chained inference pipeline.

Loads XGBoost model objects (saved as `model_explainer.json` during export)
plus the K-means background data, and constructs one TreeExplainer per model.
The classifier explainer operates over an input vector built in the model's
training column order, with predicted health values placed at their original
positions (which may be interleaved with raw features, not appended at end).
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _ModelEntry:
    """Holds a loaded XGBoost model + its TreeExplainer + metadata."""

    model: object
    explainer: object  # shap.TreeExplainer
    feature_columns: List[str]
    raw_feature_columns: List[str] = field(default_factory=list)
    health_input_columns: List[str] = field(default_factory=list)
    class_names: Optional[List[str]] = None
    n_classes: Optional[int] = None
    is_classifier: bool = False


@dataclass
class ShapResult:
    """Per-prediction SHAP explanation."""

    feature_names: List[str]
    shap_values: np.ndarray  # shape (n_features,) for regressor or (n_features,) for classifier-of-class
    base_value: float
    predicted_value: float
    top_contributors: List[Dict[str, object]]
    health_input_columns: List[str] = field(default_factory=list)


@dataclass
class ShapSummary:
    """Aggregate SHAP for a batch."""

    feature_names: List[str]
    mean_abs_shap: List[float]
    shap_values_matrix: np.ndarray  # shape (n_samples, n_features)


def _load_xgboost_model(path: Path):
    """Load an XGBoost JSON file. Tries Classifier then Regressor."""
    import xgboost as xgb

    # Peek at the JSON to detect whether it's a classifier
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}

    learner = raw.get("learner", {}) if isinstance(raw, dict) else {}
    objective = learner.get("objective", {}).get("name", "")

    if "softprob" in objective or "softmax" in objective or "binary" in objective:
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def _load_pickle_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_top_contributors(
    feature_names: List[str],
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    health_input_columns: List[str],
    top_n: int = 10,
) -> List[Dict[str, object]]:
    """Sort features by |SHAP| desc, return top_n with metadata."""
    health_set = set(health_input_columns)
    abs_vals = np.abs(shap_values)
    order = np.argsort(-abs_vals)[:top_n]
    out: List[Dict[str, object]] = []
    for idx in order:
        name = feature_names[int(idx)]
        out.append(
            {
                "feature_name": (
                    f"predicted_{name}" if name in health_set else name
                ),
                "shap_value": float(shap_values[int(idx)]),
                "feature_value": float(feature_values[int(idx)]),
                "is_predicted_health": name in health_set,
            }
        )
    return out


class ShapExplainerManager:
    """Loads and serves TreeExplainers for every exported model."""

    def __init__(self, model_repo_root: Path):
        self.model_repo_root = Path(model_repo_root)
        self._models: Dict[str, _ModelEntry] = {}

    def num_loaded(self) -> int:
        return len(self._models)

    def loaded_models(self) -> List[str]:
        return sorted(self._models.keys())

    def load_all(self) -> None:
        """Walk the model repository and load every model that has a metadata.json."""
        if not self.model_repo_root.exists():
            logger.warning(
                f"Model repository {self.model_repo_root} does not exist; "
                "SHAP explainer manager will be empty"
            )
            return
        for model_dir in sorted(self.model_repo_root.iterdir()):
            if not model_dir.is_dir():
                continue
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                self._load_one(model_dir)
            except Exception as e:
                logger.warning(f"Failed to load explainer for {model_dir.name}: {e}")

    def _load_one(self, model_dir: Path) -> None:
        import shap

        with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        name = meta["model_name"]

        explainer_path_json = model_dir / "model_explainer.json"
        explainer_path_pkl = model_dir / "model_explainer.pkl"
        if explainer_path_json.exists():
            model = _load_xgboost_model(explainer_path_json)
        elif explainer_path_pkl.exists():
            model = _load_pickle_model(explainer_path_pkl)
        else:
            logger.warning(
                f"No explainer artifact for {name}; expected model_explainer.json or .pkl"
            )
            return

        # Optional background data (K-means summary)
        bg_filename = meta.get("background_data_path")
        background = None
        if bg_filename:
            bg_path = model_dir / bg_filename
            if bg_path.exists():
                try:
                    background = np.load(bg_path)
                except Exception as e:
                    logger.warning(f"Failed to load background for {name}: {e}")

        # TreeExplainer with feature_perturbation='tree_path_dependent' is the
        # default for tree models and does not require background data. Pass
        # background only for compatibility with shap's interventional mode.
        try:
            explainer = shap.TreeExplainer(
                model,
                data=background if background is not None else None,
                feature_perturbation=(
                    "interventional" if background is not None else "tree_path_dependent"
                ),
            )
        except Exception:
            explainer = shap.TreeExplainer(model)

        self._models[name] = _ModelEntry(
            model=model,
            explainer=explainer,
            feature_columns=list(meta["feature_columns"]),
            raw_feature_columns=list(meta.get("raw_feature_columns") or []),
            health_input_columns=list(meta.get("health_input_columns") or []),
            class_names=meta.get("class_names"),
            n_classes=meta.get("n_classes"),
            is_classifier=meta.get("model_purpose") == "classifier",
        )
        logger.info(f"  loaded explainer for {name} (n_features={len(meta['feature_columns'])})")

    # ---------- explanation methods ----------

    def _entry(self, model_name: str) -> _ModelEntry:
        if model_name not in self._models:
            raise KeyError(f"No explainer loaded for {model_name}")
        return self._models[model_name]

    def explain_health(
        self,
        equipment_type: str,
        health_column: str,
        raw_features: np.ndarray,
        top_n: int = 10,
    ) -> ShapResult:
        """SHAP for one health regressor over raw features."""
        entry = self._entry(f"{equipment_type}_{health_column}")
        return self._explain_regressor(entry, raw_features, top_n)

    def explain_classifier(
        self,
        equipment_type: str,
        classifier_input: np.ndarray,
        top_n: int = 10,
        class_index: Optional[int] = None,
    ) -> ShapResult:
        """SHAP for the classifier.

        `classifier_input` must already be in the model's training column order,
        with predicted health values placed at their original positions (the
        layout produced by api.routers.inference._build_classifier_input).

        If `class_index` is None, uses the predicted class.
        """
        entry = self._entry(f"{equipment_type}_classifier")
        return self._explain_classifier(entry, classifier_input, top_n, class_index)

    def explain_full_assessment(
        self,
        equipment_type: str,
        health_features: np.ndarray,
        classifier_input: np.ndarray,
        health_columns: List[str],
        top_n: int = 10,
    ) -> Dict[str, object]:
        """Two-level explanation: classifier + per-health-component SHAP.

        `health_features` is the vector in the health regressors' training
        column order; `classifier_input` is the classifier's training-order
        vector with predicted health interleaved at the right positions.
        """
        classifier_explanation = self.explain_classifier(
            equipment_type=equipment_type,
            classifier_input=classifier_input,
            top_n=top_n,
        )
        health_explanations = {
            col: self.explain_health(equipment_type, col, health_features, top_n=top_n)
            for col in health_columns
        }
        return {
            "classifier_explanation": classifier_explanation,
            "health_explanations": health_explanations,
        }

    def explain_batch(
        self,
        model_name: str,
        features_batch: np.ndarray,
    ) -> ShapSummary:
        """Batch SHAP for summary-style visualization (mean |SHAP| per feature)."""
        entry = self._entry(model_name)
        explainer = entry.explainer
        sv = explainer.shap_values(features_batch)
        sv_matrix = self._normalize_shap_matrix(sv, entry)
        mean_abs = np.abs(sv_matrix).mean(axis=0)
        return ShapSummary(
            feature_names=list(entry.feature_columns),
            mean_abs_shap=[float(x) for x in mean_abs],
            shap_values_matrix=sv_matrix.astype(np.float32),
        )

    # ---------- internals ----------

    def _normalize_shap_matrix(
        self, shap_values, entry: _ModelEntry
    ) -> np.ndarray:
        """Reduce shap_values to shape (n_samples, n_features).

        TreeExplainer for classifiers returns a list (per-class) of (n_samples,
        n_features) arrays; we average |values| across classes for the summary.
        """
        if isinstance(shap_values, list):
            stacked = np.stack([np.abs(s) for s in shap_values], axis=0)
            return stacked.mean(axis=0)
        if shap_values.ndim == 3:
            # (n_samples, n_features, n_classes) - shap >= 0.42 returns this
            return np.abs(shap_values).mean(axis=2)
        return shap_values

    def _explain_regressor(
        self,
        entry: _ModelEntry,
        features: np.ndarray,
        top_n: int,
    ) -> ShapResult:
        explainer = entry.explainer
        sv = explainer.shap_values(features)
        # For regressors sv has shape (n_samples, n_features)
        if isinstance(sv, list):
            sv = sv[0]
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        shap_row = sv[0]
        try:
            base_value = float(np.asarray(explainer.expected_value).ravel()[0])
        except Exception:
            base_value = 0.0
        predicted_value = float(base_value + shap_row.sum())
        top = _build_top_contributors(
            feature_names=entry.feature_columns,
            shap_values=shap_row,
            feature_values=features[0],
            health_input_columns=entry.health_input_columns,
            top_n=top_n,
        )
        return ShapResult(
            feature_names=list(entry.feature_columns),
            shap_values=shap_row.astype(np.float32),
            base_value=base_value,
            predicted_value=predicted_value,
            top_contributors=top,
            health_input_columns=list(entry.health_input_columns),
        )

    def _explain_classifier(
        self,
        entry: _ModelEntry,
        features: np.ndarray,
        top_n: int,
        class_index: Optional[int],
    ) -> ShapResult:
        model = entry.model
        explainer = entry.explainer

        # Predicted class for selection
        try:
            pred_proba = model.predict_proba(features)
            predicted_idx = int(np.argmax(pred_proba[0]))
        except Exception:
            pred_proba = None
            predicted_idx = 0
        chosen_idx = predicted_idx if class_index is None else int(class_index)

        sv = explainer.shap_values(features)
        # Normalize sv -> shape (n_classes, n_samples, n_features)
        if isinstance(sv, list):
            sv_class = sv[chosen_idx]  # (n_samples, n_features)
        elif sv.ndim == 3:
            # shap >= 0.42 returns (n_samples, n_features, n_classes)
            sv_class = sv[..., chosen_idx]
        else:
            sv_class = sv

        if sv_class.ndim == 1:
            sv_class = sv_class.reshape(1, -1)
        shap_row = sv_class[0]

        # base_value per-class
        try:
            ev = np.asarray(explainer.expected_value).ravel()
            base_value = float(ev[chosen_idx]) if ev.size > 1 else float(ev[0])
        except Exception:
            base_value = 0.0

        if pred_proba is not None:
            predicted_value = float(pred_proba[0, chosen_idx])
        else:
            predicted_value = float(base_value + shap_row.sum())

        top = _build_top_contributors(
            feature_names=entry.feature_columns,
            shap_values=shap_row,
            feature_values=features[0],
            health_input_columns=entry.health_input_columns,
            top_n=top_n,
        )
        return ShapResult(
            feature_names=list(entry.feature_columns),
            shap_values=shap_row.astype(np.float32),
            base_value=base_value,
            predicted_value=predicted_value,
            top_contributors=top,
            health_input_columns=list(entry.health_input_columns),
        )
