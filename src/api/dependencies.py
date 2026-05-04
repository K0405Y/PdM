"""
FastAPI dependency injection.
Provides Database singleton, per-request sessions, MasterData access, the
Triton inference client, the SHAP explainer manager, and per-equipment
FeatureEngineer instances for stateful streaming inference.
"""
import logging
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Generator, Optional, Tuple
from sqlalchemy.orm import Session
from api.config import get_settings
from ingestion.db_setup import Database, MasterData
from src.ml.feature_prep import FeatureEngineer
from src.inference.explainer import ShapExplainerManager
from src.inference.triton_client import TritonInferenceClient

logger = logging.getLogger(__name__)

# Database (existing)

_db: Database | None = None
_master_data: MasterData | None = None


def init_database():
    """Initialize database connection. Called during app startup."""
    global _db, _master_data
    settings = get_settings()
    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL environment variable is not set")
    _db = Database(settings.postgres_url)
    _db.connect()
    _master_data = MasterData(_db)


def shutdown_database():
    """Close database connection. Called during app shutdown."""
    global _db
    if _db:
        _db.close()


def get_db() -> Database:
    """Dependency: returns the Database singleton."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


def get_db_session() -> Generator[Session, None, None]:
    """Dependency: yields a SQLAlchemy session, closes after request."""
    if _db is None:
        raise RuntimeError("Database not initialized")
    session = _db.get_session()
    try:
        yield session
    finally:
        session.close()


def get_master_data() -> MasterData:
    """Dependency: returns the MasterData singleton."""
    if _master_data is None:
        raise RuntimeError("MasterData not initialized")
    return _master_data


# Triton inference client

_triton_client = TritonInferenceClient | None = None

def init_triton() -> None:
    """Construct the Triton gRPC client singleton from env vars.

    Failures are logged but not fatal - the inference endpoints will return
    503 if the client isn't ready, while the rest of the API stays healthy.
    """

    global _triton_client
    url = os.getenv("TRITON_GRPC_URL")
    repo_root_env = os.getenv("TRITON_MODEL_REPOSITORY")
    repo_root = Path(repo_root_env) if repo_root_env else None
    try:
        _triton_client = TritonInferenceClient(url=url, model_repo_root=repo_root)
        if _triton_client.is_healthy():
            logger.info(f"Triton client connected to {url}")
        else:
            logger.warning(f"Triton client created but server at {url} is not ready")
    except Exception as e:
        logger.warning(f"Failed to initialize Triton client: {e}")
        _triton_client = None


def shutdown_triton() -> None:
    global _triton_client
    if _triton_client is not None:
        _triton_client.close()
        _triton_client = None


def get_triton_client():
    """Dependency: returns the Triton client.

    Raises if Triton was never initialized; returns the client even if the
    server is currently unreachable so endpoints can return their own 503.
    """
    if _triton_client is None:
        raise RuntimeError("Triton client not initialized")
    return _triton_client


# SHAP explainer manager

_explainer_manager = ShapExplainerManager | None = None

def init_explainers() -> None:
    """Load XGBoost model objects + SHAP TreeExplainers at startup.

    Reads from the Triton model repository (mounted read-only in the container).
    Missing/corrupt models are logged but don't block API startup.
    """

    global _explainer_manager
    repo_root_env = os.getenv("TRITON_MODEL_REPOSITORY")
    if repo_root_env:
        repo_root = Path(repo_root_env)
    else:
        repo_root = (
            Path(__file__).resolve().parents[2] / "triton" / "model_repository"
        )
    try:
        _explainer_manager = ShapExplainerManager(model_repo_root=repo_root)
        _explainer_manager.load_all()
        logger.info(
            f"SHAP explainer manager loaded {_explainer_manager.num_loaded()} models"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize SHAP explainer manager: {e}")
        _explainer_manager = None


def get_explainer_manager():
    """Dependency: returns the SHAP explainer manager."""
    if _explainer_manager is None:
        raise RuntimeError("SHAP explainer manager not initialized")
    return _explainer_manager


# FeatureEngineer cache (per equipment unit)

class _FeatureEngineerCache:
    """LRU cache keyed by (equipment_type, equipment_id).

    FeatureEngineer instances hold 30-day rolling buffers; one per equipment
    unit lets streaming inference maintain trend/stability features across
    requests. Capacity is bounded so memory does not grow without limit.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._store: "OrderedDict[Tuple[str, int], FeatureEngineer]" = OrderedDict()
        self._lock = Lock()

    def get(self, equipment_type: str, equipment_id: int) -> FeatureEngineer:
        key = (equipment_type, equipment_id)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
            fe = FeatureEngineer(equipment_type=equipment_type)
            self._store[key] = fe
            if len(self._store) > self.max_size:
                self._store.popitem(last=False)
            return fe

    def reset(self, equipment_type: str, equipment_id: int) -> None:
        with self._lock:
            self._store.pop((equipment_type, equipment_id), None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


_fe_cache: _FeatureEngineerCache | None = None


def init_feature_engineers(max_size: int = 200) -> None:
    global _fe_cache
    _fe_cache = _FeatureEngineerCache(max_size=max_size)


def get_feature_engineer_cache() -> _FeatureEngineerCache:
    if _fe_cache is None:
        raise RuntimeError("FeatureEngineer cache not initialized")
    return _fe_cache
