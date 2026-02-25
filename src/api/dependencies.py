"""
FastAPI dependency injection.
Provides Database singleton, per-request sessions, and MasterData access.
"""
from typing import Generator
from sqlalchemy.orm import Session
from ingestion.db_setup import Database, MasterData
from api.config import get_settings

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