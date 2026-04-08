import os
import subprocess
import mlflow
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from alembic import command
from alembic.config import Config

load_dotenv()


def ensure_alembic_stamped(db_uri: str) -> None:
    """
    This prevents MLflow from trying to re-apply migrations that were already
    applied when MLflow created the tables directly.
    """
    engine = create_engine(db_uri)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT version_num FROM alembic_version")).fetchall()
            if not rows:
                print("alembic_version is empty — stamping DB to current migration head...")
                migrations_dir = os.path.join(
                    os.path.dirname(mlflow.__file__),
                    "store", "db_migrations"
                )
                cfg = Config()
                cfg.set_main_option("script_location", migrations_dir)
                cfg.set_main_option("sqlalchemy.url", db_uri)
                command.stamp(cfg, "heads")
                print("Stamp complete.")
    except Exception:
        pass 
    finally:
        engine.dispose()


db_uri = os.getenv("MLFLOW_TRACKING_URI")
ensure_alembic_stamped(db_uri)

cmd = [
    "mlflow",
    "server",
    "--backend-store-uri", os.getenv("MLFLOW_TRACKING_URI"),
    "--default-artifact-root", os.getenv("MLFLOW_ARTIFACT_ROOT"),
    "--host", os.getenv("MLFLOW_HOST"),
    "--port", os.getenv("MLFLOW_PORT"),
]

print("Starting MLflow server...")
subprocess.run(cmd)
