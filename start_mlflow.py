import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

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
