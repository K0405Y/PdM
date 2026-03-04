"""
API configuration via environment variables and YAML schema config.
"""
import sys
import os
from functools import lru_cache
from typing import Any, Dict, List
import yaml
from pydantic_settings import BaseSettings

# Ensure src/ and project root are importable
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [_project_root, os.path.join(_project_root, 'src')]:
    if p not in sys.path:
        sys.path.insert(0, p)

class Settings(BaseSettings):
    postgres_url: str = ""
    weather_api_key: str = ""
    weather_api_provider: str = "weatherapi"
    allowed_origins: str = "*"
    db_schemas_dir: str = os.path.join(_project_root, "db schemas")
    ngrok_enabled: bool = False
    ngrok_auth_token: str = ""
    ngrok_domain: str = ""

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_file = os.path.join(_project_root, ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()

@lru_cache
def load_table_config() -> Dict[str, Any]:
    """Load database schema and equipment metadata from table_config.yaml."""
    config_path = os.path.join(_project_root, "table_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
