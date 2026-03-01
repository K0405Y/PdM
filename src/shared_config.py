"""
Shared configuration loader.

Provides access to the YAML schema configuration used by both the API
and ingestion layers.
"""
import os
from functools import lru_cache
from typing import Any, Dict
import yaml


@lru_cache
def load_table_config() -> Dict[str, Any]:
    """Load database schema and equipment metadata from table_config.yaml.

    Cached so the file is read only once per process.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "api", "table_config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)