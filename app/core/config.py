"""
Application configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field(default="CustomerIntent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=True, alias="DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    model_version: str = Field(default="v1", alias="MODEL_VERSION")
    default_model: str = Field(default="baseline", alias="DEFAULT_MODEL")
    confidence_threshold: float = Field(default=0.6, alias="CONFIDENCE_THRESHOLD")
    max_text_length: int = Field(default=512, alias="MAX_TEXT_LENGTH")
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="customer-intent-classification",
        alias="MLFLOW_EXPERIMENT_NAME"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", alias="LOG_FILE")
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    configs_dir: Path = base_dir / "configs"
    models_dir: Path = base_dir / "ml" / "saved_models"
    data_dir: Path = base_dir / "data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8",
                    "case_sensitive": False, "populate_by_name": True}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
