"""
Application configuration management using Pydantic Settings.
Loads from environment variables and .env file.
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central settings class. All config comes from environment or .env file.
    Using lru_cache ensures we only instantiate once (singleton pattern).
    """

    # Application
    app_name: str = Field(default="CustomerIntent", env="APP_NAME")
    app_env: str = Field(default="development", env="APP_ENV")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")

    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")

    # ML
    model_version: str = Field(default="v1", env="MODEL_VERSION")
    default_model: str = Field(default="baseline", env="DEFAULT_MODEL")
    confidence_threshold: float = Field(default=0.6, env="CONFIDENCE_THRESHOLD")
    max_text_length: int = Field(default=512, env="MAX_TEXT_LENGTH")

    # MLflow
    mlflow_tracking_uri: str = Field(default="./mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(
        default="customer-intent-classification",
        env="MLFLOW_EXPERIMENT_NAME"
    )

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    configs_dir: Path = base_dir / "configs"
    models_dir: Path = base_dir / "ml" / "saved_models"
    data_dir: Path = base_dir / "data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use this everywhere instead of instantiating Settings() directly.
    """
    return Settings()
