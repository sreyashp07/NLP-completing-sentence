"""
Structured logging setup using Loguru.
Provides consistent, searchable logs across all modules.
"""
import sys
from pathlib import Path

from loguru import logger

from app.core.config import get_settings


def setup_logging() -> None:
    """
    Configure Loguru with console and file sinks.
    Call once at application startup.
    """
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler — colored, human-readable
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — structured, rotated
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,  # Thread-safe async logging
    )

    logger.info(f"Logging initialized | env={settings.app_env} | level={settings.log_level}")


# Export logger for use across the app
__all__ = ["logger", "setup_logging"]
