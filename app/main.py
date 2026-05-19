"""
FastAPI application entrypoint.

Initializes the app, sets up CORS, mounts routers,
and handles application lifecycle (startup/shutdown).
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.v1.endpoints import predictions, health
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.model_service import ModelService

# Setup logging first
setup_logging()

settings = get_settings()

# Shared model service instance (loaded once at startup)
model_service: ModelService = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Startup: load ML models into memory.
    Shutdown: cleanup resources.
    """
    global model_service
    logger.info("Starting CustomerIntent API...")

    # Load model at startup — not on first request (avoids cold start latency)
    model_service = ModelService()
    model_service.load_model("baseline")

    # Attach to app state for access in route handlers
    app.state.model_service = model_service

    logger.success(f"{settings.app_name} API ready | v{settings.app_version}")
    yield

    # Cleanup on shutdown
    logger.info("Shutting down CustomerIntent API...")


def create_app() -> FastAPI:
    """Application factory pattern."""
    app = FastAPI(
        title="CustomerIntent API",
        description="AI-powered customer support intent classification and smart ticket routing",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow Streamlit frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, prefix=settings.api_prefix, tags=["Health"])
    app.include_router(predictions.router, prefix=settings.api_prefix, tags=["Predictions"])

    return app


app = create_app()
