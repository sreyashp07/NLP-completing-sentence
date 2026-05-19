"""Health check endpoints."""
from fastapi import APIRouter, Request
from app.schemas.prediction import HealthResponse
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Returns API health status and model readiness."""
    service = getattr(request.app.state, "model_service", None)
    model_loaded = service.is_ready if service else False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=model_loaded,
        environment=settings.app_env,
    )
