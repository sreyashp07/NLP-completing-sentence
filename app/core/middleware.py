"""
FastAPI middleware configuration.

Includes:
- Request timing middleware
- Request ID middleware for tracing
"""
import time
import uuid
from fastapi import Request
from loguru import logger


async def timing_middleware(request: Request, call_next):
    """Log request timing for every API call."""
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        f"REQUEST | id={request_id} | "
        f"method={request.method} | path={request.url.path}"
    )

    response = await call_next(request)

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"RESPONSE | id={request_id} | "
        f"status={response.status_code} | time={elapsed_ms:.1f}ms"
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"

    return response
