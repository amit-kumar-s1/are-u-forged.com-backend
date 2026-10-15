
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.guard import GuardMiddleware, SecurityHeadersMiddleware


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

log = logging.getLogger("aryouforged")
settings = get_settings()


# --------------------------------------------------
# Production security configuration
# --------------------------------------------------

if settings.malware_scan_mode == "off":
    log.warning(
        "Malware scanning is disabled. Uploaded files are not scanned."
    )

if settings.rate_limit_backend == "memory":
    log.warning(
        "Rate limiting uses in-memory storage. Limits may not be "
        "shared across workers or instances."
    )


# Only allow explicitly configured frontend origins.
# Wildcard origins are intentionally excluded.
origins = list(
    dict.fromkeys(
        origin
        for item in settings.cors_origins.split(",")
        if (origin := item.strip()) and origin != "*"
    )
)

if not origins:
    log.warning(
        "No permitted CORS origins are configured. "
        "Browser requests from the frontend may be blocked."
    )


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Are You Forged API",
    version="0.2.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url=None,
    openapi_url=(
        "/openapi.json" if settings.docs_enabled else None
    ),
    debug=False,
)

app.include_router(router)


# --------------------------------------------------
# Middleware
# --------------------------------------------------

# Middleware registration order matters.
# CORS should remain the outermost user-added middleware
# so browser clients can read relevant error responses.

app.add_middleware(GuardMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get(
    "/health",
    include_in_schema=False,
)
async def health():
    # Do not expose model status, environment variables,
    # server paths, credentials, or internal configuration.
    return {"status": "ok"}


# --------------------------------------------------
# HTTP exception handler
# --------------------------------------------------

@app.exception_handler(HTTPException)
async def http_error_handler(
    request: Request,
    exc: HTTPException,
):
    status = exc.status_code

    if status == 429:
        content = {
            "code": "rate_limited",
            "message": "Too many requests. Please try again later.",
        }

    elif status == 413:
        content = {
            "code": "file_too_large",
            "message": "The uploaded file exceeds the allowed size.",
        }

    elif status == 415:
        content = {
            "code": "unsupported_media_type",
            "message": "This file type is not supported.",
        }

    elif status == 404:
        content = {
            "code": "not_found",
            "message": "The requested resource was not found.",
        }

    elif status in (400, 422):
        content = {
            "code": "invalid_request",
            "message": "The request could not be processed.",
        }

    elif status in (401, 403):
        content = {
            "code": "access_denied",
            "message": "The request is not authorized.",
        }

    else:
        content = {
            "code": "request_failed",
            "message": "The request could not be completed.",
        }

    # Only forward a validated Retry-After header for rate limits.
    headers = {}

    if status == 429 and exc.headers:
        retry_after = exc.headers.get("Retry-After", "")
        if retry_after.isdigit():
            headers["Retry-After"] = retry_after

    return JSONResponse(
        status_code=status,
        content=content,
        headers=headers,
    )


# --------------------------------------------------
# Request validation errors
# --------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    # Do not return submitted values or detailed validation
    # internals to public clients.
    return JSONResponse(
        status_code=422,
        content={
            "code": "invalid_request",
            "message": "Invalid request payload.",
        },
    )


# --------------------------------------------------
# Unexpected server errors
# --------------------------------------------------

@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    exc: Exception,
):
    # Detailed errors remain in server logs, not API responses.
    # Never log credentials, request bodies, or uploaded files.
    log.error(
        "Unhandled server exception on %s %s (%s)",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=500,
        content={
            "code": "internal_error",
            "message": "Internal server error.",
        },
    )
