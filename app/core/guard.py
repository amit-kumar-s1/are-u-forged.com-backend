
"""Pure-ASGI security middleware for upload limits, rate limiting and headers."""

import json
import re

from starlette.concurrency import run_in_threadpool

from app.core import limits
from app.core.config import get_settings


PATH = re.compile(r"^/api/v1/detect/(image|video|text)$")
OVERHEAD = 256 * 1024  # Allow multipart framing overhead


class BodyTooLarge(Exception):
    """Raised when an upload exceeds its configured byte limit."""


async def reply(send, status, code, message, extra=None):
    body = json.dumps(
        {"code": code, "message": message}
    ).encode("utf-8")

    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
        (b"x-content-type-options", b"nosniff"),
    ]

    for key, value in (extra or {}).items():
        headers.append(
            (key.lower().encode("ascii"), str(value).encode("ascii"))
        )

    await send({
        "type": "http.response.start",
        "status": status,
        "headers": headers,
    })
    await send({
        "type": "http.response.body",
        "body": body,
        "more_body": False,
    })


def body_limit(kind: str) -> int:
    settings = get_settings()

    if kind == "image":
        return settings.max_image_mb * 1048576 + OVERHEAD

    if kind == "video":
        return settings.max_video_mb * 1048576 + OVERHEAD

    if kind == "text":
        return settings.max_text_chars * 4 + 8192

    raise ValueError(f"Unsupported upload type: {kind}")


class GuardMiddleware:
    """Enforce request limits before multipart parsing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        match = (
            PATH.match(scope.get("path", ""))
            if scope["method"] == "POST"
            else None
        )

        if not match:
            return await self.app(scope, receive, send)

        kind = match.group(1)
        limit = body_limit(kind)

        ok, retry = await run_in_threadpool(
            limits.check_rate,
            kind,
            limits.client_ip(scope),
        )

        if not ok:
            return await reply(
                send,
                429,
                "rate_limited",
                "Too many requests. Please wait and try again.",
                {"Retry-After": max(1, int(retry))},
            )

        content_length = dict(scope["headers"]).get(b"content-length")

        if content_length:
            if not content_length.isdigit():
                return await reply(
                    send,
                    400,
                    "invalid_content_length",
                    "Invalid Content-Length header.",
                )

            if int(content_length) > limit:
                return await reply(
                    send,
                    413,
                    "request_too_large",
                    "Upload exceeds the size limit.",
                )

        gate = limits.get_gate("upload")

        if not gate.acquire_nowait():
            return await reply(
                send,
                503,
                "server_busy",
                "Too many uploads in progress. Try again shortly.",
                {"Retry-After": 5},
            )

        total = 0
        exceeded = False
        response_messages = []

        async def limited_receive():
            nonlocal total, exceeded

            message = await receive()

            if message["type"] == "http.request":
                total += len(message.get("body", b""))

                if total > limit:
                    exceeded = True
                    raise BodyTooLarge()

            return message

        async def buffered_send(message):
            # Do not commit the response until body-limit enforcement
            # has finished. This lets us replace a parser error with 413.
            response_messages.append(message)

        try:
            try:
                await self.app(
                    scope,
                    limited_receive,
                    buffered_send,
                )
            except BodyTooLarge:
                # The body exceeded the limit. Discard the application's
                # buffered response and send a single 413 response below.
                pass
            except Exception:
                if exceeded:
                    # The app/parser may raise a different exception after
                    # the body limit has been exceeded.
                    pass
                else:
                    raise

            if exceeded:
                return await reply(
                    send,
                    413,
                    "request_too_large",
                    "Upload exceeds the size limit.",
                )

            # Forward the original response only if the request was within
            # the configured limit.
            for message in response_messages:
                await send(message)

        finally:
            gate.release()


class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        settings = get_settings()
        path = scope.get("path", "")

        is_docs_path = (
            path == "/docs"
            or path.startswith("/docs/")
            or path == "/redoc"
            or path.startswith("/redoc/")
            or path == "/openapi.json"
        )

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() not in (b"server", b"x-powered-by")
                ]

                existing = {key.lower() for key, _ in headers}

                additions = {
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": (
                        b"camera=(), microphone=(), geolocation=(), "
                        b"payment=(), usb=()"
                    ),
                }

                if not is_docs_path:
                    additions[b"content-security-policy"] = (
                        b"default-src 'none'; "
                        b"frame-ancestors 'none'; base-uri 'none'"
                    )

                if path.startswith("/api/"):
                    additions[b"cache-control"] = b"no-store"

                if settings.enable_hsts:
                    additions[b"strict-transport-security"] = (
                        b"max-age=31536000; includeSubDomains"
                    )

                headers.extend(
                    (key, value)
                    for key, value in additions.items()
                    if key not in existing
                )

                message = {**message, "headers": headers}

            await send(message)

        await self.app(scope, receive, send_with_headers)
