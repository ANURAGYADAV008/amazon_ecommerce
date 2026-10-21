"""PostgreSQL-backed request limiting for expensive API endpoints.

The limiter uses PostgreSQL instead of process memory so that it is shared by
every FastAPI worker and continues to work after a serverless instance starts.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import logging
import os

import psycopg2
from fastapi import HTTPException, Request, Response


logger = logging.getLogger(__name__)

_DEFAULT_DB_URI = "postgresql://user:password@localhost:5434/ecommerce"


def _positive_int_setting(name: str, default: int) -> int:
    """Read a positive integer setting without turning a typo into no limit."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid %s; using %s", name, default)
        return default
    return value if value > 0 else default


def _client_address(request: Request) -> str:
    """Return the client address, optionally trusting a deployment proxy.

    Set TRUST_PROXY_HEADERS=true only when the hosting platform removes client-
    supplied forwarding headers and sets its own (for example, a managed proxy).
    """
    trust_proxy_headers = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
    if trust_proxy_headers:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            return forwarded_for.split(",", maxsplit=1)[0].strip()
        real_ip = request.headers.get("x-real-ip", "")
        if real_ip:
            return real_ip.strip()

    return request.client.host if request.client else "unknown"


def _client_key(request: Request) -> str:
    """Hash the address before storing it in the rate-limit table."""
    address = _client_address(request)
    salt = os.getenv("RATE_LIMIT_KEY_SALT", "")
    return hashlib.sha256(f"{salt}:{address}".encode()).hexdigest()


def _window_start(now: datetime, window_seconds: int) -> datetime:
    timestamp = int(now.timestamp())
    return datetime.fromtimestamp(
        timestamp - (timestamp % window_seconds), tz=UTC
    )


def _ensure_storage(cursor: object) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS api_rate_limits (
            client_key CHAR(64) NOT NULL,
            window_start TIMESTAMPTZ NOT NULL,
            request_count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (client_key, window_start)
        );
        """
    )


def enforce_chat_rate_limit(request: Request, response: Response) -> None:
    """Limit each client before the RAG endpoint can invoke OpenAI.

    Defaults to 10 chat requests per 60 seconds.  Both settings can be changed
    without code changes through CHAT_RATE_LIMIT_REQUESTS and
    CHAT_RATE_LIMIT_WINDOW_SECONDS.
    """
    request_limit = _positive_int_setting("CHAT_RATE_LIMIT_REQUESTS", 10)
    window_seconds = _positive_int_setting("CHAT_RATE_LIMIT_WINDOW_SECONDS", 60)
    now = datetime.now(UTC)
    window_start = _window_start(now, window_seconds)
    seconds_into_window = int(now.timestamp()) % window_seconds
    retry_after = window_seconds - seconds_into_window
    database_url = os.getenv("DATABASE_URL") or os.getenv("DB_URI", _DEFAULT_DB_URI)

    try:
        with psycopg2.connect(database_url, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                _ensure_storage(cursor)
                cursor.execute(
                    """
                    INSERT INTO api_rate_limits (client_key, window_start, request_count)
                    VALUES (%s, %s, 1)
                    ON CONFLICT (client_key, window_start) DO UPDATE
                    SET request_count = api_rate_limits.request_count + 1
                    WHERE api_rate_limits.request_count < %s
                    RETURNING request_count;
                    """,
                    (_client_key(request), window_start, request_limit),
                )
                result = cursor.fetchone()
    except psycopg2.Error:
        # Failing closed is deliberate: a database outage must not expose the
        # OpenAI-backed endpoint to unbounded requests.
        logger.exception("Chat rate limiter could not reach PostgreSQL")
        raise HTTPException(
            status_code=503,
            detail="Chat service is temporarily unavailable. Please try again shortly.",
        )

    if result is None:
        raise HTTPException(
            status_code=429,
            detail="Too many chat requests. Please wait before trying again.",
            headers={"Retry-After": str(retry_after)},
        )

    request_count = result[0]
    response.headers["X-RateLimit-Limit"] = str(request_limit)
    response.headers["X-RateLimit-Remaining"] = str(max(request_limit - request_count, 0))
    response.headers["X-RateLimit-Reset"] = str(int(window_start.timestamp()) + window_seconds)
