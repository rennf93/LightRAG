"""Opt-in inbound request guard (fastapi-guard) for the API server.

Off by default: with ``LIGHTRAG_GUARD_ENABLED`` unset or false nothing here runs
and ``fastapi-guard`` is never imported. When enabled, the guard adds an HTTP
wide per-IP rate limit, IP allow/deny lists and path/query/header attack
screening in front of the application.

It is mounted as pure ASGI (``PureASGISecurityMiddleware``), not as a
``BaseHTTPMiddleware``, so streaming responses (``/query/stream``,
``/api/chat``) pass through unbuffered and a client disconnect does not cancel
in-flight LLM work. The request body is not read unless
``LIGHTRAG_GUARD_SCAN_BODY`` is true: documents, queries and prompts are free
text and are not screened by default.
"""

from __future__ import annotations

from typing import Any

# The guard's own defaults for documentation and static routes, plus /health.
# Excluded paths skip attack screening; IP policy and rate limit still apply.
DEFAULT_EXCLUDE_PATHS = (
    "/health,/docs,/redoc,/openapi.json,/openapi.yaml,/favicon.ico,/static"
)


def _split(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def build_guard_config(args: Any, api_prefix: str = "") -> Any:
    """Translate the parsed ``LIGHTRAG_GUARD_*`` settings into a ``SecurityConfig``.

    Raises ``RuntimeError`` when ``fastapi-guard`` is not installed.
    """
    try:
        from guard import SecurityConfig
        from guard_core.handlers.suspatterns_handler import ALL_DETECTION_CATEGORIES
    except ImportError as exc:
        raise RuntimeError(
            "LIGHTRAG_GUARD_ENABLED=true requires the 'fastapi-guard' package. "
            "Install it with: pip install 'lightrag-hku[api]' "
            "(or: pip install fastapi-guard)"
        ) from exc

    exclude = _split(args.guard_exclude_paths)
    if api_prefix:
        # The guard sees the un-normalized path when a prefix is configured.
        exclude += [api_prefix + path for path in exclude]

    return SecurityConfig(
        # Counters live in this process (like the login limiter); no Redis.
        enable_redis=False,
        enable_rate_limiting=args.guard_rate_limit > 0,
        rate_limit=max(args.guard_rate_limit, 1),
        rate_limit_window=args.guard_rate_limit_window,
        # Identity comes from the TCP peer unless the peer is a trusted proxy.
        trusted_proxies=_split(args.guard_trusted_proxies),
        trusted_proxy_depth=args.guard_trusted_proxy_depth,
        whitelist=_split(args.guard_whitelist) or None,
        blacklist=_split(args.guard_blacklist),
        enable_penetration_detection=args.guard_penetration_detection,
        detection_scan_body=args.guard_scan_body,
        # Browsers send the page address as Referer, and guard-core 4.4.0 flags localhost and LAN addresses in it as ssrf.
        enabled_detection_categories=frozenset(ALL_DETECTION_CATEGORIES) - {"ssrf"},
        passive_mode=args.guard_passive_mode,
        exclude_paths=exclude,
        # CORS stays with the app's CORSMiddleware, which wraps the guard.
        enable_cors=False,
        enable_ip_banning=True,
    )


def install_guard(app: Any, args: Any, api_prefix: str = "") -> None:
    """Add the guard to ``app``. Call before ``CORSMiddleware`` is added so CORS
    ends up outermost and guard rejections (403/429) carry CORS headers."""
    config = build_guard_config(args, api_prefix)
    from guard import PureASGISecurityMiddleware

    app.add_middleware(PureASGISecurityMiddleware, config=config)
