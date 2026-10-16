"""Opt-in fastapi-guard integration (LIGHTRAG_GUARD_*).

Pins the contract: default off (the package is never imported and no
middleware is added), CORS wraps the guard, client identity honors
X-Forwarded-For only from trusted proxies, bodies are scanned only on opt-in,
and streaming responses and uploads pass through unbuffered.
"""

from __future__ import annotations

import asyncio
import itertools
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

pytestmark = pytest.mark.offline

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "EMBEDDING_MODEL",
    "CORS_ORIGINS",
    "LIGHTRAG_API_PREFIX",
    "AUTH_ACCOUNTS",
    "LIGHTRAG_GUARD_ENABLED",
    "LIGHTRAG_GUARD_RATE_LIMIT",
    "LIGHTRAG_GUARD_RATE_LIMIT_WINDOW",
    "LIGHTRAG_GUARD_TRUSTED_PROXIES",
    "LIGHTRAG_GUARD_TRUSTED_PROXY_DEPTH",
    "LIGHTRAG_GUARD_WHITELIST",
    "LIGHTRAG_GUARD_BLACKLIST",
    "LIGHTRAG_GUARD_PENETRATION_DETECTION",
    "LIGHTRAG_GUARD_SCAN_BODY",
    "LIGHTRAG_GUARD_PASSIVE_MODE",
    "LIGHTRAG_GUARD_EXCLUDE_PATHS",
)

# The guard keeps process-wide handlers (rate limiter, ban list), so every test
# uses its own client IPs to stay independent of test order.
_octet = itertools.count(1)


def _peer() -> str:
    return f"203.0.113.{next(_octet)}"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3:latest")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


def _create_app(monkeypatch, **env):
    """Build the real LightRAG API app with LIGHTRAG_GUARD_* env applied."""
    from lightrag.api.config import initialize_config, parse_args

    for key, value in env.items():
        monkeypatch.setenv(f"LIGHTRAG_GUARD_{key}", str(value))

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)

    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        from lightrag.api.lightrag_server import create_app

        return create_app(args)


def _middleware_names(app) -> list[str]:
    return [m.cls.__name__ for m in app.user_middleware]


class TestDisabledByDefault:
    def test_adds_no_middleware_and_never_imports_guard(self, monkeypatch):
        # sys.modules["guard"] = None makes any `import guard` raise ImportError,
        # so create_app succeeding proves the disabled path never imports it.
        monkeypatch.setitem(sys.modules, "guard", None)

        app = _create_app(monkeypatch)

        assert "PureASGISecurityMiddleware" not in _middleware_names(app)

    def test_unlimited_when_disabled(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "guard", None)
        client = TestClient(_create_app(monkeypatch, RATE_LIMIT=1))

        codes = {client.get("/auth-status").status_code for _ in range(5)}

        assert codes == {200}


class TestEnabledWithoutPackage:
    def test_raises_a_clear_runtime_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "guard", None)

        with pytest.raises(RuntimeError, match="fastapi-guard"):
            _create_app(monkeypatch, ENABLED="true")


class TestEnabledOrderingAndRateLimit:
    def test_guard_sits_inside_cors(self, monkeypatch):
        app = _create_app(monkeypatch, ENABLED="true")

        names = _middleware_names(app)

        # user_middleware is outermost first: CORS must wrap the guard.
        assert names.index("CORSMiddleware") < names.index("PureASGISecurityMiddleware")

    def test_burst_gets_429_that_carries_cors_headers(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "*")
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true", RATE_LIMIT=3),
            client=(_peer(), 50000),
        )
        headers = {"Origin": "https://app.example.com"}

        codes = [client.get("/auth-status", headers=headers) for _ in range(5)]

        assert [r.status_code for r in codes[:3]] == [200, 200, 200]
        assert codes[-1].status_code == 429
        assert codes[-1].headers["access-control-allow-origin"] == "*"


class TestTrustedClientIp:
    @staticmethod
    def _statuses(client, forwarded_for):
        return [
            client.get("/auth-status", headers={"X-Forwarded-For": ip}).status_code
            for ip in forwarded_for
        ]

    def test_spoofed_xff_from_untrusted_peer_is_ignored(self, monkeypatch):
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true", RATE_LIMIT=2),
            client=(_peer(), 50000),
        )

        # Rotating the header must not mint fresh rate-limit buckets.
        codes = self._statuses(client, [f"198.51.100.{n}" for n in range(1, 6)])

        assert codes == [200, 200, 429, 429, 429]

    def test_xff_from_trusted_proxy_is_honored(self, monkeypatch):
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true", RATE_LIMIT=2),
            client=("127.0.0.1", 50000),
        )
        a, b = f"198.51.100.{next(_octet)}", f"198.51.100.{next(_octet)}"

        codes = self._statuses(client, [a, a, a, b])

        # Distinct forwarded clients get distinct buckets; the same one is capped.
        assert codes == [200, 200, 429, 200]

    def test_trusted_proxies_is_configurable(self, monkeypatch):
        peer = _peer()
        client = TestClient(
            _create_app(
                monkeypatch, ENABLED="true", RATE_LIMIT=1, TRUSTED_PROXIES=peer
            ),
            client=(peer, 50000),
        )
        a, b = f"198.51.100.{next(_octet)}", f"198.51.100.{next(_octet)}"

        assert self._statuses(client, [a, b]) == [200, 200]


class TestAttackScreening:
    def test_attack_pattern_in_query_is_blocked(self, monkeypatch):
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true"), client=(_peer(), 50000)
        )

        resp = client.get("/auth-status", params={"q": "1' OR '1'='1' UNION SELECT"})

        assert resp.status_code == 400 or resp.status_code == 403

    @pytest.mark.parametrize(
        "referer",
        [
            "http://localhost:9621/webui/",
            "http://127.0.0.1:9621/webui/",
            "http://192.168.1.20:9621/webui/",
        ],
    )
    def test_localhost_and_lan_referers_are_not_blocked(self, monkeypatch, referer):
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true"), client=(_peer(), 50000)
        )

        resp = client.get("/auth-status", headers={"Referer": referer})

        assert resp.status_code == 200

    def test_attack_in_referer_is_still_blocked(self, monkeypatch):
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true"), client=(_peer(), 50000)
        )

        resp = client.get(
            "/auth-status", headers={"Referer": "<script>alert(1)</script>"}
        )

        assert resp.status_code == 400

    def test_excluded_path_skips_attack_screening(self, monkeypatch):
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true", EXCLUDE_PATHS="/auth-status"),
            client=(_peer(), 50000),
        )

        resp = client.get("/auth-status", params={"q": "1' OR '1'='1' UNION SELECT"})

        assert resp.status_code == 200

    def test_screening_can_be_turned_off(self, monkeypatch):
        client = TestClient(
            _create_app(monkeypatch, ENABLED="true", PENETRATION_DETECTION="false"),
            client=(_peer(), 50000),
        )

        resp = client.get("/auth-status", params={"q": "1' OR '1'='1' UNION SELECT"})

        assert resp.status_code == 200


def _body_app(monkeypatch, **env) -> FastAPI:
    """Small app with a body-echo route behind the real guard wiring."""
    from lightrag.api.config import parse_args
    from lightrag.api.security_guard import install_guard

    for key, value in env.items():
        monkeypatch.setenv(f"LIGHTRAG_GUARD_{key}", str(value))
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    args = parse_args()
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        return {"size": len(await request.body())}

    install_guard(app, args)
    return app


_ATTACK_BODY = b"{\"query\": \"1' OR '1'='1' UNION SELECT password FROM users--\"}"


class TestBodyScanning:
    def test_body_is_not_scanned_by_default(self, monkeypatch):
        client = TestClient(
            _body_app(monkeypatch, ENABLED="true"), client=(_peer(), 50000)
        )

        resp = client.post(
            "/echo", content=_ATTACK_BODY, headers={"content-type": "application/json"}
        )

        assert resp.status_code == 200
        assert resp.json() == {"size": len(_ATTACK_BODY)}

    def test_body_scan_opt_in_blocks_and_still_replays_clean_bodies(self, monkeypatch):
        client = TestClient(
            _body_app(monkeypatch, ENABLED="true", SCAN_BODY="true"),
            client=(_peer(), 50000),
        )
        headers = {"content-type": "application/json"}

        blocked = client.post("/echo", content=_ATTACK_BODY, headers=headers)
        clean = b'{"query": "what is the capital of France?"}'
        ok = client.post("/echo", content=clean, headers=headers)

        assert blocked.status_code in (400, 403)
        assert ok.status_code == 200
        assert ok.json() == {"size": len(clean)}  # the app still sees the body


def _scope(path: str, method: str, headers: list | None = None) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": [(b"host", b"t"), *(headers or [])],
        "client": (_peer(), 50000),
        "server": ("t", 80),
    }


class TestStreamingAndUploads:
    @pytest.mark.asyncio
    async def test_streaming_response_is_not_buffered(self, monkeypatch):
        app = _body_app(monkeypatch, ENABLED="true")
        first_chunk_sent = asyncio.Event()

        @app.get("/stream")
        async def stream():
            async def gen():
                yield b"first\n"
                # Only continues once the first chunk has reached the client, so
                # a middleware that buffered the response would time out here.
                await asyncio.wait_for(first_chunk_sent.wait(), timeout=5)
                yield b"second\n"

            return StreamingResponse(gen(), media_type="text/plain")

        bodies: list[bytes] = []

        async def receive():
            await asyncio.sleep(3600)

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                bodies.append(message["body"])
                first_chunk_sent.set()

        scope = _scope("/stream", "GET")
        await asyncio.wait_for(app(scope, receive, send), timeout=10)

        assert bodies == [b"first\n", b"second\n"]

    @staticmethod
    async def _upload(app, chunks: int):
        """Drive a chunked upload; return (status, size, receive() calls the
        app had seen when its handler started, i.e. what the guard pre-read)."""
        chunk = b"x" * 65536
        calls = {"n": 0}
        seen_at_handler: list[int] = []

        @app.post("/upload")
        async def upload(request: Request):
            seen_at_handler.append(calls["n"])
            return {"size": len(await request.body())}

        async def receive():
            calls["n"] += 1
            if calls["n"] > chunks:
                return {"type": "http.disconnect"}
            return {
                "type": "http.request",
                "body": chunk,
                "more_body": calls["n"] < chunks,
            }

        sent: list[dict] = []

        async def send(message):
            sent.append(message)

        scope = _scope(
            "/upload", "POST", [(b"content-type", b"application/octet-stream")]
        )
        await app(scope, receive, send)

        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        body = b"".join(
            m.get("body", b"") for m in sent if m["type"] == "http.response.body"
        )
        assert body == b'{"size":%d}' % (len(chunk) * chunks)
        return status, seen_at_handler[0]

    @pytest.mark.asyncio
    async def test_upload_is_streamed_to_the_app_not_pre_read(self, monkeypatch):
        app = _body_app(monkeypatch, ENABLED="true")

        # 4 MiB, far beyond the 256 KiB scan window.
        status, pre_read = await self._upload(app, chunks=64)

        assert status == 200
        assert pre_read == 0

    @pytest.mark.asyncio
    async def test_opt_in_body_scan_reads_only_a_bounded_prefix(self, monkeypatch):
        app = _body_app(monkeypatch, ENABLED="true", SCAN_BODY="true")

        status, pre_read = await self._upload(app, chunks=64)

        assert status == 200  # the app still receives the complete body
        assert 0 < pre_read < 64
