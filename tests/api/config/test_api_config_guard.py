"""Offline tests for the LIGHTRAG_GUARD_* settings parsing (opt-in fastapi-guard)."""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args

pytestmark = pytest.mark.offline

_GUARD_VARS = (
    "LIGHTRAG_GUARD_ENABLED",
    "LIGHTRAG_GUARD_RATE_LIMIT",
    "LIGHTRAG_GUARD_TRUSTED_PROXIES",
    "LIGHTRAG_GUARD_SCAN_BODY",
    "LIGHTRAG_GUARD_PENETRATION_DETECTION",
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    for var in _GUARD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")


def test_guard_defaults_are_off_and_conservative():
    args = parse_args()

    assert args.guard_enabled is False
    assert args.guard_trusted_proxies == "127.0.0.1,::1"
    assert args.guard_scan_body is False
    assert args.guard_penetration_detection is True
    assert args.guard_rate_limit == 600


def test_guard_settings_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_GUARD_ENABLED", "true")
    monkeypatch.setenv("LIGHTRAG_GUARD_RATE_LIMIT", "25")
    monkeypatch.setenv("LIGHTRAG_GUARD_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("LIGHTRAG_GUARD_SCAN_BODY", "true")

    args = parse_args()

    assert args.guard_enabled is True
    assert args.guard_rate_limit == 25
    assert args.guard_trusted_proxies == "10.0.0.0/8"
    assert args.guard_scan_body is True
