"""Public API checks using synthetic inputs and isolated model registries."""

import asyncio
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def api(monkeypatch):
    deployment = Path(__file__).resolve().parents[2] / "deployment"
    monkeypatch.syspath_prepend(str(deployment))
    return importlib.import_module("api")


def request(app, path, *, body=None):
    """Call the ASGI app without private artifacts or an HTTP client dependency."""
    messages = []

    async def receive():
        return {"type": "http.request", "body": (body or "").encode(), "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST" if body is not None else "GET", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "server": ("test", 80), "client": ("test", 123), "root_path": "",
    }
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, json.loads(data)


@pytest.fixture
def request_api():
    return request
