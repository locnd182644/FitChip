"""Orchestrator API tests — skipped unless the [server] extra is installed."""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fitchip.orchestrator.app import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_targets_and_backends(client):
    targets = client.get("/v1/targets").json()
    assert any(t["id"] == "esp32s3" for t in targets)
    backends = client.get("/v1/backends").json()
    assert any(b["id"] == "tflm" for b in backends)


def test_inspect_endpoint(client, tiny_tflite):
    resp = client.post(
        "/v1/inspect",
        files={"model": ("tiny.tflite", tiny_tflite.read_bytes())},
        data={"target": "esp32s3"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"]["op_counts"] == {"ADD": 1}
    assert body["selection"]["candidates"][0]["backend"] == "tflm"


def test_compile_endpoint_returns_artifact(client, tiny_tflite):
    resp = client.post(
        "/v1/compile",
        files={"model": ("tiny.tflite", tiny_tflite.read_bytes())},
        data={"target": "esp32s3"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "done"

    artifact = client.get(body["artifact_url"])
    assert artifact.status_code == 200
    assert artifact.content[:2] == b"PK"  # a ZIP file


def test_compile_unknown_target_is_422(client, tiny_tflite):
    resp = client.post(
        "/v1/compile",
        files={"model": ("tiny.tflite", tiny_tflite.read_bytes())},
        data={"target": "nope"},
    )
    assert resp.status_code == 422
