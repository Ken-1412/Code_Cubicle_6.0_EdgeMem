import asyncio
import warnings
from uuid import uuid4

import httpx
from qdrant_client.models import SparseVector

from edgemem.api import create_app
from edgemem.changelog import ChangeLog
from edgemem.cloud import create_cloud_app
from edgemem.store import LocalStore


class ASGIClient:
    """Small synchronous wrapper around HTTPX's async ASGI transport."""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, *, json=None, headers=None):
        async def send():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app),
                base_url="http://test",
            ) as client:
                return await client.request(method, path, json=json, headers=headers)

        return asyncio.run(send())

    def get(self, path, *, headers=None):
        return self.request("GET", path, headers=headers)

    def post(self, path, *, json=None):
        return self.request("POST", path, json=json)


def fake_dense(texts):
    return [[1.0] + [0.0] * 383 for _ in texts]


def fake_sparse(texts):
    return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


def test_device_state_returns_payload_list_without_vector_data(tmp_path):
    """Device state is a sorted list of canonical payloads, with no vector blobs."""
    db_path = tmp_path / "device-state"
    store = LocalStore(
        path=db_path,
        device_id="state-device",
        dense_embedder=fake_dense,
        sparse_embedder=fake_sparse,
    )
    changelog = ChangeLog(db_path / "changes.sqlite3")
    point_id = str(uuid4())
    record = {
        "id": point_id,
        "text": "The pump seal is leaking hydraulic fluid.",
        "equipment": "hydraulic pump",
        "severity": "high",
        "error_code": "HYD-209",
        "deleted": False,
        "device_id": "state-device",
        "updated_at": 10.0,
        "field_ts": {
            field: [10.0, "state-device"]
            for field in ("text", "equipment", "severity", "error_code", "deleted")
        },
    }
    store.upsert([record])
    app = create_app(
        db_path=db_path,
        device_id="state-device",
        store=store,
        changelog=changelog,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Using `httpx` with `starlette.testclient` is deprecated.*",
                category=Warning,
            )
            from fastapi.testclient import TestClient

        with TestClient(app) as client:
            response = client.get("/state")

        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert response.json() == [record]
        assert not {"vectors", "dense", "bm25"}.intersection(response.json()[0])
    finally:
        store.close()
        changelog.close()


def test_cloud_state_is_persisted_payload_map_and_cors_accessible(tmp_path):
    """Cloud state returns canonical payloads and permits a file-origin-style request."""
    app = create_cloud_app(db_path=tmp_path / "cloud-state.sqlite3")
    client = ASGIClient(app)
    record_id = str(uuid4())
    record_fields = {
        "text": "The conveyor belt walks sideways after a roller seized.",
        "equipment": "conveyor belt",
        "severity": "high",
        "error_code": "E-4471",
        "deleted": False,
    }
    delta = {
        "point_id": record_id,
        "op": "upsert",
        "fields": {
            field: {"value": value, "stamp": [20.0, "device-a"]}
            for field, value in record_fields.items()
        },
        "ts": 20.0,
        "device_id": "device-a",
    }
    try:
        sync = client.post(
            "/sync",
            json={"device_id": "device-a", "last_cloud_seq": 0, "deltas": [delta]},
        )
        state = client.get("/state")
        preflight = client.request(
            "OPTIONS",
            "/state",
            headers={
                "Origin": "null",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert sync.status_code == 200
        assert state.json()["cloud_seq"] == 1
        assert state.json()["state"][record_id]["text"] == record_fields["text"]
        assert not {"vectors", "dense", "bm25"}.intersection(
            state.json()["state"][record_id]
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "*"
    finally:
        app.state.close()
