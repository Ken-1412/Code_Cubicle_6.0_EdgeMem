import asyncio
from copy import deepcopy
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.responses import JSONResponse
from qdrant_client import models

from edgemem.changelog import ChangeLog
from edgemem.cloud import create_cloud_app
from edgemem.store import LocalStore
from edgemem.sync import SyncClient, SyncDaemon, sync_once


def run(coroutine):
    """Run one async operation in a short-lived test event loop."""
    return asyncio.run(coroutine)


def fake_dense(texts):
    return [[1.0, 0.0, 0.0] for _ in texts]


def fake_sparse(texts):
    return [models.SparseVector(indices=[7], values=[1.0]) for _ in texts]


def make_record(record_id="11111111-1111-4111-8111-111111111111", *, device="seed", stamp=1.0, text="Pump vibration observed."):
    fields = {
        "text": text,
        "equipment": "hydraulic pump",
        "severity": "medium",
        "error_code": None,
        "deleted": False,
    }
    return {
        "id": record_id or str(uuid4()),
        **fields,
        "device_id": device,
        "updated_at": stamp,
        "field_ts": {field: [stamp, device] for field in fields},
    }


def open_store(path, device_id):
    return LocalStore(
        path=path,
        collection_name="sync_test",
        dense_dimensions=3,
        dense_embedder=fake_dense,
        sparse_embedder=fake_sparse,
        device_id=device_id,
    )


def open_sync_client(store, changelog, transport, device_id):
    return SyncClient(
        changelog=changelog,
        store=store,
        cloud_url="http://cloud",
        transport=transport,
        device_id=device_id,
    )


def edit_field(store, changelog, point_id, field, value, *, ts, device_id):
    current = store.get(point_id)
    updated = deepcopy(current)
    updated[field] = value
    updated["updated_at"] = float(ts)
    updated["device_id"] = device_id
    updated["field_ts"][field] = [float(ts), device_id]
    if field == "text":
        assert store.upsert([updated]) == 1
    else:
        patch = {
            field: value,
            "updated_at": float(ts),
            "device_id": device_id,
            "field_ts": updated["field_ts"],
        }
        assert store.update_payload(point_id, patch) is not None
    changelog.append(
        point_id,
        "update",
        {field: value},
        float(ts),
        device_id,
        {field: [float(ts), device_id]},
    )


async def cloud_request(app, method, path, body=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://cloud",
    ) as client:
        response = await client.request(method, path, json=body)
        return response


def establish_shared_baseline(tmp_path, cloud_app, transport):
    """Seed two identical stores and sync a full upsert so cloud has the base point."""
    record = make_record()
    store_a = open_store(tmp_path / "device-a-qdrant", "device-a")
    store_b = open_store(tmp_path / "device-b-qdrant", "device-b")
    log_a = ChangeLog(tmp_path / "device-a.sqlite3")
    log_b = ChangeLog(tmp_path / "device-b.sqlite3")
    store_a.upsert([record])
    store_b.upsert([record])
    log_a.append(record)
    client_a = open_sync_client(store_a, log_a, transport, "device-a")
    client_b = open_sync_client(store_b, log_b, transport, "device-b")
    run(client_a.sync_once())
    run(client_b.sync_once())
    assert store_a.get(record["id"]) == store_b.get(record["id"])
    return record, store_a, store_b, log_a, log_b, client_a, client_b


def close_device(*components):
    for component in components:
        component.close()


@pytest.fixture
def cloud(tmp_path):
    """Create a persistent cloud database for one test."""
    app = create_cloud_app(db_path=tmp_path / "cloud.sqlite3")
    try:
        yield app
    finally:
        app.state.close()


def test_offline_sync_does_nothing_and_preserves_pending_changes(tmp_path):
    """Module-level sync_once returns offline without making a request or changing state."""
    store = open_store(tmp_path / "offline-qdrant", "device-a")
    changelog = ChangeLog(tmp_path / "offline.sqlite3")
    record = make_record()
    store.upsert([record])
    changelog.append(record)
    rejected = SimpleNamespace(requests=0)

    async def fail_if_called(scope, receive, send):
        if scope["type"] == "http":
            rejected.requests += 1
            await JSONResponse(status_code=500)(scope, receive, send)

    device_state = SimpleNamespace(
        store=store,
        changelog=changelog,
        device_id="device-a",
        online=False,
    )
    try:
        report = run(
            sync_once(
                device_state,
                cloud_url="http://cloud",
                transport=httpx.ASGITransport(app=fail_if_called),
            )
        )
        assert report.skipped
        assert report.status == "offline"
        assert changelog.pending_count() == 1
        assert rejected.requests == 0
        assert device_state.last_cloud_seq == 0
    finally:
        close_device(store, changelog)


def test_two_devices_converge_independent_offline_field_edits(tmp_path, cloud):
    """A severity edit and B text edit survive alternating delta syncs."""
    transport = httpx.ASGITransport(app=cloud)
    record, store_a, store_b, log_a, log_b, client_a, client_b = establish_shared_baseline(
        tmp_path, cloud, transport
    )
    try:
        edit_field(store_a, log_a, record["id"], "severity", "critical", ts=20, device_id="device-a")
        edit_field(store_b, log_b, record["id"], "text", "Pump seal was replaced after vibration.", ts=21, device_id="device-b")

        run(client_a.sync_once())
        run(client_b.sync_once())
        run(client_a.sync_once())

        state_response = run(cloud_request(cloud, "GET", "/state"))
        cloud_record = state_response.json()["state"][record["id"]]
        record_a = store_a.get(record["id"])
        record_b = store_b.get(record["id"])
        assert record_a == record_b == cloud_record
        assert record_a["severity"] == "critical"
        assert record_a["text"] == "Pump seal was replaced after vibration."
        assert log_a.pending_count() == log_b.pending_count() == 0
        assert client_a.last_cloud_seq == client_b.last_cloud_seq == state_response.json()["cloud_seq"]
    finally:
        close_device(store_a, store_b, log_a, log_b)


def test_same_field_conflict_converges_by_timestamp_then_device_id(tmp_path, cloud):
    """Concurrent equal-time severity edits select the lexicographically greater device."""
    transport = httpx.ASGITransport(app=cloud)
    record, store_a, store_b, log_a, log_b, client_a, client_b = establish_shared_baseline(
        tmp_path, cloud, transport
    )
    try:
        edit_field(store_a, log_a, record["id"], "severity", "high", ts=30, device_id="device-a")
        edit_field(store_b, log_b, record["id"], "severity", "critical", ts=30, device_id="device-b")
        run(client_a.sync_once())
        run(client_b.sync_once())
        run(client_a.sync_once())
        cloud_record = run(cloud_request(cloud, "GET", "/state")).json()["state"][record["id"]]
        assert store_a.get(record["id"]) == store_b.get(record["id"]) == cloud_record
        assert cloud_record["severity"] == "critical"
        assert cloud_record["field_ts"]["severity"] == [30.0, "device-b"]
        assert log_a.pending_count() == log_b.pending_count() == 0
    finally:
        close_device(store_a, store_b, log_a, log_b)


def test_tombstone_propagates_and_new_record_is_searchable_on_other_device(tmp_path, cloud):
    """Deletes hide remote points and full upserts make newly logged points searchable."""
    transport = httpx.ASGITransport(app=cloud)
    record, store_a, store_b, log_a, log_b, client_a, client_b = establish_shared_baseline(
        tmp_path, cloud, transport
    )
    try:
        delete_ts = 40.0
        store_a.update_payload(
            record["id"],
            {
                "deleted": True,
                "updated_at": delete_ts,
                "device_id": "device-a",
                "field_ts": {**record["field_ts"], "deleted": [delete_ts, "device-a"]},
            },
        )
        log_a.append(
            record["id"],
            "delete",
            {"deleted": True},
            delete_ts,
            "device-a",
            {"deleted": [delete_ts, "device-a"]},
        )
        run(client_a.sync_once())
        run(client_b.sync_once())
        assert store_b.get(record["id"])["deleted"] is True
        assert record["id"] not in [hit.id for hit in store_b.search("hydraulic pump vibration")]

        new_record = make_record(
            "22222222-2222-4222-8222-222222222222",
            device="device-a",
            stamp=50,
            text="New pump seal leak repaired during night shift.",
        )
        store_a.upsert([new_record])
        log_a.append(new_record)
        run(client_a.sync_once())
        run(client_b.sync_once())
        assert store_b.get(new_record["id"]) == new_record
        hits = store_b.search("pump seal leak repaired", limit=5)
        assert new_record["id"] in [hit.id for hit in hits]
        assert log_a.pending_count() == log_b.pending_count() == 0
    finally:
        close_device(store_a, store_b, log_a, log_b)


def test_cloud_500_then_retry_preserves_pending_and_does_not_duplicate(tmp_path):
    """A transient cloud error leaves local work pending; the retry commits once."""
    cloud_app = create_cloud_app(db_path=tmp_path / "retry-cloud.sqlite3")

    class FailFirstSync:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.calls = 0

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope.get("path") == "/sync":
                self.calls += 1
                if self.calls == 1:
                    await JSONResponse(status_code=500, content={"detail": "try again"})(
                        scope, receive, send
                    )
                    return
            await self.wrapped(scope, receive, send)

    store = open_store(tmp_path / "retry-qdrant", "device-a")
    changelog = ChangeLog(tmp_path / "retry.sqlite3")
    record = make_record()
    store.upsert([record])
    changelog.append(record)
    sync_client = open_sync_client(
        store,
        changelog,
        httpx.ASGITransport(app=FailFirstSync(cloud_app)),
        "device-a",
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            run(sync_client.sync_once())
        assert changelog.pending_count() == 1
        assert sync_client.last_cloud_seq == 0

        report = run(sync_client.sync_once())
        response = run(cloud_request(cloud_app, "GET", "/state")).json()
        assert report.acknowledged_changes == 1
        assert changelog.pending_count() == 0
        assert response["cloud_seq"] == 1
        assert response["state"][record["id"]] == record
    finally:
        close_device(store, changelog)
        cloud_app.state.close()


def test_device_api_sync_now_and_status_routes(tmp_path):
    """The API reports pending/offline state and clears it after a successful sync."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
            category=Warning,
        )
        from fastapi.testclient import TestClient

    from edgemem.api import create_app

    cloud_app = create_cloud_app(db_path=tmp_path / "api-cloud.sqlite3")
    app = create_app(
        db_path=tmp_path / "api-device",
        device_id="device-api",
        cloud_url="http://cloud",
        sync_transport=httpx.ASGITransport(app=cloud_app),
    )
    try:
        with TestClient(app) as client:
            logged = client.post(
                "/log",
                json={
                    "text": "A new compressor seal leak was repaired.",
                    "equipment": "compressor",
                    "severity": "high",
                },
            )
            assert logged.status_code == 201
            before = client.get("/sync-status").json()
            assert before == {
                "pending": 1,
                "last_sync_ts": None,
                "last_cloud_seq": 0,
                "online": False,
            }
            offline = client.post("/sync-now")
            assert offline.status_code == 200
            assert offline.json()["status"] == "offline"
            assert client.get("/sync-status").json()["pending"] == 1

            assert client.post("/network", json={"online": True}).status_code == 200
            synced = client.post("/sync-now")
            assert synced.status_code == 200
            assert synced.json()["status"] == "synced"
            assert synced.json()["pending"] == 0
            assert synced.json()["last_cloud_seq"] == 1
            assert synced.json()["last_sync_ts"] is not None
    finally:
        cloud_app.state.close()


def test_sync_daemon_calls_only_when_online_and_stops_cleanly():
    """The optional daemon checks connectivity and its thread exits on stop."""
    called = Event()

    class SpyClient:
        last_cloud_seq = 0
        last_sync_ts = None

        async def sync_once(self, online=True):
            if online:
                called.set()

    state = SimpleNamespace(online=False, sync_client=SpyClient(), device_id="device-a")
    daemon = SyncDaemon(interval_s=0.01)
    daemon.start(state)
    try:
        assert not called.wait(0.04)
        state.online = True
        assert called.wait(1.0)
    finally:
        daemon.stop(timeout=1.0)
    assert not daemon.is_running
