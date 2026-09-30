import warnings

import pytest

# FastAPI's standard TestClient still works with this pinned httpx release, while
# Starlette emits a deprecation notice recommending its newer httpx2 adapter.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Using `httpx` with `starlette.testclient` is deprecated.*",
        category=Warning,
    )
    from fastapi.testclient import TestClient

from edgemem.api import create_app


@pytest.fixture
def device(tmp_path):
    """Create an isolated local API and close its Qdrant handle after each test."""
    app = create_app(db_path=tmp_path / "device-db", device_id="tablet-3")
    with TestClient(app) as client:
        yield app, client


def log_record(client, text, equipment="hydraulic pump", severity="high", error_code=None):
    body = {"text": text, "equipment": equipment, "severity": severity}
    if error_code is not None:
        body["error_code"] = error_code
    response = client.post("/log", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_log_then_paraphrased_search_returns_new_record(device):
    """A just-logged point is retrievable by a paraphrase with score breakdowns."""
    app, client = device
    record = log_record(
        client,
        "The hydraulic pump vibrates and hydraulic fluid leaks at its shaft seal. "
        "The damaged seal was replaced and the unit passed a pressure test.",
        equipment="hydraulic pump",
        error_code="HYD-209",
    )

    response = client.post(
        "/search",
        json={"query": "shaking and oil dripping", "limit": 3, "mode": "hybrid"},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert record["id"] in [hit["id"] for hit in results[:3]]
    hit = results[0]
    assert hit["payload"]["text"] == record["text"]
    assert hit["dense_score"] is not None
    assert "sparse_score" in hit
    assert hit["fused_score"] > 0
    assert client.get("/health").json() == {
        "device_id": "tablet-3",
        "points": 1,
        "online": False,
    }
    assert app.state.record_change.__name__ == "record_change"


def test_patch_severity_updates_only_that_field_and_does_not_reembed(tmp_path):
    """A metadata-only patch changes severity timestamps and reuses stored vectors."""
    timestamps = iter((1700.0, 1800.0))
    app = create_app(
        db_path=tmp_path / "patch-db",
        device_id="device_patch",
        clock=lambda: next(timestamps),
    )
    with TestClient(app) as client:
        record = log_record(
            client,
            "The motor bearing is warm but still operating normally.",
            equipment="electric motor",
            severity="medium",
        )
        old_timestamps = record["field_ts"]

        def unexpected_embedding(_texts):
            raise AssertionError("severity-only patch must not re-embed")

        app.state.store.dense_embedder = unexpected_embedding
        app.state.store.sparse_embedder = unexpected_embedding
        response = client.patch(f"/points/{record['id']}", json={"severity": "critical"})

        assert response.status_code == 200
        updated = response.json()
        assert updated["severity"] == "critical"
        assert updated["text"] == record["text"]
        assert updated["equipment"] == record["equipment"]
        assert updated["error_code"] == record["error_code"]
        assert updated["updated_at"] == 1800.0
        assert updated["field_ts"]["severity"] == [1800.0, "device_patch"]
        for field in ("text", "equipment", "error_code", "deleted"):
            assert updated["field_ts"][field] == old_timestamps[field]


def test_delete_soft_deletes_and_removes_point_from_search(device):
    """DELETE retains a tombstone but the regular query never returns it."""
    _, client = device
    record = log_record(
        client,
        "The compressor coupling rattles during a loaded run.",
        equipment="compressor",
        severity="high",
    )

    deleted = client.delete(f"/points/{record['id']}")
    search = client.post("/search", json={"query": "compressor coupling rattles"})

    assert deleted.status_code == 200
    assert deleted.json() == {"id": record["id"], "deleted": True}
    assert record["id"] not in [hit["id"] for hit in search.json()["results"]]
    assert client.get("/health").json()["points"] == 1


def test_search_filters_are_applied_inside_store(device):
    """Equipment and minimum severity constraints reach the Qdrant query."""
    _, client = device
    log_record(client, "Bearing noise noted during inspection.", "compressor", "high")
    other = log_record(client, "Bearing noise noted during inspection.", "hydraulic pump", "critical")
    below_threshold = log_record(client, "Bearing noise noted during inspection.", "compressor", "low")

    response = client.post(
        "/search",
        json={
            "query": "bearing inspection noise",
            "limit": 5,
            "filters": {"equipment": "compressor", "severity_gte": "high"},
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert all(hit["payload"]["equipment"] == "compressor" for hit in results)
    assert all(hit["payload"]["severity"] in {"high", "critical"} for hit in results)
    assert other["id"] not in [hit["id"] for hit in results]
    assert below_threshold["id"] not in [hit["id"] for hit in results]


def test_two_app_factories_keep_database_and_device_state_isolated(tmp_path):
    """Different factory paths and IDs do not share points or network flags."""
    app_a = create_app(db_path=tmp_path / "device-a", device_id="device_a")
    app_b = create_app(db_path=tmp_path / "device-b", device_id="device_b")
    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        record = log_record(client_a, "The fan belt squeals after startup.", "cooling fan", "medium")

        assert client_a.get("/health").json()["points"] == 1
        assert client_b.get("/health").json() == {
            "device_id": "device_b",
            "points": 0,
            "online": False,
        }
        assert client_b.post("/search", json={"query": "fan belt squeal"}).json()["results"] == []
        assert record["device_id"] == "device_a"


def test_network_flag_toggles_and_search_still_works_offline(device):
    """Search continues against local Qdrant while simulated connectivity is off."""
    _, client = device
    record = log_record(client, "A valve actuator sticks near the end of travel.", "valve actuator")

    offline = client.post("/network", json={"online": False})
    search = client.post("/search", json={"query": "valve sticks near travel"})
    online = client.post("/network", json={"online": True})

    assert offline.json() == {"device_id": "tablet-3", "online": False}
    assert search.status_code == 200
    assert record["id"] in [hit["id"] for hit in search.json()["results"]]
    assert online.json() == {"device_id": "tablet-3", "online": True}
    assert client.get("/health").json()["online"] is True


def test_one_record_change_hook_receives_log_patch_and_delete(device):
    """All mutation routes invoke the same hook with operation and changed fields."""
    app, client = device
    calls = []
    app.state.record_change = lambda op, point_id, fields: calls.append(
        (op, point_id, dict(fields))
    )
    record = log_record(client, "The gearbox makes a rhythmic clunk.", "gearbox", "medium")
    client.patch(f"/points/{record['id']}", json={"severity": "high"})
    client.delete(f"/points/{record['id']}")

    assert calls == [
        (
            "upsert",
            record["id"],
            {
                "text": record["text"],
                "equipment": "gearbox",
                "severity": "medium",
                "error_code": None,
            },
        ),
        ("update", record["id"], {"severity": "high"}),
        ("delete", record["id"], {"deleted": True}),
    ]


def test_log_patch_delete_each_append_one_pending_change(device):
    """Each stored mutation writes exactly one event to the device SQLite log."""
    app, client = device
    assert app.state.changelog.pending_count() == 0

    record = log_record(client, "The gearbox makes a rhythmic clunk.", "gearbox", "medium")
    assert app.state.changelog.pending_count() == 1
    client.patch(f"/points/{record['id']}", json={"severity": "high"})
    assert app.state.changelog.pending_count() == 2
    client.delete(f"/points/{record['id']}")
    assert app.state.changelog.pending_count() == 3

    changes = app.state.changelog.pending()
    assert [change.op for change in changes] == ["upsert", "update", "delete"]
    assert all(change.point_id == record["id"] for change in changes)
    assert changes[1].fields == {"severity": "high"}
    assert changes[2].fields == {"deleted": True}


def test_cors_allows_local_ui_origin(device):
    """Local UI clients can call the device API cross-origin."""
    _, client = device
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_patch_text_reembeds_and_updates_only_text_timestamp(tmp_path):
    """Changing the searchable text refreshes its timestamp and both vectors."""
    timestamps = iter((2100.0, 2200.0))
    app = create_app(
        db_path=tmp_path / "text-patch-db",
        device_id="device_text",
        clock=lambda: next(timestamps),
    )
    with TestClient(app) as client:
        record = log_record(
            client,
            "The cooling fan makes a faint squeal during startup.",
            equipment="cooling fan",
            severity="low",
        )
        old_timestamps = record["field_ts"]
        store = app.state.store
        dense_original, sparse_original = store.dense_embedder, store.sparse_embedder
        dense_inputs, sparse_inputs = [], []

        def dense_spy(texts):
            batch = list(texts)
            dense_inputs.extend(batch)
            return dense_original(batch)

        def sparse_spy(texts):
            batch = list(texts)
            sparse_inputs.extend(batch)
            return sparse_original(batch)

        store.dense_embedder = dense_spy
        store.sparse_embedder = sparse_spy
        new_text = "The cooling fan bearing squeals loudly and was lubricated."
        response = client.patch(f"/points/{record['id']}", json={"text": new_text})

        assert response.status_code == 200
        updated = response.json()
        assert updated["text"] == new_text
        assert dense_inputs == [new_text]
        assert sparse_inputs == [new_text]
        assert updated["field_ts"]["text"] == [2200.0, "device_text"]
        for field in ("equipment", "severity", "error_code", "deleted"):
            assert updated["field_ts"][field] == old_timestamps[field]


def test_sync_now_and_status_report_offline_then_synced(tmp_path):
    """The new device sync routes surface status and preserve offline changes."""
    import httpx

    from edgemem.cloud import create_cloud_app

    cloud = create_cloud_app(db_path=tmp_path / "api-cloud.sqlite3")
    app = create_app(
        db_path=tmp_path / "sync-device",
        device_id="device_sync_api",
        cloud_url="http://cloud",
        sync_transport=httpx.ASGITransport(app=cloud),
    )
    try:
        with TestClient(app) as client:
            logged = log_record(
                client,
                "Compressor coupling noise was traced to a worn seal.",
                equipment="compressor",
                severity="high",
            )
            assert client.get("/sync-status").json() == {
                "pending": 1,
                "last_sync_ts": None,
                "last_cloud_seq": 0,
                "online": False,
            }
            offline = client.post("/sync-now")
            assert offline.json()["status"] == "offline"
            assert client.get("/sync-status").json()["pending"] == 1

            client.post("/network", json={"online": True})
            synced = client.post("/sync-now")
            status = client.get("/sync-status").json()
            assert synced.status_code == 200
            assert synced.json()["status"] == "synced"
            assert status["pending"] == 0
            assert status["last_cloud_seq"] == 1
            assert status["last_sync_ts"] is not None
            assert status["online"] is True
            assert logged["id"] in cloud.state.repository.get_state()["state"]
    finally:
        cloud.state.close()
