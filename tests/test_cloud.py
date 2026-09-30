import asyncio
from copy import deepcopy
from uuid import uuid4

import httpx

from edgemem.cloud import create_cloud_app, merge_records


class ASGIClient:
    """Send synchronous test requests through HTTPX's ASGI transport."""

    def __init__(self, app):
        self.app = app

    def _request(self, method, path, json=None):
        async def send():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app),
                base_url="http://test",
            ) as client:
                return await client.request(method, path, json=json)

        return asyncio.run(send())

    def get(self, path):
        return self._request("GET", path)

    def post(self, path, *, json=None):
        return self._request("POST", path, json=json)


def make_record(record_id=None, *, device="device-a", stamp=10.0):
    """Build a canonical record with per-field last-writer metadata."""
    record_id = record_id or str(uuid4())
    fields = {
        "text": "bearing warm",
        "equipment": "pump-7",
        "severity": "medium",
        "error_code": None,
        "deleted": False,
    }
    return {
        "id": record_id,
        **fields,
        "device_id": device,
        "updated_at": stamp,
        "field_ts": {name: [stamp, device] for name in fields},
    }


def submit(client, sequence, record):
    """Submit one local change event through the legacy cloud API."""
    return client.post("/changes", json={"changes": [{"seq": sequence, "record": record}]})


def test_cloud_accepts_batches_and_returns_global_state():
    """Legacy batch callers can still submit snapshots and fetch the merged record."""
    record = make_record()
    app = create_cloud_app()
    client = ASGIClient(app)
    try:
        response = submit(client, 1, record)
        listed = client.get("/records")
        fetched = client.get(f"/records/{record['id']}")

        assert response.status_code == 200
        assert response.json() == {"accepted": 1, "record_count": 1}
        assert listed.json() == {"records": [record]}
        assert fetched.json() == record
    finally:
        app.state.close()


def test_cloud_merges_fields_and_converges_independent_of_arrival_order():
    """Field-wise LWW preserves independent edits regardless of legacy delivery order."""
    base = make_record(stamp=1.0)
    text_edit = deepcopy(base)
    text_edit.update(text="bearing noise", device_id="device-a", updated_at=2.0)
    text_edit["field_ts"]["text"] = [2.0, "device-a"]
    equipment_edit = deepcopy(base)
    equipment_edit.update(equipment="motor-4", device_id="device-b", updated_at=3.0)
    equipment_edit["field_ts"]["equipment"] = [3.0, "device-b"]

    results = []
    apps = []
    try:
        for first, second in ((text_edit, equipment_edit), (equipment_edit, text_edit)):
            app = create_cloud_app()
            apps.append(app)
            client = ASGIClient(app)
            submit(client, 1, first)
            submit(client, 1, second)
            results.append(client.get(f"/records/{base['id']}").json())

        assert results[0] == results[1]
        assert results[0]["text"] == "bearing noise"
        assert results[0]["equipment"] == "motor-4"
        assert results[0]["field_ts"]["text"] == [2.0, "device-a"]
        assert results[0]["field_ts"]["equipment"] == [3.0, "device-b"]
    finally:
        for app in apps:
            app.state.close()


def test_cloud_uses_device_id_tie_break_and_preserves_newer_tombstone():
    """Equal timestamps use device IDs and older live updates cannot erase a tombstone."""
    base = make_record(stamp=1.0)
    from_a = deepcopy(base)
    from_a.update(text="from A", device_id="device-a", updated_at=5.0)
    from_a["field_ts"]["text"] = [5.0, "device-a"]
    from_b = deepcopy(base)
    from_b.update(text="from B", device_id="device-b", updated_at=5.0)
    from_b["field_ts"]["text"] = [5.0, "device-b"]
    tombstone = deepcopy(base)
    tombstone.update(deleted=True, device_id="device-c", updated_at=6.0)
    tombstone["field_ts"]["deleted"] = [6.0, "device-c"]
    stale_live = deepcopy(base)
    stale_live.update(deleted=False, device_id="device-z", updated_at=4.0)
    stale_live["field_ts"]["deleted"] = [4.0, "device-z"]

    merged = merge_records(merge_records(from_a, from_b), tombstone)
    merged = merge_records(merged, stale_live)

    assert merged["text"] == "from B"
    assert merged["deleted"] is True


def test_merge_records_is_commutative_associative_and_idempotent():
    """The compatibility merge function retains its convergence laws."""
    first = make_record(stamp=1.0)
    second = deepcopy(first)
    second.update(text="later text", device_id="device-b", updated_at=2.0)
    second["field_ts"]["text"] = [2.0, "device-b"]
    third = deepcopy(first)
    third.update(equipment="motor-9", device_id="device-c", updated_at=3.0)
    third["field_ts"]["equipment"] = [3.0, "device-c"]

    assert merge_records(first, second) == merge_records(second, first)
    assert merge_records(merge_records(first, second), third) == merge_records(
        first, merge_records(second, third)
    )
    assert merge_records(first, first) == first


def test_cloud_replay_is_idempotent_and_unknown_record_is_404():
    """Repeated legacy events do not change state and absent IDs return 404."""
    record = make_record()
    app = create_cloud_app()
    client = ASGIClient(app)
    try:
        assert submit(client, 1, record).status_code == 200
        first = client.get("/records").json()
        assert submit(client, 1, record).status_code == 200
        assert client.get("/records").json() == first
        assert client.get(f"/records/{uuid4()}").status_code == 404
    finally:
        app.state.close()


def _record_delta(record, *, op="upsert", fields=None):
    """Encode a canonical point as a full-field stamped delta."""
    selected = fields or {
        field: record[field]
        for field in record
        if field not in {"id", "device_id", "updated_at", "field_ts"}
    }
    return {
        "point_id": record["id"],
        "op": op,
        "fields": {
            field: {"value": value, "stamp": record["field_ts"][field]}
            for field, value in selected.items()
        },
        "ts": record["updated_at"],
        "device_id": record["device_id"],
    }


def test_cloud_delta_sync_is_idempotent_persistent_and_returns_cursor_events(tmp_path):
    """Duplicate requests do not advance cloud sequence, and SQLite survives app restart."""
    db_path = tmp_path / "cloud.sqlite3"
    record = make_record(record_id="persistent-cloud")
    body = {
        "device_id": "device-a",
        "last_cloud_seq": 0,
        "deltas": [_record_delta(record)],
    }
    first_app = create_cloud_app(db_path=db_path)
    first = ASGIClient(first_app).post("/sync", json=body)
    duplicate = ASGIClient(first_app).post("/sync", json=body)
    state_before_restart = ASGIClient(first_app).get("/state").json()
    first_app.state.close()

    restarted = create_cloud_app(db_path=db_path)
    try:
        state_after_restart = ASGIClient(restarted).get("/state").json()
        assert first.status_code == 200
        assert first.json()["new_cloud_seq"] == 1
        assert len(first.json()["deltas_since_last_cloud_seq"]) == 1
        assert duplicate.json()["new_cloud_seq"] == 1
        assert len(duplicate.json()["deltas_since_last_cloud_seq"]) == 1
        assert state_before_restart == state_after_restart
        assert state_after_restart["state"] == {record["id"]: record}
        assert state_after_restart["cloud_seq"] == 1
    finally:
        restarted.state.close()
