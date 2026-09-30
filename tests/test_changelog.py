import json

import pytest

from edgemem.changelog import ChangeLog


@pytest.fixture
def changelog(tmp_path):
    instance = ChangeLog(tmp_path / "device" / "changes.sqlite3")
    try:
        yield instance
    finally:
        instance.close()


def append(changelog, point_id, op, fields, ts, device_id, field_ts=None):
    return changelog.append(
        point_id=point_id,
        op=op,
        fields=fields,
        ts=ts,
        device_id=device_id,
        field_ts=field_ts,
    )


def test_schema_has_requested_delta_columns(changelog):
    columns = {
        row[1]
        for row in changelog._connection.execute("PRAGMA table_info(changes)").fetchall()
    }
    assert columns == {"seq", "point_id", "op", "fields_json", "ts", "device_id", "synced"}


def test_append_and_pending_return_ordered_stamped_deltas(changelog):
    first = append(
        changelog,
        "point-1",
        "upsert",
        {"text": "pump vibration", "deleted": False},
        10.0,
        "device-a",
        {"text": [9.0, "device-a"], "deleted": [10.0, "device-a"]},
    )
    second = append(changelog, "point-2", "delete", {"deleted": True}, 11.0, "device-b")

    assert first.seq == 1 and second.seq == 2
    assert first.fields == {"text": "pump vibration", "deleted": False}
    assert first.field_ts["text"] == [9.0, "device-a"]
    assert first.delta["fields"]["text"] == {
        "value": "pump vibration",
        "stamp": [9.0, "device-a"],
    }
    assert [change.seq for change in changelog.pending()] == [1, 2]
    assert changelog.pending_count() == 2
    raw = changelog._connection.execute(
        "SELECT fields_json FROM changes WHERE seq = 1"
    ).fetchone()[0]
    assert json.loads(raw)["text"]["stamp"] == [9.0, "device-a"]


def test_mark_synced_up_to_sequence_is_idempotent(changelog):
    append(changelog, "p1", "upsert", {"text": "one"}, 1.0, "a")
    append(changelog, "p2", "upsert", {"text": "two"}, 2.0, "a")
    append(changelog, "p3", "upsert", {"text": "three"}, 3.0, "a")

    assert changelog.mark_synced(2) == 2
    assert changelog.mark_synced(2) == 0
    assert [item.seq for item in changelog.pending()] == [3]
    assert changelog.pending_count() == 1


def test_legacy_sequence_ack_and_snapshot_append_remain_compatible(changelog):
    timestamp = 5.0
    record = {
        "id": "legacy-1",
        "text": "bearing warm",
        "equipment": "pump-7",
        "severity": "medium",
        "error_code": None,
        "device_id": "device-a",
        "updated_at": timestamp,
        "field_ts": {
            field: [timestamp, "device-a"]
            for field in ("text", "equipment", "severity", "error_code", "deleted")
        },
        "deleted": False,
    }
    change = changelog.append(record)

    assert change.record == record
    assert changelog.mark_synced([change.seq]) == 1
    assert changelog.pending_count() == 0


def test_compact_ten_changes_to_one_latest_field_delta(changelog):
    fields = {
        "text": "initial",
        "equipment": "pump-1",
        "severity": "low",
        "error_code": None,
        "deleted": False,
    }
    stamps = {field: [1.0, "device-a"] for field in fields}
    append(changelog, "p", "upsert", fields, 1.0, "device-a", stamps)
    edits = [
        ("text", "text-2", 2.0, "device-a"),
        ("severity", "medium", 3.0, "device-b"),
        ("error_code", "E-44", 4.0, "device-a"),
        ("equipment", "pump-2", 5.0, "device-c"),
        ("text", "text-6", 6.0, "device-b"),
        ("severity", "high", 7.0, "device-a"),
        ("error_code", "E-99", 8.0, "device-c"),
        ("severity", "critical", 9.0, "device-b"),
        ("text", "final text", 10.0, "device-c"),
    ]
    for field, value, ts, device_id in edits:
        append(
            changelog,
            "p",
            "update",
            {field: value},
            ts,
            device_id,
            {field: [ts, device_id]},
        )

    compacted = changelog.compact(changelog.pending())

    assert len(compacted) == 1
    delta = compacted[0]
    assert delta["point_id"] == "p"
    assert delta["op"] == "upsert"
    values = {field: item["value"] for field, item in delta["fields"].items()}
    assert values == {
        "text": "final text",
        "equipment": "pump-2",
        "severity": "critical",
        "error_code": "E-99",
        "deleted": False,
    }
    assert delta["fields"]["text"]["stamp"] == [10.0, "device-c"]
    assert delta["fields"]["severity"]["stamp"] == [9.0, "device-b"]


def test_compaction_rejects_delete_older_than_a_content_edit(changelog):
    append(
        changelog,
        "p",
        "upsert",
        {"text": "base", "deleted": False},
        1.0,
        "device-a",
        {"text": [1.0, "device-a"], "deleted": [1.0, "device-a"]},
    )
    append(changelog, "p", "update", {"text": "later edit"}, 8.0, "device-a")
    append(
        changelog,
        "p",
        "delete",
        {"deleted": True},
        5.0,
        "device-b",
        {"deleted": [5.0, "device-b"]},
    )

    result = changelog.compact(changelog.pending())[0]
    assert result["fields"]["deleted"]["value"] is False
    assert result["fields"]["text"]["value"] == "later edit"
    assert result["op"] == "upsert"


def test_invalid_operations_values_and_ack_sequences_are_rejected(changelog):
    with pytest.raises(ValueError, match="op must be"):
        append(changelog, "p", "insert", {"text": "x"}, 1.0, "device-a")
    with pytest.raises(ValueError, match="point_id"):
        append(changelog, "", "upsert", {"text": "x"}, 1.0, "device-a")
    with pytest.raises(ValueError, match="sequence"):
        changelog.mark_synced(-1)
    with pytest.raises(ValueError, match="timestamp"):
        append(changelog, "p", "update", {"text": "x"}, float("nan"), "device-a")
    assert changelog.pending_count() == 0


def test_file_backed_changes_persist_after_reopen(tmp_path):
    path = tmp_path / "device" / "changelog.sqlite3"
    first = ChangeLog(path)
    change = append(first, "p", "upsert", {"text": "persistent"}, 4.0, "device-a")
    first.close()

    reopened = ChangeLog(path)
    try:
        restored = reopened.pending()
        assert len(restored) == 1
        assert restored[0].seq == change.seq
        assert restored[0].fields == {"text": "persistent"}
        assert reopened.pending_count() == 1
    finally:
        reopened.close()


def test_legacy_snapshot_table_migrates_to_delta_schema(tmp_path):
    """Older project databases keep event order and payload through migration."""
    import sqlite3

    path = tmp_path / "legacy" / "changes.sqlite3"
    path.parent.mkdir(parents=True)
    timestamp = 42.0
    record = {
        "id": "legacy-point",
        "text": "bearing warm",
        "equipment": "pump-7",
        "severity": "medium",
        "error_code": None,
        "device_id": "device-a",
        "updated_at": timestamp,
        "field_ts": {
            field: [timestamp, "device-a"]
            for field in ("text", "equipment", "severity", "error_code", "deleted")
        },
        "deleted": False,
    }
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE changes (
             seq INTEGER PRIMARY KEY AUTOINCREMENT,
             point_id TEXT NOT NULL,
             device_id TEXT NOT NULL,
             updated_at REAL NOT NULL,
             record_json TEXT NOT NULL,
             synced INTEGER NOT NULL DEFAULT 0
           )"""
    )
    connection.execute("CREATE INDEX idx_changes_sync_seq ON changes (synced, seq)")
    connection.execute(
        "INSERT INTO changes (seq, point_id, device_id, updated_at, record_json, synced) "
        "VALUES (7, ?, ?, ?, ?, 0)",
        (record["id"], record["device_id"], timestamp, json.dumps(record)),
    )
    connection.commit()
    connection.close()

    changelog = ChangeLog(path)
    try:
        event = changelog.pending()[0]
        columns = {
            row[1]
            for row in changelog._connection.execute("PRAGMA table_info(changes)").fetchall()
        }
        assert columns == {"seq", "point_id", "op", "fields_json", "ts", "device_id", "synced"}
        assert event.seq == 7
        assert event.op == "upsert"
        assert event.record == record
    finally:
        changelog.close()


def test_complete_sync_persists_cursor_with_acknowledgements(tmp_path):
    """A committed sync clears pending rows and restores its cloud cursor after reopen."""
    path = tmp_path / "sync-state" / "changes.sqlite3"
    first = ChangeLog(path)
    event = first.append("point-sync", "upsert", {"text": "offline note"}, 3.0, "device-a")
    assert first.complete_sync(
        event.seq,
        last_cloud_seq=17,
        last_sync_ts=1234.5,
    ) == 1
    first.close()

    reopened = ChangeLog(path)
    try:
        assert reopened.pending_count() == 0
        assert reopened.get_sync_metadata("last_cloud_seq", 0) == 17
        assert reopened.get_sync_metadata("last_sync_ts") == 1234.5
    finally:
        reopened.close()
