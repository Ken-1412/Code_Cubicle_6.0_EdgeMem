import json
from pathlib import Path
from uuid import uuid4

import pytest

from edgemem import ingest
from edgemem.store import LocalStore

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "seed_logs.jsonl"
_TIMESTAMP = 1_700_000_100.0
_FIELD_NAMES = ("text", "equipment", "severity", "error_code", "deleted")


def make_record(text="pump bearing warm"):
    """Build a canonical or seed-schema maintenance record for ingestion tests."""
    return {
        "id": str(uuid4()),
        "text": text,
        "equipment": "pump-7",
        "severity": "high",
        "error_code": "E-42",
        "device_id": "source-device",
        "updated_at": _TIMESTAMP - 1,
        "field_ts": {"text": [_TIMESTAMP - 1, "source-device"]},
        "deleted": False,
    }


def seed_record(text="pump bearing warm"):
    """Return the five-field schema emitted by scripts/make_data.py."""
    return {
        "id": str(uuid4()),
        "text": text,
        "equipment": "hydraulic pump",
        "severity": "high",
        "error_code": None,
    }


class FakeStore:
    """Capture each embedding batch handed to the storage layer."""

    def __init__(self):
        self.call = None
        self.calls = []

    def upsert_many(self, records, dense_vectors, sparse_vectors):
        self.call = (records, dense_vectors, sparse_vectors)
        self.calls.append(self.call)
        return len(records)


def write_jsonl(path, records):
    """Write records as newline-delimited JSON."""
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def fake_dense(texts):
    return [[1.0] + [0.0] * 383 for _ in texts]


def fake_sparse(texts):
    return [{"indices": [3], "values": [0.5]} for _ in texts]


def patch_fake_embeddings(monkeypatch):
    monkeypatch.setattr(ingest, "embed_dense", fake_dense)
    monkeypatch.setattr(ingest, "embed_sparse", fake_sparse)


def test_ingest_jsonl_enriches_canonical_metadata_and_batches_embeddings(monkeypatch, tmp_path):
    """Rows are enriched for the selected device and embedded in one batch."""
    rows = [make_record("bearing warms"), make_record("seal leaks")]
    path = write_jsonl(tmp_path / "logs.jsonl", rows)
    dense_inputs = []
    sparse_inputs = []

    def dense(texts):
        dense_inputs.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def sparse(texts):
        sparse_inputs.extend(texts)
        return [{"indices": [3], "values": [0.5]} for _ in texts]

    monkeypatch.setattr(ingest, "embed_dense", dense)
    monkeypatch.setattr(ingest, "embed_sparse", sparse)
    store = FakeStore()
    progress = []

    assert ingest.ingest_jsonl(
        path,
        store,
        device_id="device-a",
        batch_size=32,
        clock=lambda: _TIMESTAMP,
        progress=progress.append,
    ) == 2

    expected = []
    for row in rows:
        enriched = dict(row)
        enriched.update(
            device_id="device-a",
            updated_at=_TIMESTAMP,
            field_ts={name: [_TIMESTAMP, "device-a"] for name in _FIELD_NAMES},
            deleted=False,
        )
        expected.append(enriched)
    assert dense_inputs == [row["text"] for row in rows]
    assert sparse_inputs == [row["text"] for row in rows]
    assert store.call == (
        expected,
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [{"indices": [3], "values": [0.5]}, {"indices": [3], "values": [0.5]}],
    )
    assert progress == ["ingested 2/2"]


def test_ingest_jsonl_batches_of_32_and_reports_running_progress(monkeypatch, tmp_path):
    """A 70-row file is embedded/upserted as 32, 32, and 6 rows."""
    rows = [seed_record(f"maintenance note {index}") for index in range(70)]
    path = write_jsonl(tmp_path / "many.jsonl", rows)
    monkeypatch.setattr(ingest, "embed_dense", lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(ingest, "embed_sparse", fake_sparse)
    store = FakeStore()
    progress = []

    assert ingest.ingest_jsonl(
        path,
        store,
        device_id="device_a",
        batch_size=32,
        clock=lambda: _TIMESTAMP,
        progress=progress.append,
    ) == 70

    assert [len(call[0]) for call in store.calls] == [32, 32, 6]
    assert progress == ["ingested 32/70", "ingested 64/70", "ingested 70/70"]
    assert all(record["device_id"] == "device_a" for call in store.calls for record in call[0])


def test_ingest_jsonl_reports_invalid_json_line_without_writing(tmp_path):
    """Malformed JSON reports its line number before any storage operation."""
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(make_record()) + "\n{not json}\n", encoding="utf-8")
    store = FakeStore()

    with pytest.raises(ValueError, match=r"broken.jsonl:2: invalid JSON"):
        ingest.ingest_jsonl(path, store)

    assert store.call is None


def test_ingest_jsonl_rejects_missing_source_fields(tmp_path):
    """Rows must include the five source fields before metadata enrichment."""
    row = seed_record()
    del row["severity"]
    path = write_jsonl(tmp_path / "incomplete.jsonl", [row])

    with pytest.raises(ValueError, match="missing required fields: severity"):
        ingest.ingest_jsonl(path, FakeStore())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "not-a-uuid", "id must be a UUID string"),
        ("severity", [], "severity must be one of"),
        ("updated_at", True, "updated_at must be numeric"),
        ("field_ts", {"text": [1.0]}, r"field_ts values must be \[epoch, device_id\]"),
    ],
)
def test_ingest_jsonl_rejects_invalid_source_values(tmp_path, field, value, message):
    """Malformed optional canonical metadata is rejected before being replaced."""
    row = make_record()
    row[field] = value
    path = write_jsonl(tmp_path / "invalid.jsonl", [row])

    with pytest.raises(ValueError, match=message):
        ingest.ingest_jsonl(path, FakeStore())


def test_ingest_jsonl_rejects_embedding_count_mismatch(monkeypatch, tmp_path):
    """Embedding output counts must match parsed rows before storage is touched."""
    path = write_jsonl(tmp_path / "logs.jsonl", [seed_record()])
    monkeypatch.setattr(ingest, "embed_dense", lambda texts: [])
    monkeypatch.setattr(ingest, "embed_sparse", fake_sparse)
    store = FakeStore()

    with pytest.raises(ValueError, match="embedding counts must match"):
        ingest.ingest_jsonl(path, store)

    assert store.call is None


def test_ingest_empty_file_returns_zero_without_embedding(monkeypatch, tmp_path):
    """An empty JSONL file performs no model initialization or storage writes."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    def unexpected_embedding(texts):
        raise AssertionError("empty input must not initialize embedding models")

    monkeypatch.setattr(ingest, "embed_dense", unexpected_embedding)
    monkeypatch.setattr(ingest, "embed_sparse", unexpected_embedding)
    store = FakeStore()

    assert ingest.ingest_jsonl(path, store) == 0
    assert store.call is None


def test_cli_reset_idempotency_and_device_separation(monkeypatch, tmp_path, capsys):
    """CLI reset wipes only its DB directory; reruns are unique and devices remain distinct."""
    patch_fake_embeddings(monkeypatch)
    db_a = tmp_path / "device_a"
    db_a.mkdir()
    marker = db_a / "old-artifact.txt"
    marker.write_text("remove me", encoding="utf-8")

    assert ingest.main(
        ["--db", str(db_a), "--file", str(SEED_PATH), "--device-id", "device_a", "--reset"]
    ) == 0
    output = capsys.readouterr().out
    assert not marker.exists()
    assert all(f"ingested {number}/120" in output for number in (32, 64, 96, 120))
    assert "total time:" in output and "points/sec:" in output

    first = LocalStore(path=db_a, collection="memory")
    try:
        points_a = first.all_points()
        assert first.count() == 120
        assert {point["device_id"] for point in points_a} == {"device_a"}
    finally:
        first.close()

    assert ingest.main(["--db", str(db_a), "--file", str(SEED_PATH)]) == 0
    second = LocalStore(path=db_a, collection="memory")
    try:
        points_a_again = second.all_points()
        assert second.count() == 120
        assert {point["device_id"] for point in points_a_again} == {"device_a"}
    finally:
        second.close()

    db_b = tmp_path / "device_b"
    assert ingest.main(
        ["--db", str(db_b), "--file", str(SEED_PATH), "--device-id", "device_b"]
    ) == 0
    third = LocalStore(path=db_b, collection="memory")
    try:
        points_b = third.all_points()
        assert third.count() == 120
        assert {point["device_id"] for point in points_b} == {"device_b"}
    finally:
        third.close()

    content_a = {point["id"]: (point["text"], point["equipment"], point["severity"], point["error_code"]) for point in points_a_again}
    content_b = {point["id"]: (point["text"], point["equipment"], point["severity"], point["error_code"]) for point in points_b}
    assert content_a == content_b
