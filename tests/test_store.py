import json
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import models

from edgemem.store import LocalStore

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "seed_logs.jsonl"
EVAL_PATH = ROOT / "data" / "eval_queries.json"
SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def fake_dense(texts):
    return [[1.0, 0.0, 0.0] for _ in texts]


def fake_sparse(texts):
    return [models.SparseVector(indices=[7, 12], values=[0.8, 0.2]) for _ in texts]


@pytest.fixture
def store(tmp_path):
    """Create an isolated Qdrant local collection with deterministic query embeddings."""
    instance = LocalStore(
        path=tmp_path / "qdrant",
        collection_name="test_logs",
        dense_dimensions=3,
        dense_embedder=fake_dense,
        sparse_embedder=fake_sparse,
    )
    try:
        yield instance
    finally:
        instance.close()


def make_record(text, *, deleted=False, equipment="pump-7", severity="medium"):
    """Build a canonical maintenance point payload for storage tests."""
    now = 1_700_000_000.0
    return {
        "id": str(uuid4()),
        "text": text,
        "equipment": equipment,
        "severity": severity,
        "error_code": None,
        "device_id": "device-a",
        "updated_at": now,
        "field_ts": {
            name: [now, "device-a"]
            for name in ("text", "equipment", "severity", "error_code", "deleted")
        },
        "deleted": deleted,
    }


def test_upsert_many_and_hybrid_search_expose_channel_scores(store):
    """Stored points are found through both named vectors with score details."""
    record = make_record("bearing vibration observed")
    assert store.upsert_many(
        [record],
        [[1.0, 0.0, 0.0]],
        [{"indices": [7, 12], "values": [0.8, 0.2]}],
    ) == 1

    results = store.search(
        query_dense=[1.0, 0.0, 0.0],
        query_sparse={"indices": [7, 12], "values": [0.8, 0.2]},
        limit=5,
    )

    assert len(results) == 1
    assert results[0].id == record["id"]
    assert results[0]["id"] == record["id"]
    assert results[0].payload["text"] == "bearing vibration observed"
    assert results[0]["dense_score"] is not None
    assert results[0]["sparse_score"] is not None
    assert results[0]["score"] == results[0].rrf_score
    assert results[0]["dense_rank"] == 1
    assert results[0]["sparse_rank"] == 1


def test_rrf_combines_channel_ranks_and_keeps_score_breakdown(store):
    """RRF results expose both individual channel scores and their ranks."""
    dense_match = make_record("dense match")
    sparse_match = make_record("sparse match")
    store.upsert_many(
        [dense_match, sparse_match],
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]],
        [
            {"indices": [1], "values": [1.0]},
            {"indices": [2], "values": [1.0]},
        ],
    )

    results = store.search(
        query_dense=[1.0, 0.0, 0.0],
        query_sparse={"indices": [2], "values": [1.0]},
        limit=2,
    )

    assert results[0].id == sparse_match["id"]
    assert results[0].dense_score is not None and results[0].sparse_score is not None
    assert results[1].dense_score is not None and results[1].sparse_score is None
    assert results[0].rrf_score > results[1].rrf_score > 0
    assert results[0].dense_rank == 2
    assert results[0].sparse_rank == 1
    assert results[1].dense_rank == 1
    assert results[1].sparse_rank is None


def test_search_text_modes_and_legacy_vector_arguments(store):
    """Text queries support all three modes while prior vector calls remain valid."""
    record = make_record("bearing vibration observed")
    store.upsert_many(
        [record],
        [[1.0, 0.0, 0.0]],
        [{"indices": [7, 12], "values": [0.8, 0.2]}],
    )

    dense = store.search("bearing noise", limit=1, mode="dense")
    sparse = store.search("bearing noise", limit=1, mode="sparse")
    hybrid = store.search("bearing noise", limit=1, mode="hybrid")
    legacy = store.search([1.0, 0.0, 0.0], {"indices": [7, 12], "values": [0.8, 0.2]})

    assert dense[0]["id"] == sparse[0]["id"] == hybrid[0]["id"] == record["id"]
    assert dense[0]["sparse_score"] is None
    assert sparse[0]["dense_score"] is None
    assert legacy[0]["id"] == record["id"]
    with pytest.raises(ValueError, match="mode"):
        store.search("bearing noise", mode="unknown")


def test_upsert_fills_missing_canonical_metadata_and_get_roundtrips(store):
    """Raw seed-schema records receive canonical fields and can be retrieved by ID."""
    raw = {
        "id": str(uuid4()),
        "text": "seal leak after warm startup",
        "equipment": "hydraulic pump",
        "severity": "high",
        "error_code": None,
    }

    assert store.upsert([raw]) == 1
    found = store.get(raw["id"])

    assert found is not None
    assert {key: found[key] for key in raw} == raw
    assert found["deleted"] is False
    assert found["device_id"]
    assert isinstance(found["updated_at"], (int, float))
    assert set(found["field_ts"]) >= {"text", "equipment", "severity", "error_code", "deleted"}
    assert all(stamp[1] == found["device_id"] for stamp in found["field_ts"].values())
    assert found == next(point for point in store.all_points() if point["id"] == raw["id"])
    assert store.count() == 1
    assert store.get(str(uuid4())) is None


def test_filters_are_applied_and_deleted_points_are_always_excluded(store):
    """Equipment and severity filters constrain every result mode."""
    records = [
        make_record("compressor pressure loss", equipment="compressor", severity="high"),
        make_record("compressor valve fault", equipment="compressor", severity="critical"),
        make_record("pump pressure loss", equipment="hydraulic pump", severity="critical"),
        make_record("old compressor incident", equipment="compressor", severity="critical", deleted=True),
    ]
    store.upsert_many(
        records,
        [[1.0, 0.0, 0.0]] * len(records),
        [{"indices": [7], "values": [1.0]}] * len(records),
    )

    for mode in ("hybrid", "dense", "sparse"):
        results = store.search(
            "compressor pressure problem",
            limit=10,
            filters={"equipment": "compressor", "severity_gte": "high"},
            mode=mode,
        )
        assert {item["id"] for item in results} == {records[0]["id"], records[1]["id"]}
        assert all(item["payload"]["equipment"] == "compressor" for item in results)
        assert all(SEVERITY_RANK[item["payload"]["severity"]] >= 2 for item in results)
        assert all(item["payload"]["deleted"] is False for item in results)


def test_error_code_filter_matches_null_payloads(store):
    """Nullable error-code filters use Qdrant's null-field condition."""
    no_code = make_record("routine inspection", equipment="cooling fan")
    coded = {**make_record("diagnostic alarm", equipment="cooling fan"), "error_code": "FAN-1"}
    store.upsert_many(
        [no_code, coded],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [{"indices": [7], "values": [1.0]}, {"indices": [7], "values": [1.0]}],
    )

    results = store.search("fan observation", filters={"error_code": None})

    assert [result["id"] for result in results] == [no_code["id"]]


def test_soft_delete_keeps_tombstone_but_hides_it_from_search(store):
    """Soft deletion updates payload state without removing the stored point."""
    record = make_record("live maintenance log")
    store.upsert_many(
        [record],
        [[1.0, 0.0, 0.0]],
        [{"indices": [9], "values": [1.0]}],
    )

    assert store.soft_delete(record["id"]) is True
    assert store.soft_delete(record["id"]) is True
    assert store.count() == 1
    assert store.get(record["id"])["deleted"] is True
    assert store.search("live maintenance log", limit=5) == []
    assert store.soft_delete(str(uuid4())) is False


def test_reupsert_updates_same_point_without_changing_count(store):
    """Upserting an existing ID replaces its payload rather than duplicating it."""
    record = make_record("initial bearing noise")
    store.upsert([record])
    updated = {**record, "text": "bearing noise after replacing the coupling"}

    assert store.upsert([updated]) == 1
    assert store.count() == 1
    assert store.get(record["id"])["text"] == updated["text"]
    assert len(store.all_points()) == 1


def test_collection_schema_uses_named_bm25_and_reports_local_quantization_limits(store, caplog):
    """The collection has the requested vector names and reports unsupported INT8 use."""
    info = store.client.get_collection(store.collection_name)
    assert set(info.config.params.vectors) == {"dense"}
    assert set(info.config.params.sparse_vectors) == {"bm25"}
    assert info.config.params.vectors["dense"].quantization_config is None
    assert store.quantization_enabled is False
    assert store.payload_index_fields == set()
    setup_logs = "\n".join(item.getMessage() for item in caplog.get_records("setup"))
    assert "quantization" in setup_logs.lower()
    assert "local" in setup_logs.lower()
    assert "skipping qdrant payload-index creation" in setup_logs.lower()
    assert "indexes have no effect" in setup_logs.lower()


def test_local_store_rejects_nonpositive_dense_dimension(tmp_path):
    """A zero vector size is rejected instead of replaced with the default."""
    with pytest.raises(ValueError, match="dense_dimensions must be positive"):
        LocalStore(path=tmp_path / "invalid", dense_dimensions=0)


def test_local_store_persists_points_after_reopen(tmp_path):
    """Qdrant local mode retains points after closing and reopening its path."""
    path = tmp_path / "persistent-qdrant"
    record = make_record("persisted observation")
    first = LocalStore(path=path, collection_name="persisted", dense_dimensions=3)
    first.upsert_many(
        [record],
        [[1.0, 0.0, 0.0]],
        [{"indices": [5], "values": [1.0]}],
    )
    first.close()

    second = LocalStore(path=path, collection_name="persisted", dense_dimensions=3)
    try:
        assert second.get(record["id"])["text"] == record["text"]
        assert second.count() == 1
        results = second.search([1.0, 0.0, 0.0], {"indices": [5], "values": [1.0]})
        assert [item.id for item in results] == [record["id"]]
    finally:
        second.close()


@pytest.fixture(scope="module")
def seed_store(tmp_path_factory):
    """Index all 120 generated records with the configured real embedding models."""
    records = [
        json.loads(line)
        for line in SEED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    instance = LocalStore(
        path=tmp_path_factory.mktemp("seed-qdrant"),
        collection="seed_eval",
    )
    assert instance.upsert(records) == 120
    try:
        yield instance, records, cases
    finally:
        instance.close()


def test_seed_eval_thresholds_and_dense_vs_hybrid_exact_code(seed_store, capsys):
    """Semantic targets hit hybrid top three and all exact codes hit top one."""
    instance, _records, cases = seed_store
    metrics = {
        kind: {"count": 0, "hit@1": 0, "hit@3": 0}
        for kind in ("SEMANTIC", "EXACT-CODE", "FILTERED")
    }
    semantic_top3 = 0
    exact_top1 = 0
    dense_failed_hybrid_wins = 0

    for case in cases:
        results = instance.search(
            case["query"],
            limit=3,
            filters=case["filters"],
            mode="hybrid",
        )
        expected = set(case["expected_ids"])
        kind = case["type"]
        metrics[kind]["count"] += 1
        hit1 = bool(results) and results[0]["id"] in expected
        hit3 = any(result["id"] in expected for result in results)
        metrics[kind]["hit@1"] += int(hit1)
        metrics[kind]["hit@3"] += int(hit3)
        if kind == "SEMANTIC":
            semantic_top3 += int(hit3)
        elif kind == "EXACT-CODE":
            exact_top1 += int(hit1)
            dense = instance.search(case["query"], limit=1, mode="dense")
            dense_hit = bool(dense) and dense[0]["id"] in expected
            dense_failed_hybrid_wins += int(not dense_hit and hit1)

    assert semantic_top3 >= 6
    assert exact_top1 == 8
    assert dense_failed_hybrid_wins >= 1

    rows = ["type         hit@1  hit@3  total"]
    for kind, label in (("SEMANTIC", "SEMANTIC"), ("EXACT-CODE", "EXACT-CODE"), ("FILTERED", "FILTERED")):
        metric = metrics[kind]
        rows.append(f"{label:<12} {metric['hit@1']:>2}     {metric['hit@3']:>2}     {metric['count']:>2}")
    with capsys.disabled():
        print("\nHybrid seed-evaluation table (Qdrant local + FastEmbed)")
        print("\n".join(rows))
