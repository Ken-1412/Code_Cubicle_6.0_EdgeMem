"""Smoke test for benchmark script and metrics schema."""

import json
from pathlib import Path

import pytest

from scripts.benchmark import run_benchmarks


def test_benchmark_smoke_produces_valid_schema_and_100_percent_convergence(tmp_path):
    """Run a fast version of the benchmark suite and assert key invariants."""
    output_json = tmp_path / "results.json"
    output_md = tmp_path / "RESULTS.md"

    results = run_benchmarks(
        json_out=output_json,
        md_out=output_md,
        quick=True,
    )

    # 1. Output files exist
    assert output_json.exists()
    assert output_md.exists()

    # 2. Schema has all required top-level sections
    assert "hardware" in results
    assert "retrieval_quality" in results
    assert "latency" in results
    assert "footprint" in results
    assert "scale" in results
    assert "offline_proof" in results
    assert "sync_correctness" in results
    assert "baseline_comparison" in results

    # 3. Hardware metadata is populated
    hw = results["hardware"]
    assert hw["cpu_model"]
    assert hw["core_count"] > 0
    assert hw["total_ram_gb"] > 0
    assert hw["os"]
    assert hw["python_version"]

    # 4. Retrieval quality includes hit@1, hit@3, MRR for dense, sparse, hybrid
    rq = results["retrieval_quality"]
    for mode in ("dense", "sparse", "hybrid"):
        assert mode in rq
        assert "overall" in rq[mode]
        assert "hit@1" in rq[mode]["overall"]
        assert "hit@3" in rq[mode]["overall"]
        assert "mrr" in rq[mode]["overall"]

    # Hybrid overall MRR should be >= dense and >= sparse
    assert rq["hybrid"]["overall"]["mrr"] >= rq["dense"]["overall"]["mrr"]
    assert rq["hybrid"]["overall"]["mrr"] >= rq["sparse"]["overall"]["mrr"]

    # 5. Offline proof verifies 0 outbound network calls
    assert results["offline_proof"]["network_calls_attempted"] == 0
    assert results["offline_proof"]["hf_hub_offline"] == "1"
    assert results["offline_proof"]["verified"] is True

    # 6. Sync correctness: 100% convergence, 0 lost edits, 0 duplicates
    sync = results["sync_correctness"]
    assert sync["convergence_rate"] == 100.0
    assert sync["lost_edits"] == 0
    assert sync["duplicate_count"] == 0
    assert sync["bandwidth_saved_pct"] > 0

    # 7. Baseline comparison: naive whole-record LWW loses edits
    baseline = results["baseline_comparison"]
    assert baseline["naive_lww_lost_edits"] > 0
    assert baseline["per_field_lww_lost_edits"] == 0

    # 8. Check that results.json matches the returned dict
    on_disk_json = json.loads(output_json.read_text(encoding="utf-8"))
    assert on_disk_json["sync_correctness"]["convergence_rate"] == 100.0
