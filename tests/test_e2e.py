"""Full offline-first demo scenario, shared with scripts.demo CLI."""

import time

from scripts.demo import run_demo


def test_full_demo_scenario_converges_and_finishes_under_one_minute(tmp_path):
    """Run the same narrated workflow as the CLI against isolated local stores."""
    output = []
    started = time.perf_counter()
    result = run_demo(tmp_path / "scenario", pause=False, emit=output.append)
    elapsed = time.perf_counter() - started

    assert elapsed < 60
    assert result["semantic_expected_id"] in result["semantic_top3"]
    assert result["exact_code_top_id"] == result["exact_code_expected_id"]
    assert result["new_record_id"] in result["device_b_search_ids"]
    assert result["conflict_severity"] == "critical"
    assert result["conflict_text"] == result["expected_conflict_text"]
    assert result["offline_cloud_unchanged"] is True
    assert result["offline_pending_counts"] == {"device_a": 2, "device_b": 2}
    assert result["deleted_everywhere"] is True
    assert result["states_equal"] is True
    assert result["pending_counts"] == {"device_a": 0, "device_b": 0}
    assert result["state_count"] == 121
    assert result["cloud_seq"] > 0

    transcript = "\n".join(output)
    for step in range(1, 7):
        assert f"STEP {step}:" in transcript
    assert "dense=" in transcript and "sparse=" in transcript and "DEMO PASSED" in transcript
