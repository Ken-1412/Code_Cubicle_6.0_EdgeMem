"""Narrated, fully local end-to-end demo for EdgeMem's offline sync workflow.

The CLI and the E2E test both call :func:`run_demo`, which drives the real
FastAPI routes through HTTPX ASGI transports while keeping all databases in a
caller-provided temporary directory. No network ports or external services are
required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These imports follow path bootstrapping for direct script execution.
from edgemem.api import create_app  # noqa: E402
from edgemem.changelog import ChangeLog  # noqa: E402
from edgemem.cloud import create_cloud_app  # noqa: E402
from edgemem.ingest import ingest_jsonl  # noqa: E402
from edgemem.store import LocalStore  # noqa: E402
from edgemem.sync import SyncClient  # noqa: E402

SEED_FILE = PROJECT_ROOT / "data" / "seed_logs.jsonl"
EVAL_FILE = PROJECT_ROOT / "data" / "eval_queries.json"
SEMANTIC_QUERY = "pump shaking and oil leaking"
NEW_OBSERVATION = "replaced worn seal on primary hydraulic pump, vibration stopped"
CONFLICT_TEXT = (
    "Device B found a cracked shaft seal; replacement restored smooth pump operation."
)


def _state_by_id(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert a device's payload-list state response to an ID-keyed mapping."""
    return {str(point["id"]): point for point in points}


async def _json_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    """Issue one ASGI request and include useful response detail on failures."""
    response = await client.request(method, path, json=payload)
    if response.is_error:
        raise RuntimeError(f"{method} {path} failed ({response.status_code}): {response.text}")
    return response.json()


async def _run_demo(
    work_dir: str | Path,
    *,
    pause: bool,
    emit: Callable[[str], None],
    wait_for_enter: Callable[[], str],
) -> dict[str, Any]:
    """Execute the narrated workflow using real device and cloud API routes."""
    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    cloud_app = None
    store_a = store_b = None
    changelog_a = changelog_b = None

    def heading(step: int, text: str) -> None:
        emit(f"\nSTEP {step}: {text}")

    def pause_between_steps() -> None:
        if pause:
            emit("Press Enter to continue to the next step...")
            wait_for_enter()

    try:
        store_a = LocalStore(path=root / "device_a", collection="memory", device_id="device_a")
        store_b = LocalStore(path=root / "device_b", collection="memory", device_id="device_b")
        changelog_a = ChangeLog(root / "device_a" / "changes.sqlite3")
        changelog_b = ChangeLog(root / "device_b" / "changes.sqlite3")
        cloud_app = create_cloud_app(db_path=root / "cloud.sqlite3")

        cloud_transport = httpx.ASGITransport(app=cloud_app)
        sync_a = SyncClient(
            changelog_a,
            store_a,
            cloud_url="http://cloud",
            transport=cloud_transport,
            device_id="device_a",
        )
        sync_b = SyncClient(
            changelog_b,
            store_b,
            cloud_url="http://cloud",
            transport=cloud_transport,
            device_id="device_b",
        )
        app_a = create_app(
            db_path=root / "device_a",
            device_id="device_a",
            store=store_a,
            changelog=changelog_a,
            sync_client=sync_a,
        )
        app_b = create_app(
            db_path=root / "device_b",
            device_id="device_b",
            store=store_b,
            changelog=changelog_b,
            sync_client=sync_b,
        )

        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_a), base_url="http://device-a"
            ) as device_a,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_b), base_url="http://device-b"
            ) as device_b,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=cloud_app), base_url="http://cloud"
            ) as cloud,
        ):
            seed_rows = [
                json.loads(line)
                for line in SEED_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            eval_rows = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
            semantic_eval = next(
                row for row in eval_rows if row.get("type") == "SEMANTIC"
            )
            semantic_expected_id = semantic_eval["expected_ids"][0]
            code_record = next(row for row in seed_rows if row.get("error_code"))
            exact_code = code_record["error_code"]
            exact_code_expected_id = code_record["id"]
            conflict_point_id = semantic_expected_id
            deleted_record = next(
                row for row in seed_rows if row["id"] != conflict_point_id
            )
            deleted_point_id = deleted_record["id"]

            # STEP 1: seed two isolated local stores, establish the shared baseline,
            # then take both devices offline before making scenario edits.
            heading(1, "Boot A, B, and cloud with the seed data; establish a baseline, then go offline")
            count_a = ingest_jsonl(SEED_FILE, store_a, device_id="device_a")
            count_b = ingest_jsonl(SEED_FILE, store_b, device_id="device_b")
            for point in store_a.all_points():
                changelog_a.append(point)
            for point in store_b.all_points():
                changelog_b.append(point)
            assert count_a == count_b == 120
            assert store_a.count() == store_b.count() == 120
            emit(f"Seeded {count_a} records on device A and {count_b} on device B.")

            await _json_request(device_a, "POST", "/network", {"online": True})
            await _json_request(device_b, "POST", "/network", {"online": True})
            bootstrap_a1 = await _json_request(device_a, "POST", "/sync-now", {})
            bootstrap_b = await _json_request(device_b, "POST", "/sync-now", {})
            bootstrap_a2 = await _json_request(device_a, "POST", "/sync-now", {})
            assert all(
                result["status"] == "synced"
                for result in (bootstrap_a1, bootstrap_b, bootstrap_a2)
            )
            await _json_request(device_a, "POST", "/network", {"online": False})
            await _json_request(device_b, "POST", "/network", {"online": False})
            health_a = await _json_request(device_a, "GET", "/health")
            health_b = await _json_request(device_b, "GET", "/health")
            assert health_a["online"] is False and health_b["online"] is False
            initial_a = _state_by_id(await _json_request(device_a, "GET", "/state"))
            initial_b = _state_by_id(await _json_request(device_b, "GET", "/state"))
            initial_cloud = await _json_request(cloud, "GET", "/state")
            assert initial_a == initial_b == initial_cloud["state"]
            assert len(initial_a) == 120
            emit(
                "PASS: all three states agree on 120 points; both devices are offline "
                f"(cloud_seq={initial_cloud['cloud_seq']})."
            )
            pause_between_steps()

            # STEP 2: demonstrate hybrid semantic retrieval and expose channel scores.
            heading(2, f'Semantic hybrid search on A: "{SEMANTIC_QUERY}"')
            semantic_response = await _json_request(
                device_a,
                "POST",
                "/search",
                {"query": SEMANTIC_QUERY, "limit": 5, "mode": "hybrid"},
            )
            semantic_hits = semantic_response["results"]
            semantic_top3 = [hit["id"] for hit in semantic_hits[:3]]
            assert semantic_expected_id in semantic_top3, (
                f"expected {semantic_expected_id} in top 3; got {semantic_top3}"
            )
            for rank, hit in enumerate(semantic_hits[:3], start=1):
                emit(
                    f"  #{rank} {hit['id']} fused={hit['fused_score']:.4f} "
                    f"dense={hit['dense_score']:.4f} sparse={hit['sparse_score']:.4f}"
                )
            emit(f"PASS: expected pump record {semantic_expected_id} is in the top three.")
            pause_between_steps()

            # STEP 3: use a real seed error code and require its record at rank one.
            heading(3, f"Exact-code hybrid search on A: {exact_code}")
            code_response = await _json_request(
                device_a,
                "POST",
                "/search",
                {"query": exact_code, "limit": 5, "mode": "hybrid"},
            )
            code_hits = code_response["results"]
            assert code_hits and code_hits[0]["id"] == exact_code_expected_id, (
                f"{exact_code} expected {exact_code_expected_id} at rank one; "
                f"got {[hit['id'] for hit in code_hits[:3]]}"
            )
            emit(
                f"PASS: {exact_code} ranks {exact_code_expected_id} first "
                f"(fused={code_hits[0]['fused_score']:.4f})."
            )
            pause_between_steps()

            # STEP 4: log and find a new observation locally while A remains offline.
            heading(4, "Log a new observation on offline A and search it immediately")
            observation_response = await device_a.post(
                "/log",
                json={
                    "text": NEW_OBSERVATION,
                    "equipment": "hydraulic pump",
                    "severity": "medium",
                    "error_code": None,
                },
            )
            assert observation_response.status_code == 201, observation_response.text
            new_record = observation_response.json()
            new_search = await _json_request(
                device_a,
                "POST",
                "/search",
                {"query": NEW_OBSERVATION, "limit": 5, "mode": "hybrid"},
            )
            new_search_ids = [hit["id"] for hit in new_search["results"]]
            assert new_record["id"] in new_search_ids
            emit(f"PASS: offline A logged {new_record['id']} and found it immediately.")
            pause_between_steps()

            # STEP 5: make concurrent, different-field edits to P and delete Q on B.
            heading(5, "Edit the same point offline on both devices; delete an unrelated point on B")
            patch_a = await device_a.patch(
                f"/points/{conflict_point_id}", json={"severity": "critical"}
            )
            patch_b = await device_b.patch(
                f"/points/{conflict_point_id}", json={"text": CONFLICT_TEXT}
            )
            delete_b = await device_b.delete(f"/points/{deleted_point_id}")
            assert patch_a.status_code == 200, patch_a.text
            assert patch_a.json()["severity"] == "critical"
            assert patch_b.status_code == 200, patch_b.text
            assert patch_b.json()["text"] == CONFLICT_TEXT
            assert delete_b.status_code == 200 and delete_b.json()["deleted"] is True
            assert (await _json_request(device_a, "GET", "/health"))["online"] is False
            assert (await _json_request(device_b, "GET", "/health"))["online"] is False
            offline_cloud = await _json_request(cloud, "GET", "/state")
            offline_status_a = await _json_request(device_a, "GET", "/sync-status")
            offline_status_b = await _json_request(device_b, "GET", "/sync-status")
            offline_pending_counts = {
                "device_a": offline_status_a["pending"],
                "device_b": offline_status_b["pending"],
            }
            offline_cloud_unchanged = offline_cloud == initial_cloud
            assert offline_cloud_unchanged
            assert offline_pending_counts == {"device_a": 2, "device_b": 2}
            emit(f"PASS: A set P={conflict_point_id} severity=critical.")
            emit("PASS: B changed P's text and tombstoned unrelated Q while both were offline.")
            emit(
                "PASS: cloud is unchanged; local pending counts are "
                f"A={offline_pending_counts['device_a']}, "
                f"B={offline_pending_counts['device_b']}."
            )
            pause_between_steps()

            # STEP 6: replay the intended A, B, A order and assert full convergence.
            heading(6, "Reconnect and sync A, B, A; verify merged fields, tombstone, and convergence")
            await _json_request(device_a, "POST", "/network", {"online": True})
            await _json_request(device_b, "POST", "/network", {"online": True})
            sync_a1 = await _json_request(device_a, "POST", "/sync-now", {})
            sync_b1 = await _json_request(device_b, "POST", "/sync-now", {})
            sync_a2 = await _json_request(device_a, "POST", "/sync-now", {})
            assert all(
                result["status"] == "synced" for result in (sync_a1, sync_b1, sync_a2)
            )

            state_a = _state_by_id(await _json_request(device_a, "GET", "/state"))
            state_b = _state_by_id(await _json_request(device_b, "GET", "/state"))
            cloud_state = await _json_request(cloud, "GET", "/state")
            assert state_a == state_b == cloud_state["state"]
            assert state_a[conflict_point_id]["severity"] == "critical"
            assert state_a[conflict_point_id]["text"] == CONFLICT_TEXT

            assert state_a[deleted_point_id]["deleted"] is True
            assert state_b[deleted_point_id]["deleted"] is True
            assert cloud_state["state"][deleted_point_id]["deleted"] is True
            hidden_a = await _json_request(
                device_a,
                "POST",
                "/search",
                {"query": deleted_record["text"], "limit": 10, "mode": "hybrid"},
            )
            hidden_b = await _json_request(
                device_b,
                "POST",
                "/search",
                {"query": deleted_record["text"], "limit": 10, "mode": "hybrid"},
            )
            assert deleted_point_id not in [hit["id"] for hit in hidden_a["results"]]
            assert deleted_point_id not in [hit["id"] for hit in hidden_b["results"]]

            found_on_b = await _json_request(
                device_b,
                "POST",
                "/search",
                {"query": NEW_OBSERVATION, "limit": 5, "mode": "hybrid"},
            )
            device_b_search_ids = [hit["id"] for hit in found_on_b["results"]]
            assert new_record["id"] in device_b_search_ids

            status_a = await _json_request(device_a, "GET", "/sync-status")
            status_b = await _json_request(device_b, "GET", "/sync-status")
            pending_counts = {
                "device_a": status_a["pending"],
                "device_b": status_b["pending"],
            }
            assert pending_counts == {"device_a": 0, "device_b": 0}
            emit(
                "PASS: P contains A's critical severity and B's text; Q is tombstoned "
                "and hidden from search; the new observation is searchable on B."
            )
            emit(f"PASS: A, B, and cloud match exactly across {len(state_a)} points.")
            emit(
                f"Pending counts: device_a={pending_counts['device_a']}, "
                f"device_b={pending_counts['device_b']}"
            )
            emit("\n========== DEMO PASSED ==========")

            return {
                "semantic_expected_id": semantic_expected_id,
                "semantic_top3": semantic_top3,
                "exact_code": exact_code,
                "exact_code_expected_id": exact_code_expected_id,
                "exact_code_top_id": code_hits[0]["id"],
                "new_record_id": new_record["id"],
                "device_b_search_ids": device_b_search_ids,
                "conflict_severity": state_a[conflict_point_id]["severity"],
                "conflict_text": state_a[conflict_point_id]["text"],
                "expected_conflict_text": CONFLICT_TEXT,
                "offline_cloud_unchanged": offline_cloud_unchanged,
                "offline_pending_counts": offline_pending_counts,
                "deleted_everywhere": all(
                    state[deleted_point_id]["deleted"] is True
                    for state in (state_a, state_b, cloud_state["state"])
                ),
                "states_equal": state_a == state_b == cloud_state["state"],
                "pending_counts": pending_counts,
                "state_count": len(state_a),
                "cloud_seq": cloud_state["cloud_seq"],
            }
    finally:
        if store_a is not None:
            store_a.close()
        if store_b is not None:
            store_b.close()
        if changelog_a is not None:
            changelog_a.close()
        if changelog_b is not None:
            changelog_b.close()
        if cloud_app is not None:
            cloud_app.state.close()


def run_demo(
    work_dir: str | Path,
    *,
    pause: bool = False,
    emit: Callable[[str], None] = print,
    wait_for_enter: Callable[[], str] = input,
) -> dict[str, Any]:
    """Run the scenario synchronously; the E2E test calls this exact function."""
    return asyncio.run(
        _run_demo(
            work_dir,
            pause=pause,
            emit=emit,
            wait_for_enter=wait_for_enter,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the demo CLI parser."""
    parser = argparse.ArgumentParser(description="Run the narrated EdgeMem offline-sync demo.")
    parser.add_argument(
        "--pause",
        action="store_true",
        help="wait for Enter between scenario steps for live presenting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the full scenario in a temporary directory and return its exit status."""
    arguments = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="edgemem-demo-") as temp_dir:
        run_demo(temp_dir, pause=arguments.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
