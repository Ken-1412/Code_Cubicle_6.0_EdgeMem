# EdgeMem Project Progress

EdgeMem is an offline-first semantic memory prototype for industrial maintenance observations. This log summarizes verified work through Module 10.

## Environment and scaffold — Module 1: PASS

- Python project layout, `README.md`, `.gitignore`, pytest configuration, and venv setup instructions are in place.
- Pinned dependencies in `requirements.txt`: `pytest==9.1.1`, `fastembed==0.8.1`, `qdrant-client==1.19.1`, `fastapi==0.141.1`, `uvicorn==0.52.1`, and `httpx==0.28.1`.
- `scripts/smoke_libs.py` uses the installed FastEmbed and Qdrant APIs. The installed client exposes `query_points`; the old `search` method is not the query interface used here.

### Verified FastEmbed and Qdrant calls

```python
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

client = QdrantClient(path="./data/smoke_qdrant")
client.create_collection(
    collection_name="smoke_libs",
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
    },
)
```

- Dense model: `TextEmbedding(model_name="BAAI/bge-small-en-v1.5")`; batch outputs are converted to lists of Python floats.
- Sparse model: `SparseTextEmbedding(model_name="Qdrant/bm25")`; each result exposes `.indices` and `.values`, converted as `models.SparseVector(indices=[...], values=[...])`.
- Upsert uses `client.upsert(collection_name=..., points=[models.PointStruct(id=..., vector={"dense": dense_values, "sparse": sparse_vector}, payload=...)], wait=True)`.
- Dense and sparse single-branch queries use `client.query_points(collection_name=..., query=vector, using="dense"|"sparse", limit=..., with_payload=True)`.
- Hybrid search uses `query_points` with dense and sparse `models.Prefetch(...)` branches and `query=models.FusionQuery(fusion=models.Fusion.RRF)`; `LocalStore` names its sparse vector `bm25`.
- Payload-index API call: `client.create_payload_index(collection_name=..., field_name=..., field_schema=models.PayloadSchemaType.KEYWORD, wait=True)`. Local Qdrant accepts the operation but warns it has no filtering-performance effect. `LocalStore` skips these no-op calls and logs the limitation; the smoke harness exercises the API directly.
- Local Qdrant does not implement scalar INT8 quantization. No fake quantization setting is persisted or claimed; `LocalStore` logs the limitation and uses unquantized local vectors.
- `python scripts/smoke_libs.py` passed using the real models and local Qdrant, printing dense, sparse, and hybrid RRF top results.

## Synthetic maintenance data — Module 2: PASS

- `scripts/make_data.py` deterministically generates `data/seed_logs.jsonl` and `data/eval_queries.json` using a fixed random seed, templates, and hand-written variations; no external API is used.
- The seed contains 120 unique, schema-valid records across all eight required equipment classes. 48 records (40%) carry a verbatim alphanumeric error code.
- The evaluation manifest contains eight semantic, eight exact-code, and four filtered queries, with expected IDs validated against the seed.
- The LocalStore evaluation reports semantic hit@1 6/8 and hit@3 8/8, exact-code hit@1 8/8, and all four filtered queries passing.

## Embeddings and local Qdrant store — Module 3: PASS

- `edgemem/embeddings.py` provides lazy singleton FastEmbed dense and BM25 encoders with batch support. Sparse outputs are native Qdrant `SparseVector` objects.
- `LocalStore` configures named 384-dimensional cosine dense vectors and a named BM25 sparse vector, persists in Qdrant local mode, writes canonical record metadata, and supports get, soft delete, count, scrolling, upsert, and dense/sparse/hybrid search.
- Hybrid search uses Qdrant-side dense/sparse prefetch with RRF; dense and sparse branches are also queried to report their score/rank breakdown. Filters are applied in Qdrant and always exclude tombstones.
- Re-upserts replace the existing point ID rather than duplicating it; reopened local stores retain data. Local payload-index and quantization limitations are reported honestly as above.
- Tests cover the 120-record seed thresholds, filters, deletion, updates, persistence, and honest local-mode capability reporting.

## Batched, device-aware ingestion — Module 4: PASS

- `python -m edgemem.ingest --db <path> --file <jsonl> [--device-id <id>] [--reset]` loads the seed in batches of 32, prints `ingested N/M`, reports elapsed time and throughput, and is idempotent by point ID.
- Each record is enriched with the chosen `device_id`, `updated_at`, and per-field timestamps. `--reset` removes only the selected database directory before recreating it.
- Real CLI runs loaded all 120 rows into both `data/device_a` and `data/device_b`; a second device A run left its count at 120. Reopened stores had device-specific metadata and identical source content across devices.
- `pytest tests/test_ingest.py -q` passed.

## Per-device FastAPI app — Module 5: PASS

- `create_app(db_path, device_id)` provides isolated device instances. The module-level `app` defaults to device A at `./data/device_a`; simulated connectivity defaults to offline and does not gate local search or logging.
- Routes cover health, hybrid/dense/sparse search, immediate log creation, field-aware patching, tombstone deletion, and connectivity-state toggling. CORS is enabled for local UI development.
- `/log`, `PATCH /points/{id}`, and `DELETE /points/{id}` share `record_change(op, point_id, changed_fields)`, now wired to the SQLite changelog by Module 6.
- The API previously passed an in-process health/search check on port 8001.

## SQLite deltas and pure merge — Module 6: PASS

- `ChangeLog` stores one device's append-only events in `<db_path>/changes.sqlite3` with columns `seq`, `point_id`, `op`, `fields_json`, `ts`, `device_id`, and `synced`. Each `fields_json` field entry is `{ "value": ..., "stamp": [timestamp, device_id] }`.
- Implemented `append`, ordered `pending`, `pending_count`, `mark_synced(up_to_seq)`, and `compact(pending)`. `mark_synced` also supports the earlier explicit sequence-list form. Existing snapshot-schema files migrate while retaining sequence IDs, payloads, and acknowledgements.
- Compaction retains the latest value/stamp per field by `(ts, device_id)` with deterministic canonical-JSON conflict resolution. A tombstone is effective only when its stamp is strictly newer than every winning content-field stamp; otherwise the resolved `deleted` value is false.
- `sync.py` exposes pure `merge_point`, `merge_states`, and `apply_delta`; fields keep their winning stamps, top-level metadata is deterministic, and replay/out-of-order delivery converges. The original snapshot `SyncClient` at that checkpoint was replaced by the Module 7 delta client below.
- Device API mutations append exactly one `upsert`, `update`, or `delete` event after each successful Qdrant write. Upserts record all point values, updates record only changed fields, and deletes record the tombstone.
- `pytest tests/test_merge.py tests/test_changelog.py tests/test_api.py -q` passed at the Module 6 checkpoint, including randomized merge laws, tombstone ordering, event compaction, schema migration, and API row counts. No network sync was exercised at that checkpoint.

## Persistent cloud and delta synchronization — Module 7: PASS

- `create_cloud_app(db_path)` now persists merged point state and an append-only `cloud_events` sequence in SQLite. The importable `edgemem.cloud:app` uses `EDGEMEM_CLOUD_DB` or `./data/cloud.sqlite3`; explicit app instances can still use an isolated in-memory database for tests.
- `POST /sync` accepts `{device_id, last_cloud_seq, deltas}`, applies validated field-stamped deltas through the pure merge path, assigns a global monotonically increasing `cloud_seq`, and returns events after the submitted cursor. A stable event digest makes exact duplicate deliveries idempotent. `GET /state` returns the complete merged state and current cursor. The earlier `/changes` and `/records` routes remain as compatibility endpoints.
- `SyncClient` compacts pending local events, pushes and pulls in one request, applies returned deltas to local point state, re-embeds only if text changes, and otherwise updates payload fields without replacing vectors. It acknowledges the local sequence and persists `last_cloud_seq` plus `last_sync_ts` atomically in the device SQLite metadata only after local application succeeds.
- Failed cloud calls leave pending changes untouched. Partial local application can be safely replayed because cloud events and per-field merges are idempotent. `sync_once(device_app_state, cloud_url_or_transport)` supports FastAPI state and HTTPX ASGI transports; `SyncDaemon(interval_s)` is an opt-in stoppable background thread and runs only while the simulated device is online.
- Device API adds `POST /sync-now` and `GET /sync-status` with pending count, last-sync timestamp, cloud cursor, and online state. The default cloud URL is `http://127.0.0.1:9000`.
- Tests cover offline no-op, A/B independent-field and same-field conflicts, tombstone propagation, new-record searchability, first-call cloud failure/retry, duplicate delivery, persistent cloud restart, local cursor persistence, API status routes, and daemon start/stop behavior.

### Module 7 verification

- `pytest tests/test_sync.py -q` — **7 passed in 4.61s**.
- `pytest -q` — **79 passed in 11.77s**. The run also printed the seed-evaluation table: semantic hit@1 6/8 and hit@3 8/8; exact-code hit@1 8/8; filtered 4/4.
- `python3 -m compileall -q edgemem scripts tests` — passed.
- Cloud/device integration tests use HTTPX ASGI transport; no external service or public network was contacted.

## Two-device UI and runner — Module 8: PASS

- Added device `GET /state`, returning canonical payloads (including tombstones) without Qdrant vectors; cloud `/state` returns its merged payload map. Device and cloud APIs allow CORS for the standalone dashboard.
- `edgemem/ui/index.html` is one vanilla HTML/CSS/JS file with editable A/B/cloud URLs; it infers sibling sandbox API hosts when served through a port-mapped preview and otherwise keeps localhost defaults. It provides connectivity toggles, pending/last-sync status, local hybrid/dense/sparse search and filters, score bars, observation logging, severity edits, tombstone deletes, two-second polling, and a three-way state-hash badge.
- `scripts/run_all.py` resets only `data/run/`, seeds two independent 120-record stores through Module 4 ingestion, queues initial snapshots, starts the APIs on 8001/8002/9000, performs initial sync, checks canonical convergence, and stops child servers on Ctrl+C. `EDGEMEM_BIND_HOST` can opt into a different bind address; loopback is the default.
- `scripts/ui_checklist.md` documents ten manual dashboard checks and their API equivalents.

### Module 8 verification

- `pytest tests/test_state_endpoint.py -q` — **2 passed**; payload-only device state and cloud CORS preflight verified.
- `python scripts/run_all.py` booted all three services with 120 seed points per device; initial cloud sequence was 240, pending counts were zero, and all three canonical state hashes matched.
- All ten checklist categories passed using live HTTP API calls: endpoint/status checks, online/offline toggle, hybrid score breakdown, equipment/severity filters across all modes, offline log/search, offline no-op sync, A-to-cloud push, B pull/search, and edit/delete/tombstone propagation with final convergence.
- The browser loaded the UI directly from `file://` and connected to the temporary sandbox API URLs. UI-driven search rendered results and score bars; offline log/sync, online synchronization, severity edit, deletion, empty pending queues, and reconvergence passed. The synthetic `UI-BROWSER-1` point remains as a tombstone, excluded from search.
- The HTTP-served UI preview returned 200 locally and publicly; its inferred A/B/cloud URLs auto-connected and rendered the green convergence state.
- Final live check: A, B, and cloud each held 121 points including that tombstone, with matching canonical state hashes, zero pending changes, both devices offline, and cloud sequence 242.
- `pytest -q` — **81 passed in 11.50s**; `python3 -m compileall -q edgemem scripts tests` — passed.

## Pre-Module 9 verification checkpoint (historical)

- Latest full project gate: `pytest -q` — **81 passed in 11.50s**.
- FastEmbed models require a one-time cache download. Qdrant local payload indexes do not accelerate filtering, and scalar INT8 quantization is not implemented in Qdrant local mode.
- The simulated cloud is a local SQLite service; this module implements sync endpoints and an opt-in daemon but does not deploy or automatically start either service/daemon.

## Narrated offline-sync demo — Module 9: PASS

- `scripts/demo.py` runs the six-step scenario against real device/cloud FastAPI routes through HTTPX ASGI transports, with isolated temporary Qdrant and SQLite stores. It does not require listening ports. `tests/test_e2e.py` calls the same `run_demo` function as the CLI.
- The required semantic query (`pump shaking and oil leaking`) returns its expected pump record at rank 3; the real code `HYD-2000` returns its expected record at rank 1. The output prints dense, sparse, and fused scores.
- The scenario proves offline logging/search, independent concurrent edits to one record, an unrelated tombstone, A → B → A sync, exact A/B/cloud equality across 121 points, search visibility of the new observation on B, and zero pending changes on both devices.
- `python scripts/demo.py` and `python scripts/demo.py --pause` passed; the pause path waited five times between the six steps. The focused E2E test passed in 9.91s at its initial checkpoint.

## Documentation and clean-install validation — Module 10: PASS

- `README.md` now documents the ASCII architecture, three-command quickstart, demo walkthrough, live-service instructions, module map, and explicit limitations/next steps. It records that this is Qdrant client local mode (not Qdrant Edge), that local mode does not provide effective payload-index acceleration or scalar INT8 quantization, and that the sync model is per-field LWW rather than a full CRDT library.
- Cleanup was behavior-preserving: Ruff import organization, removal of unused imports/locals, and a targeted `E402` suppression for the demo's direct-script path bootstrap. The project passes Ruff 0.16.9 with isolated `E4,E7,E9,F,I` checks. The sandbox has user-global Ruff heuristics beyond the project lint scope; suggestions that change exception types were not applied.
- This sandbox checkout contains no `.git` directory or remote, so the fresh-clone gate was run from a clean source-only copy rather than a Git clone. A new Python 3.12.3 virtual environment installed the pinned `requirements.txt` successfully.
- In that clean copy and fresh venv, `pytest -q` passed **83 tests in 17.66s** with one expected Qdrant local-mode warning; `python scripts/demo.py` ended with **DEMO PASSED**. The seed evaluation printed semantic hit@1 **6/8**, hit@3 **8/8**; exact-code hit@1 **8/8**; filtered **4/4**.

### Per-module validation timings

These are summed per-test setup/call/teardown durations collected by pytest hooks in one instrumented run in the fresh venv. They are validation timings, not implementation time, and exclude collection and process startup; the whole-suite wall time was 18.37s in this instrumented run and 17.66s in the required plain `pytest -q` run.

| Module | Test group | Duration |
|---|---|---:|
| 1 — scaffold and library smoke | `test_config.py`, `test_smoke.py` | 1.361s |
| 2 — synthetic maintenance data | `test_data.py` | 0.011s |
| 3 — embeddings and storage | `test_embeddings.py`, `test_store.py` | 4.752s |
| 4 — ingestion | `test_ingest.py` | 0.685s |
| 5 — device API | `test_api.py` | 2.448s |
| 6 — merge and changelog | `test_merge.py`, `test_changelog.py` | 0.389s |
| 7 — cloud and sync | `test_cloud.py`, `test_sync.py` | 0.712s |
| 8 — state endpoint | `test_state_endpoint.py` | 0.053s |
| 9 — E2E demo | `test_e2e.py` | 5.880s |
| 10 — documentation and cleanup | No separate test module; covered by clean install, lint, and exit gate | N/A |

`test_api.py` also includes changelog integration coverage, and Module 8's dashboard interaction checks are documented in `scripts/ui_checklist.md` and were previously exercised through the live API/browser checklist.

## Final exit gate

- Fresh requirements install: **PASS**.
- Fresh-copy `pytest -q`: **83 passed in 17.66s**, with one expected Qdrant local-mode payload-index warning.
- Fresh-copy `python scripts/demo.py`: **DEMO PASSED**.
- Ruff isolated lint checks: **PASS**.
- Quantization and local-mode payload-index limitations are stated in README and match the Module 1 Qdrant capability checks above.

## Skeptical reviewer follow-up

- **Weak E2E assertion fixed:** the scenario previously proved convergence only after sync; an accidental early cloud write while devices were offline could have gone unnoticed. Step 5 now compares the cloud state and sequence to the post-bootstrap snapshot and requires two pending local deltas on each device. `tests/test_e2e.py` asserts these returned facts.
- **Installed-package path covered:** `tests/test_smoke.py::test_real_fastembed_and_qdrant_api_smoke` now invokes the real FastEmbed dense/sparse models and Qdrant local client, covering collection creation, payload-index call, upsert, per-vector `query_points`, and hybrid `Prefetch` + `FusionQuery(RRF)`. The previous fake-only smoke test remains for deterministic rank behavior. Its warning assertion was removed because Qdrant's local-mode warning is process/order-dependent; the warning is still visible in the suite output.
- The fresh validation venv reported `qdrant-client==1.19.1`, `fastembed==0.8.1`, `httpx==0.28.1`, `fastapi==0.141.1`, and `pytest==9.1.1`; method signatures and the presence of `Prefetch`/`FusionQuery` were inspected, then the real integration test passed against those installed packages.
- Final reviewed-code gate: **83 passed in 17.66s**, one expected Qdrant local-mode payload-index warning; Ruff passes; the fresh-copy CLI demo ends in **DEMO PASSED** and narrates unchanged cloud state plus pending counts A=2/B=2 before sync.

## Showcase polish and proof of value — Module 12: Phase 1 (Evidence): PASS

- Added `scripts/benchmark.py` measuring retrieval quality, warm/cold latencies, process RSS and on-disk footprint, scaling up to 10,000 records, zero-network offline execution, 200 randomized 3-device sync conflict scenarios, and naive whole-record LWW comparison.
- Generates `docs/results.json` and human-readable `docs/RESULTS.md`.
- Key empirical findings:
  - Hybrid search achieves **90.0% Hit@1** and **0.9417 MRR** (vs 65.0% Hit@1 / 0.7167 MRR for Dense, 85.0% Hit@1 / 0.9083 MRR for Sparse).
  - Search latency: p50 is **18.57 ms**, p95 is **22.54 ms**, p99 is **26.16 ms** across 500 queries. Write + embedding latency is **19.13 ms** with instant searchability verified.
  - Scale: grows from 120 records (0.56 MB disk, 17.07 ms p50) to 1,000 records (4.53 MB disk, 69.94 ms p50) and 10,000 records (45.13 MB disk, 629.11 ms p50).
  - Offline proof: **0 network calls attempted** under monkeypatched socket guard and `HF_HUB_OFFLINE=1`.
  - Sync correctness: **100.0% convergence rate** across 200 randomized 3-device conflict scenarios with shuffled delivery, **0 lost edits**, **0 duplicates**, and **25.0% bandwidth saved** via compaction.
  - Baseline comparison: Naive whole-record LWW lost **155 edits** out of 200 scenarios (only 22.5% preservation rate), whereas EdgeMem per-field LWW lost **0 edits** (100.0% preservation).
- `pytest tests/test_benchmark_smoke.py -q` passed in 20.46s.

