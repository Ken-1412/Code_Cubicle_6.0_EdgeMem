# EdgeMem

EdgeMem is an **offline-first semantic memory prototype for industrial maintenance technicians**. Each device searches and edits its own local maintenance log while disconnected. When connectivity returns, devices exchange compact, field-stamped changes through a simulated cloud and converge on shared point state.

It is a retrieval-and-sync prototype, not a conversational assistant: it indexes observations and returns ranked results; it does not generate maintenance advice.

## Architecture

```text
                         search / log / edit / status
[Technician] <------> [Vanilla JS dashboard, optional]
                            |                 |
                            v                 v
                    [Device A API]     [Device B API]
                     |          |       |          |
                     |          |       |          |
                Qdrant local  SQLite  Qdrant local  SQLite
                dense + BM25  deltas  dense + BM25  deltas
                     \          |       |          /
                      \  compacted field-stamped deltas
                       \       POST /sync and pull
                        v                   v
                   [Simulated cloud API: FastAPI + SQLite]
                         merged state + cloud event sequence

  On each device: BGE-small dense vectors + BM25 sparse vectors -> hybrid RRF search
```

Each device's Qdrant database and SQLite changelog are local to that device. The cloud service in this project is a local SQLite-backed simulator; it is not a hosted service. The dashboard can be opened from `file://` or served statically.

## Quickstart (3 commands)

From the repository root, with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt && pytest -q
```

The first real search or ingestion may download `BAAI/bge-small-en-v1.5` and `Qdrant/bm25` into FastEmbed's model cache. The project uses Qdrant client local mode; no Qdrant server or Docker is needed. The pinned dependencies are in [`requirements.txt`](requirements.txt).

## Run the narrated demo

```bash
python scripts/demo.py
```

The demo creates isolated temporary stores and runs the real device and cloud FastAPI routes in-process, so it does not need ports or modify the persistent `data/run` demo database. It prints six steps and ends with `DEMO PASSED` when all assertions hold. For a live presentation, pause between steps and press Enter to continue:

```bash
python scripts/demo.py --pause
```

The scenario seeds both devices, syncs their baseline, takes them offline, checks a semantic query and a real exact error code, logs a new observation on A, makes different-field concurrent edits to the same point on A and B, deletes an unrelated point on B, then syncs A → B → A. It verifies the merged fields, tombstone visibility, new-record search on B, zero pending changes, and exact state equality among A, B, and cloud.

## Run the live local services and dashboard

To start the separately running Device A (`:8001`), Device B (`:8002`), and cloud (`:9000`) services with fresh demo data:

```bash
python scripts/run_all.py
```

The runner initializes and synchronizes the devices, then stays running until Ctrl+C. In another terminal, serve the dashboard:

```bash
python -m http.server 8080 --directory edgemem/ui
```

Open `http://127.0.0.1:8080/` (or open `edgemem/ui/index.html` directly). The dashboard's API URL fields are configurable. By default, `run_all.py` binds to loopback. Setting `EDGEMEM_BIND_HOST=0.0.0.0` opts into an all-interface bind; **do not expose this unauthenticated prototype on an untrusted network or use it with sensitive data**. See the [10-step UI checklist](scripts/ui_checklist.md).

## Ingest the synthetic seed manually

The deterministic generator creates 120 realistic maintenance observations and an evaluation manifest without calling an external API:

```bash
python scripts/make_data.py
```

To load the seed into two separate local stores:

```bash
python -m edgemem.ingest --db ./data/device_a --file data/seed_logs.jsonl --device-id device_a
python -m edgemem.ingest --db ./data/device_b --file data/seed_logs.jsonl --device-id device_b
```

Ingestion writes batches of 32, reports progress and throughput, and is idempotent by point ID. `--reset` removes the selected database path before ingesting.

## API and library checks

Run a device API directly:

```bash
uvicorn edgemem.api:app --host 127.0.0.1 --port 8001
```

Useful routes are `GET /health`, `GET /state`, `POST /search`, `POST /log`, `PATCH /points/{id}`, `DELETE /points/{id}`, `POST /network`, `POST /sync-now`, and `GET /sync-status`. Network-offline mode disables sync only; local search and edits continue to work. The cloud API is available as `uvicorn edgemem.cloud:app --host 127.0.0.1 --port 9000`; `POST /sync` accepts deltas and `GET /state` returns the merged state and cursor.

To check FastEmbed and Qdrant calls independently:

```bash
python scripts/smoke_libs.py
```

The smoke script writes to `data/smoke_qdrant`. Run the complete suite with `pytest -q`; the E2E scenario is in `tests/test_e2e.py` and calls the same `run_demo` function as the CLI.

## Module map

| Module | Responsibility | Main files |
|---|---|---|
| 1 | Configuration, project scaffold, Qdrant/FastEmbed compatibility smoke test | `edgemem/config.py`, `scripts/smoke_libs.py` |
| 2 | Deterministic 120-record dataset and retrieval evaluation cases | `scripts/make_data.py`, `data/seed_logs.jsonl`, `data/eval_queries.json` |
| 3 | Dense/sparse embeddings, Qdrant local storage, hybrid RRF search | `edgemem/embeddings.py`, `edgemem/store.py` |
| 4 | Batched, idempotent JSONL ingestion | `edgemem/ingest.py` |
| 5 | Per-device FastAPI API and local mutation routes | `edgemem/api.py` |
| 6 | Persistent SQLite delta log and pure per-field LWW merge | `edgemem/changelog.py`, merge functions in `edgemem/sync.py` |
| 7 | SQLite cloud state/event log, sync client, optional daemon | `edgemem/cloud.py`, `edgemem/sync.py` |
| 8 | Two-device dashboard, state endpoint, and live service runner | `edgemem/ui/index.html`, `scripts/run_all.py` |
| 9 | Narrated offline-edit/convergence demo and E2E test | `scripts/demo.py`, `tests/test_e2e.py` |
| 10 | User/developer documentation, limitations, and final verification record | `README.md`, `PROGRESS.md` |

## Limitations and next steps

This project is a tested prototype; the following are deliberate limitations, not hidden capabilities:

- **Storage backend:** it uses `qdrant-client` **local mode**, not Qdrant Edge. Local mode is convenient for this self-contained demo but is not a substitute for evaluating the intended edge deployment/runtime.
- **Merge model:** synchronization is **CRDT-inspired per-field Last-Writer-Wins**, implemented in project code; it is not a full CRDT library or a general-purpose replicated database. Fields are resolved using `(timestamp, device_id)`, so clock skew can affect which concurrent value wins.
- **Security:** device and cloud APIs have **no authentication or transport encryption**. Keep them on loopback or a trusted isolated network; do not send sensitive operational data.
- **No generation:** there is **no local LLM** or generated troubleshooting advice. The product currently stores, searches, and synchronizes technician-authored observations.
- **Tombstone retention:** deletes are soft tombstones and are **never garbage-collected**. Long-lived installations will need a safe retention/compaction policy that preserves offline replicas' ability to observe deletes.
- **Vector conflicts:** vector state is not merged. If a synchronized winning text differs from local text, the receiver **re-embeds the merged text**; metadata-only changes keep the existing vector. This costs inference time and assumes the same model/version on every device.
- **Local-mode acceleration:** scalar **INT8 quantization is not implemented** by Qdrant local mode; the collection is created without it. Local-mode payload indexes have no filtering-performance effect, so `LocalStore` skips index creation. Filters still work, but are not index-accelerated. See the verified API notes in [`PROGRESS.md`](PROGRESS.md).
- **Synthetic evaluation:** the 120 records are deterministic synthetic examples, not a representative production benchmark. Real equipment vocabularies, privacy requirements, update rates, and device/network conditions still need field validation.

Next steps for a deployment-oriented version include testing against the intended Qdrant Edge runtime, adding authenticated and encrypted transport, defining clock and conflict policies, safely expiring tombstones, pinning model artifacts across devices, evaluating real maintenance data, and adding an optional local inference layer only if generated answers are a product requirement.
