"""Per-device FastAPI application for offline-first maintenance memory."""

import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from edgemem.changelog import ChangeLog
from edgemem.store import LocalStore
from edgemem.sync import SyncClient

Severity = Literal["low", "medium", "high", "critical"]


class ObservationInput(BaseModel):
    """Fields submitted by a technician for a new maintenance point."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000)
    equipment: str = Field(min_length=1, max_length=200)
    severity: Severity
    error_code: str | None = Field(default=None, max_length=100)


class SearchInput(BaseModel):
    """Text query, Qdrant payload filters, and search mode."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=100)
    filters: dict[str, Any] | None = None
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"


class PointPatch(BaseModel):
    """Mutable point fields; metadata and tombstone fields are server-managed."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, min_length=1, max_length=4000)
    equipment: str | None = Field(default=None, min_length=1, max_length=200)
    severity: Severity | None = None
    error_code: str | None = Field(default=None, max_length=100)


class NetworkInput(BaseModel):
    """Simulated network state; it does not gate local API operations."""

    model_config = ConfigDict(extra="forbid")

    online: bool


def create_app(
    db_path: str | Path,
    device_id: str,
    *,
    store: LocalStore | None = None,
    changelog: ChangeLog | None = None,
    sync_client: SyncClient | None = None,
    cloud_url: str | None = None,
    sync_transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.time,
    cors_origins: Sequence[str] = ("*",),
) -> FastAPI:
    """Create an isolated device app backed by its own local Qdrant path."""
    if not isinstance(device_id, str) or not device_id.strip():
        raise ValueError("device_id must be a non-empty string")
    owns_store = store is None
    owns_changelog = changelog is None
    device_db_path = Path(db_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Open per-device resources at startup and close owned handles on shutdown."""
        try:
            if application.state.store is None:
                application.state.store = LocalStore(
                    path=device_db_path,
                    collection="memory",
                    device_id=device_id,
                )
            if application.state.changelog is None:
                application.state.changelog = ChangeLog(device_db_path / "changes.sqlite3")
            if application.state.sync_client is None:
                application.state.sync_client = SyncClient(
                    changelog=application.state.changelog,
                    store=application.state.store,
                    cloud_url=cloud_url,
                    transport=sync_transport,
                    device_id=device_id,
                )
            application.state.last_cloud_seq = application.state.sync_client.last_cloud_seq
            application.state.last_sync_ts = application.state.sync_client.last_sync_ts
            yield
        finally:
            if application.state.owns_store:
                if application.state.store is not None:
                    application.state.store.close()
            if application.state.owns_changelog:
                if application.state.changelog is not None:
                    application.state.changelog.close()

    app = FastAPI(title=f"EdgeMem Device API ({device_id})", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.state.store = store
    app.state.changelog = changelog
    app.state.sync_client = sync_client
    app.state.owns_store = owns_store
    app.state.owns_changelog = owns_changelog
    app.state.db_path = device_db_path
    app.state.device_id = device_id
    app.state.online = False

    def record_change(op: str, point_id: str, changed_fields: Mapping[str, Any]) -> None:
        """Append the just-written point delta to this device's SQLite changelog."""
        point = app.state.store.get(point_id)
        if point is None:
            raise RuntimeError(f"stored point {point_id!r} disappeared before changelog append")
        if op == "upsert":
            delta_fields = {
                field: value
                for field, value in point.items()
                if field not in {"id", "device_id", "updated_at", "field_ts"}
            }
        else:
            delta_fields = {
                field: point[field]
                for field in changed_fields
                if field in point
            }
        point_stamps = point.get("field_ts") or {}
        delta_stamps = {
            field: point_stamps.get(field, [point["updated_at"], point["device_id"]])
            for field in delta_fields
        }
        app.state.changelog.append(
            point_id=point_id,
            op=op,
            fields=delta_fields,
            ts=point["updated_at"],
            device_id=point["device_id"],
            field_ts=delta_stamps,
        )

    app.state.record_change = record_change

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Report device identity, stored-point count, and simulated connectivity."""
        return {
            "device_id": app.state.device_id,
            "points": app.state.store.count(),
            "online": app.state.online,
        }

    @app.get("/state")
    def state() -> list[dict[str, Any]]:
        """Return all canonical payloads, including tombstones but no vector data."""
        return app.state.store.all_points()

    @app.post("/search")
    def search(request: SearchInput) -> dict[str, Any]:
        """Search the local store even while the simulated network is offline."""
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must not be blank")
        try:
            hits = app.state.store.search(
                request.query,
                limit=request.limit,
                filters=request.filters,
                mode=request.mode,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        results = []
        for hit in hits:
            fused_score = float(hit.get("score", hit.get("rrf_score", 0.0)))
            results.append(
                {
                    "id": hit["id"],
                    "score": fused_score,
                    "fused_score": fused_score,
                    "rrf_score": fused_score,
                    "dense_score": hit.get("dense_score"),
                    "sparse_score": hit.get("sparse_score"),
                    "dense_rank": hit.get("dense_rank"),
                    "sparse_rank": hit.get("sparse_rank"),
                    "payload": hit["payload"],
                }
            )
        return {"device_id": app.state.device_id, "results": results}

    @app.post("/log", status_code=201)
    def log_observation(request: ObservationInput) -> dict[str, Any]:
        """Embed and write one new point, then invoke the future sync hook."""
        if not request.text.strip() or not request.equipment.strip():
            raise HTTPException(status_code=422, detail="text and equipment must not be blank")
        timestamp = float(clock())
        changed_fields = {
            "text": request.text,
            "equipment": request.equipment,
            "severity": request.severity,
            "error_code": request.error_code,
        }
        point_id = str(uuid4())
        record = {
            "id": point_id,
            **changed_fields,
            "device_id": app.state.device_id,
            "updated_at": timestamp,
            "field_ts": {
                field: [timestamp, app.state.device_id]
                for field in (*changed_fields.keys(), "deleted")
            },
            "deleted": False,
        }
        written = app.state.store.upsert([record])
        if written != 1:
            raise HTTPException(status_code=500, detail="observation was not stored")
        app.state.record_change("upsert", point_id, changed_fields)
        return app.state.store.get(point_id) or record

    @app.patch("/points/{point_id}")
    def patch_point(point_id: str, request: PointPatch) -> dict[str, Any]:
        """Patch changed payload fields and re-embed only when text changes."""
        patch = request.model_dump(exclude_unset=True)
        if not patch:
            raise HTTPException(status_code=422, detail="at least one field is required")
        for field in ("text", "equipment", "severity"):
            if field in patch and (patch[field] is None or not patch[field].strip()):
                raise HTTPException(status_code=422, detail=f"{field} must not be blank or null")

        current = app.state.store.get(point_id)
        if current is None or current.get("deleted", False):
            raise HTTPException(status_code=404, detail="point not found")
        changed_fields = {
            field: value for field, value in patch.items() if current.get(field) != value
        }
        if not changed_fields:
            return current

        timestamp = float(clock())
        field_ts = dict(current.get("field_ts") or {})
        for field in changed_fields:
            field_ts[field] = [timestamp, app.state.device_id]
        metadata_patch = {
            **changed_fields,
            "updated_at": timestamp,
            "field_ts": field_ts,
        }
        if "text" in changed_fields:
            updated = {**current, **metadata_patch}
            written = app.state.store.upsert([updated])
            if written != 1:
                raise HTTPException(status_code=500, detail="point update was not stored")
        else:
            updated = app.state.store.update_payload(point_id, metadata_patch)
            if updated is None:
                raise HTTPException(status_code=404, detail="point not found")

        app.state.record_change("update", point_id, changed_fields)
        return app.state.store.get(point_id) or updated

    @app.delete("/points/{point_id}")
    def delete_point(point_id: str) -> dict[str, Any]:
        """Soft-delete one point and invoke the future sync hook exactly once."""
        current = app.state.store.get(point_id)
        if current is None:
            raise HTTPException(status_code=404, detail="point not found")
        if current.get("deleted", False):
            return {"id": point_id, "deleted": True}
        if not app.state.store.soft_delete(point_id):
            raise HTTPException(status_code=404, detail="point not found")
        app.state.record_change("delete", point_id, {"deleted": True})
        return {"id": point_id, "deleted": True}

    @app.post("/network")
    def set_network(request: NetworkInput) -> dict[str, Any]:
        """Set simulated connectivity state without disabling local operations."""
        app.state.online = request.online
        return {"device_id": app.state.device_id, "online": app.state.online}

    @app.post("/sync-now")
    async def sync_now() -> dict[str, Any]:
        """Run one push/pull pass when online; offline calls are successful no-ops."""
        result = await app.state.sync_client.sync_once(online=app.state.online)
        app.state.last_cloud_seq = app.state.sync_client.last_cloud_seq
        app.state.last_sync_ts = app.state.sync_client.last_sync_ts
        return {
            "status": result.status,
            "pushed_changes": result.pushed_changes,
            "pulled_records": result.pulled_records,
            "acknowledged_changes": result.acknowledged_changes,
            "pending": app.state.changelog.pending_count(),
            "last_sync_ts": app.state.last_sync_ts,
            "last_cloud_seq": app.state.last_cloud_seq,
            "online": app.state.online,
        }

    @app.get("/sync-status")
    def sync_status() -> dict[str, Any]:
        """Report pending local events and the persisted cloud cursor."""
        return {
            "pending": app.state.changelog.pending_count(),
            "last_sync_ts": app.state.sync_client.last_sync_ts,
            "last_cloud_seq": app.state.sync_client.last_cloud_seq,
            "online": app.state.online,
        }

    return app


app = create_app(
    db_path=os.getenv("EDGEMEM_DB", "./data/device_a"),
    device_id=os.getenv("EDGEMEM_DEVICE_ID", "device_a"),
)
