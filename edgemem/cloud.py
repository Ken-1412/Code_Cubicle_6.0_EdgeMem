"""Persistent simulated cloud API for convergent maintenance-memory sync."""

import hashlib
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from edgemem.sync import _normalize_delta, apply_delta, merge_records


class CloudChange(BaseModel):
    """Legacy full-record change event accepted by the compatibility endpoint."""

    seq: int = Field(ge=1)
    record: dict[str, Any]


class ChangeBatch(BaseModel):
    """Legacy batch of full-record changes."""

    changes: list[CloudChange]


class DeltaBatch(BaseModel):
    """Compacted device deltas submitted to the persistent cloud."""

    device_id: str = Field(min_length=1)
    last_cloud_seq: int = Field(ge=0)
    deltas: list[dict[str, Any]]


class CloudRepository:
    """SQLite state and immutable cloud sequence log for one cloud instance."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        value = ":memory:" if db_path is None else str(db_path)
        if value != ":memory:":
            path = Path(value).expanduser()
            if (path.exists() and path.is_dir()) or path.suffix.lower() not in {
                ".db",
                ".sqlite",
                ".sqlite3",
            }:
                path.mkdir(parents=True, exist_ok=True)
                path = path / "cloud.sqlite3"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
            value = str(path)

        self.db_path = value
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(value, timeout=10.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock, self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            if value != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS cloud_points (
                       point_id TEXT PRIMARY KEY,
                       record_json TEXT NOT NULL
                   )"""
            )
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS cloud_events (
                       cloud_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                       event_id TEXT NOT NULL UNIQUE,
                       point_id TEXT NOT NULL,
                       source_device_id TEXT NOT NULL,
                       delta_json TEXT NOT NULL
                   )"""
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def sync(
        self,
        *,
        device_id: str,
        last_cloud_seq: int,
        deltas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Atomically merge unseen deltas and return events after the caller cursor."""
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string")
        if not isinstance(last_cloud_seq, int) or isinstance(last_cloud_seq, bool) or last_cloud_seq < 0:
            raise ValueError("last_cloud_seq must be a non-negative integer")
        if not isinstance(deltas, list):
            raise ValueError("deltas must be a list")

        normalized_deltas = [_normalize_delta(delta) for delta in deltas]
        with self._lock, self._connection:
            for delta in normalized_deltas:
                encoded_delta = self._json(delta)
                event_id = hashlib.sha256(
                    self._json({"source_device_id": device_id, "delta": delta}).encode("utf-8")
                ).hexdigest()
                exists = self._connection.execute(
                    "SELECT 1 FROM cloud_events WHERE event_id = ?", (event_id,)
                ).fetchone()
                if exists:
                    continue

                current_row = self._connection.execute(
                    "SELECT record_json FROM cloud_points WHERE point_id = ?",
                    (delta["point_id"],),
                ).fetchone()
                current = json.loads(current_row["record_json"]) if current_row else None
                prior_state = {delta["point_id"]: current} if current is not None else {}
                merged = apply_delta(prior_state, delta)[delta["point_id"]]
                encoded_record = self._json(merged)
                self._connection.execute(
                    """INSERT INTO cloud_points (point_id, record_json)
                       VALUES (?, ?)
                       ON CONFLICT(point_id) DO UPDATE SET record_json = excluded.record_json""",
                    (delta["point_id"], encoded_record),
                )
                self._connection.execute(
                    """INSERT INTO cloud_events
                       (event_id, point_id, source_device_id, delta_json)
                       VALUES (?, ?, ?, ?)""",
                    (event_id, delta["point_id"], device_id, encoded_delta),
                )

            sequence_row = self._connection.execute(
                "SELECT COALESCE(MAX(cloud_seq), 0) AS seq FROM cloud_events"
            ).fetchone()
            new_cloud_seq = int(sequence_row["seq"])
            rows = self._connection.execute(
                """SELECT cloud_seq, point_id, source_device_id, delta_json
                   FROM cloud_events WHERE cloud_seq > ? ORDER BY cloud_seq ASC""",
                (last_cloud_seq,),
            ).fetchall()

        delivered = []
        for row in rows:
            delta = json.loads(row["delta_json"])
            delivered.append(
                {
                    **delta,
                    "cloud_seq": int(row["cloud_seq"]),
                    "source_device_id": row["source_device_id"],
                }
            )
        return {
            "new_cloud_seq": new_cloud_seq,
            "deltas_since_last_cloud_seq": delivered,
        }

    def get_state(self) -> dict[str, Any]:
        """Return the full canonical state and current global cursor."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT point_id, record_json FROM cloud_points ORDER BY point_id"
            ).fetchall()
            sequence_row = self._connection.execute(
                "SELECT COALESCE(MAX(cloud_seq), 0) AS seq FROM cloud_events"
            ).fetchone()
        return {
            "cloud_seq": int(sequence_row["seq"]),
            "state": {row["point_id"]: json.loads(row["record_json"]) for row in rows},
        }

    def close(self) -> None:
        """Close the database connection idempotently."""
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True


def create_cloud_app(db_path: str | Path | None = None) -> FastAPI:
    """Create a cloud API backed by SQLite; an omitted path uses isolated memory."""
    repository = CloudRepository(db_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Close the cloud database after the service stops."""
        try:
            yield
        finally:
            repository.close()

    app = FastAPI(title="EdgeMem Simulated Cloud API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    app.state.repository = repository
    app.state.close = repository.close

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Report cloud process status and its latest sequence."""
        return {"status": "ok", "cloud_seq": repository.get_state()["cloud_seq"]}

    @app.post("/sync")
    def sync(batch: DeltaBatch) -> dict[str, Any]:
        """Accept idempotent device deltas and return all newer cloud events."""
        try:
            return repository.sync(
                device_id=batch.device_id,
                last_cloud_seq=batch.last_cloud_seq,
                deltas=batch.deltas,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/state")
    def state() -> dict[str, Any]:
        """Return all merged cloud points, including tombstones."""
        return repository.get_state()

    # Compatibility surface for the earlier full-record sync protocol.
    @app.post("/changes")
    def accept_changes(batch: ChangeBatch) -> dict[str, int]:
        """Accept legacy snapshots by adapting each to a stamped full-field delta."""
        try:
            accepted = 0
            for change in batch.changes:
                record = change.record
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError("record id must be a non-empty string")
                merged = merge_records(None, record)
                fields = {
                    key: {
                        "value": value,
                        "stamp": merged["field_ts"].get(
                            key, [merged["updated_at"], merged["device_id"]]
                        ),
                    }
                    for key, value in merged.items()
                    if key not in {"id", "device_id", "updated_at", "field_ts"}
                }
                delta = {
                    "point_id": record_id,
                    "op": "delete" if merged["deleted"] else "upsert",
                    "fields": fields,
                    "ts": merged["updated_at"],
                    "device_id": merged["device_id"],
                }
                repository.sync(
                    device_id=merged["device_id"],
                    last_cloud_seq=0,
                    deltas=[delta],
                )
                accepted += 1
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"accepted": accepted, "record_count": len(repository.get_state()["state"])}

    @app.get("/records")
    def list_records() -> dict[str, list[dict[str, Any]]]:
        """List full records for callers using the earlier protocol."""
        records = repository.get_state()["state"]
        return {"records": [deepcopy(records[key]) for key in sorted(records)]}

    @app.get("/records/{record_id}")
    def get_record(record_id: str) -> dict[str, Any]:
        """Fetch one legacy-route record by ID."""
        record = repository.get_state()["state"].get(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="record not found")
        return deepcopy(record)

    return app


app = create_cloud_app(os.getenv("EDGEMEM_CLOUD_DB", "./data/cloud.sqlite3"))
