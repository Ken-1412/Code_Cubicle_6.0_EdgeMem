"""Device-to-cloud synchronization and deterministic field-wise record merging."""

import asyncio
import json
import logging
import math
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

import httpx

from edgemem.changelog import ChangeLog
from edgemem.config import get_settings
from edgemem.embeddings import embed_dense, embed_sparse
from edgemem.store import LocalStore

_METADATA_FIELDS = {"id", "updated_at", "device_id", "field_ts"}
_REQUIRED_FIELDS = {
    "id",
    "text",
    "equipment",
    "severity",
    "error_code",
    "device_id",
    "updated_at",
    "field_ts",
    "deleted",
}
_SEVERITIES = {"low", "medium", "high", "critical"}


class SyncProtocolError(ValueError):
    """Raised when the cloud response or sync payload violates the expected schema."""


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Summary of one upload/pull/ack synchronization pass."""

    pushed_changes: int
    pulled_records: int
    acknowledged_changes: int
    skipped: bool = False
    status: str = "synced"


def _record_stamp(record: Mapping[str, Any], field: str) -> tuple[float, str]:
    """Return the canonical (timestamp, device ID) version for one field."""
    field_ts = record.get("field_ts")
    if not isinstance(field_ts, dict):
        raise ValueError("record field_ts must be an object")
    stamp = field_ts.get(field)
    if stamp is None:
        stamp = [record["updated_at"], record["device_id"]]
    if (
        not isinstance(stamp, (list, tuple))
        or len(stamp) != 2
        or not isinstance(stamp[0], (int, float))
        or isinstance(stamp[0], bool)
        or not math.isfinite(float(stamp[0]))
        or not isinstance(stamp[1], str)
        or not stamp[1].strip()
    ):
        raise ValueError(f"invalid field timestamp for {field}")
    return float(stamp[0]), stamp[1]


def _validate_record(record: Mapping[str, Any]) -> None:
    """Validate canonical fields required for indexing and deterministic merge."""
    missing = sorted(_REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError(f"record missing required fields: {', '.join(missing)}")
    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("record id must be a non-empty string")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError("record text must be a non-empty string")
    if (
        not isinstance(record["equipment"], str)
        or not isinstance(record["device_id"], str)
        or not record["device_id"].strip()
    ):
        raise ValueError("record equipment and device_id must be strings")
    if record["severity"] not in _SEVERITIES:
        raise ValueError("record severity is invalid")
    if record["error_code"] is not None and not isinstance(record["error_code"], str):
        raise ValueError("record error_code must be a string or null")
    if not isinstance(record["updated_at"], (int, float)) or isinstance(record["updated_at"], bool):
        raise ValueError("record updated_at must be numeric")
    if not isinstance(record["field_ts"], dict):
        raise ValueError("record field_ts must be an object")
    if not isinstance(record["deleted"], bool):
        raise ValueError("record deleted must be a boolean")
    for field in record:
        if field not in _METADATA_FIELDS:
            _record_stamp(record, field)
    try:
        json.dumps(record, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("record must contain JSON-compatible values") from error


def _resolve_tombstone(record: dict[str, Any]) -> dict[str, Any]:
    """A delete is effective only when newer than every winning content field."""
    if record.get("deleted") is not True:
        return record
    deleted_stamp = _record_stamp(record, "deleted")
    content_stamps = [
        _record_stamp(record, field)
        for field in record
        if field not in _METADATA_FIELDS and field != "deleted"
    ]
    if content_stamps and deleted_stamp <= max(content_stamps):
        record["deleted"] = False
    return record


def merge_point(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two canonical point versions field by field using (ts, device_id) LWW.

    Equal stamps with conflicting JSON values use the lexicographically larger
    canonical JSON value as a deterministic final tie-break. A tombstone is
    effective only when its stamp is strictly newer than all winning content
    fields; a later content edit therefore revives a stale tombstone.
    """
    _validate_record(a)
    _validate_record(b)
    left, right = deepcopy(dict(a)), deepcopy(dict(b))
    if left["id"] != right["id"]:
        raise ValueError("cannot merge records with different IDs")

    merged: dict[str, Any] = {"id": left["id"]}
    merged_field_ts: dict[str, list[float | str]] = {}
    fields = (set(left) | set(right)) - _METADATA_FIELDS
    for field in sorted(fields):
        if field not in left:
            winner = right
        elif field not in right:
            winner = left
        else:
            left_stamp = _record_stamp(left, field)
            right_stamp = _record_stamp(right, field)
            left_key = (left_stamp, json.dumps(left[field], sort_keys=True, allow_nan=False))
            right_key = (right_stamp, json.dumps(right[field], sort_keys=True, allow_nan=False))
            winner = right if right_key > left_key else left
        merged[field] = deepcopy(winner[field])
        stamp = _record_stamp(winner, field)
        merged_field_ts[field] = [stamp[0], stamp[1]]

    latest = max(
        (float(left["updated_at"]), left["device_id"]),
        (float(right["updated_at"]), right["device_id"]),
    )
    merged["updated_at"], merged["device_id"] = latest
    merged["field_ts"] = merged_field_ts
    return _resolve_tombstone(merged)


def merge_records(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper used by the current simulated cloud API."""
    _validate_record(incoming)
    if existing is None:
        return deepcopy(dict(incoming))
    return merge_point(existing, incoming)


def merge_states(
    state_a: Mapping[str, Mapping[str, Any]],
    state_b: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Converge two ID-to-point maps without mutating either input."""
    result: dict[str, dict[str, Any]] = {}
    for point_id in sorted(set(state_a) | set(state_b)):
        left = state_a.get(point_id)
        right = state_b.get(point_id)
        if left is not None and left.get("id") != point_id:
            raise ValueError(f"state key {point_id!r} does not match its point id")
        if right is not None and right.get("id") != point_id:
            raise ValueError(f"state key {point_id!r} does not match its point id")
        if left is None:
            _validate_record(right)
            result[point_id] = deepcopy(dict(right))
        elif right is None:
            _validate_record(left)
            result[point_id] = deepcopy(dict(left))
        else:
            result[point_id] = merge_point(left, right)
    return result


def _normalize_delta(delta: Any) -> dict[str, Any]:
    """Accept a Change object or stamped/plain delta mapping."""
    if hasattr(delta, "delta"):
        delta = delta.delta
    if not isinstance(delta, Mapping):
        raise ValueError("delta must be a Change object or mapping")
    point_id = delta.get("point_id", delta.get("id"))
    op = delta.get("op")
    ts = delta.get("ts", delta.get("updated_at"))
    device_id = delta.get("device_id")
    if not isinstance(point_id, str) or not point_id.strip():
        raise ValueError("delta point_id must be a non-empty string")
    if op not in {"upsert", "update", "delete"}:
        raise ValueError("delta op must be 'upsert', 'update', or 'delete'")
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(float(ts)):
        raise ValueError("delta timestamp must be finite numeric")
    if not isinstance(device_id, str) or not device_id:
        raise ValueError("delta device_id must be a string")
    fields = delta.get("fields", delta.get("changed_fields"))
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("delta fields must be a non-empty object")
    supplied_stamps = delta.get("field_ts", {})
    if not isinstance(supplied_stamps, Mapping):
        raise ValueError("delta field_ts must be an object")
    normalized = {}
    for field, raw in fields.items():
        if not isinstance(field, str) or not field.strip() or field in _METADATA_FIELDS:
            raise ValueError(f"invalid delta field {field!r}")
        if isinstance(raw, Mapping) and "value" in raw and "stamp" in raw:
            value, stamp = raw["value"], raw["stamp"]
        else:
            value = raw
            stamp = supplied_stamps.get(field, [ts, device_id])
        if (
            not isinstance(stamp, (list, tuple))
            or len(stamp) != 2
            or not isinstance(stamp[0], (int, float))
            or isinstance(stamp[0], bool)
            or not math.isfinite(float(stamp[0]))
            or not isinstance(stamp[1], str)
            or not stamp[1].strip()
        ):
            raise ValueError(f"invalid delta timestamp for {field}")
        normalized[field] = {"value": deepcopy(value), "stamp": [float(stamp[0]), stamp[1]]}
    try:
        json.dumps(normalized, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("delta values must be JSON-compatible") from error
    return {
        "point_id": point_id,
        "op": op,
        "fields": normalized,
        "ts": float(ts),
        "device_id": device_id,
    }


def apply_delta(
    state: Mapping[str, Mapping[str, Any]],
    delta: Any,
) -> dict[str, dict[str, Any]]:
    """Return a new state with one delta applied; duplicate and reordered delivery is safe."""
    if not isinstance(state, Mapping):
        raise ValueError("state must be an ID-to-point mapping")
    change = _normalize_delta(delta)
    point_id = change["point_id"]
    current = state.get(point_id)
    if current is not None:
        _validate_record(current)
        candidate = deepcopy(dict(current))
    else:
        minimum_stamp = [-1.0e300, "__pending__"]
        candidate = {
            "id": point_id,
            "text": "__pending__",
            "equipment": "__pending__",
            "severity": "low",
            "error_code": None,
            "deleted": False,
            "device_id": change["device_id"],
            "updated_at": change["ts"],
            "field_ts": {
                field: list(minimum_stamp)
                for field in ("text", "equipment", "severity", "error_code", "deleted")
            },
        }

    for field, entry in change["fields"].items():
        candidate[field] = deepcopy(entry["value"])
        candidate.setdefault("field_ts", {})[field] = list(entry["stamp"])
    incoming_version = (change["ts"], change["device_id"])
    current_version = (float(candidate["updated_at"]), candidate["device_id"])
    if incoming_version > current_version:
        candidate["updated_at"], candidate["device_id"] = incoming_version
    _validate_record(candidate)

    result = {key: deepcopy(dict(value)) for key, value in state.items()}
    result[point_id] = (
        merge_point(current, candidate) if current is not None else _resolve_tombstone(candidate)
    )
    return result


class SyncClient:
    """Push compact local deltas, apply newer cloud deltas, then acknowledge locally."""

    def __init__(
        self,
        changelog: ChangeLog,
        store: LocalStore,
        cloud_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        dense_embedder=embed_dense,
        sparse_embedder=embed_sparse,
        timeout: float = 10.0,
        device_id: str | None = None,
    ) -> None:
        """Configure one client; text reindexing is delegated to ``LocalStore.upsert``."""
        self.changelog = changelog
        self.store = store
        self.cloud_url = (cloud_url or get_settings().cloud_url).rstrip("/")
        self.transport = transport
        # Retained as compatibility attributes; LocalStore owns its active embedders.
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.timeout = timeout
        self.device_id = device_id or getattr(store, "device_id", get_settings().device_id)
        if not isinstance(self.device_id, str) or not self.device_id.strip():
            raise ValueError("device_id must be a non-empty string")
        self.last_cloud_seq = self.changelog.get_sync_metadata("last_cloud_seq", 0)
        self.last_sync_ts = self.changelog.get_sync_metadata("last_sync_ts", None)
        if (
            not isinstance(self.last_cloud_seq, int)
            or isinstance(self.last_cloud_seq, bool)
            or self.last_cloud_seq < 0
        ):
            raise ValueError("persisted last_cloud_seq must be a non-negative integer")
        self._sync_guard = Lock()
        self.logger = logging.getLogger(__name__)

    async def sync_once(self, online: bool = True) -> SyncResult:
        """Synchronize one snapshot of pending deltas without acknowledging on failure.

        Cloud writes are idempotent by a stable event digest. Local acknowledgements and
        cursor persistence happen only after every returned delta has been applied.
        If an embedding or payload write fails part-way through, a retry replays the
        same cloud events and safely merges them into the already-partially-updated store.
        """
        if not online:
            return SyncResult(0, 0, 0, skipped=True, status="offline")
        if not self._sync_guard.acquire(blocking=False):
            return SyncResult(0, 0, 0, skipped=True, status="busy")

        try:
            pending = self.changelog.pending()
            max_pending_seq = max((change.seq for change in pending), default=0)
            deltas = ChangeLog.compact(pending)
            request_body = {
                "device_id": self.device_id,
                "last_cloud_seq": self.last_cloud_seq,
                "deltas": deltas,
            }
            async with httpx.AsyncClient(
                base_url=self.cloud_url,
                transport=self.transport,
                timeout=self.timeout,
            ) as client:
                response = await client.post("/sync", json=request_body)
                response.raise_for_status()
                body = response.json()

            if not isinstance(body, dict):
                raise SyncProtocolError("cloud response must be a JSON object")
            new_cloud_seq = body.get("new_cloud_seq")
            events = body.get("deltas_since_last_cloud_seq")
            if (
                not isinstance(new_cloud_seq, int)
                or isinstance(new_cloud_seq, bool)
                or new_cloud_seq < self.last_cloud_seq
            ):
                raise SyncProtocolError("cloud response has an invalid new_cloud_seq")
            if not isinstance(events, list):
                raise SyncProtocolError("cloud response must contain a delta list")

            originals: dict[str, dict[str, Any] | None] = {}
            merged_by_id: dict[str, dict[str, Any]] = {}
            previous_seq = self.last_cloud_seq
            for event in events:
                if not isinstance(event, Mapping):
                    raise SyncProtocolError("cloud deltas must be JSON objects")
                cloud_seq = event.get("cloud_seq")
                if (
                    not isinstance(cloud_seq, int)
                    or isinstance(cloud_seq, bool)
                    or cloud_seq <= previous_seq
                    or cloud_seq > new_cloud_seq
                ):
                    raise SyncProtocolError("cloud delta sequence is invalid or unordered")
                previous_seq = cloud_seq
                delta = event.get("delta", event)
                point_id = delta.get("point_id") if isinstance(delta, Mapping) else None
                if not isinstance(point_id, str) or not point_id.strip():
                    raise SyncProtocolError("cloud delta point_id must be a non-empty string")
                if point_id not in originals:
                    current = self.store.get(point_id)
                    originals[point_id] = deepcopy(current) if current is not None else None
                    if current is not None:
                        merged_by_id[point_id] = deepcopy(current)
                prior = merged_by_id.get(point_id)
                try:
                    merged_state = apply_delta(
                        {point_id: prior} if prior is not None else {},
                        delta,
                    )
                    merged = merged_state[point_id]
                    _validate_record(merged)
                except (KeyError, TypeError, ValueError) as error:
                    raise SyncProtocolError(f"invalid cloud delta: {error}") from error
                merged_by_id[point_id] = merged

            # Apply all planned point states. Re-read before writing so an edit that
            # arrived during the HTTP round-trip wins by its field timestamp.
            for point_id, candidate in merged_by_id.items():
                current = self.store.get(point_id)
                if current is not None:
                    candidate = merge_point(current, candidate)
                if current is None or current.get("text") != candidate.get("text"):
                    written = self.store.upsert([candidate])
                    if written != 1:
                        raise SyncProtocolError(f"local store did not apply point {point_id!r}")
                elif current != candidate:
                    payload_patch = {
                        field: deepcopy(value)
                        for field, value in candidate.items()
                        if field != "id" and current.get(field) != value
                    }
                    if self.store.update_payload(point_id, payload_patch) is None:
                        raise SyncProtocolError(f"local store lost point {point_id!r} during sync")

            sync_timestamp = time.time()
            acknowledged = self.changelog.complete_sync(
                max_pending_seq,
                last_cloud_seq=new_cloud_seq,
                last_sync_ts=sync_timestamp,
            )
            self.last_cloud_seq = new_cloud_seq
            self.last_sync_ts = sync_timestamp
            return SyncResult(
                pushed_changes=len(deltas),
                pulled_records=len(merged_by_id),
                acknowledged_changes=acknowledged,
            )
        finally:
            self._sync_guard.release()


async def sync_once(
    device_app_state: Any,
    cloud_url: str | httpx.AsyncBaseTransport | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> SyncResult:
    """Synchronize from a FastAPI ``app.state`` or compatible device-state object.

    The second positional argument may be a cloud URL or an HTTPX ASGI transport.
    """
    if cloud_url is not None and not isinstance(cloud_url, str):
        if transport is None and isinstance(cloud_url, httpx.AsyncBaseTransport):
            transport = cloud_url
            cloud_url = None
        else:
            raise TypeError("cloud_url must be a string or an HTTPX async transport")

    client = getattr(device_app_state, "sync_client", None)
    if client is None:
        changelog = getattr(device_app_state, "changelog", None)
        store = getattr(device_app_state, "store", None)
        if changelog is None or store is None:
            raise ValueError("device state must expose store and changelog")
        client = SyncClient(
            changelog=changelog,
            store=store,
            cloud_url=cloud_url,
            transport=transport,
            device_id=getattr(device_app_state, "device_id", None),
        )
        device_app_state.sync_client = client
    else:
        if cloud_url is not None:
            client.cloud_url = cloud_url.rstrip("/")
        if transport is not None:
            client.transport = transport

    result = await client.sync_once(online=bool(getattr(device_app_state, "online", False)))
    device_app_state.last_cloud_seq = client.last_cloud_seq
    device_app_state.last_sync_ts = client.last_sync_ts
    return result


class SyncDaemon:
    """Optional stoppable background thread that syncs a device while it is online."""

    def __init__(self, interval_s: float = 5.0) -> None:
        if (
            not isinstance(interval_s, (int, float))
            or isinstance(interval_s, bool)
            or not math.isfinite(float(interval_s))
            or interval_s <= 0
        ):
            raise ValueError("interval_s must be a positive finite number")
        self.interval_s = float(interval_s)
        self.device_app_state: Any = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._logger = logging.getLogger(__name__)

    @property
    def is_running(self) -> bool:
        """Whether the daemon thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, device_app_state: Any | None = None) -> "SyncDaemon":
        """Start the periodic worker, optionally binding it to a device state."""
        if device_app_state is not None:
            self.device_app_state = device_app_state
        if self.device_app_state is None:
            raise ValueError("device_app_state is required before starting the daemon")
        if self.is_running:
            return self
        self._stop_event.clear()
        device_id = getattr(self.device_app_state, "device_id", "device")
        self._thread = Thread(
            target=self._run,
            name=f"EdgeMemSync-{device_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if bool(getattr(self.device_app_state, "online", False)):
                try:
                    asyncio.run(sync_once(self.device_app_state))
                except Exception:
                    self._logger.exception("background sync failed; pending changes remain retryable")
            self._stop_event.wait(self.interval_s)

    def stop(self, timeout: float | None = None) -> None:
        """Request worker shutdown and wait up to ``timeout`` seconds."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
