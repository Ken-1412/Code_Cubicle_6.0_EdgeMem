"""Per-device SQLite change deltas and sync acknowledgements."""

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from edgemem.config import get_settings

ChangeOp = Literal["upsert", "update", "delete"]
_METADATA_FIELDS = {"id", "updated_at", "device_id", "field_ts"}
_ALLOWED_OPS = {"upsert", "update", "delete"}


@dataclass(frozen=True, slots=True)
class Change:
    """One ordered point delta with per-field (timestamp, device ID) stamps."""

    seq: int
    point_id: str
    op: ChangeOp
    fields: dict[str, Any]
    field_ts: dict[str, list[float | str]]
    ts: float
    device_id: str
    synced: bool = False

    @property
    def delta(self) -> dict[str, Any]:
        """Return the transport-neutral, stamped delta representation."""
        return {
            "point_id": self.point_id,
            "op": self.op,
            "fields": {
                field: {"value": deepcopy(value), "stamp": list(self.field_ts[field])}
                for field, value in self.fields.items()
            },
            "ts": self.ts,
            "device_id": self.device_id,
        }

    @property
    def record(self) -> dict[str, Any]:
        """Reconstruct a canonical snapshot when the event contains a full upsert."""
        fields = deepcopy(self.fields)
        stamps = deepcopy(self.field_ts)
        if "deleted" not in fields:
            fields["deleted"] = False
            stamps.setdefault("deleted", [self.ts, self.device_id])
        return {
            "id": self.point_id,
            **fields,
            "device_id": self.device_id,
            "updated_at": self.ts,
            "field_ts": stamps,
        }


class ChangeLog:
    """Append-only SQLite deltas for one device, with ordered acknowledgement."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path if path is not None else get_settings().sqlite_path
        self._database_path = str(configured_path)
        if self._database_path != ":memory:":
            Path(self._database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            self._database_path,
            timeout=5.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock, self._connection:
            self._connection.execute("PRAGMA foreign_keys = ON")
            if self._database_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create the delta schema, migrating the earlier snapshot schema if present."""
        rows = self._connection.execute("PRAGMA table_info(changes)").fetchall()
        columns = {row["name"] for row in rows}
        if columns and "fields_json" not in columns:
            if "record_json" not in columns:
                raise RuntimeError("unsupported existing changes table schema")
            legacy_rows = self._connection.execute(
                "SELECT seq, record_json, synced FROM changes ORDER BY seq"
            ).fetchall()
            self._connection.execute("DROP INDEX IF EXISTS idx_changes_sync_seq")
            self._connection.execute("ALTER TABLE changes RENAME TO changes_legacy")
            self._create_table()
            for row in legacy_rows:
                record = json.loads(row["record_json"])
                fields = {
                    key: value for key, value in record.items() if key not in _METADATA_FIELDS
                }
                event_ts = float(record["updated_at"])
                device_id = str(record["device_id"])
                encoded = self._encode_fields(fields, record.get("field_ts"), event_ts, device_id)
                op = "delete" if record.get("deleted") is True else "upsert"
                self._connection.execute(
                    """INSERT INTO changes
                       (seq, point_id, op, fields_json, ts, device_id, synced)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        int(row["seq"]),
                        str(record["id"]),
                        op,
                        encoded,
                        event_ts,
                        device_id,
                        int(row["synced"]),
                    ),
                )
            self._connection.execute("DROP TABLE changes_legacy")
        else:
            self._create_table()
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_changes_sync_seq ON changes (synced, seq)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_changes_point_seq ON changes (point_id, seq)"
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS sync_metadata (
                   key TEXT PRIMARY KEY,
                   value_json TEXT NOT NULL
               )"""
        )

    def _create_table(self) -> None:
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS changes (
                   seq INTEGER PRIMARY KEY AUTOINCREMENT,
                   point_id TEXT NOT NULL,
                   op TEXT NOT NULL CHECK (op IN ('upsert', 'update', 'delete')),
                   fields_json TEXT NOT NULL,
                   ts REAL NOT NULL,
                   device_id TEXT NOT NULL,
                   synced INTEGER NOT NULL DEFAULT 0 CHECK (synced IN (0, 1))
               )"""
        )

    @staticmethod
    def _encode_fields(
        fields: Mapping[str, Any],
        field_ts: Mapping[str, Any] | None,
        ts: float,
        device_id: str,
    ) -> str:
        """Validate and encode values together with their individual version stamps."""
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("fields must be a non-empty object")
        stamps = field_ts or {}
        if not isinstance(stamps, Mapping):
            raise ValueError("field_ts must be an object")
        encoded: dict[str, dict[str, Any]] = {}
        for field, value in fields.items():
            if not isinstance(field, str) or not field.strip():
                raise ValueError("field names must be non-empty strings")
            if field in _METADATA_FIELDS:
                raise ValueError(f"field {field!r} is changelog metadata, not a point value")
            stamp = stamps.get(field, [ts, device_id])
            if (
                not isinstance(stamp, (list, tuple))
                or len(stamp) != 2
                or not isinstance(stamp[0], (int, float))
                or isinstance(stamp[0], bool)
                or not math.isfinite(float(stamp[0]))
                or not isinstance(stamp[1], str)
                or not stamp[1].strip()
            ):
                raise ValueError(f"invalid timestamp for field {field}")
            encoded[field] = {
                "value": value,
                "stamp": [float(stamp[0]), stamp[1]],
            }
        try:
            return json.dumps(
                encoded,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("change fields must contain JSON-compatible values") from error

    def append(
        self,
        point_id: str | Mapping[str, Any],
        op: ChangeOp | None = None,
        fields: Mapping[str, Any] | None = None,
        ts: float | None = None,
        device_id: str | None = None,
        field_ts: Mapping[str, Any] | None = None,
    ) -> Change:
        """Append a stamped delta; also accepts the legacy full-record call form."""
        if isinstance(point_id, Mapping):
            if any(value is not None for value in (op, fields, ts, device_id, field_ts)):
                raise ValueError("legacy record append cannot be combined with delta arguments")
            record = dict(point_id)
            required = {"id", "device_id", "updated_at", "field_ts", "deleted"}
            missing = sorted(required - record.keys())
            if missing:
                raise ValueError(f"change record missing required fields: {', '.join(missing)}")
            point_id = record["id"]
            op = "delete" if record["deleted"] is True else "upsert"
            fields = {key: value for key, value in record.items() if key not in _METADATA_FIELDS}
            ts = record["updated_at"]
            device_id = record["device_id"]
            field_ts = record["field_ts"]

        if not isinstance(point_id, str) or not point_id.strip():
            raise ValueError("point_id must be a non-empty string")
        if op not in _ALLOWED_OPS:
            raise ValueError("op must be 'upsert', 'update', or 'delete'")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(float(ts)):
            raise ValueError("timestamp must be a finite number")
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a non-empty string")
        if fields is None:
            raise ValueError("fields must be a non-empty object")
        encoded = self._encode_fields(fields, field_ts, float(ts), device_id)

        with self._lock, self._connection:
            cursor = self._connection.execute(
                """INSERT INTO changes (point_id, op, fields_json, ts, device_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (point_id, op, encoded, float(ts), device_id),
            )
            sequence = int(cursor.lastrowid)
        return Change(
            seq=sequence,
            point_id=point_id,
            op=op,
            fields=deepcopy(dict(fields)),
            field_ts=self._decode_field_stamps(encoded),
            ts=float(ts),
            device_id=device_id,
            synced=False,
        )

    @staticmethod
    def _decode_field_stamps(fields_json: str) -> dict[str, list[float | str]]:
        fields = json.loads(fields_json)
        return {field: list(value["stamp"]) for field, value in fields.items()}

    def list_changes(
        self,
        after_seq: int = 0,
        limit: int | None = None,
        pending_only: bool = False,
    ) -> list[Change]:
        """Read events in sequence order with optional cursor, limit, and pending filter."""
        if not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < 0:
            raise ValueError("after_seq must be a non-negative integer")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("limit must be a positive integer")
        query = "SELECT seq, point_id, op, fields_json, ts, device_id, synced FROM changes"
        clauses, parameters = [], []
        if after_seq:
            clauses.append("seq > ?")
            parameters.append(after_seq)
        if pending_only:
            clauses.append("synced = 0")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY seq ASC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [self._row_to_change(row) for row in rows]

    def pending(self, limit: int | None = None) -> list[Change]:
        """Return unsynced events in ascending sequence order."""
        return self.list_changes(limit=limit, pending_only=True)

    def pending_changes(self, limit: int | None = None) -> list[Change]:
        """Compatibility alias used by the existing sync client."""
        return self.pending(limit=limit)

    def pending_count(self) -> int:
        """Return the number of events not yet acknowledged."""
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM changes WHERE synced = 0"
            ).fetchone()
        return int(row["count"])

    def get_sync_metadata(self, key: str, default: Any = None) -> Any:
        """Read one JSON-serializable sync setting, returning its default if absent."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata key must be a non-empty string")
        with self._lock:
            row = self._connection.execute(
                "SELECT value_json FROM sync_metadata WHERE key = ?", (key,)
            ).fetchone()
        return deepcopy(default) if row is None else json.loads(row["value_json"])

    def complete_sync(
        self,
        up_to_seq: int,
        *,
        last_cloud_seq: int,
        last_sync_ts: float,
    ) -> int:
        """Atomically acknowledge local events and persist the successfully applied cursor."""
        for value, name in ((up_to_seq, "up_to_seq"), (last_cloud_seq, "last_cloud_seq")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(last_sync_ts, (int, float))
            or isinstance(last_sync_ts, bool)
            or not math.isfinite(float(last_sync_ts))
        ):
            raise ValueError("last_sync_ts must be finite numeric")

        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE changes SET synced = 1 WHERE seq <= ? AND synced = 0",
                (up_to_seq,),
            )
            for key, value in (
                ("last_cloud_seq", last_cloud_seq),
                ("last_sync_ts", float(last_sync_ts)),
            ):
                self._connection.execute(
                    """INSERT INTO sync_metadata (key, value_json) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json""",
                    (key, json.dumps(value, allow_nan=False)),
                )
            return int(cursor.rowcount)

    def mark_synced(self, up_to_seq: int | Sequence[int]) -> int:
        """Acknowledge all events through a sequence, or a legacy explicit sequence list."""
        with self._lock, self._connection:
            if isinstance(up_to_seq, bool):
                raise ValueError("sequence must be a non-negative integer or integer sequence")
            if isinstance(up_to_seq, int):
                if up_to_seq < 0:
                    raise ValueError("sequence must be non-negative")
                cursor = self._connection.execute(
                    "UPDATE changes SET synced = 1 WHERE seq <= ? AND synced = 0",
                    (up_to_seq,),
                )
                return int(cursor.rowcount)
            if not isinstance(up_to_seq, Sequence) or isinstance(up_to_seq, (str, bytes)):
                raise ValueError("sequence must be a non-negative integer or integer sequence")
            values = list(up_to_seq)
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                raise ValueError("sequence values must be integers")
            if any(value < 1 for value in values):
                raise ValueError("sequence values must be positive")
            if not values:
                return 0
            cursor = self._connection.executemany(
                "UPDATE changes SET synced = 1 WHERE seq = ? AND synced = 0",
                ((value,) for value in sorted(set(values))),
            )
            return int(cursor.rowcount)

    @staticmethod
    def compact(pending: Sequence[Change | Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Collapse pending events to one stamped, minimal delta per point ID."""
        grouped: dict[str, dict[str, Any]] = {}
        for item in pending:
            delta = _normalize_delta(item)
            point_id = delta["point_id"]
            current = grouped.setdefault(
                point_id,
                {"fields": {}, "event_version": (delta["ts"], delta["device_id"]), "ops": set()},
            )
            current["event_version"] = max(
                current["event_version"], (delta["ts"], delta["device_id"])
            )
            current["ops"].add(delta["op"])
            for field, entry in delta["fields"].items():
                previous = current["fields"].get(field)
                entry_key = (
                    tuple(entry["stamp"]),
                    json.dumps(entry["value"], sort_keys=True, allow_nan=False),
                )
                previous_key = (
                    (
                        tuple(previous["stamp"]),
                        json.dumps(previous["value"], sort_keys=True, allow_nan=False),
                    )
                    if previous is not None
                    else None
                )
                if previous_key is None or entry_key > previous_key:
                    current["fields"][field] = deepcopy(entry)

        compacted = []
        for point_id in sorted(grouped):
            item = grouped[point_id]
            fields = item["fields"]
            deleted = fields.get("deleted")
            other_stamps = [
                tuple(value["stamp"])
                for name, value in fields.items()
                if name != "deleted"
            ]
            if (
                deleted is not None
                and deleted["value"] is True
                and other_stamps
                and tuple(deleted["stamp"]) <= max(other_stamps)
            ):
                deleted["value"] = False
            event_ts, event_device = item["event_version"]
            op = (
                "delete"
                if fields.get("deleted", {}).get("value") is True
                else ("upsert" if "upsert" in item["ops"] else "update")
            )
            compacted.append(
                {
                    "point_id": point_id,
                    "op": op,
                    "fields": fields,
                    "ts": event_ts,
                    "device_id": event_device,
                }
            )
        return compacted

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row_to_change(row: sqlite3.Row) -> Change:
        encoded = json.loads(row["fields_json"])
        return Change(
            seq=int(row["seq"]),
            point_id=str(row["point_id"]),
            op=row["op"],
            fields={field: deepcopy(item["value"]) for field, item in encoded.items()},
            field_ts={field: list(item["stamp"]) for field, item in encoded.items()},
            ts=float(row["ts"]),
            device_id=str(row["device_id"]),
            synced=bool(row["synced"]),
        )


def _normalize_delta(item: Change | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a Change instance or supported delta dictionary."""
    if isinstance(item, Change):
        item = item.delta
    elif hasattr(item, "delta"):
        item = item.delta
    if not isinstance(item, Mapping):
        raise ValueError("pending changes must be Change objects or delta mappings")
    point_id = item.get("point_id", item.get("id"))
    op = item.get("op")
    ts = item.get("ts", item.get("updated_at"))
    device_id = item.get("device_id")
    if not isinstance(point_id, str) or not point_id.strip():
        raise ValueError("delta point_id must be a non-empty string")
    if op not in _ALLOWED_OPS:
        raise ValueError("delta op must be 'upsert', 'update', or 'delete'")
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(float(ts)):
        raise ValueError("delta timestamp must be finite")
    if not isinstance(device_id, str) or not device_id:
        raise ValueError("delta device_id must be a string")
    raw_fields = item.get("fields", item.get("changed_fields"))
    if not isinstance(raw_fields, Mapping) or not raw_fields:
        raise ValueError("delta fields must be a non-empty object")
    field_ts = item.get("field_ts", {})
    normalized_fields = {}
    for field, raw in raw_fields.items():
        if not isinstance(field, str) or not field.strip() or field in _METADATA_FIELDS:
            raise ValueError(f"invalid delta field {field!r}")
        if isinstance(raw, Mapping) and "value" in raw and "stamp" in raw:
            value, stamp = raw["value"], raw["stamp"]
        else:
            value = raw
            stamp = field_ts.get(field, [ts, device_id])
        if (
            not isinstance(stamp, (list, tuple))
            or len(stamp) != 2
            or not isinstance(stamp[0], (int, float))
            or isinstance(stamp[0], bool)
            or not math.isfinite(float(stamp[0]))
            or not isinstance(stamp[1], str)
            or not stamp[1].strip()
        ):
            raise ValueError(f"delta has invalid stamp for field {field}")
        normalized_fields[field] = {
            "value": deepcopy(value),
            "stamp": [float(stamp[0]), stamp[1]],
        }
    try:
        json.dumps(normalized_fields, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("delta values must be JSON-compatible") from error
    return {
        "point_id": point_id,
        "op": op,
        "fields": normalized_fields,
        "ts": float(ts),
        "device_id": device_id,
    }
