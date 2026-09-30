"""Batched JSONL ingestion into a local EdgeMem Qdrant collection."""

import argparse
import json
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from edgemem.embeddings import embed_dense, embed_sparse
from edgemem.store import LocalStore

_SOURCE_FIELDS = {"id", "text", "equipment", "severity", "error_code"}
_REQUIRED_FIELDS = _SOURCE_FIELDS | {
    "device_id",
    "updated_at",
    "field_ts",
    "deleted",
}
_METADATA_FIELDS = {"id", "device_id", "updated_at", "field_ts"}
_SEVERITIES = {"low", "medium", "high", "critical"}
_DEFAULT_BATCH_SIZE = 32


def _location(source: Path, line_number: int) -> str:
    return f"{source.name}:{line_number}"


def _validate_optional_metadata(value: dict[str, Any], location: str) -> None:
    """Validate metadata when present, before replacing it with this device's stamp."""
    if "device_id" in value and not isinstance(value["device_id"], str):
        raise ValueError(f"{location}: device_id must be a string")
    if "updated_at" in value and (
        not isinstance(value["updated_at"], (int, float))
        or isinstance(value["updated_at"], bool)
    ):
        raise ValueError(f"{location}: updated_at must be numeric")
    if "deleted" in value and not isinstance(value["deleted"], bool):
        raise ValueError(f"{location}: deleted must be a boolean")
    if "field_ts" not in value:
        return
    if not isinstance(value["field_ts"], dict):
        raise ValueError(f"{location}: field_ts must be an object")
    for field_name, timestamp in value["field_ts"].items():
        if (
            not isinstance(field_name, str)
            or not isinstance(timestamp, list)
            or len(timestamp) != 2
            or not isinstance(timestamp[0], (int, float))
            or isinstance(timestamp[0], bool)
            or not isinstance(timestamp[1], str)
        ):
            raise ValueError(f"{location}: field_ts values must be [epoch, device_id]")


def _validate_record(value: Any, source: Path, line_number: int) -> dict[str, Any]:
    """Validate a fully enriched record against the canonical point schema."""
    location = _location(source, line_number)
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected a JSON object")
    missing = sorted(_REQUIRED_FIELDS - value.keys())
    if missing:
        raise ValueError(f"{location}: missing required fields: {', '.join(missing)}")
    if not isinstance(value["id"], str) or not value["id"].strip():
        raise ValueError(f"{location}: id must be a non-empty string")
    try:
        UUID(value["id"])
    except ValueError as error:
        raise ValueError(f"{location}: id must be a UUID string") from error
    if not isinstance(value["text"], str) or not value["text"].strip():
        raise ValueError(f"{location}: text must be a non-empty string")
    if not isinstance(value["severity"], str) or value["severity"] not in _SEVERITIES:
        raise ValueError(f"{location}: severity must be one of {sorted(_SEVERITIES)}")
    if not isinstance(value["equipment"], str) or not value["equipment"].strip():
        raise ValueError(f"{location}: equipment must be a non-empty string")
    if not isinstance(value["device_id"], str) or not value["device_id"].strip():
        raise ValueError(f"{location}: device_id must be a non-empty string")
    if value["error_code"] is not None and not isinstance(value["error_code"], str):
        raise ValueError(f"{location}: error_code must be a string or null")
    if not isinstance(value["updated_at"], (int, float)) or isinstance(
        value["updated_at"], bool
    ):
        raise ValueError(f"{location}: updated_at must be numeric")
    if not isinstance(value["field_ts"], dict):
        raise ValueError(f"{location}: field_ts must be an object")
    for field_name, timestamp in value["field_ts"].items():
        if (
            not isinstance(field_name, str)
            or not isinstance(timestamp, list)
            or len(timestamp) != 2
            or not isinstance(timestamp[0], (int, float))
            or isinstance(timestamp[0], bool)
            or not isinstance(timestamp[1], str)
        ):
            raise ValueError(f"{location}: field_ts values must be [epoch, device_id]")
    if not isinstance(value["deleted"], bool):
        raise ValueError(f"{location}: deleted must be a boolean")
    return value


def _enrich_record(
    value: Any,
    source: Path,
    line_number: int,
    device_id: str,
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Validate a seed row and add canonical metadata for the importing device."""
    location = _location(source, line_number)
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected a JSON object")
    missing = sorted(_SOURCE_FIELDS - value.keys())
    if missing:
        raise ValueError(f"{location}: missing required fields: {', '.join(missing)}")

    # Validate optional canonical values even though ingestion refreshes their stamps.
    _validate_optional_metadata(value, location)
    if not isinstance(value["id"], str) or not value["id"].strip():
        raise ValueError(f"{location}: id must be a non-empty string")
    try:
        UUID(value["id"])
    except ValueError as error:
        raise ValueError(f"{location}: id must be a UUID string") from error
    if not isinstance(value["text"], str) or not value["text"].strip():
        raise ValueError(f"{location}: text must be a non-empty string")
    if not isinstance(value["equipment"], str) or not value["equipment"].strip():
        raise ValueError(f"{location}: equipment must be a non-empty string")
    if not isinstance(value["severity"], str) or value["severity"] not in _SEVERITIES:
        raise ValueError(f"{location}: severity must be one of {sorted(_SEVERITIES)}")
    if value["error_code"] is not None and not isinstance(value["error_code"], str):
        raise ValueError(f"{location}: error_code must be a string or null")

    record = dict(value)
    timestamp = float(clock())
    record["device_id"] = device_id
    record["updated_at"] = timestamp
    record.setdefault("deleted", False)
    record["field_ts"] = {
        field: [timestamp, device_id]
        for field in record
        if field not in _METADATA_FIELDS
    }
    return _validate_record(record, source, line_number)


def _read_jsonl(
    path: Path,
    *,
    device_id: str = "device_a",
    clock: Callable[[], float] = time.time,
) -> list[dict[str, Any]]:
    """Parse, validate, and enrich every nonblank JSONL row before any writes."""
    records = []
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            records.append(
                _enrich_record(parsed, path, line_number, device_id, clock)
            )
    return records


def ingest_jsonl(
    path: str | Path,
    store: Any,
    *,
    device_id: str = "device_a",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    clock: Callable[[], float] = time.time,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Enrich, batch-embed, and upsert a JSONL file; return points upserted."""
    if not isinstance(device_id, str) or not device_id.strip():
        raise ValueError("device_id must be a non-empty string")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    source = Path(path)
    records = _read_jsonl(source, device_id=device_id, clock=clock)
    total = len(records)
    ingested = 0
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        texts = [record["text"] for record in batch]
        dense_vectors = list(embed_dense(texts))
        sparse_vectors = list(embed_sparse(texts))
        if len(dense_vectors) != len(batch) or len(sparse_vectors) != len(batch):
            raise ValueError("embedding counts must match the number of input records")
        written = store.upsert_many(batch, dense_vectors, sparse_vectors)
        if written != len(batch):
            raise RuntimeError(
                f"storage reported {written} writes for a batch of {len(batch)} records"
            )
        ingested += written
        if progress is not None:
            progress(f"ingested {ingested}/{total}")
    return ingested


def _reset_database_directory(path: str | Path) -> None:
    """Remove a selected database path without allowing broad destructive targets."""
    target = Path(path).expanduser()
    if target.is_symlink():
        raise ValueError("refusing to reset a symlink database path")
    resolved = target.resolve()
    project_root = Path(__file__).resolve().parents[1]
    protected = {
        Path(resolved.anchor),
        Path.home().resolve(),
        project_root,
        project_root / "data",
    }
    if resolved in protected or len(resolved.parts) < 3:
        raise ValueError(f"refusing to reset protected or broad path: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for local device ingestion."""
    parser = argparse.ArgumentParser(description="Ingest maintenance JSONL into local Qdrant.")
    parser.add_argument("--db", required=True, type=Path, help="Qdrant local database directory")
    parser.add_argument("--file", required=True, type=Path, help="Input newline-delimited JSON file")
    parser.add_argument("--device-id", default="device_a", help="Device identifier (default: device_a)")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove the selected database directory before ingestion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the batched local importer CLI and print progress and throughput."""
    arguments = build_parser().parse_args(argv)
    if not arguments.device_id.strip():
        raise ValueError("--device-id must be a non-empty string")

    started = time.perf_counter()
    if arguments.reset:
        _reset_database_directory(arguments.db)

    store = LocalStore(
        path=arguments.db,
        collection="memory",
        device_id=arguments.device_id,
    )
    try:
        ingested = ingest_jsonl(
            arguments.file,
            store,
            device_id=arguments.device_id,
            batch_size=_DEFAULT_BATCH_SIZE,
            progress=print,
        )
        total_points = store.count()
    finally:
        store.close()

    elapsed = time.perf_counter() - started
    points_per_second = ingested / elapsed if elapsed > 0 else 0.0
    print(f"total time: {elapsed:.3f}s")
    print(f"points/sec: {points_per_second:.2f}")
    print(f"total points: {total_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
