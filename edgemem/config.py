"""Configuration for the offline maintenance-memory application."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings resolved for one process."""

    project_root: Path
    data_dir: Path
    device_id: str
    cloud_url: str
    qdrant_path: Path
    sqlite_path: Path
    seed_path: Path
    collection_name: str = "maintenance_logs"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dimensions: int = 384
    dense_distance: str = "Cosine"
    sparse_model: str = "Qdrant/bm25"


def _resolve_path(value: str | None, default: Path, root: Path) -> Path:
    """Resolve an optional configured path against the project root."""
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else root / path


def get_settings() -> Settings:
    """Build settings from environment variables and stable product defaults."""
    root_value = os.getenv("EDGEMEM_ROOT")
    if root_value:
        configured_root = Path(root_value).expanduser()
        if not configured_root.is_absolute():
            configured_root = Path.cwd() / configured_root
        project_root = configured_root.resolve()
    else:
        project_root = Path(__file__).resolve().parents[1]

    data_dir = _resolve_path(os.getenv("EDGEMEM_DATA_DIR"), Path("data"), project_root)
    qdrant_path = _resolve_path(
        os.getenv("EDGEMEM_QDRANT_PATH"), data_dir / "qdrant", project_root
    )
    sqlite_path = _resolve_path(
        os.getenv("EDGEMEM_SQLITE_PATH"), data_dir / "changes.sqlite3", project_root
    )
    seed_path = _resolve_path(
        os.getenv("EDGEMEM_SEED_PATH"), data_dir / "seed_logs.jsonl", project_root
    )

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        device_id=os.getenv("EDGEMEM_DEVICE_ID", "device-a"),
        cloud_url=os.getenv("EDGEMEM_CLOUD_URL", "http://127.0.0.1:9000"),
        qdrant_path=qdrant_path,
        sqlite_path=sqlite_path,
        seed_path=seed_path,
    )
