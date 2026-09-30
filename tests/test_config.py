from pathlib import Path

from edgemem.config import Settings, get_settings


def clear_config_env(monkeypatch):
    """Remove config overrides so defaults are deterministic."""
    for name in (
        "EDGEMEM_ROOT",
        "EDGEMEM_DATA_DIR",
        "EDGEMEM_DEVICE_ID",
        "EDGEMEM_CLOUD_URL",
        "EDGEMEM_QDRANT_PATH",
        "EDGEMEM_SQLITE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_get_settings_uses_project_relative_paths(monkeypatch):
    """Default storage paths resolve beneath the project root."""
    clear_config_env(monkeypatch)

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.project_root == Path(__file__).resolve().parents[1]
    assert settings.data_dir == settings.project_root / "data"
    assert settings.qdrant_path == settings.data_dir / "qdrant"
    assert settings.sqlite_path == settings.data_dir / "changes.sqlite3"
    assert settings.seed_path == settings.data_dir / "seed_logs.jsonl"
    assert settings.cloud_url == "http://127.0.0.1:9000"


def test_get_settings_reads_environment_overrides(monkeypatch, tmp_path):
    """Environment values override defaults without creating directories."""
    root = tmp_path / "project"
    data_dir = tmp_path / "custom-data"
    qdrant_path = tmp_path / "qdrant-store"
    sqlite_path = tmp_path / "changes.db"
    monkeypatch.setenv("EDGEMEM_ROOT", str(root))
    monkeypatch.setenv("EDGEMEM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("EDGEMEM_DEVICE_ID", "pump-room-tablet")
    monkeypatch.setenv("EDGEMEM_CLOUD_URL", "http://cloud.local:8001")
    monkeypatch.setenv("EDGEMEM_QDRANT_PATH", str(qdrant_path))
    monkeypatch.setenv("EDGEMEM_SQLITE_PATH", str(sqlite_path))

    settings = get_settings()

    assert settings.project_root == root
    assert settings.data_dir == data_dir
    assert settings.device_id == "pump-room-tablet"
    assert settings.cloud_url == "http://cloud.local:8001"
    assert settings.qdrant_path == qdrant_path
    assert settings.sqlite_path == sqlite_path
    assert not data_dir.exists()


def test_get_settings_resolves_relative_overrides_from_project_root(monkeypatch, tmp_path):
    """Relative configured paths are anchored to the configured project root."""
    root = tmp_path / "project"
    monkeypatch.setenv("EDGEMEM_ROOT", str(root))
    monkeypatch.setenv("EDGEMEM_DATA_DIR", "runtime-data")
    monkeypatch.setenv("EDGEMEM_QDRANT_PATH", "vectors")
    monkeypatch.setenv("EDGEMEM_SQLITE_PATH", "logs/changes.sqlite3")

    settings = get_settings()

    assert settings.data_dir == root / "runtime-data"
    assert settings.qdrant_path == root / "vectors"
    assert settings.sqlite_path == root / "logs/changes.sqlite3"
    assert settings.seed_path == root / "runtime-data/seed_logs.jsonl"


def test_embedding_configuration_matches_fixed_product_stack():
    """Model and vector metadata match the required offline embedding stack."""
    settings = get_settings()

    assert settings.dense_model == "BAAI/bge-small-en-v1.5"
    assert settings.dense_dimensions == 384
    assert settings.dense_distance == "Cosine"
    assert settings.sparse_model == "Qdrant/bm25"
    assert settings.collection_name == "maintenance_logs"
