from scripts.smoke_libs import run_smoke


class FakeDenseModel:
    """Return three orthogonal 384-dimensional points and a matching query."""

    def embed(self, texts):
        documents = list(texts)
        if len(documents) == 3:
            return [
                [1.0] + [0.0] * 383,
                [0.0, 1.0] + [0.0] * 382,
                [0.0, 0.0, 1.0] + [0.0] * 381,
            ]
        return [[1.0] + [0.0] * 383 for _ in documents]


class FakeSparseEmbedding:
    """Expose FastEmbed's sparse index/value interface."""

    def __init__(self, index):
        self.indices = [index]
        self.values = [1.0]


class FakeSparseModel:
    """Return non-overlapping sparse vectors with a matching query vector."""

    def embed(self, texts):
        documents = list(texts)
        if len(documents) == 3:
            return [FakeSparseEmbedding(index) for index in (10, 20, 30)]
        return [FakeSparseEmbedding(10) for _ in documents]


def test_smoke_harness_returns_expected_top_results(tmp_path):
    """Smoke logic indexes both vector types and returns sensible top points."""
    results = run_smoke(
        qdrant_path=tmp_path / "smoke_qdrant",
        dense_model=FakeDenseModel(),
        sparse_model=FakeSparseModel(),
    )

    assert results["dense_top"]["id"] == "pump-seal"
    assert results["sparse_top"]["id"] == "pump-seal"
    assert results["hybrid_top"]["id"] == "pump-seal"
    assert results["dense_top"]["score"] > 0
    assert results["sparse_top"]["score"] > 0


def test_real_fastembed_and_qdrant_api_smoke(tmp_path):
    """Exercise installed FastEmbed and Qdrant calls with the real local models."""
    results = run_smoke(qdrant_path=tmp_path / "real-smoke-qdrant")

    assert results["dense_top"]["id"] == "pump-seal"
    assert results["sparse_top"]["id"] == "pump-seal"
    assert results["hybrid_top"]["id"] == "pump-seal"
    assert all(
        results[channel]["score"] > 0
        for channel in ("dense_top", "sparse_top", "hybrid_top")
    )
