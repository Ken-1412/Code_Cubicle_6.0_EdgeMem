from types import SimpleNamespace

import pytest
from qdrant_client import models

from edgemem import embeddings


@pytest.fixture(autouse=True)
def clear_model_caches():
    """Keep cached model objects isolated between offline tests."""
    embeddings._dense_model.cache_clear()
    embeddings._sparse_model.cache_clear()
    yield
    embeddings._dense_model.cache_clear()
    embeddings._sparse_model.cache_clear()


def test_embed_dense_returns_plain_float_lists(monkeypatch):
    """Dense embedding output is a list of Python float vectors."""
    class FakeDenseModel:
        def embed(self, documents):
            self.documents = list(documents)
            return iter(([1, 2.5], [3, 4]))

    model = FakeDenseModel()
    monkeypatch.setattr(embeddings, "_dense_model", lambda: model)

    result = embeddings.embed_dense(text for text in ("bearing noise", "seal leak"))

    assert model.documents == ["bearing noise", "seal leak"]
    assert result == [[1.0, 2.5], [3.0, 4.0]]
    assert all(type(value) is float for vector in result for value in vector)


def test_embed_sparse_returns_batched_qdrant_sparse_vectors(monkeypatch):
    """Sparse output is a batch of native Qdrant SparseVector models."""
    sparse = [
        SimpleNamespace(indices=(3, 17), values=(0.25, 0.75)),
        SimpleNamespace(indices=(5,), values=(1.5,)),
    ]

    class FakeSparseModel:
        def embed(self, documents):
            assert list(documents) == ["motor vibration", "bearing noise"]
            return iter(sparse)

    monkeypatch.setattr(embeddings, "_sparse_model", FakeSparseModel)

    assert embeddings.embed_sparse(["motor vibration", "bearing noise"]) == [
        models.SparseVector(indices=[3, 17], values=[0.25, 0.75]),
        models.SparseVector(indices=[5], values=[1.5]),
    ]


def test_empty_inputs_do_not_initialize_models(monkeypatch):
    """Empty input returns immediately without loading local model files."""
    def unexpected_model_load():
        raise AssertionError("empty batches must not load an embedding model")

    monkeypatch.setattr(embeddings, "_dense_model", unexpected_model_load)
    monkeypatch.setattr(embeddings, "_sparse_model", unexpected_model_load)

    assert embeddings.embed_dense([]) == []
    assert embeddings.embed_sparse([]) == []


def test_model_factories_use_configured_names_and_cache_instances(monkeypatch):
    """Each lazy model factory uses the configured model and caches it."""
    dense_calls = []
    sparse_calls = []
    dense_instance = object()
    sparse_instance = object()

    class FakeDenseEmbedding:
        def __new__(cls, *, model_name):
            dense_calls.append(model_name)
            return dense_instance

    class FakeSparseEmbedding:
        def __new__(cls, *, model_name):
            sparse_calls.append(model_name)
            return sparse_instance

    monkeypatch.setattr(embeddings, "TextEmbedding", FakeDenseEmbedding)
    monkeypatch.setattr(embeddings, "SparseTextEmbedding", FakeSparseEmbedding)

    assert embeddings._dense_model() is dense_instance
    assert embeddings._dense_model() is dense_instance
    assert embeddings._sparse_model() is sparse_instance
    assert embeddings._sparse_model() is sparse_instance
    assert dense_calls == ["BAAI/bge-small-en-v1.5"]
    assert sparse_calls == ["Qdrant/bm25"]
