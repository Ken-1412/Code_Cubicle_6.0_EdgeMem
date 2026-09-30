"""Dense and sparse embeddings for offline maintenance-log search."""

from collections.abc import Iterable
from functools import lru_cache
from typing import Protocol

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from edgemem.config import get_settings

SparseVector = models.SparseVector


class _SparseEmbeddingLike(Protocol):
    """The sparse index/value fields exposed by FastEmbed results."""

    indices: Iterable[int]
    values: Iterable[float]


@lru_cache(maxsize=1)
def _dense_model() -> TextEmbedding:
    """Create and cache the configured dense embedding model on first use."""
    return TextEmbedding(model_name=get_settings().dense_model)


@lru_cache(maxsize=1)
def _sparse_model() -> SparseTextEmbedding:
    """Create and cache the configured sparse embedding model on first use."""
    return SparseTextEmbedding(model_name=get_settings().sparse_model)


def embed_dense(texts: Iterable[str]) -> list[list[float]]:
    """Encode texts as batches of dense vectors represented by Python float lists."""
    documents = list(texts)
    if not documents:
        return []
    return [[float(value) for value in vector] for vector in _dense_model().embed(documents)]


def embed_sparse(texts: Iterable[str]) -> list[SparseVector]:
    """Encode texts as native Qdrant sparse-vector models in input order."""
    documents = list(texts)
    if not documents:
        return []
    return [
        models.SparseVector(
            indices=[int(index) for index in vector.indices],
            values=[float(value) for value in vector.values],
        )
        for vector in _sparse_model().embed(documents)
    ]
