"""Exercise FastEmbed dense/sparse models and local Qdrant queries."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
COLLECTION_NAME = "smoke_libs"
DEFAULT_QDRANT_PATH = "./data/smoke_qdrant"
SENTENCES = (
    "Hydraulic pump P-17 is leaking oil around the output seal.",
    "Compressor C-9 vibrates heavily after its motor bearing was replaced.",
    "Conveyor belt CV-4 stopped because the emergency stop was engaged.",
)
SENTENCE_IDS = ("pump-seal", "compressor-bearing", "conveyor-stop")
EQUIPMENT = ("hydraulic pump", "compressor", "conveyor belt")
QUERY_TEXT = "hydraulic pump P-17 oil leak near output seal"


class DenseModel(Protocol):
    """Minimal FastEmbed dense model interface used by the harness."""

    def embed(self, documents: Iterable[str]) -> Iterable[Iterable[float]]:
        """Return one dense vector per document."""
        ...


class SparseEmbedding(Protocol):
    """Minimal FastEmbed sparse result interface."""

    indices: Iterable[int]
    values: Iterable[float]


class SparseModel(Protocol):
    """Minimal FastEmbed sparse model interface used by the harness."""

    def embed(self, documents: Iterable[str]) -> Iterable[SparseEmbedding]:
        """Return one sparse embedding per document."""
        ...


def _dense_vectors(model: DenseModel, texts: Iterable[str]) -> list[list[float]]:
    """Convert dense model outputs to Qdrant-compatible Python floats."""
    return [[float(value) for value in vector] for vector in model.embed(texts)]


def _sparse_vectors(model: SparseModel, texts: Iterable[str]) -> list[models.SparseVector]:
    """Convert FastEmbed indices/values into Qdrant SparseVector objects."""
    return [
        models.SparseVector(
            indices=[int(index) for index in embedding.indices],
            values=[float(value) for value in embedding.values],
        )
        for embedding in model.embed(texts)
    ]


def _top_result(points: list[Any], channel: str) -> dict[str, Any]:
    """Extract a compact labeled result from one Qdrant query response."""
    if not points:
        raise RuntimeError(f"{channel} query returned no points")
    point = points[0]
    return {
        "id": point.payload["id"],
        "text": point.payload["text"],
        "score": float(point.score),
    }


def run_smoke(
    qdrant_path: str | Path = DEFAULT_QDRANT_PATH,
    *,
    dense_model: DenseModel | None = None,
    sparse_model: SparseModel | None = None,
) -> dict[str, dict[str, Any]]:
    """Create vectors, index three points, and return dense/sparse/RRF top hits."""
    dense_encoder = dense_model or TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_encoder = sparse_model or SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    dense_vectors = _dense_vectors(dense_encoder, SENTENCES)
    sparse_vectors = _sparse_vectors(sparse_encoder, SENTENCES)
    dense_query = _dense_vectors(dense_encoder, [QUERY_TEXT])[0]
    sparse_query = _sparse_vectors(sparse_encoder, [QUERY_TEXT])[0]
    if not (
        len(dense_vectors) == len(sparse_vectors) == len(SENTENCES)
        and len(dense_query) == 384
    ):
        raise RuntimeError("FastEmbed returned an unexpected embedding batch or dimension")

    client = QdrantClient(path=str(qdrant_path))
    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="equipment",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=index + 1,
                    vector={"dense": dense_vectors[index], "sparse": sparse_vectors[index]},
                    payload={
                        "id": SENTENCE_IDS[index],
                        "text": SENTENCES[index],
                        "equipment": EQUIPMENT[index],
                    },
                )
                for index in range(len(SENTENCES))
            ],
            wait=True,
        )
        dense_points = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_query,
            using="dense",
            limit=3,
            with_payload=True,
        ).points
        sparse_points = client.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_query,
            using="sparse",
            limit=3,
            with_payload=True,
        ).points
        hybrid_points = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_query, using="dense", limit=3),
                models.Prefetch(query=sparse_query, using="sparse", limit=3),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=3,
            with_payload=True,
        ).points
        return {
            "dense_top": _top_result(dense_points, "dense"),
            "sparse_top": _top_result(sparse_points, "sparse"),
            "hybrid_top": _top_result(hybrid_points, "hybrid RRF"),
        }
    finally:
        client.close()


def main() -> int:
    """Run the real-model smoke test and print each top result."""
    results = run_smoke()
    print(f"Dense top: {results['dense_top']['id']} (score={results['dense_top']['score']:.4f})")
    print(f"Sparse top: {results['sparse_top']['id']} (score={results['sparse_top']['score']:.4f})")
    print(f"Hybrid RRF top: {results['hybrid_top']['id']} (score={results['hybrid_top']['score']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
