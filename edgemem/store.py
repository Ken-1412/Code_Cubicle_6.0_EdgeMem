"""Local Qdrant storage for canonical maintenance observations and hybrid search."""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

from qdrant_client import QdrantClient, models

from edgemem.config import get_settings
from edgemem.embeddings import embed_dense, embed_sparse

SparseVectorInput: TypeAlias = (
    models.SparseVector
    | Mapping[str, Sequence[int] | Sequence[float]]
    | tuple[Sequence[int], Sequence[float]]
)
DenseEmbedder: TypeAlias = Callable[[Sequence[str]], Sequence[Sequence[float]]]
SparseEmbedder: TypeAlias = Callable[[Sequence[str]], Sequence[SparseVectorInput]]
SearchMode: TypeAlias = Literal["hybrid", "dense", "sparse"]


class SearchResult(dict[str, Any]):
    """Dictionary-shaped search hit with attribute access for existing API callers."""

    def __init__(
        self,
        *,
        id: str | int,
        payload: Mapping[str, Any],
        dense_score: float | None,
        sparse_score: float | None,
        dense_rank: int | None = None,
        sparse_rank: int | None = None,
        score: float | None = None,
        rrf_score: float | None = None,
    ) -> None:
        if score is None:
            score = rrf_score if rrf_score is not None else 0.0
        super().__init__(
            id=id,
            score=float(score),
            payload=dict(payload),
            dense_score=dense_score,
            sparse_score=sparse_score,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            # Retain the field consumed by the earlier device API.
            rrf_score=float(score),
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class LocalStore:
    """Persist canonical maintenance points in a Qdrant local collection."""

    VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "bm25"
    _SEVERITY_ORDER = ("low", "medium", "high", "critical")
    _METADATA_FIELDS = {"id", "device_id", "updated_at", "field_ts"}

    def __init__(
        self,
        path: str | Path | None = None,
        collection: str = "memory",
        *,
        collection_name: str | None = None,
        dense_dimensions: int | None = None,
        dense_embedder: DenseEmbedder = embed_dense,
        sparse_embedder: SparseEmbedder = embed_sparse,
        device_id: str | None = None,
    ) -> None:
        """Open local Qdrant storage and create the requested collection if absent."""
        settings = get_settings()
        self.collection_name = collection_name or collection
        self.dense_dimensions = (
            settings.dense_dimensions if dense_dimensions is None else dense_dimensions
        )
        if self.dense_dimensions < 1:
            raise ValueError("dense_dimensions must be positive")
        self.device_id = device_id or settings.device_id
        if not self.device_id:
            raise ValueError("device_id must be a non-empty string")
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.logger = logging.getLogger(__name__)
        self.quantization_enabled = False
        self.payload_index_fields: set[str] = set()

        storage_path = Path(path) if path is not None else settings.qdrant_path
        self.client = QdrantClient(path=str(storage_path))
        try:
            if not self.client.collection_exists(self.collection_name):
                self._create_collection()
            else:
                # This class deliberately uses Qdrant local mode. That backend does not
                # execute vector quantization even if a config object was serialized.
                self.logger.warning(
                    "Scalar INT8 quantization is unavailable in Qdrant local mode; "
                    "the existing collection is used without claiming quantization."
                )
            self._create_payload_indexes()
        except Exception:
            self.client.close()
            raise

    def _create_collection(self) -> None:
        """Try INT8, then recreate without it when local Qdrant does not apply it."""
        quantization = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
        )
        dense_with_quantization = models.VectorParams(
            size=self.dense_dimensions,
            distance=models.Distance.COSINE,
            quantization_config=quantization,
        )
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={self.VECTOR_NAME: dense_with_quantization},
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
        except Exception as error:
            self.logger.warning(
                "Qdrant local could not create the INT8-configured collection (%s); "
                "retrying without scalar quantization.",
                error,
            )
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
            self._create_unquantized_collection()
            return

        # qdrant-client 1.19.1 local mode accepts and serializes VectorParams' config,
        # but its local collection implementation has no quantization execution path.
        # Remove the probe collection so the persisted schema does not falsely imply it
        # is active, then create the actual collection without a quantization setting.
        info = self.client.get_collection(self.collection_name)
        if info.config.quantization_config is None:
            self.client.delete_collection(collection_name=self.collection_name)
            self.logger.warning(
                "Scalar INT8 quantization was attempted but is not implemented by "
                "Qdrant local mode; recreating the collection without quantization."
            )
            self._create_unquantized_collection()
            self.quantization_enabled = False
            return
        self.quantization_enabled = True

    def _create_unquantized_collection(self) -> None:
        """Create the named dense and sparse collection without unsupported settings."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.VECTOR_NAME: models.VectorParams(
                    size=self.dense_dimensions,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

    def _create_payload_indexes(self) -> None:
        """Skip indexes known to be no-ops in local mode and state the limitation."""
        self.payload_index_fields.clear()
        self.logger.warning(
            "Skipping Qdrant payload-index creation for equipment, severity, error_code, "
            "and deleted: payload indexes have no effect in local mode; filters remain "
            "correct but are not index-accelerated."
        )

    def upsert(self, records: Sequence[Mapping[str, Any]]) -> int:
        """Embed and upsert a batch of records, filling canonical metadata as needed."""
        if not records:
            return 0
        normalized = [self._canonicalize_record(record) for record in records]
        texts = [record["text"] for record in normalized]
        dense_vectors = list(self.dense_embedder(texts))
        sparse_vectors = list(self.sparse_embedder(texts))
        if len(dense_vectors) != len(normalized) or len(sparse_vectors) != len(normalized):
            raise ValueError("embedding counts must match the record count")
        return self.upsert_many(normalized, dense_vectors, sparse_vectors)

    def upsert_many(
        self,
        records: Sequence[Mapping[str, Any]],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVectorInput],
    ) -> int:
        """Write a batch of canonical payloads with paired dense and sparse vectors."""
        if not (len(records) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError("records and embedding batches must have the same number of items")
        if not records:
            return 0

        normalized = [self._canonicalize_record(record) for record in records]
        points: list[models.PointStruct] = []
        for record, dense_vector, sparse_vector in zip(
            normalized, dense_vectors, sparse_vectors, strict=True
        ):
            dense_values = [float(value) for value in dense_vector]
            if len(dense_values) != self.dense_dimensions:
                raise ValueError(
                    f"dense vector dimension {len(dense_values)} does not match "
                    f"collection dimension {self.dense_dimensions}"
                )
            indices, values = self._sparse_components(sparse_vector)
            points.append(
                models.PointStruct(
                    id=record["id"],
                    vector={
                        self.VECTOR_NAME: dense_values,
                        self.SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=indices,
                            values=values,
                        ),
                    },
                    payload=record,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def _canonicalize_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return a validated canonical payload with device-scoped field timestamps."""
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        payload = dict(record)
        identifier = payload.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("record id must be a non-empty string")
        if not isinstance(payload.get("text"), str) or not payload["text"].strip():
            raise ValueError("record text must be a non-empty string")
        if not isinstance(payload.get("equipment"), str) or not payload["equipment"].strip():
            raise ValueError("record equipment must be a non-empty string")
        if payload.get("severity") not in self._SEVERITY_ORDER:
            raise ValueError("record severity is invalid")
        if "error_code" not in payload:
            payload["error_code"] = None
        if payload["error_code"] is not None and not isinstance(payload["error_code"], str):
            raise ValueError("record error_code must be a string or null")

        device_id = payload.get("device_id", self.device_id)
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("record device_id must be a non-empty string")
        payload["device_id"] = device_id
        updated_at = payload.get("updated_at", time.time())
        if not isinstance(updated_at, (int, float)) or isinstance(updated_at, bool):
            raise ValueError("record updated_at must be numeric")
        payload["updated_at"] = float(updated_at)
        if "deleted" not in payload:
            payload["deleted"] = False
        if not isinstance(payload["deleted"], bool):
            raise ValueError("record deleted must be a boolean")

        field_ts_value = payload.get("field_ts", {})
        if not isinstance(field_ts_value, Mapping):
            raise ValueError("record field_ts must be an object")
        field_ts = dict(field_ts_value)
        for field in payload:
            if field in self._METADATA_FIELDS:
                continue
            stamp = field_ts.get(field)
            if stamp is None:
                field_ts[field] = [payload["updated_at"], device_id]
                continue
            if (
                not isinstance(stamp, (list, tuple))
                or len(stamp) != 2
                or not isinstance(stamp[0], (int, float))
                or isinstance(stamp[0], bool)
                or not isinstance(stamp[1], str)
            ):
                raise ValueError(f"invalid field timestamp for {field}")
            field_ts[field] = [float(stamp[0]), stamp[1]]
        payload["field_ts"] = field_ts
        return payload

    @staticmethod
    def _sparse_components(
        vector: SparseVectorInput,
    ) -> tuple[list[int], list[float]]:
        """Normalize Qdrant models, index/value mappings, or index/value pairs."""
        if isinstance(vector, models.SparseVector):
            indices, values = vector.indices, vector.values
        elif isinstance(vector, Mapping):
            try:
                indices, values = vector["indices"], vector["values"]
            except KeyError as error:
                raise ValueError("sparse vector requires indices and values") from error
        elif isinstance(vector, tuple) and len(vector) == 2:
            indices, values = vector
        else:
            raise TypeError("sparse vector must be a Qdrant SparseVector or indices/values pair")
        normalized_indices = [int(index) for index in indices]
        normalized_values = [float(value) for value in values]
        if len(normalized_indices) != len(normalized_values):
            raise ValueError("sparse vector indices and values must have the same length")
        return normalized_indices, normalized_values

    def get(self, id: str | int) -> dict[str, Any] | None:
        """Retrieve one payload by point ID, returning ``None`` when it is absent."""
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = dict(points[0].payload or {})
        payload.setdefault("id", str(points[0].id))
        return payload

    def update_payload(
        self,
        id: str | int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Merge payload fields without embedding or replacing stored vectors."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        current = self.get(id)
        if current is None:
            return None
        if not payload:
            return current
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=dict(payload),
            points=[id],
            wait=True,
        )
        return self.get(id)

    def soft_delete(self, id: str | int) -> bool:
        """Set the point's tombstone without removing its vectors or payload."""
        current = self.get(id)
        if current is None:
            return False
        payload = self._canonicalize_record(current)
        timestamp = time.time()
        payload["deleted"] = True
        payload["updated_at"] = timestamp
        payload["field_ts"]["deleted"] = [timestamp, payload["device_id"]]
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[id],
            wait=True,
        )
        return True

    def count(self) -> int:
        """Return an exact count of stored points, including tombstones."""
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def all_points(self) -> list[dict[str, Any]]:
        """Return all payloads, including tombstones, without loading vector data."""
        result: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                payload.setdefault("id", str(point.id))
                result.append(payload)
            if next_offset is None:
                break
            offset = next_offset
        return result

    def search(
        self,
        query: str | Sequence[float] | None = None,
        limit: int | SparseVectorInput = 5,
        filters: Mapping[str, Any] | None = None,
        mode: SearchMode = "hybrid",
        *,
        query_dense: Sequence[float] | None = None,
        query_sparse: SparseVectorInput | None = None,
    ) -> list[SearchResult]:
        """Search by text or supplied vectors and return ranked hits with score detail."""
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError("mode must be 'hybrid', 'dense', or 'sparse'")

        # Preserve the vector-based API used by the first device-API implementation.
        if query is not None and not isinstance(query, str):
            if query_dense is not None:
                raise ValueError("provide either query or query_dense, not both")
            query_dense = query
            if self._is_sparse_input(limit):
                if query_sparse is not None:
                    raise ValueError("sparse query vector was provided more than once")
                query_sparse = limit
                limit = 10
        elif isinstance(query, str):
            if not query.strip():
                raise ValueError("query text must not be empty")
            if query_dense is not None or query_sparse is not None:
                raise ValueError("text query cannot be combined with explicit vectors")
            if mode in {"hybrid", "dense"}:
                dense_batch = list(self.dense_embedder([query]))
                if len(dense_batch) != 1:
                    raise ValueError("dense query embedding must return exactly one vector")
                query_dense = dense_batch[0]
            if mode in {"hybrid", "sparse"}:
                sparse_batch = list(self.sparse_embedder([query]))
                if len(sparse_batch) != 1:
                    raise ValueError("sparse query embedding must return exactly one vector")
                query_sparse = sparse_batch[0]

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if query_dense is None and query_sparse is None:
            raise ValueError("at least one query vector is required")
        if mode == "dense" and query_dense is None:
            raise ValueError("dense mode requires a dense query vector")
        if mode == "sparse" and query_sparse is None:
            raise ValueError("sparse mode requires a sparse query vector")

        normalized_dense = (
            [float(value) for value in query_dense] if query_dense is not None else None
        )
        normalized_sparse = (
            self._sparse_vector(query_sparse) if query_sparse is not None else None
        )
        if normalized_sparse is not None and not normalized_sparse.indices:
            normalized_sparse = None
        query_filter = self._build_filter(filters)

        if mode == "dense":
            if normalized_dense is None:
                raise ValueError("dense mode requires a dense query vector")
            points = self._query_channel(
                normalized_dense, self.VECTOR_NAME, query_filter, limit
            )
            return self._single_channel_results(points, "dense")
        if mode == "sparse":
            if normalized_sparse is None:
                raise ValueError("sparse mode requires a non-empty sparse query vector")
            points = self._query_channel(
                normalized_sparse, self.SPARSE_VECTOR_NAME, query_filter, limit
            )
            return self._single_channel_results(points, "sparse")

        if normalized_dense is None:
            if normalized_sparse is None:
                return []
            points = self._query_channel(
                normalized_sparse, self.SPARSE_VECTOR_NAME, query_filter, limit
            )
            return self._single_channel_results(points, "sparse")
        if normalized_sparse is None:
            points = self._query_channel(
                normalized_dense, self.VECTOR_NAME, query_filter, limit
            )
            return self._single_channel_results(points, "dense")

        candidate_limit = max(limit, 10)
        dense_points = self._query_channel(
            normalized_dense, self.VECTOR_NAME, query_filter, candidate_limit
        )
        sparse_points = self._query_channel(
            normalized_sparse, self.SPARSE_VECTOR_NAME, query_filter, candidate_limit
        )
        fused_points = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=normalized_dense,
                    using=self.VECTOR_NAME,
                    filter=query_filter,
                    limit=candidate_limit,
                ),
                models.Prefetch(
                    query=normalized_sparse,
                    using=self.SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=candidate_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        dense_scores, dense_ranks = self._score_and_rank(dense_points)
        sparse_scores, sparse_ranks = self._score_and_rank(sparse_points)
        return [
            SearchResult(
                id=point.id,
                score=float(point.score),
                payload=point.payload or {},
                dense_score=dense_scores.get(point.id),
                sparse_score=sparse_scores.get(point.id),
                dense_rank=dense_ranks.get(point.id),
                sparse_rank=sparse_ranks.get(point.id),
            )
            for point in fused_points
        ]

    @staticmethod
    def _is_sparse_input(value: Any) -> bool:
        return (
            isinstance(value, (Mapping, models.SparseVector))
            or isinstance(value, tuple) and len(value) == 2
        )

    @classmethod
    def _sparse_vector(cls, value: SparseVectorInput) -> models.SparseVector:
        indices, values = cls._sparse_components(value)
        return models.SparseVector(indices=indices, values=values)

    def _build_filter(self, filters: Mapping[str, Any] | None) -> models.Filter:
        """Build Qdrant-side filter conditions and always exclude tombstones."""
        must: list[Any] = []
        if filters is not None:
            if not isinstance(filters, Mapping):
                raise TypeError("filters must be a mapping")
            for key, value in filters.items():
                if key == "equipment":
                    must.append(self._match_condition("equipment", value))
                elif key == "severity":
                    values = value if isinstance(value, (list, tuple, set)) else [value]
                    if any(item not in self._SEVERITY_ORDER for item in values):
                        raise ValueError("severity filter contains an unknown severity")
                    must.append(self._match_condition("severity", values))
                elif key == "severity_gte":
                    if value not in self._SEVERITY_ORDER:
                        raise ValueError("severity_gte must be low, medium, high, or critical")
                    minimum = self._SEVERITY_ORDER.index(value)
                    accepted = list(self._SEVERITY_ORDER[minimum:])
                    must.append(self._match_condition("severity", accepted))
                elif key == "error_code":
                    if value is None:
                        must.append(models.IsNullCondition(is_null=models.PayloadField(key="error_code")))
                    else:
                        must.append(self._match_condition("error_code", value))
                elif key == "deleted":
                    if value is True:
                        # Search is intentionally live-record-only in every mode.
                        must.append(models.FieldCondition(
                            key="deleted", match=models.MatchValue(value=True)
                        ))
                    elif value is not False:
                        raise ValueError("deleted filter must be a boolean")
                else:
                    raise ValueError(f"unsupported search filter: {key}")
        return models.Filter(
            must=must or None,
            must_not=[
                models.FieldCondition(
                    key="deleted",
                    match=models.MatchValue(value=True),
                )
            ],
        )

    @staticmethod
    def _match_condition(key: str, value: Any) -> models.FieldCondition:
        """Create exact or any-of payload match conditions for filter values."""
        if isinstance(value, (list, tuple, set)):
            options = list(value)
            if not options:
                raise ValueError(f"filter {key} must not be an empty collection")
            return models.FieldCondition(key=key, match=models.MatchAny(any=options))
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _query_channel(
        self,
        query: list[float] | models.SparseVector,
        using: str,
        query_filter: models.Filter,
        limit: int,
    ) -> list[Any]:
        """Run one named-vector branch while preserving its raw score and rank."""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            using=using,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

    @staticmethod
    def _score_and_rank(points: Sequence[Any]) -> tuple[dict[Any, float], dict[Any, int]]:
        """Map point IDs to raw channel scores and one-based branch ranks."""
        return (
            {point.id: float(point.score) for point in points},
            {point.id: rank for rank, point in enumerate(points, start=1)},
        )

    @classmethod
    def _single_channel_results(
        cls,
        points: Sequence[Any],
        channel: Literal["dense", "sparse"],
    ) -> list[SearchResult]:
        """Format branch-only results while leaving the other channel absent."""
        results = []
        for rank, point in enumerate(points, start=1):
            raw_score = float(point.score)
            results.append(
                SearchResult(
                    id=point.id,
                    score=raw_score,
                    payload=point.payload or {},
                    dense_score=raw_score if channel == "dense" else None,
                    sparse_score=raw_score if channel == "sparse" else None,
                    dense_rank=rank if channel == "dense" else None,
                    sparse_rank=rank if channel == "sparse" else None,
                )
            )
        return results

    def close(self) -> None:
        """Close the local Qdrant client and release its storage lock."""
        self.client.close()
