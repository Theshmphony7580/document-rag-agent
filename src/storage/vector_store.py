import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from schemas import DocumentChunk
from config import get_settings


class QdrantVectorStore:
    """Manages document chunk indexing, deduplication, and vector search in Qdrant."""

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_dim = settings.GEMINI_EMBEDDING_DIM

        # Disk storage preferred over in-memory for persistent document RAG
        if url or settings.QDRANT_URL:
            target_url = url or settings.QDRANT_URL
            target_key = api_key or settings.QDRANT_API_KEY
            self.client = QdrantClient(url=target_url, api_key=target_key)
        else:
            storage_path = path or settings.QDRANT_PATH
            os.makedirs(storage_path, exist_ok=True)
            self.client = QdrantClient(path=storage_path)

        self.ensure_collection()

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ) -> None:
        """Create Qdrant collection and payload indexes if they do not already exist."""
        target_collection = collection_name or self.collection_name
        target_dim = vector_dim or self.vector_dim

        if not self.client.collection_exists(collection_name=target_collection):
            self.client.create_collection(
                collection_name=target_collection,
                vectors_config=models.VectorParams(
                    size=target_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload keyword index on doc_hash for instantaneous deduplication checks
            self.client.create_payload_index(
                collection_name=target_collection,
                field_name="doc_hash",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def has_doc_hash(
        self,
        doc_hash: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Check if any chunks associated with doc_hash already exist in the store."""
        target_collection = collection_name or self.collection_name
        result = self.client.count(
            collection_name=target_collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_hash",
                        match=models.MatchValue(value=doc_hash),
                    )
                ]
            ),
            exact=True,
        )
        return result.count > 0

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        vectors: List[List[float]],
        collection_name: Optional[str] = None,
    ) -> None:
        """Upsert document chunks and their dense embeddings into Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(vectors)} vectors."
            )
        if not chunks:
            return

        target_collection = collection_name or self.collection_name
        points: List[models.PointStruct] = []

        for chunk, vector in zip(chunks, vectors):
            # Deterministic UUID derived from chunk ID for idempotent re-writes
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload: Dict[str, Any] = {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                # Promote critical metadata fields to top-level payload for indexing & filtering
                "doc_hash": chunk.metadata.get("doc_hash"),
                "chunk_id": chunk.metadata.get("chunk_id", chunk.id),
                "source_file": chunk.metadata.get("source_file"),
                "page_numbers": chunk.metadata.get("page_numbers", []),
                "section_path": chunk.metadata.get("section_path", ""),
                "content_type": chunk.metadata.get("content_type", "text"),
            }
            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Perform dense vector cosine similarity search."""
        target_collection = collection_name or self.collection_name

        try:
            # Modern query_points API
            response = self.client.query_points(
                collection_name=target_collection,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            scored_points = response.points
        except (AttributeError, TypeError):
            # Fallback to search API for older client bindings
            scored_points = self.client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

        retrieved: List[DocumentChunk] = []
        for point in scored_points:
            payload = point.payload or {}
            retrieved.append(
                DocumentChunk(
                    id=payload.get("id", str(point.id)),
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", payload),
                    score=point.score,
                )
            )
        return retrieved

    def delete_by_doc_hash(
        self,
        doc_hash: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """Purge all chunks associated with a specific doc_hash."""
        target_collection = collection_name or self.collection_name
        self.client.delete(
            collection_name=target_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_hash",
                            match=models.MatchValue(value=doc_hash),
                        )
                    ]
                )
            ),
            wait=True,
        )
