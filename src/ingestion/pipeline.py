from dataclasses import dataclass
from typing import List, Optional

from schemas import DocumentChunk
from storage.vector_store import QdrantVectorStore
from ingestion.parser import DocumentParser, compute_file_sha256
from ingestion.embeddings import GeminiEmbedder
from ingestion.vision import VLMVisionCaptioner


@dataclass
class IngestionResult:
    """Outcome of a document ingestion run."""
    source_file: str
    doc_hash: str
    chunk_count: int
    status: str  # "indexed" | "skipped_duplicate" | "error"
    message: str


class IngestionPipeline:
    """
    End-to-end ingestion orchestrator:
    1. Deduplication via SHA-256
    2. Structure-aware parsing via Docling
    3. Metadata enrichment
    4. Dense vector embedding via Gemini
    5. Qdrant disk storage
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        parser: Optional[DocumentParser] = None,
        embedder: Optional[GeminiEmbedder] = None,
        vision_captioner: Optional[VLMVisionCaptioner] = None,
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.vision_captioner = vision_captioner or VLMVisionCaptioner()
        self.parser = parser or DocumentParser(vision_captioner=self.vision_captioner)
        self.embedder = embedder or GeminiEmbedder()

    def ingest_file(self, file_path: str, force_reindex: bool = False) -> IngestionResult:
        """Run the ingestion pipeline for a single file."""
        doc_hash = compute_file_sha256(file_path)

        # Step 1: Deduplication check
        if not force_reindex and self.vector_store.has_doc_hash(doc_hash):
            return IngestionResult(
                source_file=file_path,
                doc_hash=doc_hash,
                chunk_count=0,
                status="skipped_duplicate",
                message=f"Document already indexed with hash {doc_hash[:10]}... Skipping.",
            )

        # If re-indexing is forced, purge old vectors first
        if force_reindex:
            self.vector_store.delete_by_doc_hash(doc_hash)

        # Step 2: Parse and chunk with metadata
        chunks = self.parser.parse_document(file_path)
        if not chunks:
            return IngestionResult(
                source_file=file_path,
                doc_hash=doc_hash,
                chunk_count=0,
                status="error",
                message="Parser produced 0 chunks. Document may be empty or unreadable.",
            )

        # Step 3: Generate dense embeddings
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_batch(texts)

        # Step 4: Upsert to Qdrant
        self.vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

        return IngestionResult(
            source_file=file_path,
            doc_hash=doc_hash,
            chunk_count=len(chunks),
            status="indexed",
            message=f"Successfully indexed {len(chunks)} chunks into Qdrant.",
        )
