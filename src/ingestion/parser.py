from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from schemas import DocumentChunk
    from config import get_settings
except ImportError:
    from schemas import DocumentChunk
    from config import get_settings


def compute_file_sha256(file_path: str) -> str:
    """Compute deterministic SHA-256 hex digest for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DocumentParser:
    """
    Parses digital documents using IBM Docling with `do_ocr=False` permanently enforced.
    Preserves document structure (headings, tables) and produces enriched DocumentChunk instances.
    """

    def __init__(self):
        self.settings = get_settings()

    def parse_document(self, file_path: str) -> List[DocumentChunk]:
        """Parse a document file into enriched DocumentChunk instances."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_hash = compute_file_sha256(file_path)
        source_name = Path(file_path).name
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.chunking import HierarchicalChunker

            # Enforce permanent do_ocr=False for high performance
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            result = converter.convert(file_path)
            doc = result.document
            chunker = HierarchicalChunker()
            docling_chunks = list(chunker.chunk(doc))

            parsed_chunks: List[DocumentChunk] = []
            for idx, c in enumerate(docling_chunks):
                chunk_id = f"{doc_hash}_{idx:04d}"
                meta: Dict[str, Any] = {
                    "doc_hash": doc_hash,
                    "chunk_id": chunk_id,
                    "source_file": source_name,
                    "page_numbers": getattr(c.meta, "page_numbers", [1]) or [1],
                    "section_path": " > ".join(getattr(c.meta, "headings", [])) or "Main",
                    "content_type": "table" if getattr(c.meta, "is_table", False) else "text",
                    "created_at": now_iso,
                }
                parsed_chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        text=c.text,
                        metadata=meta,
                    )
                )
            return parsed_chunks

        except ImportError:
            # Native fallback parser for text/markdown files when Docling is not installed in the active environment
            return self._fallback_text_parse(file_path, doc_hash, source_name, now_iso)

    def _fallback_text_parse(
        self,
        file_path: str,
        doc_hash: str,
        source_name: str,
        timestamp: str,
    ) -> List[DocumentChunk]:
        """Lightweight text splitting for plain text, markdown, or offline tests."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

        # Split into logical paragraphs
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [full_text.strip()] if full_text.strip() else []

        chunks: List[DocumentChunk] = []
        for idx, text in enumerate(paragraphs):
            chunk_id = f"{doc_hash}_{idx:04d}"
            meta: Dict[str, Any] = {
                "doc_hash": doc_hash,
                "chunk_id": chunk_id,
                "source_file": source_name,
                "page_numbers": [1],
                "section_path": "Document",
                "content_type": "text",
                "created_at": timestamp,
            }
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=text,
                    metadata=meta,
                )
            )
        return chunks
