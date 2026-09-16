from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import DocumentChunk
from config import get_settings
from ingestion.vision import VLMVisionCaptioner

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


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
    Extracts text, preserves markdown tables, and routes figures/diagrams to Gemini Vision.
    """

    def __init__(self, vision_captioner: Optional[VLMVisionCaptioner] = None):
        self.settings = get_settings()
        self.vision_captioner = vision_captioner or VLMVisionCaptioner()

    def parse_document(self, file_path: str) -> List[DocumentChunk]:
        """Parse a document or image file into enriched DocumentChunk instances."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_hash = compute_file_sha256(file_path)
        source_path = Path(file_path)
        source_name = source_path.name
        now_iso = datetime.now(timezone.utc).isoformat()

        # Handle standalone image files directly with VLM Vision
        if source_path.suffix.lower() in IMAGE_EXTENSIONS:
            return self._parse_image_file(file_path, doc_hash, source_name, now_iso)

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.chunking import HierarchicalChunker

            # Enforce permanent do_ocr=False for high speed, enable picture generation
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.generate_picture_images = True

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

            # 1. Text and Table Chunks
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

            # 2. Figure / Diagram Extraction via Gemini Vision
            if hasattr(doc, "pictures") and doc.pictures:
                for p_idx, picture in enumerate(doc.pictures):
                    try:
                        # Crop ONLY the isolated bounding box of the figure (not the whole page)
                        pil_image = picture.get_image(doc)
                        if pil_image:
                            img_byte_arr = io.BytesIO()
                            pil_image.save(img_byte_arr, format="PNG")
                            raw_bytes = img_byte_arr.getvalue()

                            # Extract exact page number from Docling provenance
                            page_no = 1
                            if hasattr(picture, "prov") and picture.prov:
                                page_no = getattr(picture.prov[0], "page_no", 1)
                            elif hasattr(picture, "page_no"):
                                page_no = picture.page_no

                            # Extract any textual caption already identified by Docling (e.g., "Figure 2: Architecture")
                            doc_caption = ""
                            if hasattr(picture, "captions") and picture.captions:
                                doc_caption = " ".join([getattr(c, "text", str(c)) for c in picture.captions if c]).strip()

                            # Call VLM on the cropped image
                            caption = self.vision_captioner.describe_figure(raw_bytes, mime_type="image/png")
                            if caption.strip():
                                fig_chunk_id = f"{doc_hash}_p{page_no}_fig_{p_idx:03d}"
                                fig_meta: Dict[str, Any] = {
                                    "doc_hash": doc_hash,
                                    "chunk_id": fig_chunk_id,
                                    "source_file": source_name,
                                    "page_numbers": [page_no],
                                    "section_path": f"Page {page_no} > Figures & Diagrams",
                                    "content_type": "diagram_vlm",
                                    "caption": doc_caption,
                                    "created_at": now_iso,
                                }

                                figure_text = (
                                    f"### Visual Figure on Page {page_no} ({doc_caption})\n{caption}"
                                    if doc_caption
                                    else f"### Visual Figure on Page {page_no}\n{caption}"
                                )

                                parsed_chunks.append(
                                    DocumentChunk(
                                        id=fig_chunk_id,
                                        text=figure_text,
                                        metadata=fig_meta,
                                    )
                                )
                    except Exception as err:
                        # Continue processing remaining chunks if a single figure fails
                        continue

            return parsed_chunks

        except ImportError:
            # Fallback parser for text/markdown files when Docling is offline
            return self._fallback_text_parse(file_path, doc_hash, source_name, now_iso)

    def _parse_image_file(
        self,
        file_path: str,
        doc_hash: str,
        source_name: str,
        timestamp: str,
    ) -> List[DocumentChunk]:
        """Directly summarize and index a standalone image file via VLM."""
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        suffix = Path(file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        caption = self.vision_captioner.describe_figure(raw_bytes, mime_type=mime_type)
        chunk_id = f"{doc_hash}_0000"
        meta: Dict[str, Any] = {
            "doc_hash": doc_hash,
            "chunk_id": chunk_id,
            "source_file": source_name,
            "page_numbers": [1],
            "section_path": "Standalone Figure",
            "content_type": "diagram_vlm",
            "created_at": timestamp,
        }
        return [
            DocumentChunk(
                id=chunk_id,
                text=f"### Visual Figure Description\n{caption}",
                metadata=meta,
            )
        ]

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
