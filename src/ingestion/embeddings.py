"""Dense embedding engines supporting local Hugging Face models and Google Gemini API.

Defaults to local execution using Hugging Face's `BAAI/bge-base-en-v1.5` via `sentence-transformers`
(768 dimensions, normalized for cosine similarity).
"""

import hashlib
import logging
import math
from typing import List, Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    """Local dense embedding model using Hugging Face sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.LOCAL_EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self.vector_dim = vector_dim or settings.EMBEDDING_DIM
        self._model = None

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first embedding request."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"[HuggingFaceEmbedder] Loading local model '{self.model_name}' onto {self.device}...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                logger.warning(
                    "[HuggingFaceEmbedder] 'sentence-transformers' not installed. "
                    "Falling back to deterministic mock embedding."
                )
                self._model = False
            except Exception as e:
                logger.warning(
                    f"[HuggingFaceEmbedder] Failed to load '{self.model_name}' ({e}). "
                    "Falling back to deterministic mock embedding."
                )
                self._model = False
        return self._model

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate 768-dim normalized embedding for a single text."""
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        model = self._load_model()
        if model:
            try:
                # BGE recommendation: prepend retrieval instruction to query for optimal ranking
                prepared_texts = texts
                if is_query and "bge" in self.model_name.lower():
                    prefix = "Represent this sentence for searching relevant passages: "
                    prepared_texts = [f"{prefix}{t}" for t in texts]

                embeddings = model.encode(
                    prepared_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [arr.tolist() for arr in embeddings]
            except Exception as e:
                logger.warning(f"[HuggingFaceEmbedder] Local inference error ({e}). Falling back to mock.")

        # Offline / deterministic fallback when model is not loaded
        return [self._generate_mock_embedding(t) for t in texts]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Deterministic 768-dimensional normalized unit vector generated from SHA-256 hash."""
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw_values = []
        for i in range(self.vector_dim):
            byte_val = seed[i % len(seed)]
            val = ((byte_val + i * 17) % 100) / 100.0 - 0.5
            raw_values.append(val)

        # Normalize to unit length for Cosine metric
        norm = math.sqrt(sum(v * v for v in raw_values)) or 1.0
        return [v / norm for v in raw_values]


class GeminiEmbedder:
    """Generates dense vector embeddings using Google Gemini API with offline test fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_EMBEDDING_MODEL
        self.vector_dim = vector_dim or settings.GEMINI_EMBEDDING_DIM
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Generate 768-dim embedding for a single text chunk."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        if self.api_key:
            try:
                return self._call_gemini_batch(texts)
            except Exception as e:
                logger.warning(f"[GeminiEmbedder] Live API call failed ({e}), falling back to mock.")

        return [self._generate_mock_embedding(t) for t in texts]

    def _call_gemini_batch(self, texts: List[str]) -> List[List[float]]:
        """Call Google Gemini batchEmbedContents endpoint via httpx."""
        url = f"{self.base_url}/models/{self.model}:batchEmbedContents?key={self.api_key}"
        requests_payload = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self.vector_dim,
            }
            for t in texts
        ]
        payload = {"requests": requests_payload}

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings_data = data.get("embeddings", [])
            return [item["values"] for item in embeddings_data]

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Deterministic 768-dimensional normalized unit vector generated from SHA-256 hash."""
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw_values = []
        for i in range(self.vector_dim):
            byte_val = seed[i % len(seed)]
            val = ((byte_val + i * 17) % 100) / 100.0 - 0.5
            raw_values.append(val)

        norm = math.sqrt(sum(v * v for v in raw_values)) or 1.0
        return [v / norm for v in raw_values]


def get_embedder():
    """Factory to instantiate the configured embedding provider."""
    settings = get_settings()
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "huggingface":
        return HuggingFaceEmbedder()
    elif provider == "gemini":
        return GeminiEmbedder()
    return HuggingFaceEmbedder()


# Backward-compatible alias
LocalEmbedder = HuggingFaceEmbedder
