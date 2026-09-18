"""Dense embedding engines supporting local Hugging Face models and Google Gemini API.

Defaults to local execution using Hugging Face's `BAAI/bge-base-en-v1.5` via `sentence-transformers`
(768 dimensions, normalized for cosine similarity).
"""

import hashlib
import logging
import math
import threading
from typing import List, Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDER = None
_EMBEDDER_LOCK = threading.RLock()


class HuggingFaceEmbedder:
    """Local dense embedding model using Hugging Face sentence-transformers."""

    _shared_model = None
    _shared_model_name: Optional[str] = None
    _shared_device: Optional[str] = None
    _lock = threading.RLock()

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
        """Load and cache the SentenceTransformer model once across the process."""
        with HuggingFaceEmbedder._lock:
            if (
                HuggingFaceEmbedder._shared_model is None
                or HuggingFaceEmbedder._shared_model_name != self.model_name
                or HuggingFaceEmbedder._shared_device != self.device
            ):
                try:
                    from sentence_transformers import SentenceTransformer
                    print(f"[HuggingFaceEmbedder] Loading local embedding model '{self.model_name}' onto {self.device}...")
                    try:
                        # Fast offline load without checking Hugging Face remote repository
                        HuggingFaceEmbedder._shared_model = SentenceTransformer(
                            self.model_name, device=self.device, local_files_only=True
                        )
                    except Exception:
                        HuggingFaceEmbedder._shared_model = SentenceTransformer(
                            self.model_name, device=self.device
                        )
                    HuggingFaceEmbedder._shared_model_name = self.model_name
                    HuggingFaceEmbedder._shared_device = self.device
                    print(f"[HuggingFaceEmbedder] Model '{self.model_name}' successfully loaded into memory.")
                except ImportError:
                    logger.warning(
                        "[HuggingFaceEmbedder] 'sentence-transformers' not installed. "
                        "Falling back to deterministic mock embedding."
                    )
                    HuggingFaceEmbedder._shared_model = False
                except Exception as e:
                    logger.warning(
                        f"[HuggingFaceEmbedder] Failed to load '{self.model_name}' ({e}). "
                        "Falling back to deterministic mock embedding."
                    )
                    HuggingFaceEmbedder._shared_model = False

            self._model = HuggingFaceEmbedder._shared_model
            return self._model

    def warmup(self):
        """Eagerly load model into memory and perform a dummy encoding to warm up execution paths."""
        model = self._load_model()
        if model:
            try:
                model.encode(["system warmup query"], normalize_embeddings=True, show_progress_bar=False)
            except Exception as e:
                logger.warning(f"[HuggingFaceEmbedder] Warmup error ({e})")

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
    """Return the thread-safe singleton instance of the configured embedding provider."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _GLOBAL_EMBEDDER is None:
                settings = get_settings()
                provider = settings.EMBEDDING_PROVIDER.lower()
                if provider == "gemini":
                    _GLOBAL_EMBEDDER = GeminiEmbedder()
                else:
                    _GLOBAL_EMBEDDER = HuggingFaceEmbedder()
    return _GLOBAL_EMBEDDER


# Backward-compatible alias
LocalEmbedder = HuggingFaceEmbedder
