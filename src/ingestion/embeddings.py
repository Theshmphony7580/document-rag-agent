import hashlib
import math
from typing import List, Optional

import httpx

from config import get_settings


class GeminiEmbedder:
    """Generates dense vector embeddings using Google Gemini (text-embedding-004) with offline test fallback."""

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

    def embed_text(self, text: str) -> List[float]:
        """Generate 768-dim embedding for a single text chunk."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Use live Gemini API if key is available
        if self.api_key:
            try:
                return self._call_gemini_batch(texts)
            except Exception as e:
                # Log or fall back if transient failure occurs during development
                print(f"[GeminiEmbedder] Live API call failed ({e}), falling back to deterministic local embedding.")

        # Offline / deterministic local mock fallback (zero cost, zero keys required)
        return [self._generate_mock_embedding(t) for t in texts]

    def _call_gemini_batch(self, texts: List[str]) -> List[List[float]]:
        """Call Google Gemini batchEmbedContents endpoint via httpx."""
        url = f"{self.base_url}/models/{self.model}:batchEmbedContents?key={self.api_key}"
        requests_payload = [
            {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}}
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

        # Normalize to unit length for Cosine metric
        norm = math.sqrt(sum(v * v for v in raw_values)) or 1.0
        return [v / norm for v in raw_values]
