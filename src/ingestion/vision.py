import base64
from typing import Optional

try:
    from config import get_settings
except ImportError:
    from config import get_settings


# Prompt kept empty as per design decision; to be refined in later milestone
DEFAULT_CHART_PROMPT = ""


class VLMVisionCaptioner:
    """Generates natural language summaries for charts, figures, and architecture diagrams."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.prompt = DEFAULT_CHART_PROMPT

    def describe_figure(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        prompt: Optional[str] = None,
    ) -> str:
        """
        Generate textual description for a cropped figure.
        Returns a placeholder if API key is missing or prompt is empty.
        """
        effective_prompt = prompt if prompt is not None else self.prompt

        if not self.api_key:
            return "[VLM: Offline / Mock figure description - Gemini API key not set]"

        # When the prompt is empty, return empty caption placeholder as instructed
        if not effective_prompt.strip():
            return ""

        # Placeholder for Gemini Vision HTTP / SDK call when prompt is finalized
        # Uses standard base64 payload
        _ = base64.b64encode(image_bytes).decode("utf-8")
        return ""
