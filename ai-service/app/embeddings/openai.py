import logging
import re
from typing import List, Optional

from app.core.config import settings
from app.embeddings.base import EmbeddingProvider, validate_embedding_input, validate_vector
from app.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    InvalidEmbeddingResponseError,
)

logger = logging.getLogger(__name__)


def _sanitize_error(msg: str) -> str:
    """Sanitize error messages to ensure API keys and bearer tokens are never leaked."""
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", msg)
    sanitized = re.sub(r"(api[_\-]?key[=:\s]+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Concrete embedding provider integrating with OpenAI's embedding API (`text-embedding-3-small`).
    Generates 1536-dimensional (or configured) dense vectors.
    """

    DEFAULT_MODEL: str = "text-embedding-3-small"
    DEFAULT_DIMENSION: int = 1536

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self._api_key = api_key or settings.EMBEDDING_API_KEY or settings.OPENAI_API_KEY
        self._model_id = model_id or (settings.EMBEDDING_MODEL if settings.EMBEDDING_PROVIDER == "openai" else self.DEFAULT_MODEL)
        self._dimension = dimension or (settings.EMBEDDING_DIMENSION if settings.EMBEDDING_PROVIDER == "openai" else self.DEFAULT_DIMENSION)

        if not self._api_key or not self._api_key.strip():
            raise EmbeddingConfigurationError(
                "OpenAI API key is required. Set OPENAI_API_KEY or EMBEDDING_API_KEY in environment."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key.strip())
        except Exception as e:
            sanitized = _sanitize_error(str(e))
            logger.error("Failed to initialize OpenAI client: %s", sanitized)
            raise EmbeddingProviderError(f"Failed to initialize OpenAI client: {sanitized}") from e

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """
        Call OpenAI embeddings API and return a validated float vector.
        """
        clean_text = validate_embedding_input(text)

        try:
            kwargs = {
                "model": self._model_id,
                "input": [clean_text],
            }
            if "text-embedding-3" in self._model_id and self._dimension:
                kwargs["dimensions"] = self._dimension

            response = self._client.embeddings.create(**kwargs)
        except Exception as e:
            sanitized = _sanitize_error(str(e))
            logger.error("OpenAI embeddings API call failed: %s", sanitized)
            raise EmbeddingProviderError(f"OpenAI embeddings API error: {sanitized}") from e

        if not response or not hasattr(response, "data") or not response.data:
            raise InvalidEmbeddingResponseError("OpenAI API returned an empty or invalid response object.")

        item = response.data[0]
        raw_embedding = getattr(item, "embedding", None)

        return validate_vector(raw_embedding, expected_dimension=self._dimension)
