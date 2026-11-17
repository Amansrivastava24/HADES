import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.connection import DatabaseManager
from app.db.memory import save_memory
from app.embeddings.base import EmbeddingProvider, validate_embedding_input
from app.embeddings.exceptions import EmbeddingConfigurationError
from app.embeddings.fake import FakeEmbeddingProvider
from app.embeddings.openai import OpenAIEmbeddingProvider
from app.schemas.models import MemoryChunk

logger = logging.getLogger(__name__)


def get_embedding_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    dimension: Optional[int] = None,
) -> EmbeddingProvider:
    """
    Factory function to instantiate the configured embedding provider.

    Parameters:
        provider_type: Provider identifier ('openai', 'fake'). Defaults to settings.EMBEDDING_PROVIDER.
        api_key: Optional API key override.
        model_id: Optional model identifier override.
        dimension: Optional dimension override.

    Returns:
        EmbeddingProvider: Configured provider instance.

    Raises:
        EmbeddingConfigurationError: If provider_type is unsupported.
    """
    ptype = (provider_type or settings.EMBEDDING_PROVIDER or "openai").lower().strip()

    if ptype == "openai":
        return OpenAIEmbeddingProvider(api_key=api_key, model_id=model_id, dimension=dimension)
    elif ptype == "fake":
        dim = dimension or settings.EMBEDDING_DIMENSION or 1536
        mid = model_id or "fake-deterministic-embed"
        return FakeEmbeddingProvider(dimension=dim, model_id=mid)
    else:
        raise EmbeddingConfigurationError(
            f"Unsupported embedding provider '{ptype}'. Supported options: 'openai', 'fake'."
        )


def generate_embedding(
    text: str,
    provider: Optional[EmbeddingProvider] = None,
) -> List[float]:
    """
    Generate an embedding vector for the provided text using the active or supplied provider.

    Parameters:
        text: Non-empty input string.
        provider: Optional EmbeddingProvider instance; defaults to factory default.

    Returns:
        List[float]: Normalized vector representation.
    """
    active_provider = provider or get_embedding_provider()
    return active_provider.embed(text)


def save_memory_with_embedding(
    external_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    provider: Optional[EmbeddingProvider] = None,
    manager: Optional[DatabaseManager] = None,
) -> MemoryChunk:
    """
    Generate an embedding vector for content and persist it to PostgreSQL pgvector storage.

    Parameters:
        external_id: External learner identifier.
        content: Text content of the memory.
        metadata: Optional dictionary of memory metadata.
        provider: Optional EmbeddingProvider instance; defaults to factory default.
        manager: Optional DatabaseManager instance; defaults to global manager.

    Returns:
        MemoryChunk: Fully populated persisted memory chunk with embedding.
    """
    clean_text = validate_embedding_input(content)
    embedding_vector = generate_embedding(clean_text, provider=provider)

    return save_memory(
        external_id=external_id,
        content=clean_text,
        metadata=metadata,
        embedding=embedding_vector,
        manager=manager,
    )
