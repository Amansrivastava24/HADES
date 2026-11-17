from unittest.mock import MagicMock, patch
import pytest

from app.core.config import Settings
from app.core.llm import get_llm_model, ConfigurationError
from app.embeddings.openai import OpenAIEmbeddingProvider
from app.embeddings.service import get_embedding_provider
from app.embeddings.exceptions import EmbeddingConfigurationError, InvalidEmbeddingInputError


def test_get_llm_model_openai_success():
    with patch("app.core.llm.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = "test-openai-key"
        mock_settings.OPENAI_MODEL_ID = "gpt-4o-mini"
        mock_settings.OPENAI_REASONING_MODEL_ID = "gpt-4o"

        model = get_llm_model(is_reasoning=False)
        assert model is not None
        assert model.id == "gpt-4o-mini"
        assert model.api_key == "test-openai-key"

        reasoning_model = get_llm_model(is_reasoning=True)
        assert reasoning_model.id == "gpt-4o"


def test_get_llm_model_missing_key_raises():
    with patch("app.core.llm.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = ""

        with pytest.raises(ConfigurationError):
            get_llm_model()


def test_openai_embedding_provider_initialization():
    with patch("app.embeddings.openai.settings") as mock_settings:
        mock_settings.EMBEDDING_API_KEY = "test-embed-key"
        mock_settings.OPENAI_API_KEY = ""
        mock_settings.EMBEDDING_PROVIDER = "openai"
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.EMBEDDING_DIMENSION = 1536

        provider = OpenAIEmbeddingProvider()
        assert provider.provider_name == "openai"
        assert provider.model_id == "text-embedding-3-small"
        assert provider.dimension == 1536


def test_openai_embedding_provider_empty_input_validation():
    with patch("app.embeddings.openai.settings") as mock_settings:
        mock_settings.EMBEDDING_API_KEY = "test-embed-key"
        mock_settings.OPENAI_API_KEY = ""
        mock_settings.EMBEDDING_PROVIDER = "openai"

        provider = OpenAIEmbeddingProvider()
        with pytest.raises(InvalidEmbeddingInputError):
            provider.embed("   ")


def test_get_embedding_provider_factory_openai():
    with patch("app.embeddings.service.settings") as mock_settings:
        mock_settings.EMBEDDING_PROVIDER = "openai"
        mock_settings.EMBEDDING_API_KEY = "test-key"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.EMBEDDING_DIMENSION = 1536

        provider = get_embedding_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIEmbeddingProvider)
