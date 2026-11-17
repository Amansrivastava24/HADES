import logging
import re
from typing import Optional
from agno.models.openai import OpenAIChat
from app.core.config import settings

logger = logging.getLogger(__name__)


def sanitize_key_error(msg: str) -> str:
    """Sanitize error messages to ensure API keys and bearer tokens are never leaked."""
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", msg)
    sanitized = re.sub(r"(api[_\-]?key[=:\s]+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized


class ConfigurationError(RuntimeError):
    """Raised when mandatory OpenAI credentials or settings are missing."""
    pass


def get_llm_model(
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
    is_reasoning: bool = False,
) -> OpenAIChat:
    """
    Instantiate an Agno OpenAIChat model.

    Parameters:
        model_id: Optional model override.
        api_key: Optional API key override.
        is_reasoning: True for complex curriculum generation (gpt-4o),
                      False for fast extraction/analysis (gpt-4o-mini).

    Returns:
        Agno OpenAIChat model instance.
    """
    key = api_key or settings.OPENAI_API_KEY
    if not key or not key.strip():
        raise ConfigurationError(
            "OPENAI_API_KEY environment variable is not configured. "
            "Live agent reasoning requires a valid OpenAI API key."
        )

    mid = model_id or (settings.OPENAI_REASONING_MODEL_ID if is_reasoning else settings.OPENAI_MODEL_ID)
    return OpenAIChat(id=mid, api_key=key.strip())
