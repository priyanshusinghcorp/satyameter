from satyameter.config import Settings
from satyameter.llm.base import LLMClient
from satyameter.llm.cache import CachedLLM
from satyameter.llm.groq_client import GroqClient
from satyameter.llm.ollama_client import OllamaClient


def get_llm(settings: Settings, use_cache: bool = True) -> LLMClient:
    """Build the configured backend and optionally wrap it in the SQLite cache."""
    backend = settings.llm_backend
    if backend == "groq":
        client: LLMClient = GroqClient(
            model=settings.groq_model,
            reasoning_effort=settings.groq_reasoning_effort,
        )
    elif backend == "ollama":
        client = OllamaClient(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout_s=settings.ollama_timeout_s,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_BACKEND '{backend}'. Choose 'groq' or 'ollama' in .env."
        )
    return CachedLLM(client, settings.cache_path) if use_cache else client


__all__ = ["LLMClient", "CachedLLM", "GroqClient", "OllamaClient", "get_llm"]
