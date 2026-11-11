"""The interface every LLM backend implements. Swap providers without touching the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    #: Identifier used in cache keys and result logs (provider + model).
    name: str = "llm"

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return the model's text reply."""
