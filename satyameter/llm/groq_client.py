"""Groq backend (free tier). Retries on rate limits and transient network errors."""

from __future__ import annotations

import os
import time

from satyameter.llm.base import LLMClient


class GroqClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        reasoning_effort: str | None = "low",
        max_retries: int = 5,
    ):
        import groq

        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Put it in your .env file.")
        self._groq = groq
        self._client = groq.Groq(api_key=api_key)
        self.model = model
        self.name = f"groq:{model}"
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        # gpt-oss models are reasoning models; "low" keeps answers fast and cheap.
        if self.reasoning_effort and self.model.startswith("openai/gpt-oss"):
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    print("  ! warning: reply was cut off (raise max_tokens)")
                return (choice.message.content or "").strip()
            except (self._groq.RateLimitError, self._groq.APIConnectionError) as exc:
                wait = min(2**attempt * 2, 60)
                print(f"  ! {type(exc).__name__}; retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Groq request failed after retries (rate limit or network).")
