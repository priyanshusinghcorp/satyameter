"""Optional local Ollama chat backend. Uses Ollama's local HTTP API."""

from __future__ import annotations

from typing import Any

import requests

from satyameter.llm.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
        timeout_s: int = 180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.name = f"ollama:{model}"
        self._session = requests.Session()

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach Ollama. Start Ollama, pull the configured model "
                f"('{self.model}'), and confirm {self.base_url} is reachable. Details: {exc}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

        content = (data.get("message") or {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama returned an empty response for model '{self.model}'.")
        return content.strip()
