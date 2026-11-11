"""Conservative query rewriting for retrieval.

The rewriter is not allowed to answer the question or add facts. It only makes
an existing information need easier to retrieve while preserving named entities,
dates, quantities, scope and constraints.
"""

from __future__ import annotations

import re

from satyameter.llm.base import LLMClient

SYSTEM_PROMPT = """You are a search-query editor for an evidence-grounded question-answering system.
Rewrite the user's question as ONE concise search query suitable for retrieving relevant
knowledge-base passages.
Rules:
- Preserve the original intent, language, named entities, dates, numbers and constraints.
- Do not answer the question.
- Do not add assumptions, facts, entities or details that were not in the question.
- Keep the query as a question or a concise search phrase.
- If the original question is already clear, return it unchanged.
- Output only the rewritten query, with no label, quotation marks or explanation."""


def rewrite_query(question: str, llm: LLMClient, max_words: int = 45) -> str:
    """Return a conservative rewritten query; fall back to the original on malformed output."""
    original = " ".join((question or "").split())
    if not original:
        raise ValueError("Question cannot be empty.")

    response = llm.generate(
        SYSTEM_PROMPT,
        f"Original question: {original}\nRewritten search query:",
        temperature=0.0,
        max_tokens=128,
    )
    candidate = (response or "").strip()
    candidate = re.sub(r"^```(?:text)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"^(?:rewritten query|search query)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip().strip("\"'` ")
    candidate = " ".join(candidate.split())

    if not candidate or len(candidate.split()) > max_words:
        return original
    if candidate.lower() in {"none", "n/a", "not applicable"}:
        return original
    return candidate
