"""Split an answer into atomic, self-contained factual claims using the LLM."""

from __future__ import annotations

import json
import re

from satyameter.llm.base import LLMClient

NOT_FOUND_MARKER = "could not find this in the knowledge base"
MAX_CLAIMS = 15
CITATION_RE = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")

SYSTEM_PROMPT = """You break an answer into atomic factual claims for fact-checking.
Rules:
- Each claim is ONE self-contained statement of fact.
- Replace pronouns and vague references ("he", "it", "the company") with the full name, using the question for context.
- Split compound sentences (for example "born in X in 1879") into separate claims.
- Keep numbers, dates and names exactly as written.
- Ignore citation markers like [1], opinions, hedges, and statements about the passages themselves.
- Output ONLY a JSON array of strings. No explanation, no markdown.

Example
Question: Who discovered penicillin?
Answer: Alexander Fleming, a Scottish physician, discovered it in 1928 [2].
Output: ["Alexander Fleming discovered penicillin.", "Alexander Fleming was a Scottish physician.", "Penicillin was discovered in 1928."]"""


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def parse_claims(raw: str) -> list[str]:
    """Parse the model's reply into a clean, de-duplicated list of claims."""
    claims: list[str] = []
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                claims = [str(item) for item in data if isinstance(item, (str, int, float))]
        except json.JSONDecodeError:
            claims = []
    if not claims:  # fallback: bullet / numbered lines
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip().strip('",')
            if len(line.split()) >= 3:
                claims.append(line)
    seen: set[str] = set()
    cleaned: list[str] = []
    for claim in claims:
        claim = strip_citations(claim)
        key = claim.lower()
        if claim and key not in seen:
            seen.add(key)
            cleaned.append(claim)
    return cleaned[:MAX_CLAIMS]


def extract_claims(question: str, answer: str, llm: LLMClient) -> list[str]:
    """Return atomic claims from an answer. Empty if the answer says 'not found'."""
    if not answer.strip() or NOT_FOUND_MARKER in answer.lower():
        return []
    user = f"Question: {question}\nAnswer: {strip_citations(answer)}\nOutput:"
    reply = llm.generate(SYSTEM_PROMPT, user, temperature=0.0, max_tokens=1024)
    return parse_claims(reply)
