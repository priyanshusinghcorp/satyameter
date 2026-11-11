"""Claim-level verification: Supported / Contradicted / Not Found, plus answer-level scores."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum

from satyameter.ingestion.chunker import Chunk
from satyameter.verification.claims import extract_claims
from satyameter.verification.nli import NLIScores


NEGATION_RE = re.compile(r"\b(never|not|no|none|neither|nor|cannot)\b|n't\b", re.IGNORECASE)


def is_negated(claim: str) -> bool:
    """True if the claim asserts an absence ('never', 'did not'...). NLI is unreliable on these."""
    return bool(NEGATION_RE.search(claim))


class Status(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds. Tuned on the hand-labelled dev set, not guessed."""

    entail: float = 0.70  # entailment >= this on any evidence chunk -> SUPPORTED
    contradict: float = 0.90  # contradiction >= this ...
    entail_ceiling: float = 0.20  # ... and no chunk entails above this -> CONTRADICTED


@dataclass(frozen=True)
class EvidenceScore:
    chunk: Chunk
    scores: NLIScores
    source: str  # "claim" (retrieved for this claim) or "answer" (retrieved for the question)


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    status: Status
    best: EvidenceScore | None  # the evidence that decided it, or the closest evidence
    evidence: list[EvidenceScore] = field(default_factory=list)


@dataclass(frozen=True)
class Verification:
    question: str
    answer: str
    verdicts: list[ClaimVerdict]
    faithfulness: float | None  # supported / total
    contradiction_rate: float | None  # contradicted / total
    coverage: float | None  # (supported + contradicted) / total: how much was decidable
    latency_s: float


def decide(evidence: list[EvidenceScore], t: Thresholds) -> tuple[Status, EvidenceScore | None]:
    """Pure decision rule over per-chunk NLI scores."""
    if not evidence:
        return Status.NOT_FOUND, None
    best_entail = max(evidence, key=lambda e: e.scores.entailment)
    if best_entail.scores.entailment >= t.entail:
        return Status.SUPPORTED, best_entail
    best_contra = max(evidence, key=lambda e: e.scores.contradiction)
    if (
        best_contra.scores.contradiction >= t.contradict
        and best_entail.scores.entailment <= t.entail_ceiling
    ):
        return Status.CONTRADICTED, best_contra
    return Status.NOT_FOUND, best_entail


def summarize(verdicts: list[ClaimVerdict]) -> tuple[float | None, float | None, float | None]:
    total = len(verdicts)
    if total == 0:
        return None, None, None
    supported = sum(v.status is Status.SUPPORTED for v in verdicts)
    contradicted = sum(v.status is Status.CONTRADICTED for v in verdicts)
    return supported / total, contradicted / total, (supported + contradicted) / total


def rethreshold(result: Verification, thresholds: Thresholds) -> Verification:
    """Re-decide every claim with new thresholds, reusing stored NLI scores (instant, no model calls)."""
    verdicts = []
    for v in result.verdicts:
        status, best = decide(v.evidence, thresholds)
        verdicts.append(ClaimVerdict(v.claim, status, best, v.evidence))
    faithfulness, contradiction_rate, coverage = summarize(verdicts)
    return replace(
        result,
        verdicts=verdicts,
        faithfulness=faithfulness,
        contradiction_rate=contradiction_rate,
        coverage=coverage,
    )


def verify_claim(
    claim: str,
    index,
    nli,
    thresholds: Thresholds = Thresholds(),
    answer_hits: list | None = None,
    claim_k: int = 3,
    max_evidence: int = 8,
) -> ClaimVerdict:
    """Verify one claim against evidence retrieved FOR THE CLAIM plus the answer's own passages.

    Retrieving per claim matters: the passages that answered the question may not
    contain the fact a particular claim asserts.
    """
    candidates: list[tuple[Chunk, str]] = []
    seen: set[str] = set()
    for hit in index.search(claim, k=claim_k):
        if hit.chunk.chunk_id not in seen:
            seen.add(hit.chunk.chunk_id)
            candidates.append((hit.chunk, "claim"))
    for hit in answer_hits or []:
        if hit.chunk.chunk_id not in seen:
            seen.add(hit.chunk.chunk_id)
            candidates.append((hit.chunk, "answer"))
    candidates = candidates[:max_evidence]

    scores = nli.predict([(chunk.embed_text, claim) for chunk, _ in candidates])
    evidence = [
        EvidenceScore(chunk=chunk, scores=score, source=source)
        for (chunk, source), score in zip(candidates, scores)
    ]
    status, best = decide(evidence, thresholds)
    return ClaimVerdict(claim=claim, status=status, best=best, evidence=evidence)


def verify_answer(
    rag_answer,
    index,
    llm,
    nli,
    thresholds: Thresholds = Thresholds(),
    claim_k: int = 3,
    max_evidence: int = 8,
) -> Verification:
    start = time.perf_counter()
    claims = extract_claims(rag_answer.question, rag_answer.answer, llm)
    verdicts = [
        verify_claim(
            claim, index, nli, thresholds,
            answer_hits=rag_answer.hits, claim_k=claim_k, max_evidence=max_evidence,
        )
        for claim in claims
    ]
    faithfulness, contradiction_rate, coverage = summarize(verdicts)
    return Verification(
        question=rag_answer.question,
        answer=rag_answer.answer,
        verdicts=verdicts,
        faithfulness=faithfulness,
        contradiction_rate=contradiction_rate,
        coverage=coverage,
        latency_s=time.perf_counter() - start,
    )
