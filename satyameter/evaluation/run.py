"""Generate the claims to be hand-labelled: unguarded LLM vs RAG, each claim verified."""

from __future__ import annotations

import csv
from pathlib import Path

from satyameter.verification.verifier import is_negated

FIELDS = [
    "system", "qid", "qtype", "question", "rewritten_query", "answer", "claim",
    "verifier_status", "negated", "entailment", "contradiction", "evidence_title",
    "evidence_url", "evidence_snippet", "human_ok",
]
# Ablation design: raw LLM baseline, RAG with the untouched query, and RAG with rewriting.
SYSTEMS = (
    ("unguarded", False, False),
    ("raw_query_rag", True, False),
    ("rewritten_rag", True, True),
)


def read_questions(path: Path, limit: int | None = None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig so Excel opens it cleanly
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_eval(pipe, questions: list[dict], progress=print) -> list[dict]:
    rows: list[dict] = []
    for i, q in enumerate(questions, start=1):
        progress(f"[{i}/{len(questions)}] {q['question']}")
        for system, guarded, rewrite in SYSTEMS:
            answer, result = pipe.ask(q["question"], guarded=guarded, rewrite=rewrite)
            for verdict in result.verdicts:
                best = verdict.best
                rows.append({
                    "system": system,
                    "qid": q["qid"],
                    "qtype": q["qtype"],
                    "question": q["question"],
                    "rewritten_query": answer.rewritten_query or "",
                    "answer": answer.answer,
                    "claim": verdict.claim,
                    "verifier_status": verdict.status.value,
                    "negated": "1" if is_negated(verdict.claim) else "0",
                    "entailment": f"{best.scores.entailment:.3f}" if best else "",
                    "contradiction": f"{best.scores.contradiction:.3f}" if best else "",
                    "evidence_title": f"{best.chunk.doc_title} - {best.chunk.section}" if best else "",
                    "evidence_url": best.chunk.url if best else "",
                    "evidence_snippet": best.chunk.text[:300] if best else "",
                    "human_ok": "",
                })
    return rows
