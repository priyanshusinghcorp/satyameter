"""Compute evaluation metrics from the manually reviewed claims CSV.

human_ok meaning: 1 = factually correct, 0 = factually incorrect or fabricated.
A NOT_FOUND verifier status is not equivalent to a false claim: it means the
retrieved evidence did not establish the claim. Reports therefore show both
(1) unsupported-evidence detection and (2) contradiction-only detection.
"""

from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_FIELDS = {"system", "verifier_status", "human_ok"}
VALID_STATUSES = {"SUPPORTED", "CONTRADICTED", "NOT_FOUND"}


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Evaluation CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    for line, row in enumerate(rows, start=2):
        status = (row.get("verifier_status") or "").strip().upper()
        if status and status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid verifier_status '{status}' on CSV row {line}. "
                f"Expected one of {sorted(VALID_STATUSES)}."
            )
        row["verifier_status"] = status
        row["human_ok"] = (row.get("human_ok") or "").strip()
        if row["human_ok"] not in ("", "0", "1"):
            raise ValueError(f"human_ok must be blank, 0, or 1 on CSV row {line}.")
    return rows


def labelled(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("human_ok", "").strip() in ("0", "1")]


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def hallucination_rate(rows: list[dict]) -> float | None:
    """Fraction of human-labelled claims judged factually incorrect."""
    return _ratio(sum(r["human_ok"].strip() == "0" for r in rows), len(rows))


def _binary_metrics(rows: list[dict], predicted_positive) -> dict:
    tp = sum(r["human_ok"] == "0" and predicted_positive(r) for r in rows)
    fp = sum(r["human_ok"] == "1" and predicted_positive(r) for r in rows)
    fn = sum(r["human_ok"] == "0" and not predicted_positive(r) for r in rows)
    tn = sum(r["human_ok"] == "1" and not predicted_positive(r) for r in rows)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def detector_prf(rows: list[dict]) -> dict:
    """Predict 'not evidence-supported' when status is NOT_FOUND or CONTRADICTED."""
    return _binary_metrics(rows, lambda r: r["verifier_status"] != "SUPPORTED")


def contradiction_detector_prf(rows: list[dict]) -> dict:
    """Conservative detector: predict a possible false claim only on CONTRADICTED."""
    return _binary_metrics(rows, lambda r: r["verifier_status"] == "CONTRADICTED")


def summarize_eval(rows: list[dict]) -> dict:
    rows = labelled(rows)
    overall = detector_prf(rows)
    contradiction_overall = contradiction_detector_prf(rows)
    out: dict = {
        "n_labelled": len(rows),
        "systems": {},
        "overall": overall,
        "contradiction_detector": contradiction_overall,
    }
    for system in sorted({r["system"] for r in rows}):
        sys_rows = [r for r in rows if r["system"] == system]
        passed = [r for r in sys_rows if r["verifier_status"] == "SUPPORTED"]
        out["systems"][system] = {
            "n": len(sys_rows),
            "hallucination_rate": hallucination_rate(sys_rows),
            "passed_by_guard": _ratio(len(passed), len(sys_rows)),
            "hallucination_rate_after_guard": hallucination_rate(passed),
            "detector": detector_prf(sys_rows),
            "contradiction_detector": contradiction_detector_prf(sys_rows),
            "statuses": {
                status: sum(r["verifier_status"] == status for r in sys_rows)
                for status in sorted(VALID_STATUSES)
            },
        }
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _metric_table(scopes: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "| Scope | TP | FP | FN | TN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in scopes:
        lines.append(
            f"| {name} | {d['tp']} | {d['fp']} | {d['fn']} | {d['tn']} "
            f"| {_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} |"
        )
    return lines


def format_report(summary: dict) -> str:
    lines = [
        f"Human-labelled claims: {summary['n_labelled']}",
        "",
        "## System-level outcomes",
        "",
        "| System | Claims | Actual hallucination rate | Marked Supported | Hallucination rate among Supported |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, s in summary["systems"].items():
        lines.append(
            f"| {name} | {s['n']} | {_pct(s['hallucination_rate'])} | "
            f"{_pct(s['passed_by_guard'])} | {_pct(s['hallucination_rate_after_guard'])} |"
        )
    lines += ["", "## Verifier as an unsupported-evidence flag", "",
              "Positive prediction = NOT_FOUND or CONTRADICTED. This measures lack of evidence support, "
              "not factual falsity; a true claim may be absent from the corpus.", ""]
    scopes = [("overall", summary["overall"])] + [(n, s["detector"]) for n, s in summary["systems"].items()]
    lines += _metric_table(scopes)
    lines += ["", "## Contradiction-only detector", "",
              "Positive prediction = CONTRADICTED only. NOT_FOUND remains an unverified/unknown outcome.", ""]
    contradiction_scopes = [("overall", summary["contradiction_detector"])] + [
        (n, s["contradiction_detector"]) for n, s in summary["systems"].items()
    ]
    lines += _metric_table(contradiction_scopes)
    lines += ["", "## Verifier status distribution", "",
              "| System | Supported | Contradicted | Not found |", "|---|---:|---:|---:|"]
    for name, s in summary["systems"].items():
        lines.append(
            f"| {name} | {s['statuses']['SUPPORTED']} | {s['statuses']['CONTRADICTED']} "
            f"| {s['statuses']['NOT_FOUND']} |"
        )
    lines += ["", "Interpretation: the evidence-grounded score is Supported / all extracted claims. "
              "It is not a measure of absolute truth."]
    return "\n".join(lines)
