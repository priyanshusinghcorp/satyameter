from satyameter.evaluation.score import detector_prf, format_report, hallucination_rate, labelled, summarize_eval


def row(system, status, ok):
    return {"system": system, "verifier_status": status, "human_ok": ok}


ROWS = [
    row("rag", "SUPPORTED", "1"),
    row("rag", "SUPPORTED", "1"),
    row("rag", "SUPPORTED", "0"),     # missed hallucination
    row("rag", "NOT_FOUND", "0"),     # caught
    row("rag", "CONTRADICTED", "1"),  # false alarm
    row("unguarded", "NOT_FOUND", "0"),
    row("unguarded", "SUPPORTED", "1"),
    row("unguarded", "SUPPORTED", ""),  # unlabelled, must be ignored
]


def test_blank_labels_are_ignored():
    assert len(labelled(ROWS)) == 7


def test_hallucination_rate():
    rag = [r for r in labelled(ROWS) if r["system"] == "rag"]
    assert hallucination_rate(rag) == 2 / 5
    assert hallucination_rate([]) is None


def test_detector_confusion_matrix_and_prf():
    d = detector_prf(labelled(ROWS))
    assert (d["tp"], d["fp"], d["fn"], d["tn"]) == (2, 1, 1, 3)
    assert d["precision"] == 2 / 3 and d["recall"] == 2 / 3
    assert abs(d["f1"] - 2 / 3) < 1e-9


def test_guard_reduces_hallucination_rate_in_summary():
    s = summarize_eval(ROWS)["systems"]["rag"]
    assert s["hallucination_rate"] == 2 / 5
    assert s["passed_by_guard"] == 3 / 5
    assert s["hallucination_rate_after_guard"] == 1 / 3


def test_report_renders_without_data_errors():
    text = format_report(summarize_eval(ROWS))
    assert "Human-labelled claims: 7" in text and "| rag |" in text
    assert "Contradiction-only detector" in text
    assert "NOT_FOUND" in text


def test_empty_detector_gives_none_not_crash():
    d = detector_prf([])
    assert d["precision"] is None and d["recall"] is None and d["f1"] is None


def test_not_found_is_separate_from_contradiction_only_detector():
    from satyameter.evaluation.score import contradiction_detector_prf

    rows = [
        row("rag", "NOT_FOUND", "1"),      # true claim absent from corpus
        row("rag", "CONTRADICTED", "0"),  # false claim with conflict
        row("rag", "SUPPORTED", "1"),
    ]
    unsupported = detector_prf(rows)
    contrad_only = contradiction_detector_prf(rows)
    assert (unsupported["tp"], unsupported["fp"]) == (1, 1)
    assert (contrad_only["tp"], contrad_only["fp"]) == (1, 0)
