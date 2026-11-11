from types import SimpleNamespace

from satyameter.evaluation.run import FIELDS, run_eval
from satyameter.verification.verifier import ClaimVerdict, Status


class FakePipe:
    def __init__(self):
        self.calls = []

    def ask(self, question, *, guarded, rewrite):
        self.calls.append((question, guarded, rewrite))
        answer = SimpleNamespace(
            answer=f"Answer for {question}",
            rewritten_query="Rewritten query" if rewrite else None,
        )
        verification = SimpleNamespace(
            verdicts=[ClaimVerdict("A test factual claim.", Status.SUPPORTED, None, [])]
        )
        return answer, verification


def test_evaluation_runs_three_ablation_conditions_and_logs_rewrites():
    pipe = FakePipe()
    rows = run_eval(pipe, [{"qid": "Q1", "qtype": "in_kb", "question": "Question?"}], progress=lambda _: None)
    assert pipe.calls == [
        ("Question?", False, False),
        ("Question?", True, False),
        ("Question?", True, True),
    ]
    assert [row["system"] for row in rows] == ["unguarded", "raw_query_rag", "rewritten_rag"]
    assert rows[-1]["rewritten_query"] == "Rewritten query"
    assert rows[0]["rewritten_query"] == ""
    assert set(rows[0]) == set(FIELDS)
