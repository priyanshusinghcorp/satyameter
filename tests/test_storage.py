from types import SimpleNamespace

from satyameter.ingestion.chunker import Chunk
from satyameter.storage import SQLiteStore
from satyameter.verification.nli import NLIScores
from satyameter.verification.verifier import ClaimVerdict, Status


def test_save_run_and_read_back_claims(tmp_path):
    chunk = Chunk("c1", "Penicillin", "https://example.org/penicillin", "Discovery", 0,
                  "Alexander Fleming discovered penicillin in 1928.")
    evidence = SimpleNamespace(
        chunk=chunk,
        scores=NLIScores(0.96, 0.02, 0.02),
    )
    answer = SimpleNamespace(
        question="Who discovered penicillin?",
        rewritten_query="Who discovered penicillin?",
        answer="Alexander Fleming discovered penicillin in 1928 [1].",
        model="fake:test",
    )
    verdict = ClaimVerdict("Alexander Fleming discovered penicillin in 1928.", Status.SUPPORTED, evidence, [evidence])
    verification = SimpleNamespace(
        verdicts=[verdict], faithfulness=1.0, contradiction_rate=0.0,
        coverage=1.0, latency_s=0.05,
    )

    store = SQLiteStore(tmp_path / "history.sqlite")
    run_id = store.save_run(answer, verification, "Guarded RAG")
    runs = store.recent_runs(5)
    claims = store.claims_for_run(run_id)

    assert len(runs) == 1
    assert runs[0]["question"] == answer.question
    assert runs[0]["rewritten_query"] == answer.rewritten_query
    assert len(claims) == 1
    assert claims[0]["status"] == "SUPPORTED"
    assert claims[0]["evidence_url"] == chunk.url


def test_history_limit_is_bounded(tmp_path):
    store = SQLiteStore(tmp_path / "history.sqlite")
    assert store.recent_runs(0) == []
    assert store.recent_runs(1000) == []
