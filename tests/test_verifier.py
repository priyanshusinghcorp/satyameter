from satyameter.ingestion.chunker import Chunk
from satyameter.retrieval.index import Hit
from satyameter.verification.nli import NLIScores, map_labels
from satyameter.verification.verifier import (
    ClaimVerdict,
    EvidenceScore,
    Status,
    Thresholds,
    decide,
    summarize,
    verify_claim,
)

T = Thresholds()


def chunk(i, text="evidence text here"):
    return Chunk(f"c{i}", f"Doc {i}", f"https://x/{i}", "Introduction", i, text)


def ev(i, ent=0.0, con=0.0):
    return EvidenceScore(chunk(i), NLIScores(ent, con, max(0.0, 1 - ent - con)), "claim")


def test_map_labels_reads_order_from_model():
    assert map_labels({0: "contradiction", 1: "entailment", 2: "neutral"}) == {
        "contradiction": 0, "entailment": 1, "neutral": 2,
    }
    assert map_labels({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"})["entailment"] == 0


def test_decide_supported_when_any_chunk_entails():
    status, best = decide([ev(1, ent=0.1, con=0.95), ev(2, ent=0.9)], T)
    assert status is Status.SUPPORTED and best.chunk.chunk_id == "c2"


def test_decide_contradicted_needs_strong_contradiction_and_no_entailment():
    status, best = decide([ev(1, ent=0.05, con=0.97)], T)
    assert status is Status.CONTRADICTED and best.chunk.chunk_id == "c1"


def test_decide_moderate_entailment_blocks_contradiction():
    status, _ = decide([ev(1, ent=0.5, con=0.95)], T)
    assert status is Status.NOT_FOUND


def test_decide_neutral_or_empty_is_not_found():
    assert decide([ev(1, ent=0.1, con=0.1)], T)[0] is Status.NOT_FOUND
    assert decide([], T) == (Status.NOT_FOUND, None)


def verdict(status):
    return ClaimVerdict("c", status, None)


def test_summarize_scores():
    verdicts = [verdict(Status.SUPPORTED)] * 6 + [verdict(Status.CONTRADICTED), verdict(Status.NOT_FOUND)]
    faithfulness, contradiction_rate, coverage = summarize(verdicts)
    assert faithfulness == 0.75 and contradiction_rate == 0.125 and coverage == 0.875


def test_summarize_empty_is_none():
    assert summarize([]) == (None, None, None)


class FakeIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k=5):
        return [Hit(c, 0.9, r) for r, c in enumerate(self.chunks[:k], start=1)]


class FakeNLI:
    """Entails when the evidence mentions 'penicillin'; contradicts when it mentions 'wrong'."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        out = []
        for evidence, _ in pairs:
            if "penicillin" in evidence.lower():
                out.append(NLIScores(0.95, 0.02, 0.03))
            elif "wrong" in evidence.lower():
                out.append(NLIScores(0.01, 0.98, 0.01))
            else:
                out.append(NLIScores(0.05, 0.05, 0.9))
        return out


def test_verify_claim_merges_claim_and_answer_evidence_without_duplicates():
    shared = chunk(1, "Penicillin was found by Fleming.")
    index = FakeIndex([shared, chunk(2, "Unrelated text.")])
    answer_hits = [Hit(shared, 0.8, 1), Hit(chunk(3, "More unrelated."), 0.7, 2)]
    nli = FakeNLI()
    v = verify_claim("Fleming found penicillin.", index, nli, answer_hits=answer_hits, claim_k=2)
    ids = [e.chunk.chunk_id for e in v.evidence]
    assert ids == ["c1", "c2", "c3"]  # claim hits first, shared chunk not repeated
    assert v.status is Status.SUPPORTED and len(nli.pairs) == 3


def test_verify_claim_caps_evidence():
    index = FakeIndex([chunk(i) for i in range(10)])
    v = verify_claim("anything", index, FakeNLI(), claim_k=10, max_evidence=4)
    assert len(v.evidence) == 4


def test_verify_claim_contradicted_and_not_found():
    wrong = FakeIndex([chunk(1, "This is wrong information.")])
    assert verify_claim("claim", wrong, FakeNLI()).status is Status.CONTRADICTED
    silent = FakeIndex([chunk(1, "Nothing relevant at all.")])
    assert verify_claim("claim", silent, FakeNLI()).status is Status.NOT_FOUND


def test_is_negated_detects_absence_claims_only():
    from satyameter.verification.verifier import is_negated

    assert is_negated("Albert Einstein never received a Nobel Prize in Chemistry.")
    assert is_negated("Fleming did not discover insulin.")
    assert is_negated("He wasn't born in Japan.")
    assert not is_negated("Albert Einstein was awarded the Nobel Prize in Physics in 1921.")
    assert not is_negated("Notable works include the Nobel lecture.")
