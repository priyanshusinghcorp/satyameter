from satyameter.config import Settings
from satyameter.ingestion.chunker import Chunk
from satyameter.llm.base import LLMClient
from satyameter.pipeline import Satyameter
from satyameter.retrieval.index import Hit
from satyameter.verification.nli import NLIScores
from satyameter.verification.verifier import Status, Thresholds, rethreshold


class FakeIndex:
    def search(self, query, k=5):
        c = Chunk("c1", "Penicillin", "https://x/p", "Discovery", 0, "Fleming discovered penicillin in 1928.")
        return [Hit(c, 0.9, 1)]


class FakeLLM(LLMClient):
    name = "fake:llm"

    def generate(self, system, user, *, temperature=0.0, max_tokens=1024):
        if "search-query editor" in system:
            return "Who discovered penicillin?"
        if "break an answer into atomic" in system:
            return '["Fleming discovered penicillin.", "Penicillin was discovered in 1945."]'
        if "ONLY the numbered passages" in system:
            return "Fleming discovered penicillin in 1928 [1]."
        return "Fleming discovered penicillin in 1945."  # unguarded answer from memory


class FakeNLI:
    def predict(self, pairs):
        out = []
        for _, claim in pairs:
            out.append(NLIScores(0.01, 0.98, 0.01) if "1945" in claim else NLIScores(0.97, 0.01, 0.02))
        return out


def make():
    return Satyameter(Settings(), index=FakeIndex(), llm=FakeLLM(), nli=FakeNLI())


def test_guarded_pipeline_end_to_end():
    answer, result = make().ask("Who discovered penicillin?", guarded=True)
    assert "[1]" in answer.answer and answer.hits
    statuses = [v.status for v in result.verdicts]
    assert statuses == [Status.SUPPORTED, Status.CONTRADICTED]
    assert result.faithfulness == 0.5 and result.coverage == 1.0


def test_unguarded_pipeline_uses_no_retrieval_but_is_still_verified():
    answer, result = make().ask("Who discovered penicillin?", guarded=False)
    assert answer.hits == [] and "1945" in answer.answer
    assert len(result.verdicts) == 2


def test_rethreshold_changes_status_without_new_model_calls():
    _, result = make().ask("Q?", guarded=True)
    strict = rethreshold(result, Thresholds(entail=0.99, contradict=0.99))
    assert strict.verdicts[0].status is Status.NOT_FOUND  # 0.97 entailment no longer enough
    assert strict.faithfulness == 0.0
