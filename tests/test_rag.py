from satyameter.generation.rag import NOT_FOUND_REPLY, answer_question, build_user_prompt, validate_citations
from satyameter.ingestion.chunker import Chunk
from satyameter.llm.base import LLMClient
from satyameter.retrieval.index import Hit


def make_hit(rank: int, title: str, text: str) -> Hit:
    chunk = Chunk(f"c-{rank}", title, f"https://example.org/{rank}", "Introduction", rank, text)
    return Hit(chunk=chunk, score=0.9 - rank * 0.1, rank=rank)


class FakeIndex:
    def __init__(self, hits):
        self._hits = hits
        self.last_k = None
        self.last_query = None

    def search(self, query, k=5):
        self.last_k = k
        self.last_query = query
        return self._hits[:k]


class RecordingLLM(LLMClient):
    name = "fake:rec"

    def __init__(self, reply="Einstein was born in 1879 [1]."):
        self.reply = reply
        self.last_user = None

    def generate(self, system, user, *, temperature=0.0, max_tokens=1024):
        self.last_user = user
        return self.reply


def test_prompt_contains_numbered_passages_and_question():
    hits = [make_hit(1, "Albert Einstein", "Einstein was born in Ulm in 1879."),
            make_hit(2, "Ulm", "Ulm is a city in Germany.")]
    prompt = build_user_prompt("Where was Einstein born?", hits)
    assert "[1] (Albert Einstein" in prompt and "[2] (Ulm" in prompt
    assert "Question: Where was Einstein born?" in prompt


def test_answer_question_returns_sources_and_respects_k():
    hits = [make_hit(i, f"Doc {i}", f"text {i}") for i in range(1, 6)]
    index, llm = FakeIndex(hits), RecordingLLM()
    result = answer_question("Q?", index, llm, k=3, rewrite=False)
    assert index.last_k == 3
    assert len(result.hits) == 3
    assert result.answer.startswith("Einstein")
    assert result.model == "fake:rec"


def test_no_hits_returns_not_found_without_calling_llm():
    llm = RecordingLLM()
    result = answer_question("Q?", FakeIndex([]), llm, rewrite=False)
    assert result.answer == NOT_FOUND_REPLY
    assert llm.last_user is None


def test_answer_question_uses_rewritten_query_for_search():
    hits = [make_hit(1, "Albert Einstein", "Einstein was born in Ulm in 1879.")]
    index = FakeIndex(hits)
    llm = RecordingLLM(reply="Where was Albert Einstein born?")
    result = answer_question("Where was Einstein born?", index, llm, rewrite=True)
    assert index.last_query == "Where was Albert Einstein born?"
    assert result.rewritten_query == index.last_query
    assert "Question: Where was Einstein born?" in llm.last_user


def test_invalid_citation_numbers_are_removed():
    assert validate_citations("Fact [1]. Another fact [3].", 2) == "Fact [1]. Another fact ."
    assert validate_citations("No source marker", 0) == "No source marker"
