from satyameter.llm.base import LLMClient
from satyameter.rewriter import rewrite_query


class FakeLLM(LLMClient):
    name = "fake:rewrite"

    def __init__(self, reply):
        self.reply = reply
        self.last_system = ""
        self.last_user = ""

    def generate(self, system, user, *, temperature=0.0, max_tokens=1024):
        self.last_system = system
        self.last_user = user
        return self.reply


def test_rewriter_returns_trimmed_query_and_preserves_question():
    llm = FakeLLM("  Search query: When did Chandrayaan-3 land on the Moon?  ")
    result = rewrite_query("When did Chandrayaan-3 land?", llm)
    assert result == "When did Chandrayaan-3 land on the Moon?"
    assert "Do not answer" in llm.last_system
    assert "Chandrayaan-3" in llm.last_user


def test_rewriter_falls_back_on_empty_or_overlong_output():
    assert rewrite_query("  Who built the Taj Mahal?  ", FakeLLM("")) == "Who built the Taj Mahal?"
    too_long = "word " * 50
    assert rewrite_query("Who built the Taj Mahal?", FakeLLM(too_long)) == "Who built the Taj Mahal?"


def test_rewriter_rejects_non_answer_sentinel():
    assert rewrite_query("What is photosynthesis?", FakeLLM("N/A")) == "What is photosynthesis?"
