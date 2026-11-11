from satyameter.llm.base import LLMClient
from satyameter.verification.claims import extract_claims, parse_claims, strip_citations


class ScriptedLLM(LLMClient):
    name = "fake:scripted"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.last_user = None

    def generate(self, system, user, *, temperature=0.0, max_tokens=1024):
        self.calls += 1
        self.last_user = user
        return self.reply


def test_parse_plain_json_array():
    raw = '["Fleming discovered penicillin.", "It was discovered in 1928."]'
    assert parse_claims(raw) == ["Fleming discovered penicillin.", "It was discovered in 1928."]


def test_parse_json_inside_markdown_fence_and_chatter():
    raw = 'Sure! Here you go:\n```json\n["A is B.", "C is D."]\n```\nHope that helps.'
    assert parse_claims(raw) == ["A is B.", "C is D."]


def test_parse_falls_back_to_bullet_lines():
    raw = "- Fleming was Scottish.\n- Penicillin is an antibiotic.\n"
    assert parse_claims(raw) == ["Fleming was Scottish.", "Penicillin is an antibiotic."]


def test_parse_strips_citations_and_dedupes():
    raw = '["Fleming discovered it [2].", "fleming discovered it.", "Another fact here [1, 3]."]'
    assert parse_claims(raw) == ["Fleming discovered it.", "Another fact here."]


def test_strip_citations():
    assert strip_citations("It happened in 1928 [2]. Also true [1, 2].") == "It happened in 1928. Also true."


def test_not_found_answer_makes_no_llm_call():
    llm = ScriptedLLM('["should not be used"]')
    assert extract_claims("Q?", "I could not find this in the knowledge base.", llm) == []
    assert llm.calls == 0


def test_extract_claims_sends_question_and_clean_answer():
    llm = ScriptedLLM('["Fleming discovered penicillin."]')
    claims = extract_claims("Who discovered penicillin?", "Fleming did [2].", llm)
    assert claims == ["Fleming discovered penicillin."]
    assert "Who discovered penicillin?" in llm.last_user and "[2]" not in llm.last_user
