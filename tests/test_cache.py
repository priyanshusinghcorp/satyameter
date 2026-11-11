from satyameter.llm.base import LLMClient
from satyameter.llm.cache import CachedLLM


class FakeLLM(LLMClient):
    name = "fake:test"

    def __init__(self):
        self.calls = 0

    def generate(self, system, user, *, temperature=0.0, max_tokens=1024):
        self.calls += 1
        return f"echo: {user}"


def test_second_identical_call_hits_cache(tmp_path):
    inner = FakeLLM()
    llm = CachedLLM(inner, tmp_path / "cache.sqlite")
    assert llm.generate("sys", "hello") == "echo: hello"
    assert llm.generate("sys", "hello") == "echo: hello"
    assert inner.calls == 1
    assert (llm.hits, llm.misses) == (1, 1)


def test_different_prompt_or_params_miss_cache(tmp_path):
    inner = FakeLLM()
    llm = CachedLLM(inner, tmp_path / "cache.sqlite")
    llm.generate("sys", "a")
    llm.generate("sys", "b")
    llm.generate("sys", "a", temperature=0.7)
    assert inner.calls == 3


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "cache.sqlite"
    CachedLLM(FakeLLM(), path).generate("sys", "persist me")
    second_inner = FakeLLM()
    CachedLLM(second_inner, path).generate("sys", "persist me")
    assert second_inner.calls == 0


def test_cache_can_be_used_from_another_thread(tmp_path):
    """Regression test: Streamlit calls the cache from a different thread than the one that created it."""
    import threading

    llm = CachedLLM(FakeLLM(), tmp_path / "cache.sqlite")
    llm.generate("sys", "made in main thread")
    result = {}

    def worker():
        try:
            result["value"] = llm.generate("sys", "made in main thread")
            result["new"] = llm.generate("sys", "made in worker thread")
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert "error" not in result, result.get("error")
    assert result["value"] == "echo: made in main thread"
