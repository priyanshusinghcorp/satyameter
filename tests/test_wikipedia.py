import pytest
import requests

from satyameter.ingestion import wikipedia


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def article_payload(title="Penicillin"):
    return {"query": {"pages": [{"title": title, "extract": "Penicillin was discovered in 1928."}]}}


def test_retries_after_429_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(wikipedia.time, "sleep", lambda s: sleeps.append(s))
    session = FakeSession([
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, article_payload()),
    ])
    article = wikipedia.fetch_article("Penicillin", session)
    assert article["title"] == "Penicillin"
    assert article["url"].endswith("/Penicillin")
    assert session.calls == 2
    assert sleeps and sleeps[0] >= 7


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(wikipedia.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse(429)] * wikipedia.MAX_RETRIES)
    with pytest.raises(requests.RequestException):
        wikipedia.fetch_article("X", session)


def test_user_agent_includes_contact(monkeypatch):
    monkeypatch.setenv("WIKI_CONTACT", "me@example.com")
    assert "me@example.com" in wikipedia.user_agent()


def test_missing_article_returns_none():
    page = {"title": "Zzz", "missing": True}
    session = FakeSession([FakeResponse(200, {"query": {"pages": [page]}})])
    assert wikipedia.fetch_article("Zzz", session) is None
