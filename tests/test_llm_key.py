"""The hosted demo gives every visitor their own Gemini key; one visitor's key must never be used for another."""
import threading

from core import llm


class FakeChat:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeChat.last_kwargs = kwargs

    def with_retry(self, **kwargs):
        return self


def _key_used():
    llm.get_llm()
    return FakeChat.last_kwargs["google_api_key"]


def test_session_key_overrides_environment(monkeypatch):
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", FakeChat)
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    llm.set_api_key("visitor-key")
    try:
        assert _key_used() == "visitor-key"
    finally:
        llm.set_api_key(None)


def test_blank_session_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", FakeChat)
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    llm.set_api_key("   ")
    assert _key_used() == "env-key"


def test_key_set_in_one_thread_does_not_leak_to_another(monkeypatch):
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", FakeChat)
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    llm.set_api_key(None)

    def other_visitor():
        llm.set_api_key("other-visitor-key")
        assert _key_used() == "other-visitor-key"

    t = threading.Thread(target=other_visitor)
    t.start()
    t.join()
    assert _key_used() == "env-key"
