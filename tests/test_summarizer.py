"""Tests for the summariser's chunking and its single-call vs map-reduce switch (no API key needed)."""
from core import summarizer


class FakeLLMChain:
    def __init__(self, log):
        self.log = log

    def invoke(self, payload):
        self.log.append(payload["text"])
        return "- point"


def _patch_llm(monkeypatch, log):
    """Replace `prompt | llm | parser` with a recorder so no model is called."""
    class FakeStep:                      # result of `prompt | llm`; `| parser` completes the chain
        def __or__(self, other):
            return FakeLLMChain(log)

    class FakePrompt:
        def __or__(self, other):
            return FakeStep()

    monkeypatch.setattr(summarizer, "_prompt", lambda system: FakePrompt())
    monkeypatch.setattr(summarizer, "get_llm", lambda: object())
    monkeypatch.setattr(summarizer, "PAUSE_SECONDS", 0)


def test_split_transcript_respects_chunk_size():
    text = ("This is a sentence about the project. " * 600).strip()
    chunks = summarizer.split_transcript(text)
    assert len(chunks) > 1
    assert all(len(c) <= summarizer.CHUNK_SIZE for c in chunks)


def test_short_transcript_is_summarised_in_one_call(monkeypatch):
    log = []
    _patch_llm(monkeypatch, log)
    assert summarizer.summarize("short meeting") == "- point"
    assert log == ["short meeting"]


def test_long_transcript_uses_map_reduce(monkeypatch):
    log = []
    _patch_llm(monkeypatch, log)
    long_text = ("Budget review and hiring plan. " * 1500)[: summarizer.SINGLE_CALL_LIMIT + 5000]
    summarizer.summarize(long_text)
    n_chunks = len(summarizer.split_transcript(long_text))
    assert len(log) == n_chunks + 1               # one call per chunk plus the final merge
