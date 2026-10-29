"""Tests for the action-item / decision / open-question extraction logic. No API key or network needed:
the LLM chain is replaced by a fake that returns canned text."""
import pytest

from core import extractor


class FakeChain:
    """Stands in for `prompt | llm | parser`: records calls and returns the next canned answer."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload["text"])
        return self.answers.pop(0)


@pytest.fixture(autouse=True)
def clear_cache():
    extractor._extract_all.cache_clear()
    yield
    extractor._extract_all.cache_clear()


COMBINED = """### ACTION ITEMS
1. Send the report - Priya - Friday

### KEY DECISIONS
1. Launch moves to March

### OPEN QUESTIONS
1. Who owns the budget?
"""


def test_parse_sections_reads_all_three_headings():
    out = extractor.parse_sections(COMBINED)
    assert set(out) == {"ACTION ITEMS", "KEY DECISIONS", "OPEN QUESTIONS"}
    assert "Priya" in out["ACTION ITEMS"]
    assert out["OPEN QUESTIONS"].startswith("1. Who owns the budget?")


@pytest.mark.parametrize("heading_line", ["### ACTION ITEMS", "**ACTION ITEMS**", "ACTION ITEMS:", "## action items"])
def test_parse_sections_tolerates_heading_styles(heading_line):
    out = extractor.parse_sections(f"{heading_line}\n1. Do the thing\n")
    assert out == {"ACTION ITEMS": "1. Do the thing"}


def test_short_transcript_uses_one_llm_call(monkeypatch):
    fake = FakeChain([COMBINED])
    monkeypatch.setattr(extractor, "_chain", lambda system_prompt: fake)
    assert "Priya" in extractor.extract_action_items("short transcript")
    assert "March" in extractor.extract_key_decisions("short transcript")   # served from the cache
    assert len(fake.calls) == 1


def test_missing_section_is_extracted_separately(monkeypatch):
    only_two = "### ACTION ITEMS\n1. A\n\n### KEY DECISIONS\n1. B\n"
    fakes = {}

    def fake_chain(system_prompt):
        key = "combined" if "THREE sections" in system_prompt else "open"
        fakes.setdefault(key, FakeChain([only_two] if key == "combined" else ["1. Fallback question"]))
        return fakes[key]

    monkeypatch.setattr(extractor, "_chain", fake_chain)
    assert extractor.extract_questions("transcript") == "1. Fallback question"
    assert len(fakes["combined"].calls) == 1 and len(fakes["open"].calls) == 1


def test_long_transcript_is_split_then_merged(monkeypatch):
    monkeypatch.setattr(extractor, "MAX_CHARS", 1000)
    monkeypatch.setattr(extractor, "PAUSE_SECONDS", 0)
    seen = []

    class Recorder:
        def invoke(self, payload):
            seen.append(len(payload["text"]))
            return "1. merged result"

    monkeypatch.setattr(extractor, "_chain", lambda system_prompt: Recorder())
    long_text = ("word " * 1000).strip()         # about 5,000 characters, well above MAX_CHARS
    assert extractor.extract_action_items(long_text) == "1. merged result"
    assert len(seen) > 3                          # several parts per heading, plus the merge call
