import json
from pathlib import Path

from eval.scoring import (NOT_FOUND, coverage, find_span, is_not_found, keyword_match,
                          score_answer, score_retrieval, summarise)

EVAL = Path(__file__).resolve().parent.parent / "eval"


def test_coverage_union_of_chunks():
    # span 10..20; chunks cover 0..14 and 14..18 -> 8 of 10 characters
    assert coverage((10, 20), [(0, 14), (14, 18)]) == 0.8
    assert coverage((10, 20), [(30, 40)]) == 0.0
    assert coverage((10, 20), [(0, 100)]) == 1.0
    assert coverage((10, 20), [(8, 16), (12, 20)]) == 1.0  # overlapping chunks are not double counted


def test_score_retrieval_hit_vs_full():
    spans = [(0, 10), (100, 110)]
    r = score_retrieval(spans, [(0, 20)])
    assert r["hit"] and not r["full"] and r["coverage"] == 0.5
    r = score_retrieval(spans, [(0, 20), (95, 120)])
    assert r["hit"] and r["full"]
    r = score_retrieval(spans, [(200, 300)])
    assert not r["hit"] and not r["full"]


def test_find_span_requires_exactly_one_match():
    assert find_span("abc def abc", "def") == (4, 7)
    assert find_span("abc def abc", "abc") is None
    assert find_span("abc", "xyz") is None


def test_keyword_match_is_any_of_per_group():
    groups = [["sql"], ["clarification", "clarify"]]
    assert keyword_match("It is text to SQL with a Clarification engine", groups)
    assert not keyword_match("It is text to SQL", groups)


def test_score_answer_rules():
    g = [["61"], ["85"]]
    assert score_answer("From 61% to 85%.", True, g)
    assert not score_answer(NOT_FOUND, True, g)           # refusing an answerable question fails
    assert not score_answer("About 60%.", True, g)
    assert score_answer(NOT_FOUND, False, [])              # refusing an unanswerable one passes
    assert not score_answer("It is blue.", False, [])
    assert is_not_found(NOT_FOUND.upper())


def test_summarise():
    assert summarise([]) == {"n": 0, "hit_rate": 0.0, "full_rate": 0.0}
    s = summarise([{"hit": True, "full": True}, {"hit": True, "full": False}, {"hit": False, "full": False}, {"hit": True, "full": True}])
    assert s["n"] == 4 and s["hit_rate"] == 0.75 and s["full_rate"] == 0.5


def test_qa_set_is_consistent_with_transcript():
    text = (EVAL / "transcript_ai_projects.txt").read_text(encoding="utf-8")
    qa = json.loads((EVAL / "qa_set.json").read_text(encoding="utf-8"))
    assert 25 <= len(qa) <= 50
    assert len({q["id"] for q in qa}) == len(qa)
    for q in qa:
        if q["answerable"]:
            assert q["evidence"] and q["keywords"], q["id"]
            for span in q["evidence"]:
                assert find_span(text, span), f"{q['id']}: evidence must appear exactly once: {span!r}"
        else:
            assert not q["evidence"], q["id"]
