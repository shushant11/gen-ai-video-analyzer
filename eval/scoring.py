"""Pure scoring helpers for the RAG evaluation. No LangChain / Chroma imports, so they are
fast to unit-test and run in CI."""

NOT_FOUND = "I could not find this information in the meeting transcript."


def find_span(text: str, span: str):
    """(start, end) of `span` in `text`, or None. The span must occur exactly once."""
    if text.count(span) != 1:
        return None
    start = text.index(span)
    return start, start + len(span)


def coverage(span: tuple, intervals: list) -> float:
    """Fraction of the span's characters that lie inside the union of retrieved chunks."""
    s, e = span
    if e <= s:
        return 0.0
    covered = 0
    cur = s
    for a, b in sorted(intervals):
        a, b = max(a, cur), min(b, e)
        if b > a:
            covered += b - a
            cur = b
    return covered / (e - s)


def score_retrieval(spans: list, intervals: list, threshold: float = 0.5) -> dict:
    """hit  = at least one evidence span is >= threshold covered by the retrieved chunks
       full = every evidence span is >= threshold covered"""
    covs = [coverage(sp, intervals) for sp in spans]
    return {
        "hit": any(c >= threshold for c in covs),
        "full": all(c >= threshold for c in covs),
        "coverage": sum(covs) / len(covs),
    }


def keyword_match(answer: str, groups: list) -> bool:
    """Every group needs at least one of its alternatives in the answer (case-insensitive)."""
    a = answer.lower()
    return all(any(alt.lower() in a for alt in g) for g in groups)


def is_not_found(answer: str) -> bool:
    return NOT_FOUND.lower() in answer.lower()


def score_answer(answer: str, answerable: bool, groups: list) -> bool:
    """Answerable question: must not be NOT_FOUND and must contain the keywords.
       Unanswerable question: must say NOT_FOUND."""
    if answerable:
        return (not is_not_found(answer)) and keyword_match(answer, groups)
    return is_not_found(answer)


def summarise(rows: list) -> dict:
    """rows: list of dicts with 'hit' and 'full' booleans (answerable questions only)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "hit_rate": 0.0, "full_rate": 0.0}
    return {
        "n": n,
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "full_rate": sum(r["full"] for r in rows) / n,
    }
