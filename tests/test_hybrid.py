from core.hybrid import BM25, HybridRetriever, rrf, tokenize

TEXTS = [
    "Project number four, RAG with Eval. A company has 10,000 documents.",
    "The system first searches for three or four documents and sends them to the AI.",
    "Project number five is Agentic Code Reviewer. It gives a structured review.",
    "Install Ollama and pull a model. Compare three models on your own laptop.",
]


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("What is the Project NUMBER four?") == ["project", "number", "four"]


def test_bm25_finds_exact_words():
    bm = BM25(TEXTS)
    assert bm.top("What is project number five?", 1) == [2]
    assert bm.top("Which tool runs local models, Ollama?", 1) == [3]


def test_bm25_ignores_texts_without_any_query_word():
    assert BM25(TEXTS).top("favourite colour", 3) == []


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([[1, 2, 3], [2, 4, 1]])[:2] == [2, 1]
    assert rrf([[7], []]) == [7]
    assert rrf([]) == []


class _Doc:
    def __init__(self, text):
        self.page_content = text


class FakeStore:
    """Dense search that always puts chunk 1 first, like an embedding model that misses 'number four'."""

    def get(self):
        return {"documents": TEXTS, "metadatas": [{"start_index": i * 100} for i in range(len(TEXTS))]}

    def similarity_search(self, query, k=4):
        return [_Doc(TEXTS[i]) for i in (1, 3, 2, 0)][:k]


def test_hybrid_recovers_a_chunk_that_dense_search_ranks_last(monkeypatch):
    import sys, types

    mod = types.ModuleType("langchain_core.documents")
    mod.Document = lambda page_content, metadata: types.SimpleNamespace(page_content=page_content, metadata=metadata)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", mod)

    h = HybridRetriever(FakeStore(), k=2, pool=4)
    got = h.retrieve("What is project number four?")
    assert TEXTS[0] in [d.page_content for d in got]       # found by keywords
    assert 0 in [d.metadata["start_index"] for d in got]
