"""Hybrid retrieval: BM25 keyword search fused with the embedding search (reciprocal rank fusion).

Embeddings match meaning but can miss exact tokens such as "number four"; BM25 matches exact
words but not paraphrases. Fusing the two rankings covers both. No extra dependencies.
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"\w+")
_STOP = frozenset(
    "the a an is are was were of to and in on for it its that this what how which who do does did "
    "you your me my i we they he she be been with as at by from or about".split()
)


def tokenize(text: str) -> list:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = (sum(self.dl) / len(texts)) if texts else 0.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(texts)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> list:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list:
        """Indices of the best-matching texts, best first; texts with no matching word are left out."""
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        return [i for i in order[:k] if sc[i] > 0]


def rrf(rankings: list, c: int = 60) -> list:
    """Reciprocal rank fusion. rankings: lists of ids, best first. Returns fused ids, best first."""
    score = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1.0 / (c + rank + 1)
    return sorted(score, key=lambda i: (-score[i], i))


class HybridRetriever:
    """Works on any Chroma-like store that has .get() (all chunks) and .similarity_search()."""

    def __init__(self, vector_store, k: int = 6, pool: int = 20):
        data = vector_store.get()
        self.store = vector_store
        self.texts = list(data["documents"])
        self.metas = list(data.get("metadatas") or [{}] * len(self.texts))
        self.k, self.pool = k, pool
        self.bm25 = BM25(self.texts)
        self._index = {}
        for i, t in enumerate(self.texts):
            self._index.setdefault(t, i)

    def retrieve(self, query: str, k: int = None) -> list:
        from langchain_core.documents import Document

        k = k or self.k
        dense = [self._index[d.page_content] for d in self.store.similarity_search(query, k=self.pool)
                 if d.page_content in self._index]
        keyword = self.bm25.top(query, self.pool)
        fused = rrf([dense, keyword])[:k]
        return [Document(page_content=self.texts[i], metadata=self.metas[i] or {}) for i in fused]
