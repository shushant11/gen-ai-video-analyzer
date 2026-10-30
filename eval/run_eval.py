"""Measure how well the RAG pipeline retrieves (and optionally answers) for a known transcript.

    python -m eval.run_eval                  # retrieval only: free, no API calls
    python -m eval.run_eval --answers        # also ask Gemini the questions with the default settings
    python -m eval.run_eval --answers --hybrid   # same, but with embeddings + BM25 keyword retrieval

Uses an in-memory Chroma index, so the app's vector_db/ folder is never touched.
"""
import argparse
import json
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.hybrid import HybridRetriever
from core.vector_store import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_K, get_embeddings
from eval.scoring import find_span, score_answer, score_retrieval, summarise

HERE = Path(__file__).parent
CHUNK_SIZES = [300, 500, 1000, 1500]
KS = [2, 4, 6, 8]


def load_inputs():
    text = (HERE / "transcript_ai_projects.txt").read_text(encoding="utf-8")
    qa = json.loads((HERE / "qa_set.json").read_text(encoding="utf-8"))
    for q in qa:
        q["spans"] = [find_span(text, s) for s in q["evidence"]]
        assert all(q["spans"]), f"{q['id']}: evidence not found exactly once"
    return text, qa


def build_index(text, size, overlap, embeddings, name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=min(overlap, size - 1), add_start_index=True
    )
    docs = splitter.create_documents([text])
    store = Chroma.from_documents(docs, embeddings, collection_name=name)  # in memory
    return store, len(docs)


def _intervals(docs):
    return [(d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)) for d in docs]


def retrieval_grid(text, qa, embeddings):
    """Dense (embeddings only) vs hybrid (embeddings + BM25 keyword search, rank-fused)."""
    answerable = [q for q in qa if q["answerable"]]
    results = []
    for size in CHUNK_SIZES:
        store, n_chunks = build_index(text, size, CHUNK_OVERLAP, embeddings, f"eval_{size}")
        hybrid = HybridRetriever(store, k=max(KS))
        ranked = {"dense": {}, "hybrid": {}}
        for q in answerable:
            ranked["dense"][q["id"]] = _intervals(store.similarity_search(q["question"], k=max(KS)))
            ranked["hybrid"][q["id"]] = _intervals(hybrid.retrieve(q["question"], k=max(KS)))
        for k in KS:
            row = dict(chunk_size=size, k=k, chunks=n_chunks, chance=min(1.0, k / n_chunks))
            for mode in ("dense", "hybrid"):
                s = summarise([score_retrieval(q["spans"], ranked[mode][q["id"]][:k]) for q in answerable])
                row[mode] = {"hit_rate": s["hit_rate"], "full_rate": s["full_rate"]}
                row["n"] = s["n"]
            results.append(row)
    return results


def answer_check(text, qa, embeddings, use_hybrid=False):
    from core.rag_engine import _chain_from, ask_question  # needs GOOGLE_API_KEY in .env

    store, _ = build_index(text, CHUNK_SIZE, CHUNK_OVERLAP, embeddings, "eval_answers")
    from langchain_core.runnables import RunnableLambda

    from core.vector_store import get_retriever

    if use_hybrid:
        retriever = RunnableLambda(HybridRetriever(store, k=DEFAULT_K).retrieve)
    else:
        retriever = get_retriever(store, k=DEFAULT_K)  # embeddings only
    chain = _chain_from(store, retriever)
    rows = []
    for q in qa:
        try:
            ans = ask_question(chain, q["question"])
        except Exception as e:  # rate limit etc.: record, don't crash the run
            ans = f"[ERROR] {e}"
        ok = score_answer(ans, q["answerable"], q["keywords"])
        rows.append({"id": q["id"], "answerable": q["answerable"], "question": q["question"], "answer": ans, "pass": ok})
        time.sleep(4)  # stay under free-tier rate limits
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also call Gemini (default settings only)")
    ap.add_argument("--hybrid", action="store_true", help="with --answers: use hybrid retrieval instead of embeddings only")
    args = ap.parse_args()

    text, qa = load_inputs()
    n_ans = sum(q["answerable"] for q in qa)
    print(f"Transcript: {len(text):,} characters | questions: {n_ans} answerable + {len(qa) - n_ans} unanswerable")
    embeddings = get_embeddings()

    grid = retrieval_grid(text, qa, embeddings)
    print("\nRetrieval hit rate (answerable questions only)")
    print("hit = a retrieved chunk covers >=50% of at least one evidence sentence")
    print("dense = embeddings only | hybrid = embeddings + BM25 keyword search | chance = k / chunks\n")
    print(f"{'chunk':>6} {'k':>3} {'chunks':>7} {'dense':>7} {'hybrid':>7} {'d-full':>7} {'h-full':>7} {'chance':>7}")
    for r in grid:
        mark = "  <- current app setting" if (r["chunk_size"], r["k"]) == (CHUNK_SIZE, DEFAULT_K) else ""
        print(f"{r['chunk_size']:>6} {r['k']:>3} {r['chunks']:>7} {r['dense']['hit_rate']:>7.0%} {r['hybrid']['hit_rate']:>7.0%} "
              f"{r['dense']['full_rate']:>7.0%} {r['hybrid']['full_rate']:>7.0%} {r['chance']:>7.0%}{mark}")

    out = {"transcript_chars": len(text), "retrieval": grid}
    if args.answers:
        rows = answer_check(text, qa, embeddings, use_hybrid=args.hybrid)
        a = [r for r in rows if r["answerable"]]
        u = [r for r in rows if not r["answerable"]]
        pa, pu = sum(r["pass"] for r in a), sum(r["pass"] for r in u)
        print(f"\nAnswers ({'hybrid' if args.hybrid else 'dense'}, chunk {CHUNK_SIZE}, k {DEFAULT_K}): answerable {pa}/{len(a)} keyword-match | "
              f"unanswerable {pu}/{len(u)} correctly refused")
        for r in rows:
            if not r["pass"]:
                print(f"  FAIL {r['id']}: {r['question']}\n       -> {r['answer'][:160]!r}")
        out["answers"] = rows

    (HERE / ("results_hybrid.json" if args.hybrid else "results.json")).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved results file in eval/")


if __name__ == "__main__":
    main()
