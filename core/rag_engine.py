from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.hybrid import HybridRetriever
from core.llm import get_llm as _get_llm
from core.vector_store import DEFAULT_K, build_vector_store, load_vector_store

NOT_FOUND = "I could not find this information in the meeting transcript."

SYSTEM_PROMPT = (
    "You are an expert meeting assistant. Answer the user's question "
    "based ONLY on the meeting transcript context provided below.\n\n"
    f'If the answer is not found in the context, say: "{NOT_FOUND}"\n\n'
    "Always be concise and precise. If quoting someone, mention it clearly.\n\n"
    "Context from meeting transcript:\n{context}"
)


def get_llm():
    return _get_llm(temperature=0.3)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _chain_from(vector_store, retriever=None):
    """retrieve -> fill the prompt -> Gemini -> plain text (LangChain LCEL).
    Default retrieval is hybrid (embeddings + BM25 keyword search, rank-fused); pass a runnable
    such as get_retriever(vector_store) to use embeddings only."""
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    retriever = retriever or RunnableLambda(HybridRetriever(vector_store, k=DEFAULT_K).retrieve)
    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    """Index this transcript (replacing any earlier one) and return a question-answering chain."""
    return _chain_from(build_vector_store(transcript))


def load_rag_chain():
    """Rebuild the chain from the transcript already stored in vector_db/."""
    return _chain_from(load_vector_store())


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer
