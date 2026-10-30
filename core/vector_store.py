from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # older installs
    from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_K = 6


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:
    """Index ONE transcript. Any previously stored transcript is removed first, so chat
    answers only ever come from the video that was just analysed."""
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(transcript)
    docs = [Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)]

    embeddings = get_embeddings()

    # Drop the old collection (if any). Without this, Chroma appends and a second video's
    # questions can retrieve chunks from the first one.
    try:
        Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        ).delete_collection()
    except Exception as e:  # nothing to delete on the first run
        print(f"(no previous collection to reset: {e})")

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = DEFAULT_K):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
