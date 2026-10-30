import contextvars
import os

from langchain_google_genai import ChatGoogleGenerativeAI

# A hosted demo has many visitors in one process, so a key from the environment would be shared.
# set_api_key() stores the visitor's own key for the current thread of execution only.
_api_key = contextvars.ContextVar("gemini_api_key", default=None)


def set_api_key(key):
    """Use this Gemini key for LLM calls made from the current session. Empty means: use GOOGLE_API_KEY."""
    _api_key.set((key or "").strip() or None)


def get_llm(temperature: float = 0.3):
    """Gemini LLM. Uses the key from set_api_key() if one was set, else GOOGLE_API_KEY from .env
    (optional: GEMINI_MODEL)."""
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=_api_key.get() or os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )
    # retry automatically (waiting longer each time) if Gemini rate-limits us
    return llm.with_retry(
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 10, "max": 60},
        stop_after_attempt=6,
    )
