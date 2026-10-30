import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(temperature: float = 0.3):
    """Gemini LLM. Needs GOOGLE_API_KEY in .env (optional: GEMINI_MODEL)."""
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )
    # retry automatically (waiting longer each time) if Gemini rate-limits us
    return llm.with_retry(
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 10, "max": 60},
        stop_after_attempt=6,
    )
