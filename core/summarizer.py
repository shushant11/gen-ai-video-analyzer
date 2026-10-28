import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm as _get_llm

SINGLE_CALL_LIMIT = 20000   # up to this size the whole transcript is summarised in ONE call
CHUNK_SIZE = 8000           # longer transcripts: summarise parts, then merge
PAUSE_SECONDS = 1


def get_llm():
    return _get_llm(temperature=0.3)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)
    return splitter.split_text(transcript)


def _prompt(system: str):
    return ChatPromptTemplate.from_messages([("system", system), ("human", "{text}")])


def summarize(transcript: str) -> str:
    start = time.time()
    print("Summarising...")
    llm = get_llm()

    final_system = (
        "You are an expert meeting summarizer. Write one professional meeting "
        "summary in bullet points."
    )

    if len(transcript) <= SINGLE_CALL_LIMIT:
        summary = (_prompt(final_system) | llm | StrOutputParser()).invoke({"text": transcript})
    else:
        map_chain = _prompt("Summarize this portion of a meeting transcript concisely.") | llm | StrOutputParser()
        partial = []
        for chunk in split_transcript(transcript):
            partial.append(map_chain.invoke({"text": chunk}))
            time.sleep(PAUSE_SECONDS)
        combine_system = (
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points."
        )
        summary = (_prompt(combine_system) | llm | StrOutputParser()).invoke(
            {"text": "\n\n".join(partial)}
        )

    print(f"Summary done in {time.time() - start:.0f}s")
    return summary


def generate_title(transcript: str) -> str:
    start = time.time()
    print("Generating title...")
    chain = _prompt(
        "Based on the meeting transcript, generate a short professional meeting title "
        "(max 8 words). Only return the title, nothing else."
    ) | get_llm() | StrOutputParser()
    title = chain.invoke({"text": transcript[:2000]})
    print(f"Title done in {time.time() - start:.0f}s")
    return title