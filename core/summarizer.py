import os
import time

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAUSE_SECONDS = 1.5  # short pause between LLM calls to stay under rate limits


def get_llm():
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
        timeout=60,
    )
    # Retry on rate limits (429) and network errors, waiting longer each time
    return llm.with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=6,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = []
    for chunk in split_transcript(transcript):
        chunk_summaries.append(map_chain.invoke({"text": chunk}))
        time.sleep(PAUSE_SECONDS)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert meeting summarizer. Combine these partial summaries "
         "into one final professional meeting summary in bullet points."),
        ("human", "{text}"),
    ])
    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )
    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system",
             "Based on the meeting transcript, generate a short professional meeting title "
             "(max 8 words). Only return the title, nothing else."),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])