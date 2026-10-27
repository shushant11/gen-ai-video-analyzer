# Action items, decisions, questions

import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import time

# Long transcripts are split so each request stays inside the model's context window
MAX_CHARS = 12000
PAUSE_SECONDS = 1.5


def get_llm():
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
        timeout=60,
    )
    # Retry on rate limits (429) and network errors, waiting longer each time
    return llm.with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=6,
    )


def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return prompt | get_llm() | StrOutputParser()


def run_on_transcript(system_prompt: str, transcript: str) -> str:
    chain = build_chain(system_prompt)

    if len(transcript) <= MAX_CHARS:
        return chain.invoke({"text": transcript})

    # Map: run on each part. Reduce: merge into one clean list.
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=300)
    parts = []
    for part in splitter.split_text(transcript):
        parts.append(chain.invoke({"text": part}))
        time.sleep(PAUSE_SECONDS)  # stay under the provider's per-minute limit

    merge_chain = build_chain(
        "You are given several partial results extracted from different parts of one "
        "meeting. Merge them into a single numbered list, removing duplicates and "
        "keeping the same format. Ignore parts that say nothing was found, unless "
        "all of them say so.\n\nInstructions used for each part:\n"
        + system_prompt.replace("{", "{{").replace("}", "}}")
    )
    return merge_chain.invoke({"text": "\n\n---\n\n".join(parts)})


def extract_action_items(transcript: str) -> str:
    return run_on_transcript(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        transcript,
    )


def extract_key_decisions(transcript: str) -> str:
    return run_on_transcript(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'",
        transcript,
    )


def extract_questions(transcript: str) -> str:
    return run_on_transcript(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'",
        transcript,
    )