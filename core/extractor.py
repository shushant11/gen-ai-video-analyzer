# Action items, decisions, questions
#
# Speed: for normal-length transcripts all three are extracted in ONE LLM call
# (instead of three). Very long transcripts are split into parts first.

import re
import time
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm as _get_llm

MAX_CHARS = 20000          # above this, split the transcript into parts
PAUSE_SECONDS = 1          # small pause between calls for very long transcripts

SECTIONS = {
    "ACTION ITEMS": "No action items found.",
    "KEY DECISIONS": "No key decisions found.",
    "OPEN QUESTIONS": "No open questions found.",
}

PROMPTS = {
    "ACTION ITEMS": (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    ),
    "KEY DECISIONS": (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    ),
    "OPEN QUESTIONS": (
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    ),
}

COMBINED_PROMPT = (
    "You are an expert meeting analyst. From the meeting transcript produce THREE sections, "
    "each starting with its heading on its own line, exactly as written:\n\n"
    "### ACTION ITEMS\n"
    "A numbered list. For each item give: task, owner (who is responsible), and deadline "
    "(write 'Not specified' if not mentioned). If none, write 'No action items found.'\n\n"
    "### KEY DECISIONS\n"
    "A numbered list of the key decisions made. If none, write 'No key decisions found.'\n\n"
    "### OPEN QUESTIONS\n"
    "A numbered list of unresolved questions or topics needing follow-up. "
    "If none, write 'No open questions found.'\n\n"
    "Output only these three sections."
)


def get_llm():
    return _get_llm(temperature=0.2)


def _chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return prompt | get_llm() | StrOutputParser()


def parse_sections(text: str) -> dict:
    """Split the combined answer into {HEADING: body}. Tolerates '###', '**' and ':'."""
    found, current = {}, None
    for line in text.splitlines():
        heading = line.strip().strip("#*: ").upper()
        if heading in SECTIONS:
            current = heading
            found[current] = []
        elif current:
            found[current].append(line)
    return {k: "\n".join(v).strip() for k, v in found.items() if "\n".join(v).strip()}


def _run_one(heading: str, transcript: str) -> str:
    """One extraction type on its own (used for fallback and very long transcripts)."""
    chain = _chain(PROMPTS[heading])
    if len(transcript) <= MAX_CHARS:
        return chain.invoke({"text": transcript})

    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=300)
    parts = []
    for part in splitter.split_text(transcript):
        parts.append(chain.invoke({"text": part}))
        time.sleep(PAUSE_SECONDS)

    merge = _chain(
        "You are given several partial results extracted from different parts of one "
        "meeting. Merge them into a single numbered list, removing duplicates and keeping "
        "the same format. Ignore parts that say nothing was found, unless all of them say so."
    )
    return merge.invoke({"text": "\n\n---\n\n".join(parts)})


@lru_cache(maxsize=4)
def _extract_all(transcript: str) -> dict:
    start = time.time()
    print("Extracting action items, decisions and open questions...")
    result = {}

    if len(transcript) <= MAX_CHARS:
        answer = _chain(COMBINED_PROMPT).invoke({"text": transcript})
        result = parse_sections(answer)

    # fallback: anything missing (or a very long transcript) is extracted separately
    for heading in SECTIONS:
        if heading not in result:
            result[heading] = _run_one(heading, transcript)

    print(f"Extraction done in {time.time() - start:.0f}s")
    return result


def extract_action_items(transcript: str) -> str:
    return _extract_all(transcript)["ACTION ITEMS"]


def extract_key_decisions(transcript: str) -> str:
    return _extract_all(transcript)["KEY DECISIONS"]


def extract_questions(transcript: str) -> str:
    return _extract_all(transcript)["OPEN QUESTIONS"]