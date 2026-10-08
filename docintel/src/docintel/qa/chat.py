"""Conversation on top of single-question QA.

Retrieval only sees the latest message, so a follow-up such as "and at
Central?" finds nothing useful on its own. When there is history, the
question is first rewritten into a standalone one ("How many people
joined Digital skills at Central?") using the recent turns, then
retrieved and answered exactly like a single question.
"""
from pathlib import Path

from .. import config
from .answer import answer_from_chunks

REWRITE_PROMPT = """\
You rewrite the user's latest message into a standalone question for a \
document search. Use the conversation only to resolve what the message \
refers to (pronouns, "that table", "and at Central?", "what about 2024?"). \
Keep the user's wording and every name, number and year. Do not answer \
the question and do not add facts. If the message is already standalone, \
return it unchanged."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"standalone_question": {"type": "string"}},
    "required": ["standalone_question"],
    "additionalProperties": False,
}

# Enough context to resolve references without sending the whole chat.
HISTORY_TURNS = 3


def format_history(history: list[dict]) -> str:
    lines = []
    for message in history[-2 * HISTORY_TURNS:]:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content'].strip()}")
    return "\n".join(lines)


def rewrite_question(question: str, history: list[dict], llm) -> str:
    """A standalone version of `question`; unchanged without history."""
    if not history:
        return question

    prompt = (
        f"Conversation:\n{format_history(history)}\n\n"
        f"Latest message: {question}"
    )
    data = llm.generate_json(REWRITE_PROMPT, prompt, REWRITE_SCHEMA)
    rewritten = str(data.get("standalone_question", "")).strip()
    return rewritten or question


def chat_turn(
    question: str,
    history: list[dict],
    llm,
    session_id: str | None = None,
    top_k: int = config.QA_TOP_K,
    rewrite: bool = True,
    db_path: Path = config.DB_PATH,
    search=None,
) -> dict:
    """Answer one chat message. `history` holds earlier messages as
    {"role": "user" | "assistant", "content": text}.

    Searches the library, or with `session_id` that session's documents.
    The result is answer_from_chunks()'s plus "asked" (the message as
    typed) and "searched_for" (the question actually retrieved).
    """
    if search is None:
        from ..retrieval.search import search

    standalone = (
        rewrite_question(question, history, llm) if rewrite else question
    )
    if session_id is not None:
        from ..sessions import touch_session
        touch_session(session_id, db_path)

    chunks = search(standalone, top_k=top_k, db_path=db_path,
                    session_id=session_id)
    result = answer_from_chunks(standalone, chunks, llm)
    result.update(asked=question, searched_for=standalone)
    return result
