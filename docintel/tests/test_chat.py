"""Follow-up rewriting and chat turns, with a fake LLM and fake search.

Run from the project folder:  pytest
"""
from docintel.qa.chat import (
    HISTORY_TURNS,
    REWRITE_SCHEMA,
    chat_turn,
    format_history,
    rewrite_question,
)

CHUNK = {
    "chunk_id": "doc:v1:p2:t1", "document_name": "test.pdf",
    "page_number": 2, "content_type": "table",
    "text": "Program: Digital skills | Location: Central | Participants: 120",
}

HISTORY = [
    {"role": "user", "content": "How many joined Digital skills at East?"},
    {"role": "assistant", "content": "85 people joined [1]."},
]


class ScriptedLLM:
    """Replies by schema: rewrite requests vs. answer requests."""

    provider, model = "fake", "fake-1"

    def __init__(self, rewrite="", answerable=True):
        self.rewrite = rewrite
        self.answerable = answerable
        self.prompts = []

    def generate_json(self, system, prompt, schema):
        self.prompts.append(prompt)
        if schema is REWRITE_SCHEMA:
            return {"standalone_question": self.rewrite}
        return {"answerable": self.answerable, "answer": "120 [1].",
                "citations": [{"source": 1, "quote": "Participants: 120"}]}


class FakeSearch:
    def __init__(self):
        self.calls = []

    def __call__(self, question, top_k, db_path, session_id):
        self.calls.append({"question": question, "top_k": top_k,
                           "session_id": session_id})
        return [CHUNK]


def test_no_history_means_no_rewrite_call():
    llm = ScriptedLLM(rewrite="should not be used")
    assert rewrite_question("Who prepared it?", [], llm) == "Who prepared it?"
    assert llm.prompts == []


def test_follow_up_is_rewritten_with_history():
    llm = ScriptedLLM(rewrite="How many joined Digital skills at Central?")
    question = rewrite_question("and at Central?", HISTORY, llm)

    assert question == "How many joined Digital skills at Central?"
    assert "User: How many joined Digital skills at East?" in llm.prompts[0]
    assert llm.prompts[0].endswith("Latest message: and at Central?")


def test_empty_rewrite_falls_back_to_the_original():
    llm = ScriptedLLM(rewrite="   ")
    assert rewrite_question("and at Central?", HISTORY, llm) == (
        "and at Central?")


def test_history_is_limited_to_recent_turns():
    history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    lines = format_history(history).splitlines()
    assert len(lines) == 2 * HISTORY_TURNS
    assert lines[-1] == "User: q19"


def test_chat_turn_searches_with_the_standalone_question():
    llm = ScriptedLLM(rewrite="How many joined Digital skills at Central?")
    search = FakeSearch()

    result = chat_turn("and at Central?", HISTORY, llm, top_k=4,
                       search=search)

    assert search.calls == [{
        "question": "How many joined Digital skills at Central?",
        "top_k": 4, "session_id": None,
    }]
    assert result["status"] == "answered"
    assert result["asked"] == "and at Central?"
    assert result["searched_for"] == (
        "How many joined Digital skills at Central?")
    assert result["citations"][0]["page"] == 2


def test_rewrite_can_be_turned_off():
    llm = ScriptedLLM(rewrite="unused")
    search = FakeSearch()
    result = chat_turn("and at Central?", HISTORY, llm, rewrite=False,
                       search=search)
    assert search.calls[0]["question"] == "and at Central?"
    assert result["searched_for"] == "and at Central?"
    assert len(llm.prompts) == 1  # only the answer call


def test_session_turn_searches_the_session_and_marks_it_active(monkeypatch):
    touched = []
    monkeypatch.setattr("docintel.sessions.touch_session",
                        lambda session_id, db_path: touched.append(session_id))
    search = FakeSearch()

    chat_turn("Who prepared it?", [], ScriptedLLM(), session_id="abc",
              search=search)

    assert search.calls[0]["session_id"] == "abc"
    assert touched == ["abc"]
