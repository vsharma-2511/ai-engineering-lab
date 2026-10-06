"""QA tests with a fake LLM (no API key, no network).

Run from the project folder:  pytest
"""
from types import SimpleNamespace

import pytest

from docintel import config
from docintel.qa import llm as llm_module
from docintel.qa.answer import answer_from_chunks, build_prompt
from docintel.qa.evaluate import judge
from docintel.qa.llm import LLMError, LLMQuotaError, get_llm

CHUNKS = [
    {"chunk_id": "doc:v1:p2:t1:part1", "document_name": "test.pdf",
     "page_number": 2, "content_type": "table",
     "text": "Table 3. Program participation, 2025\n"
             "Program: Digital skills | Location: East | Participants: 85 "
             "| Completion rate: 76%"},
    {"chunk_id": "doc:v1:p2:g8", "document_name": "test.pdf",
     "page_number": 2, "content_type": "text",
     "text": "Source: Fictional Northport Service Office, DocIntel test\n"
             "dataset, prepared September 2026."},
]


class FakeLLM:
    provider = "fake"
    model = "fake-1"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_json(self, system, prompt, schema):
        self.calls.append(prompt)
        return self.reply


def reply(answer, *citations, answerable=True):
    return {
        "answerable": answerable,
        "answer": answer,
        "citations": [{"source": s, "quote": q} for s, q in citations],
    }


def test_prompt_numbers_sources_with_page_and_type():
    prompt = build_prompt("Who prepared it?", CHUNKS)
    assert "[1] test.pdf, page 2 (table)" in prompt
    assert "[2] test.pdf, page 2 (text)" in prompt
    assert prompt.endswith("Question: Who prepared it?")


def test_verified_citation_maps_to_page():
    llm = FakeLLM(reply("85 people joined [1].",
                        (1, "Location: East | Participants: 85")))
    result = answer_from_chunks("How many joined at East?", CHUNKS, llm)

    assert result["status"] == "answered"
    assert result["answer"] == "85 people joined [1]."
    assert result["citations"] == [{
        "source": 1, "document": "test.pdf", "page": 2,
        "chunk_id": "doc:v1:p2:t1:part1",
        "quote": "Location: East | Participants: 85",
    }]


def test_quote_across_pdf_line_break_and_curly_quotes_is_accepted():
    llm = FakeLLM(reply("The Northport Service Office [2].",
                        (2, "“DocIntel test dataset, prepared "
                            "September 2026”")))
    result = answer_from_chunks("Who prepared it?", CHUNKS, llm)
    assert result["status"] == "answered"
    assert result["citations"][0]["page"] == 2


def test_invented_quote_is_rejected_and_answer_flagged():
    llm = FakeLLM(reply("120 people [1].", (1, "Participants: 120"),
                        (7, "anything")))
    result = answer_from_chunks("How many joined at East?", CHUNKS, llm)

    assert result["status"] == "unsupported"
    assert result["citations"] == []
    assert [r["reason"] for r in result["rejected_citations"]] == [
        "quote not in source", "no such source",
    ]


def test_quote_from_the_wrong_source_is_rejected():
    llm = FakeLLM(reply("85 [2].", (2, "Participants: 85")))
    result = answer_from_chunks("How many joined at East?", CHUNKS, llm)
    assert result["status"] == "unsupported"


def test_table_quote_with_skipped_cells_maps_to_full_row():
    llm = FakeLLM(reply("76% [1].",
                        (1, "Location: East | Completion rate: 76%")))
    result = answer_from_chunks("Completion rate at East?", CHUNKS, llm)

    assert result["status"] == "answered"
    assert result["citations"][0]["quote"] == (
        "Program: Digital skills | Location: East | Participants: 85 "
        "| Completion rate: 76%"
    )


@pytest.mark.parametrize("quote", [
    "Location: East | Participants: 120",   # cells from different rows
    "Location: East | Participants: 8",     # truncated number
    "Participants: 8",                      # truncated, single cell
])
def test_table_quotes_must_match_whole_cells_of_one_row(quote):
    llm = FakeLLM(reply("x [1].", (1, quote)))
    result = answer_from_chunks("q", CHUNKS, llm)
    assert result["status"] == "unsupported"


def test_cell_matching_only_applies_to_tables():
    llm = FakeLLM(reply("x [2].", (2, "Source: Fictional | prepared")))
    result = answer_from_chunks("q", CHUNKS, llm)
    assert result["status"] == "unsupported"


def test_not_answerable_returns_not_found_and_drops_text():
    llm = FakeLLM(reply("Probably 40 staff.", answerable=False))
    result = answer_from_chunks("How many staff?", CHUNKS, llm)
    assert result["status"] == "not_found"
    assert result["answer"] == ""


def test_no_chunks_skips_the_llm():
    llm = FakeLLM(reply("x"))
    result = answer_from_chunks("Anything?", [], llm)
    assert result["status"] == "not_found"
    assert llm.calls == []


def test_duplicate_citations_are_kept_once():
    quote = (1, "Participants: 85")
    llm = FakeLLM(reply("85 [1].", quote, quote))
    result = answer_from_chunks("How many?", CHUNKS, llm)
    assert len(result["citations"]) == 1


def test_judge():
    by_id = {c["chunk_id"]: c for c in CHUNKS}
    answered = answer_from_chunks(
        "q", CHUNKS, FakeLLM(reply("85 [1].", (1, "Participants: 85"))))
    not_found = answer_from_chunks(
        "q", CHUNKS, FakeLLM(reply("", answerable=False)))

    expects_east = {"expect_any": [["Location: East", "Participants: 85"]]}
    unanswerable = {"expect_any": []}

    assert judge(expects_east, answered, by_id) == "PASS"
    assert judge(expects_east, not_found, by_id) == "FAIL"
    assert judge(unanswerable, not_found, by_id) == "PASS"
    assert judge(unanswerable, answered, by_id) == "FAIL"


# --- provider selection -------------------------------------------------

def test_unknown_provider():
    with pytest.raises(LLMError, match="Unknown provider"):
        get_llm("llama")


@pytest.mark.parametrize("provider, keys", [
    ("gemini", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    ("openai", ["OPENAI_API_KEY"]),
])
def test_missing_key_gives_clear_error(monkeypatch, provider, keys):
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(LLMError, match=keys[0]):
        get_llm(provider)


def test_model_selection(monkeypatch):
    built = []

    class Recorder:
        def __init__(self, model):
            built.append(model)

    monkeypatch.setattr(llm_module, "PROVIDERS",
                        {"gemini": Recorder, "openai": Recorder})
    monkeypatch.setattr(config, "QA_PROVIDER", "gemini")
    monkeypatch.setattr(config, "QA_MODEL", "gemini-custom")

    get_llm()                       # configured provider + configured model
    get_llm("openai")               # other provider: its own default
    get_llm("gemini", "explicit")   # argument wins

    assert built == ["gemini-custom", config.QA_DEFAULT_MODELS["openai"],
                     "explicit"]


# --- provider request/response wiring (SDK clients replaced) -------------

SCHEMA = {"type": "object"}


def test_gemini_wiring(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = get_llm("gemini", "gemini-test")
    sent = {}

    def generate_content(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(text='{"answerable": false}')

    llm._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content))

    assert llm.generate_json("sys", "prompt", SCHEMA) == {"answerable": False}
    assert sent["model"] == "gemini-test"
    assert sent["config"].system_instruction == "sys"
    assert sent["config"].response_mime_type == "application/json"


def test_openai_wiring(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = get_llm("openai", "gpt-test")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(output_text='{"answerable": true}')

    llm._client = SimpleNamespace(responses=SimpleNamespace(create=create))

    assert llm.generate_json("sys", "prompt", SCHEMA) == {"answerable": True}
    assert sent["instructions"] == "sys"
    assert sent["text"]["format"]["strict"] is True


def test_claude_wiring_and_refusal(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = get_llm("claude", "claude-test")
    sent = {}
    responses = [
        SimpleNamespace(stop_reason="end_turn", content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text='{"answerable": true}'),
        ]),
        SimpleNamespace(stop_reason="refusal", content=[]),
    ]

    def create(**kwargs):
        sent.update(kwargs)
        return responses.pop(0)

    llm._client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create)))

    assert llm.generate_json("sys", "prompt", SCHEMA) == {"answerable": True}
    assert sent["output_config"]["format"]["schema"] == SCHEMA
    assert sent["fallbacks"] == "default"

    with pytest.raises(LLMError, match="declined"):
        llm.generate_json("sys", "prompt", SCHEMA)


def test_invalid_json_is_reported(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = get_llm("openai", "gpt-test")
    llm._client = SimpleNamespace(responses=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(output_text="not json")))
    with pytest.raises(LLMError, match="invalid JSON"):
        llm.generate_json("sys", "prompt", SCHEMA)


# --- rate limits / quota ------------------------------------------------

def test_gemini_429_becomes_quota_error(monkeypatch):
    pytest.importorskip("google.genai")
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = get_llm("gemini", "gemini-test")

    def generate_content(**kwargs):
        raise errors.ClientError(429, {"error": {
            "code": 429, "status": "RESOURCE_EXHAUSTED",
            "message": "You exceeded your current quota"}})

    llm._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content))

    with pytest.raises(LLMQuotaError, match="quota"):
        llm.generate_json("sys", "prompt", SCHEMA)


def test_gemini_other_errors_pass_through(monkeypatch):
    pytest.importorskip("google.genai")
    from google.genai import errors

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = get_llm("gemini", "gemini-test")

    def generate_content(**kwargs):
        raise errors.ServerError(503, {"error": {
            "code": 503, "status": "UNAVAILABLE", "message": "busy"}})

    llm._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content))

    with pytest.raises(errors.ServerError):
        llm.generate_json("sys", "prompt", SCHEMA)


@pytest.mark.parametrize("provider, key, sdk_attr, path", [
    ("openai", "OPENAI_API_KEY", "_openai", ("responses", "create")),
    ("claude", "ANTHROPIC_API_KEY", "_anthropic",
     ("beta", "messages", "create")),
])
def test_rate_limit_becomes_quota_error(monkeypatch, provider, key,
                                        sdk_attr, path):
    pytest.importorskip("openai" if provider == "openai" else "anthropic")
    monkeypatch.setenv(key, "test-key")
    llm = get_llm(provider, "test-model")

    class FakeRateLimit(Exception):
        pass

    def create(**kwargs):
        raise FakeRateLimit("429")

    setattr(llm, sdk_attr, SimpleNamespace(RateLimitError=FakeRateLimit))
    client = create
    for name in reversed(path):
        client = SimpleNamespace(**{name: client})
    llm._client = client

    with pytest.raises(LLMQuotaError):
        llm.generate_json("sys", "prompt", SCHEMA)


# --- Ollama (HTTP calls stubbed; no Ollama install needed) --------------

class FakeOllama:
    """Stands in for OllamaLLM._request: records calls, replays replies."""

    def __init__(self, installed=("qwen3:4b",), replies=()):
        self.installed = installed
        self.replies = list(replies)
        self.bodies = []

    def __call__(self, path, body=None):
        if path == "/api/tags":
            return {"models": [{"name": n} for n in self.installed]}
        self.bodies.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def ollama_reply(content, done_reason="stop"):
    return {"message": {"role": "assistant", "content": content},
            "done_reason": done_reason}


def test_ollama_request_and_response(monkeypatch):
    fake = FakeOllama(replies=[ollama_reply('{"answerable": false}')])
    monkeypatch.setattr(llm_module.OllamaLLM, "_request", fake)
    monkeypatch.setattr(config, "QA_OLLAMA_THINK", False)

    llm = get_llm("ollama", "qwen3:4b")
    assert llm.generate_json("sys", "prompt", SCHEMA) == {"answerable": False}

    body = fake.bodies[0]
    assert body["model"] == "qwen3:4b"
    assert body["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "prompt"}]
    assert body["format"] == SCHEMA
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_ctx"] == config.QA_OLLAMA_NUM_CTX


def test_ollama_missing_model_says_how_to_pull(monkeypatch):
    monkeypatch.setattr(llm_module.OllamaLLM, "_request",
                        FakeOllama(installed=("llama3.2:3b",)))
    with pytest.raises(LLMError, match="ollama pull qwen3:4b") as error:
        get_llm("ollama", "qwen3:4b")
    assert "llama3.2:3b" in str(error.value)


def test_ollama_untagged_name_means_latest(monkeypatch):
    monkeypatch.setattr(llm_module.OllamaLLM, "_request",
                        FakeOllama(installed=("mistral:latest",)))
    assert get_llm("ollama", "mistral").model == "mistral"


def test_ollama_retries_without_think_for_non_thinking_models(monkeypatch):
    fake = FakeOllama(installed=("llama3.2:3b",), replies=[
        LLMError('Ollama error 400: "llama3.2:3b" does not support thinking'),
        ollama_reply('{"answerable": true}'),
    ])
    monkeypatch.setattr(llm_module.OllamaLLM, "_request", fake)
    monkeypatch.setattr(config, "QA_OLLAMA_THINK", False)

    llm = get_llm("ollama", "llama3.2:3b")
    assert llm.generate_json("s", "p", SCHEMA) == {"answerable": True}
    assert "think" in fake.bodies[0] and "think" not in fake.bodies[1]

    # Remembered: later calls don't send it again.
    fake.replies.append(ollama_reply('{"answerable": true}'))
    llm.generate_json("s", "p", SCHEMA)
    assert "think" not in fake.bodies[2]


def test_ollama_other_errors_are_not_retried(monkeypatch):
    fake = FakeOllama(replies=[LLMError("Ollama error 500: out of memory")])
    monkeypatch.setattr(llm_module.OllamaLLM, "_request", fake)
    llm = get_llm("ollama", "qwen3:4b")
    with pytest.raises(LLMError, match="out of memory"):
        llm.generate_json("s", "p", SCHEMA)
    assert len(fake.bodies) == 1


def test_ollama_truncated_answer_is_reported(monkeypatch):
    fake = FakeOllama(replies=[ollama_reply('{"answ', done_reason="length")])
    monkeypatch.setattr(llm_module.OllamaLLM, "_request", fake)
    llm = get_llm("ollama", "qwen3:4b")
    with pytest.raises(LLMError, match="cut off"):
        llm.generate_json("s", "p", SCHEMA)


def test_ollama_not_running_gives_clear_error(monkeypatch):
    # Nothing listens on port 9 (discard) locally: connection refused.
    monkeypatch.setattr(config, "QA_OLLAMA_HOST", "http://127.0.0.1:9")
    with pytest.raises(LLMError, match="not running|Could not reach"):
        get_llm("ollama", "qwen3:4b")
