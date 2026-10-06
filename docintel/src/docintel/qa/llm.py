"""One interface over Ollama, Gemini, OpenAI and Claude: prompt in, JSON out.

Every provider is asked for the same JSON schema, so the QA code never
depends on which model answered. Ollama runs models locally (free, no
API key) and needs no extra package. The cloud SDKs are imported lazily:
install only the one you use (pip install -e ".[gemini]", ".[openai]"
or ".[claude]").

    llm = get_llm("ollama")                 # provider default model
    llm = get_llm("openai", "gpt-5.4-mini")
    data = llm.generate_json(system, prompt, schema)
"""
import json
import os
import urllib.error
import urllib.request

from .. import config


class LLMError(RuntimeError):
    """Setup problem or unusable response; the message says what to fix."""


class LLMQuotaError(LLMError):
    """Rate limit or quota still exceeded after the SDK's own retries.

    Further requests will fail the same way, so batch jobs should stop.
    """

    def __init__(self, provider: str, detail: object):
        super().__init__(
            f"{provider} rate limit or quota exceeded (after retries). "
            f"Wait and retry, or switch model/provider.\n  {detail}"
        )


def _require_key(*names: str) -> None:
    if not any(os.environ.get(name) for name in names):
        raise LLMError(
            f"No API key found. Set {' or '.join(names)}, e.g.\n"
            f"  export {names[0]}=..."
        )


def _parse(text: str | None, provider: str) -> dict:
    if not text:
        raise LLMError(f"{provider} returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{provider} returned invalid JSON: {exc}") from exc


class GeminiLLM:
    provider = "gemini"

    def __init__(self, model: str):
        _require_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise LLMError(
                'Gemini SDK missing: pip install -e ".[gemini]"'
            ) from exc

        self.model = model
        self._types = types
        self._errors = errors
        # Retries 408/429/5xx with backoff; free-tier rate limits are low.
        self._client = genai.Client(http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2.0)
        ))

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            response = self._generate(system, prompt, schema)
        except self._errors.APIError as exc:
            if exc.code == 429:
                raise LLMQuotaError(self.provider, exc) from exc
            raise
        return _parse(response.text, self.provider)

    def _generate(self, system: str, prompt: str, schema: dict):
        return self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema,
                # No tools are used; also silences the SDK's AFC warning.
                automatic_function_calling=(
                    self._types.AutomaticFunctionCallingConfig(disable=True)
                ),
            ),
        )


class OpenAILLM:
    provider = "openai"

    def __init__(self, model: str):
        _require_key("OPENAI_API_KEY")
        try:
            import openai
        except ImportError as exc:
            raise LLMError(
                'OpenAI SDK missing: pip install -e ".[openai]"'
            ) from exc

        self.model = model
        self._openai = openai
        self._client = openai.OpenAI(max_retries=4)

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            response = self._create(system, prompt, schema)
        except self._openai.RateLimitError as exc:
            raise LLMQuotaError(self.provider, exc) from exc
        return _parse(response.output_text, self.provider)

    def _create(self, system: str, prompt: str, schema: dict):
        return self._client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            text={"format": {
                "type": "json_schema",
                "name": "grounded_answer",
                "schema": schema,
                "strict": True,
            }},
        )


class ClaudeLLM:
    provider = "claude"

    def __init__(self, model: str):
        # No key check: the SDK also accepts ANTHROPIC_AUTH_TOKEN or an
        # `ant auth login` profile, and reports clearly if none is found.
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                'Anthropic SDK missing: pip install -e ".[claude]"'
            ) from exc

        self.model = model
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=4)

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            response = self._create(system, prompt, schema)
        except self._anthropic.RateLimitError as exc:
            raise LLMQuotaError(self.provider, exc) from exc

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this request")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude's answer was cut off (max_tokens)")

        text = next(
            (block.text for block in response.content if block.type == "text"),
            None,
        )
        return _parse(text, self.provider)

    def _create(self, system: str, prompt: str, schema: dict):
        return self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                # Short grounded answers need little reasoning.
                "effort": config.QA_CLAUDE_EFFORT,
                "format": {"type": "json_schema", "schema": schema},
            },
            # If a safety classifier declines, the API re-runs the
            # request on a fallback model instead of returning a refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )


class OllamaLLM:
    """A model served by a local Ollama (https://ollama.com).

    Uses Ollama's REST API directly, so no Python package is needed.
    """

    provider = "ollama"

    def __init__(self, model: str):
        self.model = model
        self.host = config.QA_OLLAMA_HOST.rstrip("/")
        self._think = config.QA_OLLAMA_THINK
        self._check_model_available()

    def _request(self, path: str, body: dict | None = None) -> dict:
        request = urllib.request.Request(
            self.host + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=config.QA_OLLAMA_TIMEOUT
            ) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except json.JSONDecodeError:
                pass
            raise LLMError(f"Ollama error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, ConnectionRefusedError):
                raise LLMError(
                    f"Ollama is not running at {self.host}. Start the "
                    "Ollama app, or run: ollama serve"
                ) from exc
            raise LLMError(f"Could not reach Ollama: {reason}") from exc

    def _check_model_available(self) -> None:
        names = {m["name"] for m in self._request("/api/tags")["models"]}
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted not in names:
            raise LLMError(
                f"Ollama model {self.model!r} is not downloaded. Run:\n"
                f"  ollama pull {self.model}\n"
                f"Installed: {', '.join(sorted(names)) or 'none'}"
            )

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "format": schema,
            "stream": False,
            "options": {
                "temperature": 0,
                # Ollama's default context can be smaller than our prompt
                # and it truncates silently, so set it explicitly.
                "num_ctx": config.QA_OLLAMA_NUM_CTX,
            },
        }
        if self._think is not None:
            body["think"] = self._think

        try:
            data = self._request("/api/chat", body)
        except LLMError as exc:
            # Models without a thinking mode may reject the field.
            if "think" not in body or "think" not in str(exc).lower():
                raise
            self._think = None
            body = {key: value for key, value in body.items()
                    if key != "think"}
            data = self._request("/api/chat", body)

        if data.get("done_reason") == "length":
            raise LLMError(
                "Ollama's answer was cut off; raise DOCINTEL_OLLAMA_NUM_CTX"
            )
        return _parse(data.get("message", {}).get("content"), self.provider)


PROVIDERS = {
    "ollama": OllamaLLM,
    "gemini": GeminiLLM,
    "openai": OpenAILLM,
    "claude": ClaudeLLM,
}


def get_llm(provider: str | None = None, model: str | None = None):
    """Provider and model from the arguments, else from config/env vars."""
    provider = (provider or config.QA_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise LLMError(
            f"Unknown provider {provider!r}; use one of {sorted(PROVIDERS)}"
        )
    # DOCINTEL_LLM_MODEL names a model of the configured provider only.
    configured = config.QA_MODEL if provider == config.QA_PROVIDER else None
    model = model or configured or config.QA_DEFAULT_MODELS[provider]
    return PROVIDERS[provider](model)
