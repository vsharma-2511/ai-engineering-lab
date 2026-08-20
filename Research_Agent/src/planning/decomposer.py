from google import genai
from google.genai import types
from openai import OpenAI
from Research_Agent.src.config.settings import settings
from Research_Agent.src.schemas.query_schema import ExecutionPlan

SYSTEM_PROMPT = """
You are the Query Decomposition module for an evidence-grounded research agent.
Your job is to:
1. Parse natural language questions into distinct indicators, locations, timeframes, and official target domains.
2. Break down the question into clear, independent sub-tasks (queries) that can be searched individually.
"""


def _decompose_with_openai(user_query: str) -> ExecutionPlan:
    """Run decomposition using OpenAI API."""
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.beta.chat.completions.parse(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        response_format=ExecutionPlan,
    )
    return response.choices[0].message.parsed


def _decompose_with_gemini(user_query: str) -> ExecutionPlan:
    """Run decomposition using Google Gemini API via Chat session."""
    client = genai.Client(api_key=settings.gemini_api_key)

    # Use client.chats.create to handle structured generation cleanly without AFC warnings
    chat = client.chats.create(
        model=settings.gemini_model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ExecutionPlan,  # Native Pydantic schema validation
        ),
    )

    response = chat.send_message(message=user_query)
    return ExecutionPlan.model_validate_json(response.text)


def decompose_query(user_query: str) -> ExecutionPlan:
    """Factory router: Directs requests based on the configured LLM_PROVIDER."""
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        return _decompose_with_gemini(user_query)
    elif provider == "openai":
        return _decompose_with_openai(user_query)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Choose 'openai' or 'gemini'."
        )


if __name__ == "__main__":
    sample_query = (
        "Compare the unemployment rate and average home prices between "
        "Toronto and Vancouver from 2022 to 2025."
    )
    print(f"Using Provider: {settings.llm_provider.upper()}")
    plan = decompose_query(sample_query)
    print(f"Extracted {len(plan.sub_tasks)} sub-tasks successfully.")
    print("\n✅ Extracted Parameters:")
    print(f" • Indicators : {plan.extracted_params.indicators}")
    print(f" • Locations  : {plan.extracted_params.locations}")
    print(f" • Timeframe  : {plan.extracted_params.timeframe}")
    print(f" • Domains    : {plan.extracted_params.target_domains}\n")