from google import genai
from google.genai import types
from openai import OpenAI
from Research_Agent.src.config.settings import settings
from Research_Agent.src.schemas.query_schema import ExecutionPlan
from Research_Agent.src.utils.token_tracker import token_tracker

SYSTEM_PROMPT = """
You are an expert Query Decomposition System for an advanced multi-agent research architecture.

Parse the user query into structured search parameters and discrete atomic SubTask items.

RULES FOR SUB-TASK GENERATION:
1. For data retrieval requests, create atomic `SubTask` items with `task_type="DATA_RETRIEVAL"` and `depends_on=[]`.
2. FOR COMPARATIVE/EVALUATIVE QUERIES (containing keywords like 'compare', 'versus', 'difference between', 'trend evaluation'):
   - Create individual `DATA_RETRIEVAL` tasks for each metric-location combination (`depends_on=[]`).
   - MANDATORY: Create a final synthesis `SubTask` with `task_type="SYNTHESIS_COMPARISON"`.
   - Set the final synthesis task's `depends_on` list to include ALL preceding `DATA_RETRIEVAL` `task_id` values (e.g., ["TASK_1", "TASK_2", "TASK_3", "TASK_4"]).
   - Set `sub_query` on the final task to explicitly instruct the agent on how to compare and contrast the collected evidence.
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
    usage = response.usage
    token_tracker.log_usage(
        stage="Decomposer",  # or "Executor: TASK_1", "Synthesizer"
        prompt_tok=usage.prompt_tokens,
        completion_tok=usage.completion_tokens,
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