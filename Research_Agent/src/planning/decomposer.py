from openai import OpenAI
from src.schemas.query_schema import ExecutionPlan


def decompose_query(user_query: str) -> ExecutionPlan:
    """Takes a natural language query and decomposes it into structured parameters

    and atomic sub-queries for execution.
    """
    client = OpenAI()

    system_prompt = """
    You are the Query Decomposition module for an evidence-grounded research agent.
    Your job is to:
    1. Parse natural language questions into distinct indicators, locations, timeframes, and official target domains.
    2. Break down the question into clear, independent sub-tasks (queries) that can be searched individually.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        response_format=ExecutionPlan,
    )

    return response.choices[0].message.parsed


if __name__ == "__main__":
    sample_query = (
        "Compare the unemployment rate and average home prices between "
        "Toronto and Vancouver from 2022 to 2025."
    )

    print(f"\n📥 User Query:\n'{sample_query}'\n")
    print("⏳ Decomposing query...\n")

    plan: ExecutionPlan = decompose_query(sample_query)

    print("✅ EXTRACTED PARAMETERS:")
    print(f" • Indicators : {plan.extracted_params.indicators}")
    print(f" • Locations  : {plan.extracted_params.locations}")
    print(f" • Timeframe  : {plan.extracted_params.timeframe}")
    print(f" • Domains    : {plan.extracted_params.target_domains}\n")

    print("📋 GENERATED SUB-TASKS:")
    for task in plan.sub_tasks:
        print(
            f" [{task.task_id}] Query: '{task.sub_query}' | Domain: {task.target_domain}"
        )