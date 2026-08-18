from src.config.settings import settings
from src.planning.decomposer import decompose_query
from src.planning.planner import build_dag_plan


def main():
    # 1. Input query
    user_query = "What is the inflation rate and GDP growth in Canada for 2024?"
    print(f"\n📥 Input Query:\n'{user_query}'\n")

    # 2. Step 1.1: Decompose query into parameters and atomic sub-tasks
    print("⏳ Running Step 1.1: Query Decomposition...")
    execution_plan = decompose_query(user_query)

    print("\n✅ Extracted Parameters:")
    print(f" • Indicators : {execution_plan.extracted_params.indicators}")
    print(f" • Locations  : {execution_plan.extracted_params.locations}")
    print(f" • Timeframe  : {execution_plan.extracted_params.timeframe}")
    print(f" • Domains    : {execution_plan.extracted_params.target_domains}\n")

    # 3. Step 1.2: Build the DAG execution plan
    print("⏳ Running Step 1.2: Dynamic Execution Planning...")
    dag_plan = build_dag_plan(execution_plan)

    print(f"\n📋 Generated Execution Plan [{dag_plan.plan_id}]:")
    for node in dag_plan.nodes:
        parallel_flag = "Parallel" if node.is_parallelizable else "Sequential"
        print(
            f" • [{node.task.task_id}] Query: '{node.task.sub_query}' "
            f"| Domain: {node.task.target_domain} | Mode: {parallel_flag}"
        )

    print("\n✨ Phase 1 Complete! Ready for Phase 2 Tool Execution.\n")


if __name__ == "__main__":
    main()