"""
Test runner for DAGExecutor.
Run with: python -m tests.test_executor
"""

from Research_Agent.src.schemas.query_schema import (
    ExecutionPlan,
    SearchParameters,
    SubTask,
    TaskType,
)
from Research_Agent.src.planning.planner import build_dag_plan
from Research_Agent.src.planning.executor import DAGExecutor


def test_dag_execution():
    # 1. Mock an ExecutionPlan matching query_schema.py
    mock_execution_plan = ExecutionPlan(
        extracted_params=SearchParameters(
            indicators=["population density", "land area"],
            locations=["Toronto", "Vancouver"],
            timeframe="2021",
            target_domains=["statcan.gc.ca"],
        ),
        sub_tasks=[
            SubTask(
                task_id="TASK_1",
                task_type=TaskType.DATA_RETRIEVAL,
                sub_query="Find official 2021 Census population density and land area for Toronto.",
                depends_on=[],
            ),
            SubTask(
                task_id="TASK_2",
                task_type=TaskType.DATA_RETRIEVAL,
                sub_query="Find official 2021 Census population density and land area for Vancouver.",
                depends_on=[],
            ),
            SubTask(
                task_id="TASK_3",
                task_type=TaskType.SYNTHESIS_COMPARISON,
                sub_query="Synthesize and compare the population density and land area figures between Toronto and Vancouver.",
                depends_on=["TASK_1", "TASK_2"],
            ),
        ],
    )

    # 2. Build the DAGExecutionPlan using planner.py logic
    dag_plan = build_dag_plan(mock_execution_plan)

    print("\n--- GENERATED DAG PLAN METADATA ---")
    print(f"Plan ID: {dag_plan.plan_id}")
    print(f"Execution Waves: {dag_plan.execution_waves}")
    for node in dag_plan.nodes:
        print(
            f"Node '{node.task.task_id}' -> Wave {node.execution_wave} | "
            f"Parallelizable: {node.is_parallelizable} | Deps: {node.depends_on}"
        )
    print("-" * 50 + "\n")

    # 3. Execute the DAG Plan
    executor = DAGExecutor(max_workers=2)
    final_output = executor.execute_dag(dag_plan)

    # 4. Print Final Results Summary
    print("\n--- FINAL EXECUTOR SUMMARY ---")
    for task_id, output in final_output.get("results", {}).items():
        print(f"\n[Node: {task_id}] Status: {output.get('status')}")
        print(f"Sub-Query: {output.get('description')}")
        print(f"Result:\n{output.get('result')}\n" + "-" * 50)


if __name__ == "__main__":
    test_dag_execution()