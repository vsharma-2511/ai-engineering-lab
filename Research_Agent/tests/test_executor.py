"""
Test runner for DAGExecutor
Run with: python -m tests.test_executor
"""

from Research_Agent.src.planning.executor import DAGExecutor


def test_dag_execution():
    # Sample DAG plan matching Phase 1 output schema
    mock_dag_plan = {
        "plan_id": "toronto_demographics_investigation",
        "query": "Compare Toronto population density and land area with Vancouver.",
        "tasks": [
            {
                "id": "task_1",
                "description": "Find official 2021 Census population density and land area for Toronto.",
                "dependencies": [],
            },
            {
                "id": "task_2",
                "description": "Find official 2021 Census population density and land area for Vancouver.",
                "dependencies": [],
            },
            {
                "id": "task_3",
                "description": "Synthesize and compare the population density and land area figures between Toronto and Vancouver.",
                "dependencies": ["task_1", "task_2"],
            },
        ],
    }

    executor = DAGExecutor(max_workers=2)
    final_output = executor.execute_dag(mock_dag_plan)

    print("\n--- FINAL EXECUTOR SUMMARY ---")
    for task_id, output in final_output.get("results", {}).items():
        print(f"\n[Node: {task_id}] Status: {output['status']}")
        print(f"Result:\n{output['result']}\n" + "-" * 40)


if __name__ == "__main__":
    test_dag_execution()