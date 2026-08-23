"""
End-to-End Test runner for Decomposer + DAG Planner
Run with: python -m tests.test_planner
"""

from Research_Agent.src.planning.decomposer import decompose_query
from Research_Agent.src.planning.planner import build_dag_plan


def test_end_to_end_planning():
    # Sample complex query
    # user_query = "Find the total annual revenue of Apple for the most recent fiscal year, calculate its average per-month revenue from that figure, and then compare that monthly average to the total monthly revenue of a direct competitor like Microsoft."
    # user_query = (
    #     "Compare the unemployment rate and average home prices between "
    #     "Toronto and Vancouver from 2022 to 2025."
    # )

    user_query = (
        "Compare the unemployment rate of "
        "Toronto and Vancouver in 2026."
    )
    print("=" * 60)
    print(f"🧪 [End-to-End Planning Test] Query: '{user_query}'")
    print("=" * 60)

    try:
        # Step 1: Pass query to Decomposer to generate sub-tasks using the LLM
        print("\n⏳ Step 1: Running Decomposer (Calling LLM)...")
        execution_plan = decompose_query(user_query)
        print(f"✅ Success! Decomposed into {len(execution_plan.sub_tasks)} sub-task(s).")

        # Step 2: Pass the ExecutionPlan to the Planner to build the DAG nodes
        print("\n⏳ Step 2: Running DAG Planner...")
        dag_plan = build_dag_plan(execution_plan)
        print(f"✅ Success! DAG Plan '{dag_plan.plan_id}' constructed with {len(dag_plan.nodes)} node(s).\n")

        print("=" * 60)
        print("🗺️ [Final Structured DAG Plan Output]")
        print("=" * 60)

        for node in dag_plan.nodes:
            task_id = getattr(node.task, "task_id", None) or getattr(node.task, "id", "unknown")
            query_text = getattr(node.task, "query", None) or getattr(node.task, "description", str(node.task))
            print(f"Node ID: {task_id}")
            print(f" ├── Query: {query_text}")
            print(f" ├── Dependencies: {node.depends_on}")
            print(f" └── Parallelizable: {node.is_parallelizable}\n")

    except Exception as e:
        print(f"\n❌ [Planning Test Failed]: {e}")


if __name__ == "__main__":
    test_end_to_end_planning()