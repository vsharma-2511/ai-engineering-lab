"""
Test runner for Query Decomposer
Run with: python -m tests.test_decomposer
"""

from Research_Agent.src.planning.decomposer import decompose_query


def test_decomposer():
    # Sample complex query
    user_query = (
        "Compare the unemployment rate and average home prices between "
        "Toronto and Vancouver from 2022 to 2025."
    )

    print("=" * 60)
    print(f"🧪 [Decomposer Test] Processing Query: '{user_query}'")
    print("=" * 60)

    try:
        execution_plan = decompose_query(user_query)

        print("\n✅ [Decomposition Successful]")
        print(f"Generated {len(execution_plan.sub_tasks)} Sub-Task(s):\n")

        for idx, subtask in enumerate(execution_plan.sub_tasks, 1):
            task_id = getattr(subtask, "task_id", None) or getattr(subtask, "id", f"task_{idx}")
            query_desc = getattr(subtask, "query", None) or getattr(subtask, "description", str(subtask))
            print(f"  [{task_id}] -> {query_desc}")

    except Exception as e:
        print(f"\n❌ [Decomposition Failed]: {e}")


if __name__ == "__main__":
    test_decomposer()