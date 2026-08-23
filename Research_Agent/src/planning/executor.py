import concurrent.futures
from typing import Any, Dict, List, Optional, Set

from Research_Agent.src.agents.research_agent import ResearchAgent
from Research_Agent.src.config.settings import settings
from Research_Agent.src.planning.planner import DAGExecutionPlan, TaskNode


class DAGExecutor:
    """Orchestrates multi-agent execution consuming Phase 1 DAGExecutionPlan Pydantic models."""

    def __init__(
            self,
            max_workers: int = 3,
            provider: Optional[str] = None,
            max_iterations_per_agent: int = 6,
    ):
        self.max_workers = max_workers
        self.provider = provider or settings.llm_provider
        self.max_iterations_per_agent = max_iterations_per_agent

    def _extract_task_info(self, node: TaskNode, idx: int) -> Dict[str, Any]:
        """Extracts task ID, sub-query, and dependencies from a TaskNode."""
        subtask = node.task

        # Updated to check sub_query first (matching query_schema.py SubTask)
        task_id = getattr(subtask, "task_id", None) or f"TASK_{idx}"
        sub_query = getattr(subtask, "sub_query", None) or str(subtask)

        return {
            "task_id": str(task_id),
            "description": sub_query,
            "dependencies": node.depends_on,
            "is_parallelizable": node.is_parallelizable,
            "raw_subtask": subtask,
        }

    def _build_context_for_task(
            self, dependencies: List[str], completed_results: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Gathers outputs from parent dependency tasks to form context for child tasks."""
        if not dependencies:
            return None

        context_blocks = []
        for parent_id in dependencies:
            parent_data = completed_results.get(parent_id, {})
            parent_result = parent_data.get("result", "No output returned.")
            context_blocks.append(
                f"--- Prior Findings from Dependency Task [{parent_id}] ---\n{parent_result}"
            )

        return "\n\n".join(context_blocks)

    def _execute_single_task(
            self, task_info: Dict[str, Any], context: Optional[str]
    ) -> Dict[str, Any]:
        """Instantiates an isolated ResearchAgent worker to run a single DAG node."""
        task_id = task_info["task_id"]
        description = task_info["description"]

        print(f"\n🚀 [Executor] Starting Node: '{task_id}'")
        agent = ResearchAgent(
            provider=self.provider, max_iterations=self.max_iterations_per_agent
        )

        execution_result = agent.run_task(
            task_description=description, task_context=context
        )

        return {
            "task_id": task_id,
            "description": description,
            "status": execution_result.get("status", "completed"),
            "result": execution_result.get("result", execution_result.get("error", "")),
            "provider": execution_result.get("provider", self.provider),
        }

    def execute_dag(self, dag_plan: DAGExecutionPlan) -> Dict[str, Any]:
        """Executes a DAGExecutionPlan wave-by-wave in topological order."""
        nodes: List[TaskNode] = dag_plan.nodes
        if not nodes:
            return {"status": "error", "error": "No nodes provided in DAG execution plan."}

        # Index nodes by task_id
        task_map: Dict[str, Dict[str, Any]] = {}
        for idx, node in enumerate(nodes, 1):
            info = self._extract_task_info(node, idx)
            task_map[info["task_id"]] = info

        completed_results: Dict[str, Dict[str, Any]] = {}

        print("=" * 60)
        print(
            f"🎯 [DAG Executor] Executing Plan '{dag_plan.plan_id}' ({len(nodes)} total nodes across {len(dag_plan.execution_waves)} waves)")
        print("=" * 60)

        # Iterate sequentially wave-by-wave as dictated by planner.py
        for wave_idx, wave_task_ids in enumerate(dag_plan.execution_waves, start=1):
            print(f"\n🌊 [Execution Wave {wave_idx}/{len(dag_plan.execution_waves)}] Running tasks: {wave_task_ids}")

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
            ) as executor:
                future_to_id = {}

                for t_id in wave_task_ids:
                    task_info = task_map[t_id]
                    deps = task_info["dependencies"]

                    # Context is automatically passed from prior waves!
                    context = self._build_context_for_task(deps, completed_results)

                    future = executor.submit(self._execute_single_task, task_info, context)
                    future_to_id[future] = t_id

                for future in concurrent.futures.as_completed(future_to_id):
                    t_id = future_to_id[future]
                    try:
                        res = future.result()
                        completed_results[t_id] = res
                        print(f"✅ [Node Completed] '{t_id}' finished.")
                    except Exception as e:
                        print(f"❌ [Node Failed] '{t_id}' failed: {e}")
                        completed_results[t_id] = {
                            "task_id": t_id,
                            "status": "error",
                            "error": str(e),
                        }

        print("\n" + "=" * 60)
        print("🎉 [DAG Execution Complete] All waves executed.")
        print("=" * 60)

        return {"status": "completed", "plan_id": dag_plan.plan_id, "results": completed_results}