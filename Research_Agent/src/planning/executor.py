import concurrent.futures
from typing import Any, Dict, List, Optional, Set, Union

from Research_Agent.src.agents.research_agent import ResearchAgent
from Research_Agent.src.config.settings import settings
from Research_Agent.src.planning.planner import DAGExecutionPlan, TaskNode


class DAGExecutor:
    """Orchestrates multi-agent execution directly consuming Phase 1 DAGExecutionPlan Pydantic models."""

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
        """Extracts task ID, description/query, and dependencies from a TaskNode."""
        subtask = node.task

        # Flexibly handle SubTask attribute naming (e.g., task_id vs id, query vs description)
        task_id = getattr(subtask, "task_id", None) or getattr(subtask, "id", None) or f"task_{idx}"
        description = getattr(subtask, "query", None) or getattr(subtask, "description", None) or str(subtask)

        return {
            "task_id": str(task_id),
            "description": description,
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
        """Executes a Phase 1 DAGExecutionPlan in topological order."""
        nodes: List[TaskNode] = dag_plan.nodes
        if not nodes:
            return {"status": "error", "error": "No nodes provided in DAG execution plan."}

        # Parse nodes into a normalized lookup dictionary
        task_map: Dict[str, Dict[str, Any]] = {}
        for idx, node in enumerate(nodes, 1):
            info = self._extract_task_info(node, idx)
            task_map[info["task_id"]] = info

        completed_results: Dict[str, Dict[str, Any]] = {}
        pending_task_ids: Set[str] = set(task_map.keys())
        completed_task_ids: Set[str] = set()

        print("=" * 60)
        print(f"🎯 [DAG Executor] Executing Plan '{dag_plan.plan_id}' with {len(nodes)} Node(s)")
        print("=" * 60)

        while pending_task_ids:
            # Identify nodes whose dependencies are satisfied
            ready_ids = [
                t_id
                for t_id in pending_task_ids
                if set(task_map[t_id]["dependencies"]).issubset(completed_task_ids)
            ]

            if not ready_ids:
                print("❌ [Executor Error] Cyclic dependency or deadlock detected in DAG.")
                return {
                    "status": "deadlock",
                    "completed_tasks": completed_results,
                    "unresolved_tasks": list(pending_task_ids),
                }

            print(f"\n📌 [Executor Batch] Ready Nodes: {ready_ids}")

            # Execute ready nodes concurrently
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
            ) as executor:
                future_to_id = {}

                for t_id in ready_ids:
                    task_info = task_map[t_id]
                    deps = task_info["dependencies"]
                    context = self._build_context_for_task(deps, completed_results)

                    future = executor.submit(self._execute_single_task, task_info, context)
                    future_to_id[future] = t_id

                for future in concurrent.futures.as_completed(future_to_id):
                    t_id = future_to_id[future]
                    try:
                        res = future.result()
                        completed_results[t_id] = res
                        completed_task_ids.add(t_id)
                        pending_task_ids.remove(t_id)
                        print(f"✅ [Executor Node Completed] '{t_id}' finished.")
                    except Exception as e:
                        print(f"❌ [Executor Node Failed] '{t_id}' failed: {e}")
                        completed_results[t_id] = {
                            "task_id": t_id,
                            "status": "error",
                            "error": str(e),
                        }
                        completed_task_ids.add(t_id)
                        pending_task_ids.remove(t_id)

        print("\n" + "=" * 60)
        print("🎉 [DAG Execution Complete] All nodes executed.")
        print("=" * 60)

        return {"status": "completed", "plan_id": dag_plan.plan_id, "results": completed_results}