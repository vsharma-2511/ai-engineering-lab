from typing import List, Dict
from pydantic import BaseModel, Field
from Research_Agent.src.schemas.query_schema import ExecutionPlan, SubTask


class TaskNode(BaseModel):
    task: SubTask
    depends_on: List[str] = Field(
        default=[],
        description="IDs of tasks that must finish before this task runs"
    )
    is_parallelizable: bool = Field(
        default=True,
        description="Can this task run concurrently with other tasks in its wave?"
    )
    execution_wave: int = Field(
        default=0,
        description="0-indexed execution wave indicating topological sequence level"
    )


class DAGExecutionPlan(BaseModel):
    plan_id: str
    nodes: List[TaskNode]
    execution_waves: List[List[str]] = Field(
        default_factory=list,
        description="Grouped task IDs ordered by execution wave sequence"
    )


def build_dag_plan(execution_plan: ExecutionPlan) -> DAGExecutionPlan:
    """
    Transforms decomposed sub-tasks into a structured DAG execution plan.
    Validates dependencies, detects cycles, and calculates wave-based execution order.
    """
    sub_tasks = execution_plan.sub_tasks
    task_map: Dict[str, SubTask] = {t.task_id: t for t in sub_tasks}

    # 1. Validate all dependencies exist
    for task in sub_tasks:
        for dep in task.depends_on:
            if dep not in task_map:
                raise ValueError(
                    f"Task '{task.task_id}' depends on non-existent task '{dep}'."
                )

    # 2. Compute in-degree and adjacency graph
    in_degree: Dict[str, int] = {t.task_id: len(t.depends_on) for t in sub_tasks}
    graph: Dict[str, List[str]] = {t.task_id: [] for t in sub_tasks}

    for task in sub_tasks:
        for dep in task.depends_on:
            graph[dep].append(task.task_id)

    # 3. Calculate topological execution waves
    waves: List[List[str]] = []
    current_wave = [task_id for task_id, deg in in_degree.items() if deg == 0]

    processed_count = 0
    wave_index = 0
    node_wave_map: Dict[str, int] = {}

    while current_wave:
        waves.append(current_wave)
        next_wave = []
        for task_id in current_wave:
            processed_count += 1
            node_wave_map[task_id] = wave_index
            for dependent_id in graph[task_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    next_wave.append(dependent_id)
        current_wave = next_wave
        wave_index += 1

    if processed_count != len(sub_tasks):
        raise ValueError("Circular dependency detected in execution plan sub-tasks.")

    # 4. Construct TaskNode objects
    nodes: List[TaskNode] = []
    for task in sub_tasks:
        wave_num = node_wave_map[task.task_id]
        same_wave_tasks = waves[wave_num]

        # Parallelizable if wave contains > 1 task and node has no blocking dependencies
        is_parallel = len(same_wave_tasks) > 1 and len(task.depends_on) == 0

        node = TaskNode(
            task=task,
            depends_on=task.depends_on,
            is_parallelizable=is_parallel,
            execution_wave=wave_num,
        )
        nodes.append(node)

    return DAGExecutionPlan(
        plan_id="PLAN_001",
        nodes=nodes,
        execution_waves=waves,
    )