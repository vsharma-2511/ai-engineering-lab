from typing import List
from pydantic import BaseModel, Field
from src.schemas.query_schema import ExecutionPlan, SubTask


class TaskNode(BaseModel):
    task: SubTask
    depends_on: List[str] = Field(default=[], description="IDs of tasks that must finish before this task runs")
    is_parallelizable: bool = Field(default=True, description="Can this run concurrently with other tasks?")


class DAGExecutionPlan(BaseModel):
    plan_id: str
    nodes: List[TaskNode]


def build_dag_plan(execution_plan: ExecutionPlan) -> DAGExecutionPlan:
    """
    Takes the decomposed sub-tasks and creates a structured DAG plan
    determining task order and parallelism.
    """
    nodes = []
    for task in execution_plan.sub_tasks:
        # Simple heuristic: Independent sub-queries can run in parallel
        node = TaskNode(
            task=task,
            depends_on=[],
            is_parallelizable=True
        )
        nodes.append(node)

    return DAGExecutionPlan(plan_id="PLAN_001", nodes=nodes)