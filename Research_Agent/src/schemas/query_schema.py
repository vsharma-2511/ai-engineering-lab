from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    DATA_RETRIEVAL = "DATA_RETRIEVAL"
    SYNTHESIS_COMPARISON = "SYNTHESIS_COMPARISON"


class SearchParameters(BaseModel):
    indicators: List[str] = Field(
        description="Key numerical, statistical, or research variables requested. e.g., ['unemployment rate', 'housing prices']"
    )
    locations: List[str] = Field(
        description="Cities, provinces, countries, or specific regions mentioned in the query. e.g., ['Toronto', 'Vancouver']"
    )
    timeframe: Optional[str] = Field(
        default="latest",
        description="The years, date range, or period requested. e.g., '2021-2025' or '2023'",
    )
    target_domains: List[str] = Field(
        description="High-authority, official web domains likely to hold official stats for these indicators/locations. e.g., ['statcan.gc.ca', 'toronto.ca']"
    )


class SubTask(BaseModel):
    task_id: str = Field(
        description="Unique identifier for the sub-task, e.g., 'TASK_1'"
    )
    sub_query: str = Field(
        description="A focused, self-contained search query string or comparison instruction"
    )
    target_domain: Optional[str] = Field(
        default="",
        description="The primary domain to restrict search to (optional for comparison tasks)"
    )
    task_type: TaskType = Field(
        default=TaskType.DATA_RETRIEVAL,
        description="Type of task: DATA_RETRIEVAL for fetching evidence, or SYNTHESIS_COMPARISON for cross-entity evaluation"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of task_ids that must complete before this task can execute"
    )


class ExecutionPlan(BaseModel):
    extracted_params: SearchParameters
    sub_tasks: List[SubTask] = Field(
        description="Breakdown of the user query into individual sub-tasks, including data retrieval and comparative synthesis steps"
    )