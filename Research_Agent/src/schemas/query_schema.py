from typing import List, Optional
from pydantic import BaseModel, Field


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
        description="A focused, self-contained search query string"
    )
    target_domain: str = Field(
        description="The primary domain to restrict the search to"
    )


class ExecutionPlan(BaseModel):
    extracted_params: SearchParameters
    sub_tasks: List[SubTask] = Field(
        description="Breakdown of the user query into individual, single-focus sub-queries for downstream tools"
    )