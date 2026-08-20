from typing import Optional
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(description="The search engine query string")
    domain: Optional[str] = Field(
        default=None,
        description="Optional specific domain to restrict results (e.g. statcan.gc.ca)",
    )
    num_results: int = Field(
        default=5, description="Number of top search results to return"
    )


class ScraperInput(BaseModel):
    url: str = Field(
        description="The exact HTTP/HTTPS URL to fetch and scrape"
    )
    max_length: int = Field(
        default=4000,
        description="Maximum length of extracted textual content to avoid context overflow",
    )


class ExtractedContent(BaseModel):
    url: str
    title: str
    content: str
    status_code: int