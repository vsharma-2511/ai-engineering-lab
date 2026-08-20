import os
from typing import Dict, List
import httpx
from Research_Agent.src.config import settings
from Research_Agent.src.tools.schemas import SearchInput


def web_search(
    query: str, domain: str = None, num_results: int = 5
) -> List[Dict[str, str]]:
    """Performs a web search using SerpAPI (or fallback mock/API) and returns structured snippets."""
    print(f"🔧 [TOOL EXECUTED] web_search | query='{query}' | domain='{domain}'")
    full_query = f"site:{domain} {query}" if domain else query
    serpapi_key = getattr(settings, "SERPAPI_API_KEY", None) or os.getenv("SERPAPI_API_KEY")

    if serpapi_key:
        url = "https://serpapi.com/search"
        params = {
            "q": full_query,
            "api_key": serpapi_key,
            "num": num_results,
            "engine": "google",
        }
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return results
    else:
        # Fallback/Development notice if API key is not set
        return [{
            "title": f"Search Result for: {full_query}",
            "link": "https://example.com",
            "snippet": f"SERPAPI_API_KEY not found. Placeholder search snippet for query '{full_query}'.",
        }]