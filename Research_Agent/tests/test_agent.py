"""
Test runner for ResearchAgent
Run with: python -m tests.test_agent
"""

from Research_Agent.src.agents.research_agent import ResearchAgent


def test_research_agent():
    # ResearchAgent automatically picks up settings.LLM_PROVIDER and settings.get_api_key()
    agent = ResearchAgent(max_iterations=4)

    task = (
        "Find the 2021 Census URL for Toronto, scrape the full webpage, "
        "and extract both the total land area in sq km and the population density."
    )
    res = agent.run_task(task_description=task)

    print("\n" + "=" * 60)
    print(f"Agent Execution Status: {res.get('status')}")
    print(f"Provider Used:          {res.get('provider')}")
    print(f"Total Iterations Taken: {res.get('iterations', 'N/A')}")
    print("=" * 60)
    print("Final Output:\n")
    print(res.get("result", res.get("error")))


if __name__ == "__main__":
    test_research_agent()