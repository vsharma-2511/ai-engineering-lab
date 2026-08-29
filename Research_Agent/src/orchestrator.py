"""
Main Orchestration Pipeline for Research Agent Project.
Integrates: Decomposer -> SemanticCache -> Planner -> DAGExecutor (ResearchAgent) -> Synthesizer
"""

import time
from typing import Any, Dict, Optional

from Research_Agent.src.agents.research_agent import ResearchAgent
from Research_Agent.src.cache.semantic_cache import SemanticCache
from Research_Agent.src.planning.decomposer import Decomposer
from Research_Agent.src.planning.executor import DAGExecutor
from Research_Agent.src.planning.planner import Planner
from Research_Agent.src.schemas.query_schema import SearchParameters
from Research_Agent.src.synthesis.synthesizer import Synthesizer


class ResearchPipeline:
    def __init__(
        self,
        qdrant_path: str = "./data/cache_db",
        collection_name: str = "research_cache",
        similarity_threshold: float = 0.90,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize the top-level research pipeline with caching and execution agents.

        Parameters:
        -----------
        qdrant_path : str
            Directory path on disk where Qdrant vector database files are stored.
        collection_name : str
            Qdrant collection namespace for storing research cache vectors.
        similarity_threshold : float
            Cosine similarity cutoff score (0.0 - 1.0) required to trigger a hit.
        llm_provider : Optional[str]
            Override LLM provider ('gemini', 'openai', 'anthropic') from settings.
        llm_model : Optional[str]
            Override target model name.
        """
        self.decomposer = Decomposer()
        self.planner = Planner()
        self.executor = DAGExecutor()
        self.synthesizer = Synthesizer()

        # Instantiate sub-task ReAct worker agent using your existing multi-provider implementation
        self.worker_agent = ResearchAgent(
            provider=llm_provider,
            model=llm_model,
        )

        # Persistent Semantic Cache layer
        self.cache = SemanticCache(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            similarity_threshold=similarity_threshold,
        )

    async def run(self, user_query: str) -> Dict[str, Any]:
        """
        Execute full multi-agent search workflow with semantic caching.

        Parameters:
        -----------
        user_query : str
            Raw natural language query submitted by user.

        Returns:
        --------
        Dict[str, Any]
            Execution metadata, performance stats, source, and synthesized report.
        """
        start_time = time.time()
        print(f"\n🚀 [PIPELINE START] Processing query: '{user_query}'")

        # Step 1: Decompose query into parameter guardrails
        print("\n🔍 [STEP 1/5] Extracting query parameters (Decomposer)...")
        extracted_params: SearchParameters = await self.decomposer.decompose(user_query)

        # Step 2: Semantic Cache Lookup
        print("\n⚡ [STEP 2/5] Checking Semantic Cache...")
        cached_response = self.cache.lookup(user_query, extracted_params)

        if cached_response is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            print(f"✅ [CACHE HIT] Pipeline completed in {elapsed_ms:.2f} ms")
            return {
                "source": "semantic_cache",
                "latency_ms": elapsed_ms,
                "parameters": extracted_params.model_dump(),
                "response": cached_response,
            }

        # Step 3: Cache Miss -> Generate execution plan
        print("\n📋 [STEP 3/5] Cache Miss. Generating execution DAG (Planner)...")
        dag_plan = await self.planner.create_plan(user_query, extracted_params)

        # Step 4: Execute Sub-Tasks using DAGExecutor and ResearchAgent
        print("\n⚙️ [STEP 4/5] Executing sub-tasks with DAGExecutor & ResearchAgent...")
        raw_results = await self.executor.execute(
            dag_plan=dag_plan,
            agent_runner=self.worker_agent.run_task,
        )

        # Step 5: Synthesize results into final output
        print("\n📝 [STEP 5/5] Synthesizing report (Synthesizer)...")
        final_response = await self.synthesizer.synthesize(
            user_query=user_query,
            execution_results=raw_results,
        )

        # Store result in cache
        print("\n💾 Storing response into SemanticCache...")
        self.cache.store(
            user_query=user_query,
            extracted_params=extracted_params,
            response=final_response,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"✨ [PIPELINE COMPLETE] Executed in {elapsed_ms:.2f} ms")

        return {
            "source": "live_execution",
            "latency_ms": elapsed_ms,
            "parameters": extracted_params.model_dump(),
            "execution_plan": dag_plan,
            "response": final_response,
        }

    def close(self):
        """Release disk locks and database client handles."""
        self.cache.close()