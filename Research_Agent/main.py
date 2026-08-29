import sys
from Research_Agent.src.cache.semantic_cache import SemanticCache
from Research_Agent.src.config.settings import settings
from Research_Agent.src.planning.decomposer import decompose_query
from Research_Agent.src.planning.executor import DAGExecutor
from Research_Agent.src.planning.planner import build_dag_plan
from Research_Agent.src.synthesis.synthesizer import Synthesizer
from Research_Agent.src.utils.token_tracker import token_tracker


def display_report(report):
    """Utility to print the synthesized or cached report formatted to stdout."""
    print("\n" + "=" * 70)
    # Handles both Pydantic model output and dictionary output seamlessly
    title = getattr(report, "title", None) or report.get("title", "RESEARCH REPORT")
    print(f"📊 {str(title).upper()}")
    print("=" * 70)

    exec_summary = getattr(report, "executive_summary", None) or report.get(
        "executive_summary", ""
    )
    print("\n📌 EXECUTIVE SUMMARY")
    print(exec_summary)

    key_metrics = getattr(report, "key_metrics", []) or report.get("key_metrics", [])
    if key_metrics:
        print("\n📈 KEY METRICS")
        for metric in key_metrics:
            if isinstance(metric, dict):
                m_name = metric.get("metric_name")
                m_val = metric.get("value")
                m_ctx = metric.get("context")
            else:
                m_name = metric.metric_name
                m_val = metric.value
                m_ctx = metric.context
            print(f" • {m_name}: {m_val} ({m_ctx})")

    detailed_analysis = getattr(report, "detailed_analysis", None) or report.get(
        "detailed_analysis", ""
    )
    if detailed_analysis:
        print("\n📝 DETAILED ANALYSIS")
        print(detailed_analysis)

    sources = getattr(report, "sources", []) or report.get("sources", [])
    if sources:
        print("\n🔗 CITED SOURCES")
        for src in sources:
            if isinstance(src, dict):
                s_title = src.get("source_title")
                s_url = src.get("url")
            else:
                s_title = src.source_title
                s_url = src.url
            print(f" • [{s_title}]({s_url})")

    print("\n" + "=" * 70 + "\n")


def run_research_pipeline(user_query: str):
    """
    Executes the research agent pipeline.

    1. Checks SemanticCache upfront using raw text vector embedding.
    2. On Cache Miss: Decomposes query, builds DAG plan, executes tasks, synthesizes report,
       and stores the result into SemanticCache with extracted metadata parameters.
    """
    cache = SemanticCache()

    # --- Step 1: Upfront Semantic Cache Lookup (Zero LLM Calls) ---
    print(f"\n📥 Input Query: '{user_query}'")
    print("⚡ Step 1: Checking Upfront Semantic Cache...")

    # Check cache using raw user query embedding
    cached_response = cache.lookup_raw_query(user_query)
    if cached_response:
        print("\n🚀 [FAST PATH HIT] Retracted response directly from Semantic Cache!")
        print("💡 Zero LLM calls executed for decomposition or execution.")
        display_report(cached_response)
        return cached_response

    print("❌ Cache Miss. Proceeding with full execution pipeline...\n")

    # --- Step 2: Query Decomposition & Parameter Extraction ---
    print("⏳ Phase 1: Running Query Decomposition & Parameter Extraction...")
    execution_plan = decompose_query(user_query)
    extracted_params = execution_plan.extracted_params

    print("\n✅ Extracted Parameters:")
    print(f" • Indicators : {extracted_params.indicators}")
    print(f" • Locations  : {extracted_params.locations}")
    print(f" • Timeframe  : {extracted_params.timeframe}")
    print(f" • Domains    : {extracted_params.target_domains}\n")

    # --- Step 3: Dynamic DAG Execution Planning ---
    print("⏳ Phase 2: Building Dynamic DAG Execution Plan...")
    dag_plan = build_dag_plan(execution_plan)

    print(f"\n📋 Generated Execution Plan [{dag_plan.plan_id}]:")
    for node in dag_plan.nodes:
        parallel_flag = "Parallel" if node.is_parallelizable else "Sequential"
        print(
            f" • [{node.task.task_id}] Query: '{node.task.sub_query}' "
            f"| Domain: {node.task.target_domain} | Mode: {parallel_flag}"
        )

    # --- Step 4: Dynamic DAG Waves Execution via Workers ---
    print("\n⏳ Phase 3: Executing DAG Tasks with ReAct Workers...")
    executor = DAGExecutor(max_workers=2)
    execution_results = executor.execute_dag(dag_plan)

    # --- Step 5: Report Synthesis ---
    print("\n⏳ Phase 4: Synthesizing Final Report...")
    synthesizer = Synthesizer()
    final_report = synthesizer.synthesize(user_query, execution_plan, execution_results)

    # Display live generated output
    display_report(final_report)

    # --- Step 6: Store Result in Semantic Cache for Future Lookups ---
    print("💾 Storing final synthesized response into SemanticCache...")
    cache.store(
        user_query=user_query,
        extracted_params=extracted_params,
        response=final_report,
    )

    # print("\n✨ Pipeline execution complete! Full lifecycle finished.")
    # print(token_tracker.summary())
    return final_report


if __name__ == "__main__":
    query = "What is the inflation rate and GDP growth in Canada for 2024?"

    # Run 1: Cold Execution (Cache Miss)
    print("\n=================== RUN 1: COLD EXECUTION ===================")
    run_research_pipeline(user_query=query)

    # # Run 2: Warm Execution (Cache Hit - Upfront Short-Circuit)
    # print("\n=================== RUN 2: WARM CACHE HIT ===================")
    # run_research_pipeline(user_query=query)

    print("\n✨ Pipeline execution complete! Full lifecycle finished.")
    print(token_tracker.summary())