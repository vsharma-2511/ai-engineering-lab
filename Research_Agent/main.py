from src.config.settings import settings
from src.planning.decomposer import decompose_query
from src.planning.planner import build_dag_plan
from src.planning.executor import DAGExecutor
from src.synthesis.synthesizer import Synthesizer


def main():
    # 1. Input query
    user_query = "What is the inflation rate and GDP growth in Canada for 2024?"
    print(f"\n📥 Input Query:\n'{user_query}'\n")

    # 2. Step 1.1: Decompose query into parameters and atomic sub-tasks
    print("⏳ Running Step 1.1: Query Decomposition...")
    execution_plan = decompose_query(user_query)

    print("\n✅ Extracted Parameters:")
    print(f" • Indicators : {execution_plan.extracted_params.indicators}")
    print(f" • Locations  : {execution_plan.extracted_params.locations}")
    print(f" • Timeframe  : {execution_plan.extracted_params.timeframe}")
    print(f" • Domains    : {execution_plan.extracted_params.target_domains}\n")

    # 3. Step 1.2: Build the DAG execution plan
    print("⏳ Running Step 1.2: Dynamic Execution Planning...")
    dag_plan = build_dag_plan(execution_plan)

    print(f"\n📋 Generated Execution Plan [{dag_plan.plan_id}]:")
    for node in dag_plan.nodes:
        parallel_flag = "Parallel" if node.is_parallelizable else "Sequential"
        print(
            f" • [{node.task.task_id}] Query: '{node.task.sub_query}' "
            f"| Domain: {node.task.target_domain} | Mode: {parallel_flag}"
        )

    # 4. Phase 3 & 2: Dynamic Execution of the DAG via Workers
    print("\n⏳ Running Phase 3 & 2: Executing DAG Tasks with ReAct Workers...")
    executor = DAGExecutor(max_workers=2)
    execution_output = executor.execute_dag(dag_plan)

    # 5. Phase 4: Final Synthesis
    print("\n⏳ Running Phase 4: Report Synthesis...")
    synthesizer = Synthesizer()
    report = synthesizer.synthesize(user_query, execution_plan, execution_output)

    # 6. Display Final Formatted Report
    print("\n" + "=" * 70)
    print(f"📊 {report.title.upper()}")
    print("=" * 70)

    print("\n📌 EXECUTIVE SUMMARY")
    print(report.executive_summary)

    print("\n📈 KEY METRICS")
    for metric in report.key_metrics:
        print(f" • {metric.metric_name}: {metric.value} ({metric.context})")

    print("\n📝 DETAILED ANALYSIS")
    print(report.detailed_analysis)

    print("\n🔗 CITED SOURCES")
    for src in report.sources:
        print(f" • [{src.source_title}]({src.url})")

    print("\n✨ Pipeline execution complete! Phases 1 through 4 operational.\n")


if __name__ == "__main__":
    main()