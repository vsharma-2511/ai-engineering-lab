import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from Research_Agent.src.cache.semantic_cache import SemanticCache
from Research_Agent.src.planning.decomposer import decompose_query
from Research_Agent.src.planning.executor import DAGExecutor
from Research_Agent.src.planning.planner import build_dag_plan
from Research_Agent.src.synthesis.synthesizer import Synthesizer

app = FastAPI(
    title="Research Agent API",
    version="1.0.0",
    description="Asynchronous multi-agent research pipeline API with upfront semantic cache and SSE progress streaming.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Formats payload according to the W3C Server-Sent Events standard."""
    payload = json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def pipeline_event_generator(
    query: str, provider: str
) -> AsyncGenerator[str, None]:
    """Asynchronous generator yielding pipeline stage updates via SSE."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def publish_status(
        status_type: str, message: str, payload: Dict[str, Any] = None
    ):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {
                "event": status_type,
                "data": {"message": message, "payload": payload or {}},
            },
        )

    yield format_sse_event("init", {"status": "started", "query": query})

    async def run_pipeline():
        cache = SemanticCache()
        try:
            # Step 0: Upfront Semantic Cache Lookup (Zero LLM Calls on Hit)
            publish_status("stage_start", "Checking Semantic Cache...")
            cached_response = await loop.run_in_executor(
                None, cache.lookup_raw_query, query
            )

            if cached_response:
                # Convert Pydantic model to dict if needed
                report_data = (
                    cached_response.model_dump()
                    if hasattr(cached_response, "model_dump")
                    else cached_response
                )
                publish_status(
                    "cache_hit",
                    "Retrieved response directly from Semantic Cache.",
                    report_data,
                )
                publish_status(
                    "report_complete", "Research report retrieved.", report_data
                )
                return

            publish_status(
                "cache_miss",
                "Cache miss. Executing full decomposition and research pipeline...",
            )

            # Step 1: Decomposition
            publish_status("stage_start", "Decomposing input query...")
            execution_plan = await loop.run_in_executor(
                None, decompose_query, query
            )

            extracted_info = {
                "indicators": execution_plan.extracted_params.indicators,
                "locations": execution_plan.extracted_params.locations,
                "timeframe": execution_plan.extracted_params.timeframe,
            }
            publish_status(
                "decomposition_complete",
                "Query parameters extracted.",
                extracted_info,
            )

            # Step 2: Planning
            publish_status("stage_start", "Building DAG execution plan...")
            dag_plan = await loop.run_in_executor(
                None, build_dag_plan, execution_plan
            )

            nodes_summary = [
                {
                    "task_id": n.task.task_id,
                    "query": n.task.sub_query,
                    "mode": "parallel" if n.is_parallelizable else "sequential",
                }
                for n in dag_plan.nodes
            ]
            publish_status(
                "plan_complete",
                "DAG generated successfully.",
                {"plan_id": dag_plan.plan_id, "nodes": nodes_summary},
            )

            # Step 3: Execution
            publish_status(
                "stage_start",
                "Executing DAG nodes concurrently with ReAct workers...",
            )
            executor = DAGExecutor(max_workers=2)
            execution_output = await loop.run_in_executor(
                None, executor.execute_dag, dag_plan
            )
            publish_status(
                "execution_complete",
                "All research tasks completed.",
                execution_output,
            )

            # Step 4: Synthesis
            publish_status("stage_start", "Synthesizing research findings...")
            synthesizer = Synthesizer(provider=provider)
            report = await loop.run_in_executor(
                None,
                synthesizer.synthesize,
                query,
                execution_plan,
                execution_output,
            )

            report_data = (
                report.model_dump()
                if hasattr(report, "model_dump")
                else report
            )

            # Step 5: Store in Cache
            await loop.run_in_executor(
                None,
                cache.store,
                query,
                execution_plan.extracted_params,
                report,
            )

            # Final Output
            publish_status(
                "report_complete", "Research report compiled.", report_data
            )

        except Exception as e:
            publish_status("error", f"Pipeline failure: {str(e)}")
        finally:
            cache.close()
            publish_status("close", "Pipeline finished.")

    pipeline_task = asyncio.create_task(run_pipeline())

    while True:
        item = await queue.get()
        event_name = item["event"]
        data = item["data"]

        yield format_sse_event(event_name, data)

        if event_name in ["close", "error"]:
            break

    await pipeline_task


@app.get("/api/v1/research/stream")
async def stream_research(query: str, provider: str = "openai"):
    """Executes the 4-phase research agent pipeline and streams updates via SSE."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Input query cannot be empty.")

    return StreamingResponse(
        pipeline_event_generator(query, provider),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Research Agent API"}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)