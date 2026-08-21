🔬 Autonomous Multi-Agent Research System
A production-grade, asynchronous AI Research Agent platform built in Python. The system converts high-level user queries into parallelized Directed Acyclic Graphs (DAGs), deploys autonomous ReAct worker agents equipped with web search and scraping tools, and compiles grounded research reports with live Server-Sent Events (SSE) streaming and an interactive web console.

🛠️ Architecture & Core Mechanics
The pipeline is structured into 4 decoupled phases to ensure control, parallel execution, and strict data grounding.

+-----------------------------------+
                          |         Incoming User Query       |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          | Phase 1.1: Query Decomposition    |
                          |  • Parameters & Sub-tasks Extracted|
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          | Phase 1.2: Dynamic DAG Planner    |
                          |  • Constructs Task Nodes & Graph  |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          | Phase 3: Concurrent DAG Executor  |
                          |  • ThreadPoolExecutor Batches     |
                          +-----------------------------------+
                                    /               \
                                   v                 v
                      +-------------------+   +-------------------+
                      | Phase 2: Worker A |   | Phase 2: Worker B |
                      | (ReAct Search/    |   | (ReAct Search/    |
                      |  Scrape Loop)     |   |  Scrape Loop)     |
                      +-------------------+   +-------------------+
                                   \                 /
                                    v               v
                          +-----------------------------------+
                          | Phase 4: Report Synthesizer       |
                          |  • Grounded Summary & Metrics     |
                          +-----------------------------------+
                                            |
                                            v
                          +-----------------------------------+
                          | Real-Time FastAPI SSE API / UI    |
                          +-----------------------------------+
📋 System Phases & Component Breakdown
Phase 1: Query Decomposition & Dynamic DAG Planning
src/planning/decomposer.py (decompose_query)

What it does: Converts unstructured queries into structured parameters (indicators, locations, timeframe, target_domains) and atomic SubTask objects.

How it does it: Uses structured output parsing (Pydantic) via OpenAI or Gemini to guarantee strict schema validation.

src/planning/planner.py (build_dag_plan)

What it does: Converts atomic sub-tasks into an executable DAGExecutionPlan containing TaskNode elements.

How it does it: Analyzes task dependencies, assigns unique IDs, and flags nodes as parallelizable or sequential to maximize concurrent throughput.

Phase 2: ReAct Worker Agents & Tools
src/agents/research_agent.py (ResearchWorkerAgent)

What it does: Runs an isolated Reasoning + Action (ReAct) execution loop to fulfill a single assigned task node.

How it does it: Evaluates search results, decides whether to scrape specific URLs, and iteratively gathers evidence up to a max iteration ceiling (default: 6).

src/tools/web_tools.py (web_search, scrape_webpage)

web_search: Queries search engines (e.g., SerpAPI/Google) restricted to requested domain parameters.

scrape_webpage: Fetches webpage text, cleans raw HTML, and extracts relevant body content.

Phase 3: Concurrent DAG Execution Engine
src/planning/executor.py (DAGExecutor)

What it does: Orchestrates the execution of all nodes in a DAGExecutionPlan.

How it does it: Identifies ready nodes without unfulfilled dependencies and executes them concurrently using concurrent.futures.ThreadPoolExecutor. Stores results in a unified state dictionary for synthesis.

Phase 4: Structured Report Synthesis
src/synthesis/synthesizer.py (Synthesizer)

What it does: Compiles gathered findings into a publication-ready report.

How it does it: Maps execution results to a SynthesizedReport Pydantic model containing an Executive Summary, Key Metrics, Detailed Sectional Analysis, and Cited Source Links.

FastAPI Layer & Real-Time SSE Streaming
src/api/app.py

What it does: Exposes the pipeline over HTTP and streams stage-by-stage execution updates.

How it does it: Uses asyncio.Queue and StreamingResponse to push W3C-compliant Server-Sent Events (init, stage_start, plan_complete, report_complete) in real time without blocking.

src/api/index.html

What it does: Offers a browser interface for running research queries and monitoring progress live.

📂 Directory Structure
Plaintext
Research_Agent/
├── config/
│   └── settings.py          # Environment settings & API keys
├── schemas/
│   └── query_schema.py      # Core Pydantic schemas (ExecutionPlan, SubTask, DAG)
├── planning/
│   ├── decomposer.py        # Phase 1.1: Query decomposition
│   ├── planner.py           # Phase 1.2: Dynamic DAG generation
│   └── executor.py          # Phase 3: Concurrent thread-pool executor
├── agents/
│   └── research_agent.py    # Phase 2: ReAct execution worker loop
├── tools/
│   └── web_tools.py         # Search & Web scraping tools
├── synthesis/
│   └── synthesizer.py       # Phase 4: Final report synthesizer
├── api/
│   ├── app.py               # FastAPI server with SSE endpoint
│   └── index.html           # Interactive live dashboard
└── main.py                  # CLI pipeline entrypoint
🚀 Setup & Execution Guide
1. Prerequisites
Python 3.10+

API Keys for OpenAI (OPENAI_API_KEY) and/or Google Gemini (GEMINI_API_KEY).

2. Installation
Clone the repository and set up your virtual environment:

Bash
git clone https://github.com/your-username/Research_Agent.git
cd Research_Agent

python3 -m venv .venv
source .venv/bin/activate

pip install fastapi uvicorn pydantic requests google-genai openai
Configure your API keys in your environment or a .env file:

Bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export LLM_PROVIDER="openai" # Options: 'openai' or 'gemini'
3. Running the System
Option A: Terminal CLI Pipeline
To run the full 4-phase pipeline end-to-end directly in your terminal:

Bash
python -m Research_Agent.main
Option B: FastAPI Server & Live Interactive Dashboard
Launch the API server using Uvicorn:

Bash
uvicorn Research_Agent.src.api.app:app --reload --port 8000
Open your web browser and go to:

Plaintext
http://localhost:8000
Type your query into the search console and click Run Research to watch each execution stage stream in real time.

Option C: Stream via curl
You can also connect to the SSE endpoint directly via terminal:

Bash
curl -N "http://localhost:8000/api/v1/research/stream?query=What%20is%20the%20inflation%20rate%20and%20GDP%20growth%20in%20Canada%20for%202024%3F"
🗺️ Roadmap
[ ] Semantic Vector Caching: Check historical query embeddings in Vector DB before execution to return instant cached responses.

[ ] RAG Tool Integration: Store scraped web pages as vector chunks to let workers search internal memory before making web requests.

[ ] Evaluator/Verifier Agent: Add an automated verification step to validate retrieved data quality before generating final reports.

[ ] Disk Exporters: Automatically save generated reports as Markdown (.md) and PDF files under a /reports directory.