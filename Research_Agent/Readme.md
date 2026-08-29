# Multi-Agent Research & Data Extraction Engine

An asynchronous, DAG-based research pipeline that leverages Large Language Models (LLMs), ReAct-based web agent workers, semantic caching, and FastAPI streaming to process complex multi-variable analytical research queries.

## 🛠 Project Capabilities & Architecture Overview

The system processes complex queries through a multi-stage execution pipeline:

- **LLM Query Decomposition & Metadata Extraction:** Deconstructs broad user queries into structured JSON parameters, identifying key target indicators (e.g., inflation, GDP growth), locations, and temporal constraints.
- **ReAct Agent Workers for URL Discovery & Extraction:** Uses ReAct (Reasoning + Acting) decision-making loops to continuously search, evaluate search results, select high-confidence URLs, and extract target metrics.
- **SerpAPI Search Engine Integration:** Executes real-time Google search queries via SerpAPI to obtain live web search results and top-ranking destination URLs.
- **BeautifulSoup Content Extraction & Parsing:** Fetches and parses raw HTML pages from candidate URLs, stripping boilerplate code to isolate pertinent body text and data points for the ReAct workers.
- **Parallel Execution Engine (`concurrent.futures` / `asyncio`):** Utilizes thread pools and asynchronous execution graphs (DAGs) to run independent research tasks concurrently, minimizing end-to-end pipeline latency.
- **FastAPI Async Streaming Server:** Exposes research execution as an asynchronous API featuring real-time Server-Sent Events (SSE) log streaming (`/api/v1/research/stream`).
- **Semantic Caching with Vector Embeddings (Qdrant & Gemini):** Stores raw research outputs as high-dimensional embeddings using Google Gemini models and Qdrant vector database. Incoming queries undergo cosine similarity checks to serve cached results instantly, bypassing LLM pipeline execution when similarity thresholds are met.

---

## 🔄 System Workflow & Low-Level Technical Mechanics

```text
User Query ──► [Step 0: Semantic Cache Lookup] ──(Hit)──► Return Cached Output
                       │
                    (Miss)
                       ▼
            [Step 1: Query Decomposition]
                       │
                       ▼
         [Step 2: DAG Execution Planning]
                       │
                       ▼
      [Step 3: Parallel ReAct Worker Execution]
       ├── SerpAPI Search
       ├── BeautifulSoup Scraping
       └── ReAct LLM Reasoning Loop
                       │
                       ▼
           [Step 4: Synthesis & Output]
                       │
                       ▼
       [Step 5: Qdrant Cache Ingestion]
```

### Step 0: Semantic Vector Cache Check

Before running LLM decomposition, the pipeline converts the raw string query into a dense mathematical vector representation.

- **Embedding Model:** Google's `gemini-embedding-001` generates a **768-dimensional** vector representing the semantic meaning of the user query.
- **Vector Database (Qdrant):** The system connects to a local disk-backed Qdrant instance stored at `Research_Agent/data/cache_db`.
- **Similarity Matching:** Qdrant performs a cosine similarity check against existing stored vectors. If the similarity score is **≥ 0.85**, the cache yields a `CACHE_HIT` event and returns the saved payload instantly. If the similarity score is **< 0.85**, a `CACHE_MISS` event triggers full pipeline execution.

### Step 1: Query Decomposition & Metadata Parsing

The incoming prompt is routed to the primary LLM with strict JSON schema enforcement. The LLM extracts:

1. `indicators`: Primary metrics to discover (e.g., `["inflation rate", "GDP growth"]`).
2. `locations`: Target geographical entities (e.g., `["Canada"]`).
3. `timeframe`: Target years or periods (e.g., `"2024"`).

### Step 2: Directed Acyclic Graph (DAG) Generation

The engine constructs a DAG of discrete execution tasks. Independent tasks (e.g., searching for **Canada Inflation 2024** vs. **Canada GDP Growth 2024**) are marked as **parallel** execution nodes, while final summary and synthesis tasks are marked as **sequential** dependent nodes.

### Step 3: ReAct Worker Execution Loop

Tasks marked for web execution are assigned to ReAct agent instances. Each agent operates in an iterative loop:

1. **Thought:** The agent reasons about what web queries are needed.
2. **Action (`serp_search`):** Calls SerpAPI to retrieve top organic Google search results.
3. **Observation:** Evaluates returned titles, snippets, and target URLs.
4. **Action (`scrape_url`):** Utilizes `BeautifulSoup4` with HTTP request handling to download target HTML content, stripping clean text from paragraphs and tables.
5. **Thought & Final Answer:** Synthesizes the extracted text into a structured response with direct source attribution.

### Step 4: Output Synthesis & Stream Dispatch

The results from parallel worker tasks are merged into a single context block. A synthesis LLM pass creates an executive summary comparing all requested metrics. The response is streamed to the frontend UI via FastAPI's EventSource interface.

### Step 5: Vector Ingestion

The newly generated output, along with the user query's **768-dimensional embedding**, is saved as a new payload point in the local Qdrant collection for future query matching.

---

## 📦 Required Dependencies & Libraries

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Then install the required packages:

```bash
pip install fastapi uvicorn qdrant-client google-genai requests beautifulsoup4 pydantic pydantic-settings python-dotenv
```

### Key Libraries Used

- **`fastapi` & `uvicorn[standard]`**: Asynchronous web framework and ASGI server for SSE streaming.
- **`qdrant-client`**: Vector search engine interface for embedding storage and cosine similarity lookups.
- **`google-genai`**: Official Google SDK for Gemini LLM prompting and `gemini-embedding-001` vector generation.
- **`beautifulsoup4` & `requests`**: HTML DOM parsing and HTTP communication for web scraping.
- **`pydantic-settings`**: Type-safe configuration management reading from `.env` files.

---

## 🔑 Environment Setup (`.env`)

Create a `.env` file in the project root directory containing your API credentials:

```env
# Gemini API Key for LLM reasoning and embedding generation
GEMINI_API_KEY="your-google-gemini-api-key"

# SerpAPI Key for Google search scraping execution
SERPAPI_KEY="your-serpapi-api-key"

# Optional: Override default logging/environment flags
ENVIRONMENT="development"
```

---

## 🚀 How to Run the Project

### 1. Start the FastAPI Application Server

Run Uvicorn from the project root directory:

```bash
uvicorn Research_Agent.src.app:app --reload
```

The server will start at:

```text
http://127.0.0.1:8000
```

### 2. Access the Interactive UI & Documentation

- **Interactive Web UI Console:** `http://127.0.0.1:8000/`
- **Swagger API Documentation:** `http://127.0.0.1:8000/docs`

### 3. Execute a Research Query via API

Call the streaming SSE API directly using `curl`:

```bash
curl -N "http://127.0.0.1:8000/api/v1/research/stream?query=What%20is%20the%20inflation%20rate%20and%20GDP%20growth%20in%20Canada%20for%202024%3F"
```
