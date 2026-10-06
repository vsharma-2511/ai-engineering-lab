# DocIntel

A retrieval-augmented generation (RAG) pipeline for PDFs containing both
prose and tables. Documents dropped into `documents/` are registered with
versioning, parsed, split into validated, token-limited chunks, embedded,
and answered from with verified page citations.

## Pipeline

```
documents/*.pdf            (inbox; processed PDFs move to archive/)
  -> ingestion/watcher      register (checksum + version) in data/registry.db
  -> parsing/pdf_parser     text blocks + tables (PyMuPDF), header/footer removal
  -> chunking/pipeline      text chunks + table chunks, lossless-coverage checks
  -> chunks table           data/registry.db
  -> embeddings             all-MiniLM-L6-v2, cached in chunk_vectors table
  -> retrieval/search       hybrid: vectors + BM25 keywords, merged by rank
  -> qa                     Ollama (local) / Gemini / OpenAI / Claude answer with verified page citations
```

Table rows are rendered with their column names so embeddings can tell
which number answers which question:

```
Table 3. Program participation, 2025
Program: Digital skills | Location: East | Participants: 85 | Completion rate: 76%
```

## Setup

```bash
cd docintel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

Run these from the `docintel` folder (after `pip install -e .` they work
from anywhere):

| Task | Command |
|---|---|
| Watch `documents/` and process new PDFs | `python -m docintel.ingestion.watcher` |
| Re-run parsing/chunking for registered PDFs | `python -m docintel.ingestion.reprocess` |
| List library documents | `python -m docintel.ingestion.library list` |
| Remove a document and all its data | `python -m docintel.ingestion.library remove "Report.pdf"` |
| List / end / clean up chat sessions | `python -m docintel.sessions list` (or `end <id>`, `cleanup`) |
| Ask about one session's documents | `python -m docintel.qa.ask "question" --session <id>` |
| Check embeddings + retrieval scenarios | `python -m docintel.embeddings.check_embeddings` |
| Same, printing the top chunks for misses | `python -m docintel.embeddings.check_embeddings --show-misses` |
| Search the documents | `python -m docintel.retrieval.search "your question"` |
| Compare hybrid / vector / BM25 retrieval | `python -m docintel.retrieval.evaluate` |
| Ask a question | `python -m docintel.qa.ask "your question"` |
| Run the QA eval (one LLM call per case) | `python -m docintel.qa.evaluate --sleep 4` |
| Unit tests | `pytest` |

Without installing, you can run the same modules from the folder above
the project with the longer path, e.g.
`python -m docintel.src.docintel.ingestion.watcher`. Once the package is
installed, use the short form only.

## Retrieval

```python
from docintel.retrieval.search import search

for hit in search("Who prepared this dataset?", top_k=5):
    print(hit["score"], hit["document_name"], hit["page_number"], hit["text"])
```

`search()` covers the latest version of every `CHUNKED` document. It
ranks chunks two ways: by embedding similarity, which handles
paraphrases, and by BM25 keyword score, which handles exact terms such
as names and labels. It then merges the two rankings with reciprocal
rank fusion. Use `mode="vector"` or `mode="bm25"` to get one ranking
alone. Each result also carries `vector_similarity`, `bm25_score` and
both ranks for debugging.

Chunk vectors are cached in `data/registry.db`, so only new or changed
chunks are embedded. The in-memory index rebuilds automatically when
documents are added or reprocessed.

## Question answering

`qa` retrieves the top chunks, sends them to an LLM as numbered sources,
and asks for JSON: whether the sources answer the question, the answer
with `[n]` markers, and an exact quote for each source used. Every quote
is then checked against its chunk, so a citation is kept only if those
words really appear on that page. The result `status` is:

- `answered`: at least one citation was verified
- `not_found`: the documents don't answer the question
- `unsupported`: the model answered, but none of its quotes are in the
  sources, so don't trust the answer

### Choosing the model

| Provider | Install | API key | Default model |
|---|---|---|---|
| `ollama` (default) | [Ollama app](https://ollama.com/download), then `ollama pull qwen3:4b` | none, runs locally | `qwen3:4b` |
| `gemini` | `pip install -e ".[gemini]"` | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| `openai` | `pip install -e ".[openai]"` | `OPENAI_API_KEY` | `gpt-5.4-mini` |
| `claude` | `pip install -e ".[claude]"` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |

Choose with environment variables, or per command with `--provider` and
`--model`:

```bash
export DOCINTEL_LLM_PROVIDER=ollama      # ollama | gemini | openai | claude
export DOCINTEL_LLM_MODEL=qwen3:4b       # optional
python -m docintel.qa.ask "Who prepared this dataset?" --provider gemini
```

Ollama runs the model on your own machine, so it's free and nothing
leaves your computer. Optional settings: `OLLAMA_HOST` (default
`http://localhost:11434`), `DOCINTEL_OLLAMA_NUM_CTX` (context window,
default 4096 tokens), `DOCINTEL_OLLAMA_TIMEOUT` (seconds, default 300)
and `DOCINTEL_OLLAMA_THINK` (default `true`: thinking models such as
qwen3 reason before answering; without it qwen3:4b missed answers that
were in the sources). On an 8 GB Mac, stay with models of about 4B
parameters or fewer, and expect roughly 10–20 seconds per answer.

Gemini's free tier is enough for this prototype. Google may use
free-tier prompts to improve its products, so don't send confidential
documents on it.

## Configuration

Settings live in `src/docintel/config.py`: paths, embedding model, chunk
token limit (200 for MiniLM's 256-token input), and rows per table chunk.
Set `DOCINTEL_HOME` to use a different project folder.

## Evaluation

`eval/retrieval_cases.json` lists questions with the text the correct
chunk must contain, grouped by scenario (table cell lookup, distractor
column, year headers, notes, narrative text, definitions, trends,
unanswerable). Add cases whenever you add documents.

## Document library and chat sessions

There are two places a document can live.

**The library** holds documents you keep. Drop a PDF into `documents/`
while the watcher runs. Once it is processed (`CHUNKED`), the PDF moves
to `archive/<name>.v<version>.pdf`. The original is kept so `reprocess`
can re-read it after parser or chunker changes. Files that end up
`NEEDS_REVIEW` or `FAILED` stay in `documents/`. Plain searches and
questions use the library.

**Chat sessions** are for chat and forget. A file uploaded into a session
(`sessions.add_session_document`) is searchable only inside that
session, never from the library. Ending the session deletes it.
Sessions idle for more than 24 hours (`DOCINTEL_SESSION_IDLE_HOURS`) are
ended by `python -m docintel.sessions cleanup`. If an upload is identical
to a library document, the session reuses the library's chunks, and
ending the session leaves the library copy alone.

**Removing a document** deletes everything derived from it, for every
version: chunks, vectors, parsed JSON, registry rows and its PDF. Run
`python -m docintel.ingestion.library remove "Report.pdf"`, or delete
the PDF from `archive/` (or from `documents/` if it was never archived).
The watcher notices deletions while it runs, and catches the rest the
next time it starts.

## Document statuses

`DISCOVERED -> PROCESSING -> CHUNKED`, or `NEEDS_REVIEW` (parser warnings)
or `FAILED` (error stored in `error_message`).
