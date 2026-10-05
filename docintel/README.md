# DocIntel

A retrieval-augmented generation (RAG) pipeline for PDFs containing both
prose and tables. Documents dropped into `documents/` are registered with
versioning, parsed, and split into validated, token-limited chunks ready
for embedding.

## Pipeline

```
documents/*.pdf
  -> ingestion/watcher      register (checksum + version) in data/registry.db
  -> parsing/pdf_parser     text blocks + tables (PyMuPDF), header/footer removal
  -> chunking/pipeline      text chunks + table chunks, lossless-coverage checks
  -> chunks table           data/registry.db
  -> embeddings             all-MiniLM-L6-v2, cached in chunk_vectors table
  -> retrieval/search       hybrid: vectors + BM25 keywords, merged by rank
  -> qa                     (to do)
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
| Check embeddings + retrieval scenarios | `python -m docintel.embeddings.check_embeddings` |
| Same, printing the top chunks for misses | `python -m docintel.embeddings.check_embeddings --show-misses` |
| Search the documents | `python -m docintel.retrieval.search "your question"` |
| Compare hybrid / vector / BM25 retrieval | `python -m docintel.retrieval.evaluate` |
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

## Configuration

Settings live in `src/docintel/config.py`: paths, embedding model, chunk
token limit (200 for MiniLM's 256-token input), and rows per table chunk.
Set `DOCINTEL_HOME` to use a different project folder.

## Evaluation

`eval/retrieval_cases.json` lists questions with the text the correct
chunk must contain, grouped by scenario (table cell lookup, distractor
column, year headers, notes, narrative text, definitions, trends,
unanswerable). Add cases whenever you add documents.

## Document statuses

`DISCOVERED -> PROCESSING -> CHUNKED`, or `NEEDS_REVIEW` (parser warnings)
or `FAILED` (error stored in `error_message`).
