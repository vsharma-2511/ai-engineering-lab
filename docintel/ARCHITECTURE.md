# DocIntel architecture

DocIntel answers questions about PDFs (typed, scanned or both) and backs
every answer with quotes that are checked against the page they cite.
This document describes every module and follows a PDF from upload to a
chat answer. For setup and commands, see [README.md](README.md).

```
                         ┌──────────────── app.py (Streamlit) ─────────────────┐
                         │  Chat page                        Library page      │
                         └──────┬─────────────────────────────────┬────────────┘
                                │ upload into a chat              │ add / remove
  documents/ ── watcher ───┐    │                                 │
                           ▼    ▼                                 ▼
                    1. INGEST  registry: checksum, version, scope (library or session)
                           │
                    2. PARSE   page_analysis → text layer (PyMuPDF) and/or OCR
                           │   → pages of text blocks + tables  → data/parsed/<id>.json
                           │
                    3. CHUNK   text chunks + table chunks, coverage-validated
                           │   → chunks table (SQLite)          → archive/<name>.v<N>.pdf
                           │
  question ──► 6. UNDERSTAND (chat follow-ups rewritten)
                           │
               5. RETRIEVE  ◄── 4. EMBED (lazily, cached in chunk_vectors)
                           │   embeddings + BM25, merged by rank, top 5
                           │
               7. ANSWER    LLM (Ollama / Gemini / OpenAI / Claude) → JSON with quotes
                           │
               8. VERIFY    every quote checked against its chunk → citation cards
```

## Contents

1. [Repository layout](#1-repository-layout)
2. [Where data lives](#2-where-data-lives)
3. [End-to-end flow](#3-end-to-end-flow)
4. [Module reference](#4-module-reference)
5. [Design rules that hold everywhere](#5-design-rules-that-hold-everywhere)
6. [Configuration](#6-configuration)
7. [Tests and evaluation](#7-tests-and-evaluation)
8. [Known limitations](#8-known-limitations)
9. [Lessons learned](#9-lessons-learned)

---

## 1. Repository layout

```
docintel/
├── app.py                      Streamlit chat app (Chat and Library pages)
├── pyproject.toml              package, dependencies, optional extras, pytest settings
├── eval/retrieval_cases.json   16 questions for the retrieval and QA evals
├── src/docintel/
│   ├── config.py               all settings (paths, models, OCR, sessions)
│   ├── tokens.py               token counting with the embedding model's tokenizer
│   ├── sessions.py             chat sessions and their private documents
│   ├── ingestion/              getting documents in and out
│   │   ├── registry.py         documents table: checksum, version, status, scope
│   │   ├── processor.py        parse → chunk → store one document version
│   │   ├── watcher.py          watches documents/, processes, archives, notices deletions
│   │   ├── library.py          archive, add, list, delete library documents
│   │   └── reprocess.py        re-run parsing/chunking for registered documents
│   ├── parsing/                PDF → pages of text blocks and tables
│   │   ├── page_analysis.py    classify each page: text / scanned / mixed / blank
│   │   ├── pdf_parser.py       read each page (text layer and/or OCR)
│   │   ├── ocr.py              EasyOCR and Docling engines, OCR layout reconstruction
│   │   └── cleanup.py          mark repeated headers/footers ("page furniture")
│   ├── chunking/               pages → validated, token-limited chunks
│   │   ├── text_chunker.py     text blocks → text chunks with source spans
│   │   ├── table_chunker.py    tables → row-group chunks rendered with column names
│   │   ├── pipeline.py         runs both and proves nothing was lost or changed
│   │   └── chunk_storage.py    chunks table; loads the latest chunks per scope
│   ├── embeddings/
│   │   ├── encoder.py          sentence-transformers wrapper (all-MiniLM-L6-v2)
│   │   └── check_embeddings.py embedding health check + vector-only retrieval eval
│   ├── retrieval/
│   │   ├── vector_store.py     chunk_vectors cache table
│   │   ├── bm25.py             keyword scoring
│   │   ├── search.py           hybrid search (embeddings + BM25, rank fusion)
│   │   └── evaluate.py         compares hybrid / vector / BM25 on the eval cases
│   └── qa/
│       ├── llm.py              one JSON interface over Ollama, Gemini, OpenAI, Claude
│       ├── answer.py           prompt, LLM call, citation verification, statuses
│       ├── chat.py             follow-up question rewriting, one chat turn
│       ├── ask.py              command line: ask one question
│       └── evaluate.py         QA eval over eval/retrieval_cases.json
└── tests/                      unit tests (no model downloads, no LLM calls)
    └── fixtures/               a parsed test PDF and real EasyOCR output
```

---

## 2. Where data lives

All paths are relative to the project folder (`config.PROJECT_ROOT`, or
`DOCINTEL_HOME` if set).

| Location | Contents | Written by |
|---|---|---|
| `documents/` | Library inbox. New PDFs, and files that ended `NEEDS_REVIEW` or `FAILED`. | you; `watcher`, `library.add_to_library` |
| `archive/<name>.v<N>.pdf` | Originals of processed library documents, one file per version. | `library.archive_document` |
| `data/registry.db` | SQLite database with five tables (below). | everything |
| `data/parsed/<document_id>.json` | Full parse result per document version: pages, blocks, tables, OCR info, quality. | `pdf_parser.save_parse_result` |
| `data/uploads/<session_id>/` | Files uploaded into a chat session. | `sessions.add_session_document` |
| `data/uploads/library/` | Short-lived staging copy while the app adds a library document. | `library.add_to_library` |
| `~/.EasyOCR/`, Hugging Face cache | OCR and embedding models, downloaded once. | EasyOCR, Docling, sentence-transformers |

**Database tables** (`data/registry.db`):

| Table | One row per | Key columns | Defined in |
|---|---|---|---|
| `documents` | registered document **version** | `document_id`, `filename` (registry key), `checksum`, `version`, `status`, `source_path`, `archive_path`, `session_id` | `ingestion/registry.py` |
| `chunks` | chunk | `chunk_id`, `document_id`, `document_version`, `page_number`, `content_type` (`text`/`table`), `text`, `metadata_json` | `chunking/chunk_storage.py` |
| `chunk_vectors` | chunk × embedding model | `chunk_id`, `model`, `text_sha256`, `vector` (float32 bytes) | `retrieval/vector_store.py` |
| `sessions` | chat session | `session_id`, `created_at`, `last_active_at` | `ingestion/registry.py` |
| `session_documents` | document a session may search | `session_id`, `document_id` | `ingestion/registry.py` |

**Document status** (`documents.status`): `DISCOVERED → PROCESSING →
CHUNKED`, or `NEEDS_REVIEW` (the parser flagged a page), or `FAILED`
(an error, stored in `error_message`). Only `CHUNKED` documents are
searched.

**Library vs. session documents.** `documents.session_id` is `NULL` for
library documents. For a session upload it holds the session id, and
the registry key in `filename` becomes `session:<session_id>/<name>`
(`registry.registry_key`). This keeps a chat upload called
`Report.pdf` from becoming "version 2" of the library's `Report.pdf`.

---

## 3. End-to-end flow

### 3.1 Ingest: four ways in, one processing path

| Entry point | Scope | Code |
|---|---|---|
| Drop a PDF into `documents/` while the watcher runs | library | `ingestion/watcher.py` |
| Library page → *Add to library* | library | `library.add_to_library` |
| Chat sidebar upload, or `python -m docintel.sessions add` | session | `sessions.add_session_document` |
| `python -m docintel.ingestion.reprocess` | existing documents | `ingestion/reprocess.py` |

The first three paths do the same two things. `reprocess` skips step 1:
it re-runs step 2 for the latest registered version of each document,
and skips any file whose contents no longer match the registered
checksum.

1. **Register** (`registry.register_document`). A SHA-256 checksum of
   the file is compared within the same scope (library, or that one
   session):
   - same name, same contents → `already_registered`, nothing to do
   - same contents under another name → `duplicate_content`, skipped
   - otherwise a new row with `version` = previous + 1 and status
     `DISCOVERED`

   The watcher only registers a file once it has been unchanged for
   `WATCHER_QUIET_SECONDS` (3 s), so half-copied files are never read.
   A session upload whose contents match a library document is not
   processed again: the session is linked to the library's chunks
   (`REUSED`).
2. **Process** (`processor.process_document`), described next.

### 3.2 `process_document`: parse, chunk, store

```
status = PROCESSING
parsed = parse_pdf(...)                      → saved to data/parsed/<id>.json
if any page has warnings: status = NEEDS_REVIEW, stop    (nothing is chunked)
chunks = create_validated_chunks(parsed)     (raises if anything is lost)
if the file changed meanwhile: raise         (never store chunks of a different file)
save_chunks(chunks); status = CHUNKED
on any exception: status = FAILED with the error message
```

After `CHUNKED`, the watcher and the Library page move library files to
`archive/<name>.v<N>.pdf` (`library.archive_document`). `reprocess` does
not move files: a `CHUNKED` file it leaves in `documents/` is archived
the next time the watcher starts. Session uploads stay in
`data/uploads/<session>/` until the session ends.

### 3.3 Parse: PDF → pages of blocks and tables

`parsing/pdf_parser.parse_pdf` handles one page at a time.

**Step 1: classify the page** (`page_analysis.analyze_page`). This is
cheap and runs no OCR.

| Kind | Rule | How the page is read |
|---|---|---|
| `text` | ≥ 20 characters in the text layer, and no large text-free image | PyMuPDF text layer only |
| `scanned` | < 20 characters, plus at least one image or ≥ 50 vector drawings | OCR of the whole page |
| `mixed` | a text layer, plus image(s) covering ≥ 10% of the page with < 20 characters of text over them | text layer, plus OCR of just those image regions |
| `blank` | none of the above | nothing (no warning) |

An image with a text layer on top of it, as in an already-OCR'd scan,
is read from that text layer and never OCR'd twice. The document type
(`page_analysis.document_type`) is `digital` (only text pages),
`scanned` (every non-blank page scanned) or `mixed`.

**Step 2: read the page.**
- *Text layer* (`_read_text_layer`): `page.find_tables()` extracts
  tables (rows of cells plus PyMuPDF's header guess), and
  `page.get_text("blocks")` gives text blocks with bounding boxes.
- *OCR* (`_run_ocr` → `ocr.get_ocr_engine().recognize(...)`): either
  the whole page or each image region. The engine returns an
  `OcrResult` with blocks and tables, in **PDF points with a top-left
  origin**, the same as PyMuPDF. That shared shape is what lets
  everything downstream ignore where text came from.
  - **EasyOCR** (default): Pillow renders the page in grayscale at
    `OCR_DPI` and stretches the contrast. EasyOCR returns text segments.
    Segments below `OCR_MIN_LINE_CONFIDENCE` are dropped, and an "S"
    misread for "$" directly before an amount (`S185,000`) is fixed
    (`fix_currency`). `ocr.layout_lines` then rebuilds the layout:
    1. segments become lines, split into pieces at wide gaps
    2. a run of lines becomes a **table** when its pieces line up in
       columns, look like cells (narrow), and the first line is
       header-like (text, not numbers). Each segment goes to the column
       it starts in, so close header words ("Annual Cost" | "Term")
       still separate.
    3. with no table and only short labels (on average fewer than 8
       characters per segment, as in a chart's axis ticks and legend),
       the region becomes one **figure block**
    4. everything else becomes paragraph blocks. Section headings,
       "Table N" titles and Note/Footnote/Source lines always start
       their own block, because the chunkers find titles and notes by
       how a block begins.
  - **Docling** (optional): converts that one page with its layout and
    TableFormer models (EasyOCR inside). `ocr.items_to_result` maps its
    items to blocks and tables, flipping from Docling's bottom-left
    origin.
- OCR blocks are numbered after the text-layer blocks, so
  `(page, block_number)` stays unique.

**Step 3: classify blocks** (`classify_block`): each text block is
`OUTSIDE_TABLE`, `INSIDE_TABLE` or `PARTIAL_OVERLAP` relative to the
page's tables. An **OCR** block that partly overlaps a table becomes
`OUTSIDE_TABLE`. OCR builds tables and blocks from different words, so
the overlap is only geometry, such as a chart label beside a table. A
text-layer `PARTIAL_OVERLAP` still stops chunking, so the document is
reviewed.

**Step 4: page warnings.** Any warning sends the document to
`NEEDS_REVIEW`. The possible warnings are `ocr_disabled`, `ocr_failed`,
`ocr_found_no_text`, `low_ocr_confidence` (mean below
`OCR_MIN_PAGE_CONFIDENCE`), `little_or_no_extractable_text`, table
extraction problems, and `no_text_in_document`.
`library.processing_message` turns them into plain sentences for the
user.

**Step 5: page furniture** (`cleanup.mark_page_furniture`). A block is
marked `PAGE_FURNITURE` and left out of chunks if it meets three
conditions:
- it is among the first or last two text blocks on its page
- it is at most 200 characters
- it is a bare page number, or it repeats (ignoring digits) on at least
  half the pages, with a minimum of two

Each page records `page_type`, `method` (`embedded_text`, `ocr`,
`embedded_text+ocr`, `none`), `ocr` details and `warnings`. Each block
records its `source` (`text_layer` or `ocr`). The document's `quality`
records `document_type`, `page_types`, `ocr_pages` and
`pages_needing_review`.

### 3.4 Chunk: pages → validated chunks

`chunking/pipeline.create_validated_chunks` runs both chunkers, then
proves the result is complete.

- **Text chunks** (`text_chunker.create_grouped_text_chunks`). Text
  blocks outside tables, in page order, are grouped up to
  `CHUNK_MAX_TOKENS` (200, measured with MiniLM's own tokenizer). A
  block that is too long is split at sentence ends, then whitespace,
  then a hard cut. Each chunk keeps `source_spans`, mapping every
  character back to its block, and the current numbered section.
  `INSIDE_TABLE` blocks end a chunk, `PAGE_FURNITURE` is skipped, and a
  `PARTIAL_OVERLAP` block stops processing with an error (the document
  becomes `FAILED`).
- **Table chunks** (`table_chunker.create_table_chunks`). The header
  row is PyMuPDF's header when it sits above the table body. Otherwise
  it is the first row, if every cell is filled and none is a number
  (years allowed). Otherwise there is no header. Each data row is
  rendered with its column names, e.g. `Location: East | Participants:
  85`. The title ("Table N …" ending above the table) and the notes are
  attached. Notes are Note/Footnote/Source blocks below the table; the
  search stops at the next section heading, "Table N" title, table
  text, or the top of the next table. Rows are grouped to fit the
  token budget and `TABLE_MAX_ROWS_PER_CHUNK`, and each group repeats
  the title and column names.
- **Validation.** Every text block's characters must appear in exactly
  one chunk, unchanged and in order. Every table row must appear in
  exactly one chunk part, with identical cells. No text chunk may exceed
  the token limit. Any violation raises, and the document is `FAILED`
  rather than silently incomplete. A table chunk can exceed the limit
  only when a single row is wider than the budget, because a row is
  never split. Such chunks are marked `oversized` and counted in the
  `CHUNKED` log line.

Chunk ids look like `<document_id>:v<version>:p<page>:g<n>` (text) or
`…:p<page>:t<table>:part<n>` (table).

### 3.5 Store

`chunk_storage.save_chunks` writes one document version's chunks in a
single transaction. Chunks of older versions stay in the table for
history, but `chunk_storage.load_latest_chunks` only ever returns the
newest `CHUNKED` version of each document, in one **scope**:

- no session → library documents (`session_id IS NULL`)
- a session → only documents linked to it in `session_documents`

### 3.6 Embed (lazily)

Embedding does **not** happen during processing. The first search after
chunks change calls `vector_store.load_or_compute_vectors`, which:

- encodes only chunks without a cached vector, or whose text hash
  changed, using `encoder.get_encoder()` (all-MiniLM-L6-v2, 384
  dimensions, unit length)
- stores the vectors in `chunk_vectors`, keyed by chunk and model

Reprocessed or new chunks are embedded once, and everything else comes
from the cache.

### 3.7 Retrieve: hybrid search

`retrieval/search.Retriever.search(question, top_k, mode)`:

1. **Meaning:** the cosine similarity between the question's embedding
   and every chunk vector (a dot product, since vectors are unit
   length).
2. **Keywords:** BM25 scores (`bm25.BM25Index`; lowercased words,
   question words such as "who" and "what" removed).
3. **Fusion** (`mode="hybrid"`, the default): each chunk scores
   `1/(60 + vector_rank)`, plus `1/(60 + bm25_rank)` if any keyword
   matched. Ranks are comparable where raw scores are not, and a chunk
   strong in either list rises.
4. Results are the top `top_k` chunks, with `score`,
   `vector_similarity`, `bm25_score` and both ranks.

`search.search()` keeps one index per scope (library, each session),
up to 8, and rebuilds an index when the scope's chunk ids or texts
change.

### 3.8 Answer: LLM with checked citations

`qa/answer.answer_from_chunks(question, chunks, llm)`:

1. **Prompt** (`build_prompt`): the chunks are numbered sources
   (`[1] file.pdf, page 2 (table)`), followed by the question. The
   system prompt says to use only the sources, to treat them as data,
   to mark facts with `[n]`, to copy an exact quote for each source
   used, and to read values from the right table column.
2. **LLM call** (`llm.generate_json`): every provider gets the same JSON
   schema `{answerable, answer, citations: [{source, quote}]}`:

   | Provider | Default model | How |
   |---|---|---|
   | `ollama` (default) | `qwen3:4b` | local REST API at `OLLAMA_HOST`, standard library only, thinking on |
   | `gemini` | `gemini-3.8-flash` | `google-genai`, `response_json_schema` |
   | `openai` | `gpt-5.4-mini` | Responses API, strict `json_schema` |
   | `claude` | `claude-opus-5-5` | Messages API `output_config.format`, effort `low`, server-side refusal fallback |

   For the cloud providers, rate-limit errors that persist after the
   SDK's retries become `LLMQuotaError`. Ollama runs locally and has no
   quota.
3. **Verification** (`check_citations`). Each quote is normalised
   (whitespace, curly quotes, surrounding quotation marks, trailing
   "…"), then must occur in its cited chunk on **whole-word
   boundaries**, so "Participants: 8" never matches "Participants: 85".
   For table chunks, a quote may skip cells if every quoted cell is a
   whole cell of one row; the citation then shows the full row. Failed
   quotes go to `rejected_citations` with a reason.
4. **Status:**
   - `answered`: at least one citation was verified
   - `not_found`: the model said the sources don't answer (or there
     were no chunks); any answer text is dropped
   - `unsupported`: the model answered, but no quote checked out, so
     don't trust the answer

### 3.9 Chat: follow-ups and the app

`qa/chat.chat_turn(question, history, llm, session_id, ...)`:

1. If there is history and rewriting is on, `rewrite_question` asks the
   LLM to turn the message into a **standalone question**, using the
   last 3 turns. For example, "and at Central?" becomes "How many people
   joined Digital skills at Central?". An empty rewrite falls back to
   the original.
2. If the turn is in a session, the session is marked active
   (`sessions.touch_session`).
3. It searches with the standalone question (library, or the session's
   documents), then calls `answer_from_chunks`.
4. The result adds `asked` (what was typed) and `searched_for` (what was
   searched).

`app.py` (Streamlit, `st.navigation` with two pages):

- **Sidebar, all pages:** provider and model (`load_llm`, cached per
  pair; errors such as "Ollama is not running" are shown instead of
  raised), the follow-up toggle, and sources per question.
- **Chat page:**
  - The sidebar takes uploads for *this chat only*. A docintel session
    is created on the first upload and files go through
    `add_session_document`.
  - A *Search in* toggle picks this chat's files or the library, and
    *End chat* calls `sessions.end_session`.
  - Answers render with citation cards (file, page, verified quote),
    the searched-for question (when the rewrite changed it), model and
    time, and the sources searched. The chat input is pinned to the
    bottom.
- **Library page:** a table of documents (version, type, status,
  chunks, file), an uploader that calls `add_to_library`, and *Remove*
  with a confirmation step (`delete_document`).
- **On server start**, sessions idle for more than
  `SESSION_IDLE_HOURS` are ended (`cleanup_expired_sessions`).

### 3.10 Lifecycle: archive, delete, forget

- **Delete a library document**
  (`library.delete_document(db, registry_key)`). For **every version**,
  in one transaction: `chunk_vectors`, `chunks`, `session_documents`
  and `documents` rows. Then the archived PDFs, the parsed JSON and the
  inbox copy (only if its checksum matches a registered version, so a
  newer unregistered file survives). Triggered by:
  - `library remove`, or the Library page
  - deleting the PDF from `archive/` (or from `documents/` before it was
    archived) while the watcher runs (`watcher.process_removals`)
  - at watcher start, `remove_missing_documents` for files deleted while
    it was off. This is skipped if `archive/` is missing entirely.
- **Watcher safety.** Its own move from inbox to archive also produces
  a "removed" event. `process_removals` ignores it because the document
  still has a current file (`library.current_file`). Locations are
  always resolved from the configured folders, never from stored
  absolute paths, so moving the project folder breaks nothing.
- **End a session** (`sessions.end_session`). `delete_document` runs for
  each document the session uploaded (identified by
  `documents.session_id`). Library documents it merely reused are only
  unlinked. Then the session row and `data/uploads/<session>/` are
  deleted.

---

## 4. Module reference

### Top level

| Module | Purpose | Main entry points |
|---|---|---|
| `config.py` | Every setting, read from environment variables where useful. Paths resolve from the project root. | constants (section 6) |
| `tokens.py` | Exact token counts with the embedding model's tokenizer (special tokens included), plus a fast word-based estimate for tests. | `get_token_counter`, `approximate_token_count` |
| `sessions.py` | Chat-and-forget sessions: create, upload, touch, end, idle cleanup. CLI: `new`, `add`, `list`, `end`, `cleanup`. | `create_session`, `add_session_document`, `end_session`, `cleanup_expired_sessions` |
| `app.py` | Streamlit UI (section 3.9). | `main` |

### `ingestion/`

| Module | Purpose | Main entry points |
|---|---|---|
| `registry.py` | `documents`, `sessions` and `session_documents` tables. Checksum and version registration per scope. Adds the session columns to older databases. | `initialize_registry`, `register_document`, `registry_key`, `get_latest_document(s)`, `get_document_versions`, `update_document_status`, `set_archive_path` |
| `processor.py` | Parse → (review gate) → chunk → checksum re-check → store, with status updates. Shared by every entry point. | `process_document`, `file_checksum` |
| `watcher.py` | Watches `documents/` and `archive/`. Processes stable inbox PDFs, archives `CHUNKED` ones, deletes documents whose PDF was removed, and catches up at start. | `main`, `process_ready_candidates`, `process_removals` |
| `library.py` | Archive naming and moves, adding from the UI, listing, full deletion, plain-language processing messages. CLI: `list`, `remove`. | `archive_document`, `add_to_library`, `delete_document`, `list_library`, `processing_message`, `current_file` |
| `reprocess.py` | Re-parses and re-chunks the latest version of every registered document (after parser or chunker changes). Reads from the archive, then the inbox, then the registered path. | `main` |

### `parsing/`

| Module | Purpose | Main entry points |
|---|---|---|
| `page_analysis.py` | Page kind (`text`/`scanned`/`mixed`/`blank`) and the OCR regions, from text length, image coverage and drawings. Document type. | `analyze_page`, `document_type` |
| `pdf_parser.py` | Reads each page by its kind, merges text-layer and OCR output, classifies blocks, collects warnings and quality, writes the parse JSON atomically. | `parse_pdf`, `save_parse_result`, `classify_block` |
| `ocr.py` | `EasyOcrEngine` (Pillow preprocessing, line → paragraph and table layout) and `DoclingEngine` (layout and table models). Both return `OcrResult`. | `get_ocr_engine`, `layout_lines`, `items_to_result`, `render_page`, `prepare_image` |
| `cleanup.py` | Marks repeated headers and footers as `PAGE_FURNITURE`. | `mark_page_furniture` |

### `chunking/`

| Module | Purpose | Main entry points |
|---|---|---|
| `text_chunker.py` | Groups text blocks into ≤ 200-token chunks with character-exact source spans and section names. | `create_grouped_text_chunks`, `split_text_block`, `is_section_heading` |
| `table_chunker.py` | Header detection, row rendering with column names, titles and notes, token-aware row groups. | `create_table_chunks`, `detect_header`, `render_row`, `find_table_title`, `find_table_notes` |
| `pipeline.py` | Runs both chunkers and validates complete, unchanged, non-overlapping coverage. | `create_validated_chunks` |
| `chunk_storage.py` | `chunks` table. Saves one version atomically, and loads the latest `CHUNKED` chunks per scope. | `save_chunks`, `load_latest_chunks`, `initialize_chunk_storage` |

### `embeddings/`

| Module | Purpose | Main entry points |
|---|---|---|
| `encoder.py` | Loads all-MiniLM-L6-v2 once per process and returns unit-length float32 vectors. | `get_encoder`, `SentenceTransformerBackend` |
| `check_embeddings.py` | Health check (empty or overlong chunks, non-finite or non-unit vectors, unstable encodings, duplicates) and a vector-only retrieval eval. `--backend tfidf` gives an offline baseline. | `main`, `is_relevant` |

### `retrieval/`

| Module | Purpose | Main entry points |
|---|---|---|
| `vector_store.py` | `chunk_vectors` cache. Re-embeds only new chunks or chunks whose text changed. | `load_or_compute_vectors` |
| `bm25.py` | Okapi BM25 (k1 = 1.5, b = 0.75) in plain Python. | `BM25Index`, `tokenize` |
| `search.py` | Hybrid retriever with reciprocal rank fusion, one cached index per scope, and a CLI. | `search`, `build_retriever`, `Retriever` |
| `evaluate.py` | Rank of the first relevant chunk per eval case, for hybrid, vector and BM25. | `main` |

### `qa/`

| Module | Purpose | Main entry points |
|---|---|---|
| `llm.py` | Provider classes with one method, `generate_json(system, prompt, schema)`. Clear setup errors, quota errors, Ollama checks (server running, model pulled, thinking fallback, truncation). | `get_llm`, `LLMError`, `LLMQuotaError` |
| `answer.py` | Prompt, LLM call, citation verification, statuses. | `answer_from_chunks`, `answer`, `check_citations` |
| `chat.py` | Follow-up rewriting and a full chat turn. | `chat_turn`, `rewrite_question` |
| `ask.py` | CLI for one question (`--session`, `--provider`, `--model`, `--show-sources`). | `main` |
| `evaluate.py` | QA eval. It stops at the first quota error and prints a command to resume. | `main`, `judge` |

---

## 5. Design rules that hold everywhere

1. **Nothing is lost silently.** Unreadable pages flag `NEEDS_REVIEW`.
   Chunking proves full coverage or fails. A file that changes during
   processing is never stored.
2. **Every answer is checkable.** The LLM must quote, and the code
   verifies each quote against the cited chunk before it is shown.
3. **One shape for every source.** Text layer, EasyOCR and Docling all
   produce the same blocks and tables, so chunking, search and
   citations don't care how a page was read.
4. **Only the newest version is searched.** Older versions stay for
   history.
5. **Scopes never mix.** A search covers the library or one session,
   never both.
6. **Deletion is total.** Removing a document removes every derived
   artefact, for every version.
7. **The expensive work is cached.** Embeddings are cached per chunk
   text, models load once per process, and search indexes are kept per
   scope.

---

## 6. Configuration

All settings live in `src/docintel/config.py`. These can be set from the
environment:

| Variable | Default | Meaning |
|---|---|---|
| `DOCINTEL_HOME` | the project folder | root for `documents/`, `archive/`, `data/`, `eval/` |
| `DOCINTEL_LLM_PROVIDER` | `ollama` | `ollama`, `gemini`, `openai` or `claude` |
| `DOCINTEL_LLM_MODEL` | provider default | a model of the configured provider |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `DOCINTEL_OLLAMA_NUM_CTX` / `_TIMEOUT` / `_THINK` | `4096` / `300` / `true` | Ollama context window, timeout in seconds, thinking |
| `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | none | cloud provider credentials |
| `DOCINTEL_OCR_ENGINE` | `easyocr` | `easyocr`, `docling` or `none` |
| `DOCINTEL_OCR_LANGUAGES` | `en` | comma-separated OCR languages |
| `DOCINTEL_OCR_DPI` | `200` | render resolution for OCR |
| `DOCINTEL_OCR_GPU` | `0` | EasyOCR on CUDA or Apple MPS |
| `DOCINTEL_SESSION_IDLE_HOURS` | `24` | idle time before a chat's files are deleted |

Fixed in code: `CHUNK_MAX_TOKENS` 200, `TABLE_MAX_ROWS_PER_CHUNK` 25,
`QA_TOP_K` 5, `EMBEDDING_MODEL` all-MiniLM-L6-v2,
`OCR_MIN_LINE_CONFIDENCE` 0.3, `OCR_MIN_PAGE_CONFIDENCE` 0.5,
`OCR_MIN_REGION_FRACTION` 0.10, `WATCHER_QUIET_SECONDS` 3.

---

## 7. Tests and evaluation

- **Unit tests** (`pytest`, about 3 s). These need no network, LLM or
  embedding model. Expensive parts are replaced with stand-ins: a
  `FakeLLM` returns prepared JSON, fake encoders return fixed vectors,
  and a fake OCR engine or recorded real EasyOCR output stands in for
  OCR. Library, session and parser tests build real PDFs in a temporary
  project folder.

  | File | Covers |
  |---|---|
  | `test_chunking.py` | chunkers, coverage validation |
  | `test_retrieval.py` | BM25, fusion, vector cache |
  | `test_qa.py` | citation checks, providers |
  | `test_chat.py` | follow-up rewriting |
  | `test_library.py` | archive, delete, sessions |
  | `test_page_analysis.py` | page kinds, OCR routing |
  | `test_ocr.py` | OCR layout, engine wiring |
  | `test_app.py` | app smoke test |

- **Evaluations with real models**, all over
  `eval/retrieval_cases.json`:
  - `embeddings.check_embeddings`: vector-only retrieval
  - `retrieval.evaluate`: hybrid vs. vector vs. BM25
  - `qa.evaluate`: the full answer step, one LLM call per case. On a
    laptop, run only a case or two with `--only`.

---

## 8. Known limitations

- **One test document.** The evals use one 2-page synthetic PDF, so
  their scores say little about real documents yet.
- **OCR speed.** About 20–35 s per scanned page on an M2. In the app the
  upload waits for processing, with no background queue.
- **OCR coverage.** Scans are not deskewed or rotated. Handwriting reads
  poorly. EasyOCR's table rebuilding needs column-aligned cells and a
  text header row. Charts are kept only as their labels' text: axis
  numbers add some noise, and the plotted values are not read.
- **Strict review gate.** One unreadable page puts the whole document
  on `NEEDS_REVIEW`.
- **Speed of local answers.** With Ollama `qwen3:4b` on 8 GB, answers
  take about 20–60 s, and follow-ups take longer because the rewrite is
  a second call.
- **Layout assumptions.** Reading order is top-to-bottom, then
  left-to-right per block. Complex multi-column layouts may interleave.

---

## 9. Lessons learned

What running the pipeline taught us, and why the code is shaped the way
it is. The measurements come from the 2-page test PDF, the user's
3-page test document (charts plus a scanned appendix), and an 8 GB M2
MacBook Air.

### Retrieval

- **Embeddings alone miss exact-term questions.** "Who prepared this
  dataset?" ranked its answer 11th of 12 with vectors only. The answer
  sits on a "Source: … prepared …" line that shares words, not meaning,
  with the question. BM25 ranked it 1st. That is why search is hybrid:

  | Mode | hit@1 | hit@3 | MRR |
  |---|---|---|---|
  | vector only | 12/14 | 13/14 | 0.899 |
  | BM25 only | 9/14 | 14/14 | 0.810 |
  | **hybrid (RRF)** | 12/14 | **14/14** | **0.929** |

  The cost: one case (`text_trend`) slipped from rank 1 to 2. It was
  not tuned away on a 12-chunk corpus.
- **Similarity scores cannot detect unanswerable questions.** The best
  score for an unanswerable question (0.539) beat the lowest score of a
  correct answer (0.380). "Not in the documents" is therefore decided by
  the LLM and the citation check, never by a score threshold.

### Answering and citations

- **A small local model needs to think before structured output.** With
  thinking off, `qwen3:4b` returned `answerable: false` in about 1 s,
  even with the answer in the sources. Putting the answer before
  `answerable` in the JSON fields did not help. With thinking on it
  answered correctly (16/16 on the QA eval) in about 15–50 s per
  question. Hence `DOCINTEL_OLLAMA_THINK=true` by default.
- **Quote checks must respect token boundaries.** A plain substring
  check accepted "Participants: 8" as a quote of "Participants: 85". The
  check now requires whole words and numbers.
- **Models shorten table rows when they quote them.** `qwen3:4b` quoted
  "Category: Capital | 2025: 1.8" from a row that also has a 2024 cell.
  That was a correct answer rejected by an exact-match check. Table
  quotes may now skip cells, as long as every quoted cell belongs to one
  row.
- **Follow-up rewriting doubles the latency.** The rewrite is a second
  thinking call: a follow-up took 163 s against 52 s for a first
  question. It can be switched off in the app. Running the rewrite
  without thinking is the open idea for making it faster.
- **Free cloud tiers fail mid-run.** Gemini's free tier returned
  503 (overloaded) and 429 (quota) during the QA eval. Quotas are per
  model and reset at midnight Pacific. The SDKs retry first. A
  persistent 429 becomes `LLMQuotaError`, which stops `qa.evaluate`
  and prints a resume command.

### OCR and layout

- **OCR text needs layout reconstruction to keep table meaning.**
  EasyOCR returns loose segments, so a scanned table first came out as
  separate words ("85" with no column). `layout_lines` rebuilds tables
  from column alignment, so scanned tables chunk like typed ones.
- **OCR blocks must look like text-layer blocks.** The chunkers find
  table titles and notes by how a block *starts*. When OCR grouping
  glued "Table 1. …" onto the paragraph above, titles fell back to
  "Table". When OCR tables had no `INSIDE_TABLE` blocks, a table's
  notes picked up the next table's footnote. The fixes:
  - headings, titles and notes always start their own block
  - the note search stops at the next table's top edge, which also
    hardens typed PDFs
- **Charts are not tables.** A chart's axis labels formed a fake 2×2
  table. A label overlapping it then failed the whole document. Two
  measurements on real regions decided the fix:

  | Region | text-area density | chars per segment |
  |---|---|---|
  | chart, page 1 | 0.071 | 3.8 |
  | chart, page 2 | 0.111 | 4.4 |
  | scanned appendix | 0.131 | 17.6 |
  | scanned table pages | 0.120–0.142 | 24.5–25.3 |

  Density does not separate charts from scans, but characters per
  segment does. Hence the figure rule (fewer than 8 characters on
  average and no table), plus header-like first rows for tables, plus
  OCR blocks never failing on overlap.
- **EasyOCR is the default, not Docling.** On the scanned table page,
  EasyOCR with our table reconstruction got all 20 cells right. Docling
  lost one cell. Docling's punctuation was cleaner, and its models are
  heavier.
- **Known EasyOCR misreads.** "$" is read as "S", which is fixed before
  amounts only. "." is often read as ":" or ";", which is harmless for
  search but visible in quotes. Expect about 20–35 s per page on an M2
  CPU.

### Operating notes (development environment)

- **Keep the virtualenv outside iCloud.** The repository lives in
  iCloud-synced `~/Documents`. A venv there had the macOS `hidden` flag
  re-applied to its files, and Python 3.12 skips hidden `.pth` files,
  so `import docintel` broke minutes after a working install. The venv
  lives at `~/.venvs/docintel`. iCloud also creates conflict copies like
  `llm 2.py` and `test_qa 2.py`, which pytest then collects. Compare such
  copies with git before deleting them.
- **Always run modules, not files.** Use `python -m docintel.…`; running
  `src/docintel/…/watcher.py` as a script fails on relative imports. In
  PyCharm, set run configurations to *Module name*, with the
  `~/.venvs/docintel` interpreter registered and the working directory
  `docintel/`.
- **Start Streamlit through the venv's Python.** After a package
  install, zsh may still run a previously found `streamlit` (Anaconda's)
  from its command cache. `python -m streamlit run app.py` always uses
  the active venv.
- **The Streamlit file watcher is off on purpose.** Scanning
  `transformers` makes it import vision modules that need `torchvision`,
  printing dozens of harmless tracebacks. `.streamlit/config.toml` sets
  `fileWatcherType = "none"`, so restart the app after code changes.
- **Chat UI layout.** `st.chat_input` inside `st.tabs` renders inline,
  not pinned to the bottom. Chat and Library are therefore separate
  pages (`st.navigation`). Streamlit's `AppTest` cannot switch to
  function-based pages, so the app smoke test covers the Chat page only.
- **"REMOVED (file missing)" at watcher start is expected** when a
  registered library file was moved or deleted while the watcher was
  off. Startup cleanup deletes that document's data. It refuses to run
  if `archive/` is missing altogether.
- **Respect the laptop.** Live checks with the local model use 1–2 eval
  cases (`qa.evaluate --only …`). A full 16-case run took about 13
  minutes at high memory pressure. Unit tests cover the rest without
  models.

