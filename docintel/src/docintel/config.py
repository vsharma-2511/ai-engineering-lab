"""Central settings. Every path is resolved from the project root, so
commands work no matter which directory you run them from.

Set DOCINTEL_HOME to point the pipeline at a different project folder.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("DOCINTEL_HOME", Path(__file__).resolve().parents[2])
)

# Library inbox: drop PDFs here. Once processed (CHUNKED) they move to
# ARCHIVE_DIR, which keeps the originals for reprocessing.
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "registry.db"
PARSED_DIR = DATA_DIR / "parsed"
# Files uploaded into chat sessions; deleted when the session ends.
UPLOADS_DIR = DATA_DIR / "uploads"
EVAL_DIR = PROJECT_ROOT / "eval"

SUPPORTED_EXTENSIONS = {".pdf"}
WATCHER_QUIET_SECONDS = 3

# Embedding model. Chunk sizes are measured with this model's tokenizer.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 accepts 256 tokens (it was trained on ~128).
# 200 leaves headroom and keeps chunks focused.
CHUNK_MAX_TOKENS = 200

# Upper bound on rows in one table chunk; the token budget usually
# splits tables earlier.
TABLE_MAX_ROWS_PER_CHUNK = 25

# Question answering. Choose the LLM with environment variables:
#   DOCINTEL_LLM_PROVIDER = ollama | gemini | openai | claude
#   DOCINTEL_LLM_MODEL    = a model of that provider (optional)
# Ollama runs locally and needs no key. The cloud providers read
# GEMINI_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY.
QA_PROVIDER = os.environ.get("DOCINTEL_LLM_PROVIDER", "ollama").lower()
QA_MODEL = os.environ.get("DOCINTEL_LLM_MODEL") or None
QA_DEFAULT_MODELS = {
    # Fits an 8 GB Mac (2.5 GB download) and handles JSON output well.
    "ollama": "qwen3:4b",
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-5.4-mini",
    "claude": "claude-opus-5-5",
}
QA_CLAUDE_EFFORT = "low"

# Ollama (local models).
QA_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if "://" not in QA_OLLAMA_HOST:  # OLLAMA_HOST may be just host:port
    QA_OLLAMA_HOST = "http://" + QA_OLLAMA_HOST
# Context window in tokens: room for 5 chunks, instructions and answer.
QA_OLLAMA_NUM_CTX = int(os.environ.get("DOCINTEL_OLLAMA_NUM_CTX", "4096"))
# Seconds; the first call also loads the model into memory.
QA_OLLAMA_TIMEOUT = float(os.environ.get("DOCINTEL_OLLAMA_TIMEOUT", "300"))
# Let thinking models (qwen3, deepseek-r1, ...) reason before answering.
# Without it qwen3:4b answered "not found" even with the answer in the
# sources; with it, ~15 s per question on an 8 GB M2 instead of ~1 s.
# Models without a thinking mode ignore this. DOCINTEL_OLLAMA_THINK=false
# turns it off.
QA_OLLAMA_THINK = (
    os.environ.get("DOCINTEL_OLLAMA_THINK", "true").lower()
    in {"1", "true", "yes"}
)

# Chunks sent to the LLM per question.
QA_TOP_K = 5

# Chat sessions: documents uploaded into a session are private to it and
# deleted with everything derived from them when the session ends, or
# after this many hours without activity.
SESSION_IDLE_HOURS = float(os.environ.get("DOCINTEL_SESSION_IDLE_HOURS", "24"))

# OCR for scanned pages. Each page is classified first (see
# parsing/page_analysis.py); only scanned pages and image regions on
# mixed pages are sent to OCR.
#   DOCINTEL_OCR_ENGINE = easyocr | docling | none
# easyocr reads text lines (lighter). docling also recovers table
# structure from scans (heavier; downloads layout and table models on
# first use). none leaves scanned pages unread (flagged NEEDS_REVIEW).
OCR_ENGINE = os.environ.get("DOCINTEL_OCR_ENGINE", "easyocr").lower()
OCR_LANGUAGES = os.environ.get("DOCINTEL_OCR_LANGUAGES", "en").split(",")
# EasyOCR on the GPU (CUDA, or Apple's MPS): faster, but on an 8 GB Mac
# the GPU shares memory with the embedding model and Ollama.
OCR_GPU = os.environ.get("DOCINTEL_OCR_GPU", "0").lower() in {"1", "true", "yes"}
# Render resolution for OCR. 200 dpi reads normal print well and keeps
# an A4 page around 1650 x 2340 pixels (memory-friendly on 8 GB).
OCR_DPI = int(os.environ.get("DOCINTEL_OCR_DPI", "200"))
# OCR lines below this confidence are dropped as likely noise.
OCR_MIN_LINE_CONFIDENCE = 0.3
# A scanned page whose kept lines average below this is flagged for
# review instead of being indexed silently.
OCR_MIN_PAGE_CONFIDENCE = 0.5
# Images covering at least this share of a page that has text elsewhere
# are OCR'd as regions (screenshots, pasted scans, photographed tables).
OCR_MIN_REGION_FRACTION = 0.10
