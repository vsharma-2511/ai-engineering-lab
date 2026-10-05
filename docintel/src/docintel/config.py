"""Central settings. Every path is resolved from the project root, so
commands work no matter which directory you run them from.

Set DOCINTEL_HOME to point the pipeline at a different project folder.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("DOCINTEL_HOME", Path(__file__).resolve().parents[2])
)

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "registry.db"
PARSED_DIR = DATA_DIR / "parsed"
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
