"""
config.py — Central Configuration for IPO Prospectus RAG System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with strict, predictable values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ===========================================================
# Paths
# ===========================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Indexes (ChromaDB + BM25)
INDEXES_DIR         = DATA_DIR / "indexes"
CHROMA_DIR          = INDEXES_DIR / "chroma_db"
BM25_INDEX_PATH     = INDEXES_DIR / "bm25_index.pkl"

# Registry
CACHE_REGISTRY_PATH = DATA_DIR / "registry" / "index_registry.json"

# Raw prospectus PDFs
PROSPEKTUS_DIR      = DATA_DIR / "raw" / "prospectuses"
PARSED_CACHE_DIR    = DATA_DIR / "parsed"

# Ensure directories exist
INDEXES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "registry").mkdir(parents=True, exist_ok=True)
PROSPEKTUS_DIR.mkdir(parents=True, exist_ok=True)
PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================
# API Keys
# ===========================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# Parser modes:
# - strict_llamaparse: research mode; stop if LlamaParse fails or credits run out
# - llamaparse_fallback: operational mode; fall back to pdfplumber
# - pdfplumber: fully local and reproducible baseline
PARSER_MODE = os.getenv("PARSER_MODE", "strict_llamaparse").strip().lower()
PARSER_CACHE_VERSION = "v3"

# ===========================================================
# Text Preprocessing (conservative; financial notation is preserved)
# ===========================================================
PREPROCESSING_VERSION = "v1"
PREPROCESSING_ENABLED = _env_bool("PREPROCESSING_ENABLED", True)
PREPROCESS_REMOVE_REPEATED_EDGES = _env_bool(
    "PREPROCESS_REMOVE_REPEATED_EDGES", True
)
PREPROCESS_EDGE_LINES = max(1, int(os.getenv("PREPROCESS_EDGE_LINES", "3")))
PREPROCESS_REPEAT_RATIO = min(
    1.0, max(0.0, float(os.getenv("PREPROCESS_REPEAT_RATIO", "0.65")))
)
PREPROCESS_MIN_REPEAT_PAGES = max(
    2, int(os.getenv("PREPROCESS_MIN_REPEAT_PAGES", "3"))
)
PREPROCESS_MAX_BLANK_LINES = max(
    1, int(os.getenv("PREPROCESS_MAX_BLANK_LINES", "2"))
)
INDEX_SCHEMA_VERSION = "v4-cosine-bgem3-no-prefix-bm25-per-doc"

# ===========================================================
# Embedding Model (BAAI/bge-m3 via OpenRouter or Local)
# ===========================================================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openrouter").strip().lower()
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL", "baai/bge-m3"
).strip()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
# A conservative character cap keeps unusually large atomic Markdown tables
# below BGE-M3's 8,192-token context window. Stored chunk content remains full.
EMBEDDING_MAX_INPUT_CHARS = int(os.getenv("EMBEDDING_MAX_INPUT_CHARS", "8000"))

# ===========================================================
# Chunking Settings
# ===========================================================
NARRATIVE_CHUNK_SIZE = 800         # chars per narrative chunk
NARRATIVE_CHUNK_OVERLAP = 150      # overlap between narrative chunks
MIN_CHUNK_LENGTH = 50              # discard chunks shorter than this

# ===========================================================
# Retrieval Settings
# ===========================================================
DENSE_TOP_K = 10                   # candidates from ChromaDB
SPARSE_TOP_K = 10                  # candidates from BM25
FINAL_TOP_K = 6                    # final chunks sent to LLM (after RRF)
RRF_K = 60                         # RRF constant (higher = smoother merge)
# With k=60, a single retriever contributes at most 1/61 = 0.01639.
# The proposal uses 0.02 on the best fused score, requiring support from both lists.
MIN_RRF_SCORE = float(os.getenv("MIN_RRF_SCORE", "0.02"))

# ===========================================================
# ChromaDB
# ===========================================================
CHROMA_COLLECTION_NAME = "ipo_prospectus"
CHROMA_DISTANCE_SPACE = "cosine"

# BM25 is built independently for every prospectus so corpus statistics
# (N, document frequency, and avgdl) always belong to the active issuer.
BM25_INDEX_SCHEMA_VERSION = "per-document-v1"

# ===========================================================
# LLM (OpenRouter - Qwen)
# ===========================================================
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip().rstrip("/")
OPENROUTER_MODEL_NAME = os.getenv(
    "OPENROUTER_MODEL_NAME", "qwen/qwen3.7-flash"
).strip()
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "2048"))
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "60"))
OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))

# ===========================================================
# RAGAS Evaluation
# ===========================================================
# Separate settings keep the answer generator and LLM-as-a-judge reproducible.
RAGAS_EVALUATOR_MODEL = os.getenv(
    "RAGAS_EVALUATOR_MODEL", OPENROUTER_MODEL_NAME
).strip()
RAGAS_EVALUATOR_TEMPERATURE = 0.0
RAGAS_EVALUATOR_MAX_TOKENS = int(os.getenv("RAGAS_EVALUATOR_MAX_TOKENS", "2048"))

# ===========================================================
# Streamlit UI
# ===========================================================
APP_TITLE = "Sistem Tanya Jawab Prospektus IPO"
APP_SUBTITLE = f"Berbasis RAG + Hybrid Search (BAAI/bge-m3 + BM25 + {OPENROUTER_MODEL_NAME})"
