from pathlib import Path

# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# DATA
# ==========================================================

DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"

# Qdrant local storage
QDRANT_PATH = DATA_DIR / "qdrant_storage"


# ==========================================================
# MODELS (OFFLINE MODELS)
# ==========================================================

MODELS_DIR = BASE_DIR / "models"

# Embeddings model
EMBED_MODEL_PATH = MODELS_DIR / "bge-m3"

# Reranker (если понадобится cross-encoder)
RERANKER_MODEL_PATH = MODELS_DIR / "models--cross-encoder--ms-marco-MiniLM-L-6-v2"

# Альтернативная папка reranker
RERANKER_ALT_PATH = MODELS_DIR / "reranker"


# ==========================================================
# DATABASE
# ==========================================================

DB_PATH = BASE_DIR / "ai.db"


# ==========================================================
# LOGGING
# ==========================================================

LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "app.log"


# ==========================================================
# LLM SERVER
# ==========================================================

# llama.cpp / OpenAI-compatible endpoint
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

MAX_TOKENS = 3000
TEMPERATURE = 0.2

# Максимальное количество одновременных LLM-запросов
LLM_CONCURRENCY_LIMIT = 2


# ==========================================================
# QDRANT
# ==========================================================

COLLECTION_NAME = "documents"

# Батч загрузки при индексации
QDRANT_BATCH_SIZE = 256


# ==========================================================
# RETRIEVAL SETTINGS
# ==========================================================

# Базовый top-k (может динамически изменяться)
TOP_K = 3
BASE_THRESHOLD = 0.4
# Минимальный score для допуска к генерации
MIN_SCORE_THRESHOLD = 0.25

# Соотношение score относительно лучшего документа
MIN_SCORE_RATIO = 0.8

# Максимальный размер контекста (символы)
MAX_CONTEXT_CHARS = 8000


# ==========================================================
# HYBRID RERANKING
# ==========================================================

HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_KEYWORD_WEIGHT = 0.3


# ==========================================================
# SEMANTIC TOPIC DETECTION
# ==========================================================

SIMILARITY_STRONG = 0.75
SIMILARITY_WEAK = 0.55


# ==========================================================
# FUZZY MATCHING
# ==========================================================

FUZZY_MATCH_CUTOFF = 0.75


# ==========================================================
# RETRIEVAL SELF-CHECK
# ==========================================================

BASE_RETRIEVAL_THRESHOLD = 0.20

ADAPTIVE_THRESHOLD_MIN = 0.2
ADAPTIVE_THRESHOLD_MAX = 0.6
# ==========================================================
# RATE LIMIT
# ==========================================================

RATE_LIMIT = 5
RATE_WINDOW = 60  # seconds


# ==========================================================
# METRICS / CLEANUP
# ==========================================================

# Через сколько дней удалять старые данные
DATA_RETENTION_DAYS = 7