import os

KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", "knowledge/")
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.1"))
SCORE_SCALE: float = float(os.getenv("SCORE_SCALE", "10.0"))
MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "3"))
WATCHER_DEBOUNCE_MS: int = int(os.getenv("WATCHER_DEBOUNCE_MS", "500"))
MAX_SYNONYMS_PER_TOKEN: int = int(os.getenv("MAX_SYNONYMS_PER_TOKEN", "2"))
