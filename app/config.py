import os

KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", "knowledge/")
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.1"))
SCORE_SCALE: float = float(os.getenv("SCORE_SCALE", "25.0"))
MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "3"))
WATCHER_DEBOUNCE_MS: int = int(os.getenv("WATCHER_DEBOUNCE_MS", "500"))
MAX_SYNONYMS_PER_TOKEN: int = int(os.getenv("MAX_SYNONYMS_PER_TOKEN", "2"))
HEADING_BOOST_FACTOR: float = float(os.getenv("HEADING_BOOST_FACTOR", "2.0"))
HEADING_OVERLAP_THRESHOLD: float = float(os.getenv("HEADING_OVERLAP_THRESHOLD", "0.5"))
FILENAME_BOOST_FACTOR: float = float(os.getenv("FILENAME_BOOST_FACTOR", "1.4"))
REFERENCE_PATH_BOOST: float = float(os.getenv("REFERENCE_PATH_BOOST", "1.2"))
# Fuzzy correction — used only when a query token is absent from the indexed
# vocabulary. Cap the displayed confidence when ANY correction fired so users
# see a lower number for typo-only matches instead of a confident wrong answer.
FUZZY_CORRECTION_ENABLED: bool = os.getenv("FUZZY_CORRECTION_ENABLED", "true").lower() == "true"
FUZZY_MAX_CANDIDATES: int = int(os.getenv("FUZZY_MAX_CANDIDATES", "2"))
FUZZY_CONFIDENCE_CAP: float = float(os.getenv("FUZZY_CONFIDENCE_CAP", "0.6"))
