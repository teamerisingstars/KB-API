from app.config import (
    KNOWLEDGE_DIR,
    CONFIDENCE_THRESHOLD,
    SCORE_SCALE,
    MAX_RESULTS,
    WATCHER_DEBOUNCE_MS,
    MAX_SYNONYMS_PER_TOKEN,
)

def test_config_defaults():
    assert KNOWLEDGE_DIR == "knowledge/"
    assert CONFIDENCE_THRESHOLD == 1.0
    assert SCORE_SCALE == 10.0
    assert MAX_RESULTS == 3
    assert WATCHER_DEBOUNCE_MS == 500
    assert MAX_SYNONYMS_PER_TOKEN == 2
