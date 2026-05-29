import os
import tempfile
import pytest
from app.indexer import IndexStore, build_index
from app.searcher import search


@pytest.fixture()
def loaded_store():
    """IndexStore pre-loaded with knowledge/sample.md content."""
    store = IndexStore()
    build_index("knowledge", store)
    return store


@pytest.fixture()
def mini_store():
    """IndexStore with a single known section for deterministic assertions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "facts.md"), "w") as f:
            f.write(
                "## What is a Car\n\n"
                "A car is a wheeled motor vehicle used for transportation. "
                "Cars run on petrol, diesel, or electricity.\n\n"
                "## What is Python\n\n"
                "Python is a high-level programming language created by Guido van Rossum.\n"
            )
        store = IndexStore()
        build_index(tmpdir, store)
        return store


def test_search_finds_relevant_section(mini_store):
    response = search("what is a car", mini_store)
    assert response.answer is not None
    assert "car" in response.answer.lower() or "vehicle" in response.answer.lower()


def test_search_no_match_returns_null_answer(mini_store):
    response = search("quantum entanglement in neutrino physics", mini_store)
    assert response.answer is None
    assert response.message is not None
    assert response.confidence == 0.0


def test_search_confidence_is_absolute_not_forced_to_one(mini_store):
    response = search("what is a car", mini_store)
    if response.answer is not None:
        assert response.confidence > 0.0
        # Absolute scoring: raw_score / SCORE_SCALE — top result must NOT be forced to 1.0
        assert response.confidence < 1.0


def test_search_returns_alternatives(mini_store):
    response = search("programming language", mini_store)
    if response.answer is not None:
        assert isinstance(response.alternatives, list)


def test_search_alternatives_also_above_threshold(mini_store):
    response = search("what is a car", mini_store)
    for alt in response.alternatives:
        assert alt.confidence > 0.0


def test_search_empty_index_returns_no_match():
    empty_store = IndexStore()
    response = search("anything", empty_store)
    assert response.answer is None
    assert response.confidence == 0.0


def test_search_empty_question_after_stopword_removal(mini_store):
    # "the a is" are all stopwords; after removal query is empty
    response = search("the a is", mini_store)
    assert response.answer is None


def test_search_synonym_expansion_finds_automobile(mini_store):
    # "automobile" is a synonym of "car"; should find the car section
    response = search("what is an automobile", mini_store)
    assert response.answer is not None
    assert "car" in response.answer.lower() or "vehicle" in response.answer.lower()


@pytest.fixture()
def python_sections_store():
    """Store with two Python-related sections to exercise heading boost.

    Without the boost, BM25 on "what is python" scores both sections similarly
    because "python" appears in both bodies and IDF collapses on a 2-doc corpus.
    The boost lifts "What is Python" (100% heading overlap) over "Python Data Types"
    (≈33% heading overlap after lemmatization: {'python'} ∩ {'python','datum','type'}).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "python.md"), "w") as f:
            f.write(
                "## What is Python\n\n"
                "Python is a high-level programming language with clear syntax.\n\n"
                "## Python Data Types\n\n"
                "Python has integers, floats, strings, lists, tuples, and dictionaries.\n"
            )
        store = IndexStore()
        build_index(tmpdir, store)
        return store


def test_heading_boost_prefers_exact_heading_match(python_sections_store):
    # "What is Python" heading lemmatizes to ["python"] → 1/1 = 100% overlap → boosted
    # "Python Data Types" heading lemmatizes to ["python","datum","type"] → 1/3 ≈ 33% → not boosted
    response = search("what is python", python_sections_store)
    assert response.answer is not None
    assert response.section == "What is Python"


def test_exact_heading_match_bonus_beats_partial_heading(loaded_store):
    # knowledge/sample.md has "What is Python" (heading lemmas: {python})
    # and "History of Python" (heading lemmas: {history, python}).
    # query lemmas = {python}; exact match bonus (+1.0) applied only to "What is Python".
    # "History of Python" only gets the multiplicative boost (1/2 heading overlap < threshold).
    response = search("what is python", loaded_store)
    assert response.answer is not None
    assert response.section == "What is Python"
