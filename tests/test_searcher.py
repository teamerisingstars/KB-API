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
        assert response.confidence <= 1.0
        assert response.confidence > 0.0


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
