from app.fuzzy import BKTree, _levenshtein, build_bk_tree, fuzzy_candidates


def test_levenshtein_basic():
    assert _levenshtein("cors", "cors") == 0
    assert _levenshtein("cors", "corss") == 1
    assert _levenshtein("cors", "corts") == 1
    assert _levenshtein("apple", "orange") == 5
    assert _levenshtein("", "") == 0


def test_bktree_add_and_contains():
    t = BKTree()
    t.add("cors")
    t.add("cross")
    t.add("origin")
    assert "cors" in t
    assert "origin" in t
    assert "nope" not in t


def test_bktree_search_exact():
    t = BKTree()
    for w in ["cors", "cross", "origin"]:
        t.add(w)
    results = t.search("cors", 0)
    assert results == [("cors", 0)]


def test_bktree_search_within_distance():
    t = BKTree()
    for w in ["cors", "cross", "origin", "apirouter"]:
        t.add(w)
    results = t.search("corss", 1)
    found = {w for w, _ in results}
    assert "cors" in found
    assert "apirouter" not in found


def test_build_bk_tree_skips_short_words():
    tree = build_bk_tree([["a", "ab", "cors", "origin"]])
    assert "cors" in tree
    assert "origin" in tree
    assert "a" not in tree
    assert "ab" not in tree


def test_fuzzy_candidates_corrects_short_typo():
    tree = build_bk_tree([["cors", "cross", "origin", "response_model"]])
    cands = fuzzy_candidates(tree, "corss")
    # "cross" is also within distance 1, both are acceptable corrections
    assert "cors" in cands


def test_fuzzy_candidates_corrects_long_typo():
    tree = build_bk_tree([["oauth2passwordbearer", "apirouter", "response_model"]])
    suggestions = fuzzy_candidates(tree, "oauth2passwrdbearer")
    assert "oauth2passwordbearer" in suggestions


def test_fuzzy_candidates_skips_exact_match():
    tree = build_bk_tree([["cors", "cross"]])
    # exact match should not return itself as a "correction"
    assert fuzzy_candidates(tree, "cors") == []


def test_fuzzy_candidates_ignores_very_short_tokens():
    tree = build_bk_tree([["cors", "cross"]])
    assert fuzzy_candidates(tree, "ab") == []


def test_fuzzy_candidates_returns_nothing_for_unrelated_word():
    tree = build_bk_tree([["cors", "origin", "background"]])
    assert fuzzy_candidates(tree, "xylophone") == []
