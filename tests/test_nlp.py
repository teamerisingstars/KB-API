from app.nlp import penn_to_wordnet, lemmatize_text, lemmatize_text_with_pos, expand_synonyms
from nltk.corpus import wordnet


def test_penn_to_wordnet_noun():
    assert penn_to_wordnet("NN") == wordnet.NOUN
    assert penn_to_wordnet("NNS") == wordnet.NOUN
    assert penn_to_wordnet("NNP") == wordnet.NOUN


def test_penn_to_wordnet_verb():
    assert penn_to_wordnet("VB") == wordnet.VERB
    assert penn_to_wordnet("VBZ") == wordnet.VERB


def test_penn_to_wordnet_adj():
    assert penn_to_wordnet("JJ") == wordnet.ADJ


def test_penn_to_wordnet_adv():
    assert penn_to_wordnet("RB") == wordnet.ADV


def test_penn_to_wordnet_unknown_defaults_to_noun():
    assert penn_to_wordnet("XX") == wordnet.NOUN


def test_lemmatize_text_removes_stopwords():
    result = lemmatize_text("the is a and")
    assert result == []


def test_lemmatize_text_lemmatizes_verbs():
    result = lemmatize_text("running cars")
    assert "run" in result
    assert "car" in result


def test_lemmatize_text_lowercases():
    result = lemmatize_text("Python")
    assert all(t == t.lower() for t in result)


def test_lemmatize_text_with_pos_returns_tuples():
    result = lemmatize_text_with_pos("fast cars")
    assert isinstance(result, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in result)


def test_lemmatize_text_with_pos_includes_tag():
    result = lemmatize_text_with_pos("cars")
    lemmas = [lemma for lemma, _ in result]
    assert "car" in lemmas


def test_expand_synonyms_includes_original():
    tokens = [("car", "NN")]
    result = expand_synonyms(tokens)
    assert "car" in result


def test_expand_synonyms_skips_proper_nouns():
    tokens = [("Python", "NNP")]
    result = expand_synonyms(tokens)
    # "Python" should not expand to snake-related terms
    snake_terms = {"serpent", "ophidian", "snake"}
    assert not snake_terms.intersection(set(result))


def test_expand_synonyms_deduplicates():
    tokens = [("car", "NN"), ("car", "NN")]
    result = expand_synonyms(tokens)
    assert len(result) == len(set(result))


def test_expand_synonyms_max_synonyms_respected():
    tokens = [("vehicle", "NN")]
    result = expand_synonyms(tokens, max_synonyms=2)
    # Original + max 2 synonyms = at most 3 tokens from this word
    assert len(result) <= 3
