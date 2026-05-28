import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))


def penn_to_wordnet(tag: str) -> str:
    if tag.startswith("NN"):
        return wordnet.NOUN
    if tag.startswith("VB"):
        return wordnet.VERB
    if tag.startswith("JJ"):
        return wordnet.ADJ
    if tag.startswith("RB"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_text(text: str) -> list[str]:
    """Tokenize, POS-tag, lemmatize, remove stopwords. Used for indexing."""
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    result = []
    for word, tag in tagged:
        if word.isalpha() and word not in _STOPWORDS:
            lemma = _LEMMATIZER.lemmatize(word, pos=penn_to_wordnet(tag))
            result.append(lemma)
    return result


def lemmatize_text_with_pos(text: str) -> list[tuple[str, str]]:
    """Tokenize, POS-tag, lemmatize, remove stopwords. Returns (lemma, penn_tag) tuples.
    Used for query-side synonym expansion."""
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    result = []
    for word, tag in tagged:
        if word.isalpha() and word not in _STOPWORDS:
            lemma = _LEMMATIZER.lemmatize(word, pos=penn_to_wordnet(tag))
            result.append((lemma, tag))
    return result


def expand_synonyms(tokens_with_pos: list[tuple[str, str]], max_synonyms: int = 2) -> list[str]:
    """Expand noun/verb tokens with WordNet synonyms (query side only).
    Skips proper nouns to avoid polysemy traps (Python->snake, Apple->fruit)."""
    result = []
    for lemma, tag in tokens_with_pos:
        result.append(lemma)

        if tag in ("NNP", "NNPS"):
            continue

        if tag.startswith("NN"):
            wn_pos = wordnet.NOUN
        elif tag.startswith("VB"):
            wn_pos = wordnet.VERB
        else:
            continue

        synsets = wordnet.synsets(lemma, pos=wn_pos)
        if not synsets:
            continue

        count = 0
        for syn_lemma in synsets[0].lemmas():
            synonym = syn_lemma.name().replace("_", " ").lower()
            if synonym != lemma and count < max_synonyms:
                result.append(synonym)
                count += 1

    return list(dict.fromkeys(result))
