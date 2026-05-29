import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))

# Split at CamelCase boundaries: aA, a1A, AAa
_CAMEL_BOUNDARY = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')

# Query-side acronym expansions. NEVER applied at index time (would inflate IDF).
_ACRONYMS = {
    "cors": "cross origin resource sharing",
    "jwt": "json web token",
    "api": "application programming interface",
    "url": "uniform resource locator",
    "sse": "server sent events",
    "csrf": "cross site request forgery",
    "xss": "cross site scripting",
    "orm": "object relational mapping",
    "rest": "representational state transfer",
    "tls": "transport layer security",
    "ssl": "secure sockets layer",
    "html": "hypertext markup language",
    "json": "javascript object notation",
    "yaml": "yet another markup language",
    "sql": "structured query language",
}


def split_identifier(token: str) -> list[str]:
    """Split a programming identifier into its sub-parts.

    response_model       -> [response_model, response, model]
    OAuth2PasswordBearer -> [OAuth2PasswordBearer, OAuth2, Password, Bearer]
    Cross-Origin         -> [Cross-Origin, Cross, Origin]
    CORS                 -> [CORS]
    plain                -> [plain]
    """
    if not token:
        return []
    parts = [token]
    if "_" in token or "-" in token:
        for p in re.split(r"[_-]", token):
            if not p:
                continue
            parts.append(p)
            if any(c.isupper() for c in p[1:]):
                parts.extend(_CAMEL_BOUNDARY.split(p))
    elif any(c.isupper() for c in token[1:]):
        parts.extend(_CAMEL_BOUNDARY.split(token))
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _is_indexable(word: str) -> bool:
    """Allow alphanumeric tokens (plus underscores) with at least one letter,
    length 2-50. Filters out punctuation, lone digits, and very long blobs."""
    if len(word) < 2 or len(word) > 50:
        return False
    if not any(c.isalpha() for c in word):
        return False
    return all(c.isalnum() or c == "_" for c in word)


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


def _process_tokens(text: str) -> list[tuple[str, str]]:
    """Tokenize, split identifiers (case-preserving), lowercase, POS-tag, lemmatize.
    Returns (lemma, penn_tag) tuples."""
    raw = nltk.word_tokenize(text)
    expanded = []
    for tok in raw:
        for part in split_identifier(tok):
            expanded.append(part)
    lowered = [t.lower() for t in expanded]
    tagged = nltk.pos_tag(lowered)
    result = []
    for word, tag in tagged:
        if not _is_indexable(word):
            continue
        if word in _STOPWORDS:
            continue
        lemma = _LEMMATIZER.lemmatize(word, pos=penn_to_wordnet(tag))
        result.append((lemma, tag))
    return result


def lemmatize_text(text: str) -> list[str]:
    """Tokenize, split identifiers, POS-tag, lemmatize, remove stopwords.
    Used for indexing."""
    return [lemma for lemma, _ in _process_tokens(text)]


def lemmatize_text_with_pos(text: str) -> list[tuple[str, str]]:
    """Same pipeline as lemmatize_text but keeps the Penn tag — used for
    query-side synonym expansion which needs noun/verb signals."""
    return _process_tokens(text)


def expand_acronyms_for_query(text: str) -> list[str]:
    """Detect acronyms in the raw query text and return their lemmatized
    expansion tokens. Always query-side only — NEVER call this at index time."""
    expansion_text_parts = []
    for word in re.findall(r"\b\w+\b", text):
        key = word.lower()
        if key in _ACRONYMS:
            expansion_text_parts.append(_ACRONYMS[key])
    if not expansion_text_parts:
        return []
    return lemmatize_text(" ".join(expansion_text_parts))


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
