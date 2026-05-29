import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    FILENAME_BOOST_FACTOR,
    FUZZY_CONFIDENCE_CAP,
    FUZZY_CORRECTION_ENABLED,
    FUZZY_MAX_CANDIDATES,
    HEADING_BOOST_FACTOR,
    HEADING_OVERLAP_THRESHOLD,
    MAX_RESULTS,
    MAX_SYNONYMS_PER_TOKEN,
    REFERENCE_PATH_BOOST,
    SCORE_SCALE,
)
from app.fuzzy import fuzzy_candidates
from app.indexer import IndexStore
from app.models import AskResponse, SectionResult
from app.nlp import (
    expand_acronyms_for_query,
    expand_synonyms,
    lemmatize_text,
    lemmatize_text_with_pos,
)


def _no_match() -> AskResponse:
    return AskResponse(
        confidence=0.0,
        alternatives=[],
        message="I don't have enough information to answer that.",
    )


def _tf_scores(bm25, query_tokens):
    """Term-frequency fallback when BM25 IDF collapses to 0 (tiny corpora)."""
    scores = np.zeros(len(bm25.doc_freqs), dtype=float)
    for i, doc_freq in enumerate(bm25.doc_freqs):
        raw_tf = sum(doc_freq.get(t, 0) for t in query_tokens)
        doc_len = bm25.doc_len[i] if bm25.doc_len[i] > 0 else 1
        scores[i] = raw_tf * SCORE_SCALE / doc_len
    return scores


def _apply_heading_boost(scores, sections, query_lemmas):
    """Multiplicative boost for sections whose heading overlaps the query,
    plus an additive bonus when the heading EXACTLY matches the query."""
    boosted = scores.copy()
    for i, section in enumerate(sections):
        heading_tokens = set(lemmatize_text(section.heading))
        if not heading_tokens:
            continue
        overlap = len(query_lemmas & heading_tokens) / len(heading_tokens)
        if overlap >= HEADING_OVERLAP_THRESHOLD:
            boosted[i] *= HEADING_BOOST_FACTOR
        if query_lemmas == heading_tokens:
            boosted[i] += 1.0
    return boosted


def _apply_filename_boost(scores, sections, query_lemmas):
    """Boost sections whose source filename overlaps the query subject —
    'what is response_model' should favour response-model.md."""
    boosted = scores.copy()
    for i, section in enumerate(sections):
        path = section.source_file
        filename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        filename_lemmas = set(lemmatize_text(filename.replace("-", " ").replace("_", " ")))
        if filename_lemmas and (query_lemmas & filename_lemmas):
            boosted[i] *= FILENAME_BOOST_FACTOR
        if "/reference/" in path or path.startswith("reference/"):
            boosted[i] *= REFERENCE_PATH_BOOST
    return boosted


def _apply_typo_correction(query_tokens, bk_tree) -> tuple[list[str], set[str], bool]:
    """For each query token NOT in the indexed vocab, find nearest-neighbour
    candidates and append them to the query stream.

    Returns (new_tokens, fuzzy_set, used_fuzzy):
      - new_tokens: full token list including the corrections
      - fuzzy_set: ONLY the corrections that were appended (so callers can
        feed them to heading/filename boost checks)
      - used_fuzzy: True if any correction fired (caller caps confidence)
    """
    if bk_tree is None or not FUZZY_CORRECTION_ENABLED:
        return query_tokens, set(), False
    fuzzy_set: set[str] = set()
    corrected = list(query_tokens)
    for tok in query_tokens:
        if tok in bk_tree:
            continue
        suggestions = fuzzy_candidates(bk_tree, tok, max_candidates=FUZZY_MAX_CANDIDATES)
        if suggestions:
            corrected.extend(suggestions)
            fuzzy_set.update(suggestions)
    return list(dict.fromkeys(corrected)), fuzzy_set, bool(fuzzy_set)


def search(question: str, index_store: IndexStore) -> AskResponse:
    sections, bm25, bk_tree = index_store.snapshot()
    if not sections or bm25 is None:
        return _no_match()

    tokens_with_pos = lemmatize_text_with_pos(question)
    query_tokens = expand_synonyms(tokens_with_pos, max_synonyms=MAX_SYNONYMS_PER_TOKEN)
    acronym_tokens = expand_acronyms_for_query(question)
    if acronym_tokens:
        query_tokens = list(dict.fromkeys(query_tokens + acronym_tokens))
    if not query_tokens:
        return _no_match()

    # Typo correction — fires only for tokens not in the indexed vocab.
    query_tokens, fuzzy_corrections, used_fuzzy = _apply_typo_correction(query_tokens, bk_tree)

    # Out-of-domain guard: if 3+ user-supplied tokens are unrecognized, the
    # query is almost certainly out of domain (e.g. "quantum entanglement
    # neutrino flux" against a FastAPI corpus). Returning a fuzzy-corrected
    # result there is worse than honestly returning null. A single typo'd
    # token is allowed — that's the common case the correction was built for.
    if used_fuzzy and bk_tree is not None:
        # Count from the USER'S question only, not synonym expansion —
        # WordNet synonyms ("carrier", "toter" for "bearer") often aren't in
        # the indexed vocab and would inflate the count.
        user_lemmas = {lemma for lemma, _ in tokens_with_pos}
        unrecognized = sum(
            1 for t in user_lemmas
            if t not in bk_tree and " " not in t and len(t) >= 3
        )
        if unrecognized >= 3:
            return _no_match()

    scores = bm25.get_scores(query_tokens)
    if scores.size and scores.max() <= 0.0:
        scores = _tf_scores(bm25, query_tokens)

    # Include fuzzy corrections in the lemma set so heading/filename boost
    # checks recognise that "corss" → "cors" should boost CORS sections.
    query_lemmas = {lemma for lemma, _ in tokens_with_pos} | fuzzy_corrections
    scores = _apply_heading_boost(scores, sections, query_lemmas)
    scores = _apply_filename_boost(scores, sections, query_lemmas)

    if not scores.size or scores.max() < CONFIDENCE_THRESHOLD:
        return _no_match()

    ranked = sorted(
        ((i, float(s)) for i, s in enumerate(scores) if s >= CONFIDENCE_THRESHOLD),
        key=lambda x: x[1],
        reverse=True,
    )[:MAX_RESULTS]

    # Cap the displayed confidence when fuzzy fired. The raw score still
    # wins the ranking — we just signal lower trust to the caller.
    confidence_cap = FUZZY_CONFIDENCE_CAP if used_fuzzy else 1.0

    results = [
        SectionResult(
            answer=sections[idx].body,
            section=sections[idx].heading,
            source=sections[idx].source_file,
            confidence=round(min(score / SCORE_SCALE, confidence_cap), 4),
        )
        for idx, score in ranked
    ]

    best = results[0]
    msg = (
        "Result based on a fuzzy match — your query contained tokens not in "
        "the indexed vocabulary. Verify the source before trusting the answer."
    ) if used_fuzzy else None
    return AskResponse(
        answer=best.answer,
        section=best.section,
        source=best.source,
        confidence=best.confidence,
        alternatives=results[1:],
        message=msg,
    )
