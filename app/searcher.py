import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    FILENAME_BOOST_FACTOR,
    HEADING_BOOST_FACTOR,
    HEADING_OVERLAP_THRESHOLD,
    MAX_RESULTS,
    MAX_SYNONYMS_PER_TOKEN,
    REFERENCE_PATH_BOOST,
    SCORE_SCALE,
)
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
    """Multiplicative boost for sections whose heading overlaps the query, plus
    an additive bonus when the heading EXACTLY matches the query subject."""
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
    'what is response_model' should favour response-model.md. Plus a small
    extra boost for sections under reference/ (canonical definitions)."""
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


def search(question: str, index_store: IndexStore) -> AskResponse:
    sections, bm25 = index_store.snapshot()
    if not sections or bm25 is None:
        return _no_match()

    tokens_with_pos = lemmatize_text_with_pos(question)
    query_tokens = expand_synonyms(tokens_with_pos, max_synonyms=MAX_SYNONYMS_PER_TOKEN)
    acronym_tokens = expand_acronyms_for_query(question)
    if acronym_tokens:
        query_tokens = list(dict.fromkeys(query_tokens + acronym_tokens))
    if not query_tokens:
        return _no_match()

    scores = bm25.get_scores(query_tokens)
    if scores.size and scores.max() <= 0.0:
        scores = _tf_scores(bm25, query_tokens)

    query_lemmas = {lemma for lemma, _ in tokens_with_pos}
    scores = _apply_heading_boost(scores, sections, query_lemmas)
    scores = _apply_filename_boost(scores, sections, query_lemmas)

    if not scores.size or scores.max() < CONFIDENCE_THRESHOLD:
        return _no_match()

    ranked = sorted(
        ((i, float(s)) for i, s in enumerate(scores) if s >= CONFIDENCE_THRESHOLD),
        key=lambda x: x[1],
        reverse=True,
    )[:MAX_RESULTS]

    results = [
        SectionResult(
            answer=sections[idx].body,
            section=sections[idx].heading,
            source=sections[idx].source_file,
            confidence=round(min(score / SCORE_SCALE, 1.0), 4),
        )
        for idx, score in ranked
    ]

    best = results[0]
    return AskResponse(
        answer=best.answer,
        section=best.section,
        source=best.source,
        confidence=best.confidence,
        alternatives=results[1:],
        message=None,
    )
