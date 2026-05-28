import numpy as np

from app.config import CONFIDENCE_THRESHOLD, SCORE_SCALE, MAX_RESULTS, MAX_SYNONYMS_PER_TOKEN
from app.indexer import IndexStore
from app.models import AskResponse, SectionResult
from app.nlp import expand_synonyms, lemmatize_text_with_pos

def _no_match() -> AskResponse:
    return AskResponse(
        confidence=0.0,
        alternatives=[],
        message="I don't have enough information to answer that.",
    )


def _tf_scores(bm25, query_tokens: list[str]) -> np.ndarray:
    """Term-frequency fallback when BM25 IDF collapses to 0 (tiny corpora).

    Returns raw TF counts (sum of query-token hits per document) normalised
    by each document's length so shorter docs aren't unfairly penalised.
    """
    scores = np.zeros(len(bm25.doc_freqs), dtype=float)
    for i, doc_freq in enumerate(bm25.doc_freqs):
        raw_tf = sum(doc_freq.get(t, 0) for t in query_tokens)
        doc_len = bm25.doc_len[i] if bm25.doc_len[i] > 0 else 1
        # Scale to produce scores comparable to the BM25 range
        scores[i] = raw_tf * SCORE_SCALE / doc_len
    return scores


def search(question: str, index_store: IndexStore) -> AskResponse:
    sections, bm25 = index_store.snapshot()

    if not sections or bm25 is None:
        return _no_match()

    tokens_with_pos = lemmatize_text_with_pos(question)
    query_tokens = expand_synonyms(tokens_with_pos, max_synonyms=MAX_SYNONYMS_PER_TOKEN)

    if not query_tokens:
        return _no_match()

    scores = bm25.get_scores(query_tokens)

    # BM25Okapi IDF collapses to 0 when every term appears in exactly half
    # the corpus (e.g. a 2-section index).  Fall back to TF-based scoring so
    # that small knowledge bases still return useful results.
    if scores.size and scores.max() == 0.0:
        scores = _tf_scores(bm25, query_tokens)

    if not scores.size or scores.max() < CONFIDENCE_THRESHOLD:
        return _no_match()

    ranked = sorted(
        ((i, float(score)) for i, score in enumerate(scores) if score >= CONFIDENCE_THRESHOLD),
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
