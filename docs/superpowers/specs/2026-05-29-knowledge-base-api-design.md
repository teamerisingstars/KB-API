# Knowledge Base API — Design Spec

**Date:** 2026-05-29
**Status:** Approved

---

## 1. Overview

A REST API that answers natural-language questions by searching a local knowledge base of Markdown files. No AI or ML models are used. Retrieval is powered by BM25 ranking over POS-aware lemmatized text, with query-time synonym expansion for improved recall.

The system behaves like a knowledgeable assistant: it answers confidently when it has relevant information, and explicitly says "I don't know" when it does not.

---

## 2. Goals

- Answer questions from `.md` files stored in a `knowledge/` directory
- No AI/ML — purely algorithmic (BM25 + NLTK linguistic tools)
- Handle paraphrasing via POS-aware lemmatization and controlled synonym expansion
- Auto-reindex when knowledge files are added or changed
- Return a meaningful confidence score and refuse to answer when no good match exists

---

## 3. Project Structure

```
knowledge/              ← drop .md files here (any subdirectory structure)
  general/
    python.md
    history.md
  ...

app/
  main.py               ← FastAPI app, route definitions
  indexer.py            ← parses .md files → Section objects, builds BM25 index
  searcher.py           ← query pipeline: lemmatize → expand → score → threshold
  models.py             ← Pydantic request/response schemas
  watcher.py            ← watchdog-based file watcher with 500ms debounce
  nlp.py                ← shared NLP utilities (POS tagging, lemmatization, synonym expansion)
  config.py             ← CONFIDENCE_THRESHOLD, SCORE_SCALE, KNOWLEDGE_DIR, etc.

docs/
  superpowers/specs/    ← this file lives here

requirements.txt
README.md
```

---

## 4. Data Model

### Section
The atomic unit of knowledge. One section = one heading + its body text in a `.md` file.

```python
@dataclass
class Section:
    heading: str        # e.g. "What is Python"
    body: str           # raw text under the heading (stored for response)
    source_file: str    # relative path, e.g. "knowledge/programming/python.md"
    tokens: list[str]   # POS-lemmatized tokens (used for BM25 indexing)
```

Content that appears before the first heading in a file is stored as a section with `heading = "Introduction"`.

A file with no headings becomes a single section using the filename (without extension) as the heading.

---

## 5. Indexer Pipeline (runs at startup + on file change)

```
knowledge/ directory
  → walk all .md files recursively
  → parse each file with markdown-it-py
      (correctly handles fenced code blocks — # inside code is not treated as a heading)
  → split into Section objects on headings (##, ###, etc.)
  → for each section body:
      → NLTK word_tokenize
      → pos_tag (Penn Treebank tagset)
      → map PB tags → WordNet POS (N/V/ADJ/ADV); unknown tags → noun default
      → WordNetLemmatizer.lemmatize(word, pos=wordnet_pos)
      → remove stopwords (NLTK english stopwords)
      → store as Section.tokens
  → build BM25Okapi(all_sections_tokens)
  → atomically swap index reference (thread-safe)
```

**Thread safety:** The index is held in a single object reference. A new index is built entirely in a background thread, then the reference is swapped atomically. In-flight `/ask` requests complete against the old index; new requests pick up the new one.

**Debouncing:** The file watcher waits 500ms after the last file-system event before triggering a reindex. This prevents repeated reindexes during rapid save bursts (e.g. editor auto-save).

---

## 6. Query Pipeline (runs on each POST /ask)

```
raw question string
  → word_tokenize + pos_tag
  → POS-aware lemmatization (same as indexer)
  → remove stopwords
  → synonym expansion (query side only — never expand at index time):
      → for each token where POS is noun (NN*) or verb (VB*):
          → if Penn tag is NNP/NNPS (proper noun): SKIP expansion
              (prevents "Python" → "snake", "Apple" → "fruit")
          → WordNet synsets filtered to matching POS
          → take first synset only (most common sense, avoids polysemy)
          → extract up to 2 lemma names, skip the original word
      → append expanded terms to token list
      → deduplicate
  → BM25 score all sections against expanded token list
  → if max raw score < CONFIDENCE_THRESHOLD: return "no match" response
  → else: return top 3 sections above threshold
```

**Why query-side expansion only:** Expanding at index time pollutes BM25's term statistics — IDF degrades because terms appear artificially often, and document lengths inflate. BM25 is fundamentally a term-statistics model; flooding the index with synonyms breaks it. Expanding only at query time is the conventional, lower-damage choice.

---

## 7. Confidence Score

Raw BM25 scores are unbounded and corpus-dependent. The score is mapped to a human-readable confidence as follows:

```
confidence = min(raw_score / SCORE_SCALE, 1.0)
```

Where `SCORE_SCALE` is a configurable constant (default: `10.0`).

**Critical:** Confidence is based on the **absolute** raw score, not normalized within the result set. This means:
- A score of `0.0` correctly signals "no match"
- The top result is NOT forced to `1.0`
- The threshold comparison is against the raw score, not a relative rank

**CONFIDENCE_THRESHOLD** (default: `1.0`) — sections with raw score below this are excluded entirely. The threshold should be tuned during initial testing against the knowledge base.

---

## 8. "I Don't Know" Behaviour

If the maximum BM25 score across all sections is below `CONFIDENCE_THRESHOLD`, the API returns:

```json
{
  "answer": null,
  "section": null,
  "source": null,
  "confidence": 0.0,
  "alternatives": [],
  "message": "I don't have enough information to answer that."
}
```

This is the trust boundary of the system: it never fabricates an answer.

---

## 9. API Endpoints

### POST /ask
Ask a question against the knowledge base.

**Request:**
```json
{ "question": "what is an automobile?" }
```

**Response (match found):**
```json
{
  "answer": "A car is a wheeled motor vehicle used for transportation...",
  "section": "What is a Car",
  "source": "knowledge/vehicles.md",
  "confidence": 0.87,
  "alternatives": [
    {
      "answer": "Trucks are large motor vehicles...",
      "section": "Types of Vehicles",
      "source": "knowledge/vehicles.md",
      "confidence": 0.42
    }
  ],
  "message": null
}
```

**Response (no match):**
```json
{
  "answer": null,
  "section": null,
  "source": null,
  "confidence": 0.0,
  "alternatives": [],
  "message": "I don't have enough information to answer that."
}
```

---

### GET /health
Returns API status and current index statistics.

```json
{
  "status": "ok",
  "indexed_sections": 142,
  "indexed_files": 18,
  "last_indexed": "2026-05-29T10:32:11Z"
}
```

---

### POST /reload
Manually triggers a full reindex of the `knowledge/` directory. Useful after bulk file changes.

```json
{ "status": "reindexing", "message": "Reindex triggered." }
```

---

### GET /sections
Returns a list of all indexed sections. Intended for debugging and knowledge base inspection.

```json
{
  "sections": [
    { "heading": "What is Python", "source": "knowledge/programming/python.md" },
    ...
  ]
}
```

---

## 10. Configuration (config.py)

| Key | Default | Description |
|-----|---------|-------------|
| `KNOWLEDGE_DIR` | `"knowledge/"` | Root directory for `.md` files |
| `CONFIDENCE_THRESHOLD` | `1.0` | Min raw BM25 score to include a section in results |
| `SCORE_SCALE` | `10.0` | Divisor for mapping raw score to 0.0–1.0 confidence |
| `MAX_RESULTS` | `3` | Maximum number of results returned |
| `WATCHER_DEBOUNCE_MS` | `500` | Milliseconds to wait after last file event before reindexing |
| `MAX_SYNONYMS_PER_TOKEN` | `2` | Max synonyms added per query token |

---

## 11. Dependencies

```
fastapi
uvicorn
markdown-it-py      ← markdown parser (handles fenced code blocks correctly)
rank-bm25           ← BM25Okapi implementation
nltk                ← POS tagging, lemmatization, stopwords, WordNet
watchdog            ← file system watcher
pydantic            ← request/response models (included with FastAPI)
```

NLTK data downloads required at first run: `punkt`, `averaged_perceptron_tagger`, `wordnet`, `stopwords`.

---

## 12. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Empty question | 422 Unprocessable Entity |
| `knowledge/` dir missing | 503 on startup with clear error message |
| No `.md` files found | API starts but `/ask` always returns "no match" |
| Malformed `.md` file | Skip file, log warning, continue indexing others |
| Reindex in progress + `/ask` called | Serve from current index; swap completes in background |

---

## 13. Out of Scope (v1)

- Multi-language support
- Persistent index cache (index is rebuilt in memory on each startup)
- Authentication / rate limiting
- Extracted (synthesized) answers — `answer` field is always the full section body
- Conversational context / follow-up questions
