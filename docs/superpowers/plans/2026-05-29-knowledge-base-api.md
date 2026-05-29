# Knowledge Base API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a REST API that answers natural-language questions by searching a BM25-indexed knowledge base of Markdown files — no AI/ML, purely algorithmic.

**Architecture:** markdown-it-py parses `.md` files into heading-delimited Sections; BM25Okapi ranks sections against POS-lemmatized queries; query-time WordNet synonym expansion handles paraphrasing without polluting index statistics. A watchdog watcher auto-reindexes on file changes with a 500ms debounce.

**Tech Stack:** Python 3.11+, FastAPI, rank-bm25, NLTK (WordNetLemmatizer + pos_tag + WordNet), markdown-it-py, watchdog, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `app/config.py` | All tunable constants (thresholds, paths, limits) |
| `app/models.py` | `Section` dataclass + all Pydantic request/response schemas |
| `app/nlp.py` | `penn_to_wordnet`, `lemmatize_text`, `lemmatize_text_with_pos`, `expand_synonyms` |
| `app/indexer.py` | `parse_sections`, `IndexStore`, `build_index` |
| `app/searcher.py` | `search(question, index_store) → AskResponse` |
| `app/watcher.py` | `DebounceHandler`, `KnowledgeWatcher` |
| `app/main.py` | FastAPI app, lifespan, all four routes |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Test-only dependencies |
| `download_nltk_data.py` | One-time NLTK corpus downloader |
| `tests/conftest.py` | NLTK download fixture + shared test helpers |
| `tests/test_nlp.py` | Unit tests for NLP utilities |
| `tests/test_indexer.py` | Unit tests for markdown parsing + index building |
| `tests/test_searcher.py` | Unit tests for search pipeline |
| `tests/test_api.py` | Integration tests for all FastAPI routes |
| `knowledge/sample.md` | Sample knowledge file for tests and development |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `download_nltk_data.py`
- Create: `knowledge/sample.md`
- Create: `app/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
markdown-it-py>=3.0.0
rank-bm25>=0.2.2
nltk>=3.8.1
watchdog>=4.0.0
```

- [ ] **Step 2: Create requirements-dev.txt**

```
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 3: Create download_nltk_data.py**

```python
import nltk

REQUIRED = [
    "punkt_tab",
    "punkt",
    "averaged_perceptron_tagger_eng",
    "averaged_perceptron_tagger",
    "wordnet",
    "stopwords",
    "omw-1.4",
]

for corpus in REQUIRED:
    nltk.download(corpus, quiet=True)

print("NLTK data downloaded.")
```

- [ ] **Step 4: Create app/__init__.py and tests/__init__.py** (both empty)

- [ ] **Step 5: Create knowledge/sample.md**

```markdown
## What is Python

Python is a high-level, interpreted programming language known for its clear syntax
and readability. It supports multiple programming paradigms including procedural,
object-oriented, and functional programming.

## History of Python

Python was created by Guido van Rossum and first released in 1991. The name comes
from the British comedy group Monty Python, not the snake.

## What is a Car

A car is a wheeled motor vehicle used for transportation. Most cars have four wheels
and run on petrol, diesel, or electricity. They are also called automobiles or autos.

## Python Data Types

Python has several built-in data types: integers, floats, strings, lists, tuples,
dictionaries, and sets. Everything in Python is an object.
```

- [ ] **Step 6: Install dependencies**

```
pip install -r requirements.txt -r requirements-dev.txt
python download_nltk_data.py
```

- [ ] **Step 7: Commit**

```bash
git init
git add requirements.txt requirements-dev.txt download_nltk_data.py knowledge/sample.md app/__init__.py tests/__init__.py
git commit -m "chore: project scaffold, dependencies, sample knowledge"
```

---

## Task 2: Configuration

**Files:**
- Create: `app/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from app.config import (
    KNOWLEDGE_DIR,
    CONFIDENCE_THRESHOLD,
    SCORE_SCALE,
    MAX_RESULTS,
    WATCHER_DEBOUNCE_MS,
    MAX_SYNONYMS_PER_TOKEN,
)

def test_config_defaults():
    assert KNOWLEDGE_DIR == "knowledge/"
    assert CONFIDENCE_THRESHOLD == 1.0
    assert SCORE_SCALE == 10.0
    assert MAX_RESULTS == 3
    assert WATCHER_DEBOUNCE_MS == 500
    assert MAX_SYNONYMS_PER_TOKEN == 2
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: Write implementation**

Create `app/config.py`:

```python
import os

KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", "knowledge/")
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "1.0"))
SCORE_SCALE: float = float(os.getenv("SCORE_SCALE", "10.0"))
MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "3"))
WATCHER_DEBOUNCE_MS: int = int(os.getenv("WATCHER_DEBOUNCE_MS", "500"))
MAX_SYNONYMS_PER_TOKEN: int = int(os.getenv("MAX_SYNONYMS_PER_TOKEN", "2"))
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_config.py -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add configuration module"
```

---

## Task 3: Data Models

**Files:**
- Create: `app/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
from app.models import Section, AskRequest, SectionResult, AskResponse, HealthResponse, SectionsResponse, SectionInfo, ReloadResponse

def test_section_dataclass():
    s = Section(heading="Test", body="body text", source_file="knowledge/test.md", tokens=["body", "text"])
    assert s.heading == "Test"
    assert s.tokens == ["body", "text"]

def test_ask_request_requires_question():
    req = AskRequest(question="what is python?")
    assert req.question == "what is python?"

def test_ask_request_rejects_empty():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AskRequest(question="")

def test_ask_response_match():
    r = AskResponse(
        answer="Python is a language.",
        section="What is Python",
        source="knowledge/sample.md",
        confidence=0.87,
        alternatives=[],
        message=None,
    )
    assert r.answer == "Python is a language."
    assert r.confidence == 0.87

def test_ask_response_no_match():
    r = AskResponse(
        answer=None,
        section=None,
        source=None,
        confidence=0.0,
        alternatives=[],
        message="I don't have enough information to answer that.",
    )
    assert r.answer is None
    assert r.message is not None

def test_health_response():
    h = HealthResponse(status="ok", indexed_sections=10, indexed_files=2, last_indexed="2026-05-29T10:00:00Z")
    assert h.status == "ok"

def test_sections_response():
    sr = SectionsResponse(sections=[SectionInfo(heading="Intro", source="test.md")])
    assert len(sr.sections) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: Write implementation**

Create `app/models.py`:

```python
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, field_validator


@dataclass
class Section:
    heading: str
    body: str
    source_file: str
    tokens: list[str]


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class SectionResult(BaseModel):
    answer: str
    section: str
    source: str
    confidence: float


class AskResponse(BaseModel):
    answer: Optional[str]
    section: Optional[str]
    source: Optional[str]
    confidence: float
    alternatives: list[SectionResult]
    message: Optional[str]


class HealthResponse(BaseModel):
    status: str
    indexed_sections: int
    indexed_files: int
    last_indexed: Optional[str]


class SectionInfo(BaseModel):
    heading: str
    source: str


class SectionsResponse(BaseModel):
    sections: list[SectionInfo]


class ReloadResponse(BaseModel):
    status: str
    message: str
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_models.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add data models (Section dataclass + Pydantic schemas)"
```

---

## Task 4: NLP Utilities

**Files:**
- Create: `app/nlp.py`
- Create: `tests/conftest.py`
- Create: `tests/test_nlp.py`

- [ ] **Step 1: Create tests/conftest.py** (downloads NLTK data once per session)

```python
import pytest
import nltk

@pytest.fixture(scope="session", autouse=True)
def download_nltk_data():
    for corpus in ["punkt_tab", "punkt", "averaged_perceptron_tagger_eng",
                   "averaged_perceptron_tagger", "wordnet", "stopwords", "omw-1.4"]:
        nltk.download(corpus, quiet=True)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_nlp.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/test_nlp.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.nlp'`

- [ ] **Step 4: Write implementation**

Create `app/nlp.py`:

```python
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))


def penn_to_wordnet(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
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
    Skips proper nouns to avoid polysemy traps (Python→snake, Apple→fruit)."""
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
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_nlp.py -v
```

Expected: all `PASSED`

- [ ] **Step 6: Commit**

```bash
git add app/nlp.py tests/conftest.py tests/test_nlp.py
git commit -m "feat: add NLP utilities (POS lemmatization + synonym expansion)"
```

---

## Task 5: Markdown Parser

**Files:**
- Create: `app/indexer.py` (parse_sections only — IndexStore added in Task 6)
- Create: `tests/test_indexer.py` (parse_sections tests only)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_indexer.py`:

```python
from app.indexer import parse_sections
from app.models import Section


SIMPLE_MD = """## What is Python

Python is a high-level programming language.

## History

Python was created by Guido van Rossum in 1991.
"""

CODE_BLOCK_MD = """## Example

Here is some code.

```python
# This is a comment, not a heading
def hello():
    pass
```

More text after the code.
"""

PRE_HEADING_MD = """This content appears before any heading.

## First Section

Section body.
"""

NO_HEADING_MD = "This file has no headings at all. Just plain text content."


def test_parse_sections_returns_list_of_sections():
    sections = parse_sections(SIMPLE_MD, "knowledge/test.md")
    assert isinstance(sections, list)
    assert all(isinstance(s, Section) for s in sections)


def test_parse_sections_splits_on_headings():
    sections = parse_sections(SIMPLE_MD, "knowledge/test.md")
    headings = [s.heading for s in sections]
    assert "What is Python" in headings
    assert "History" in headings


def test_parse_sections_captures_body():
    sections = parse_sections(SIMPLE_MD, "knowledge/test.md")
    python_section = next(s for s in sections if s.heading == "What is Python")
    assert "Python" in python_section.body
    assert "high-level" in python_section.body


def test_parse_sections_skips_hash_inside_code_block():
    sections = parse_sections(CODE_BLOCK_MD, "knowledge/test.md")
    headings = [s.heading for s in sections]
    # "This is a comment, not a heading" must NOT become a heading
    assert not any("comment" in h.lower() for h in headings)
    assert "Example" in headings


def test_parse_sections_content_before_first_heading_is_introduction():
    sections = parse_sections(PRE_HEADING_MD, "knowledge/test.md")
    headings = [s.heading for s in sections]
    assert "Introduction" in headings


def test_parse_sections_no_headings_uses_filename():
    sections = parse_sections(NO_HEADING_MD, "knowledge/my_topic.md")
    assert len(sections) == 1
    assert sections[0].heading == "my_topic"


def test_parse_sections_stores_source_file():
    sections = parse_sections(SIMPLE_MD, "knowledge/sub/test.md")
    assert all(s.source_file == "knowledge/sub/test.md" for s in sections)


def test_parse_sections_tokens_are_lemmatized():
    sections = parse_sections(SIMPLE_MD, "knowledge/test.md")
    python_section = next(s for s in sections if s.heading == "What is Python")
    # "programming" should be lemmatized; stopwords removed
    assert "the" not in python_section.tokens
    assert "program" in python_section.tokens or "programming" in python_section.tokens
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_indexer.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.indexer'`

- [ ] **Step 3: Write implementation** (parse_sections only)

Create `app/indexer.py`:

```python
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from markdown_it import MarkdownIt
from rank_bm25 import BM25Okapi

from app.models import Section
from app.nlp import lemmatize_text

_MD_PARSER = MarkdownIt()
logger = logging.getLogger(__name__)


def parse_sections(content: str, source_file: str) -> list[Section]:
    """Parse a markdown string into Section objects split on headings.
    Fenced code blocks are skipped so # in code is never treated as a heading."""
    tokens = _MD_PARSER.parse(content)
    sections: list[Section] = []
    current_heading: str | None = None
    current_body_parts: list[str] = []
    inside_heading = False

    def flush() -> None:
        heading = current_heading or "Introduction"
        body = " ".join(current_body_parts).strip()
        if body:
            sections.append(Section(
                heading=heading,
                body=body,
                source_file=source_file,
                tokens=lemmatize_text(body),
            ))

    for token in tokens:
        if token.type == "heading_open":
            flush()
            current_heading = None
            current_body_parts.clear()
            inside_heading = True
        elif token.type == "heading_close":
            inside_heading = False
        elif token.type == "fence":
            pass  # Skip code block content
        elif token.type == "inline":
            if inside_heading:
                current_heading = token.content
            else:
                current_body_parts.append(token.content)

    flush()

    if not sections and content.strip():
        filename = Path(source_file).stem
        body = content.strip()
        sections.append(Section(
            heading=filename,
            body=body,
            source_file=source_file,
            tokens=lemmatize_text(body),
        ))

    return sections
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_indexer.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add app/indexer.py tests/test_indexer.py
git commit -m "feat: add markdown section parser"
```

---

## Task 6: BM25 Index Store

**Files:**
- Modify: `app/indexer.py` (add `IndexStore` class + `build_index` function)
- Modify: `tests/test_indexer.py` (add IndexStore + build_index tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_indexer.py`)

```python
import os
import tempfile
from app.indexer import IndexStore, build_index


def test_index_store_starts_empty():
    store = IndexStore()
    assert store.sections == []
    assert store.bm25 is None
    assert store.file_count == 0


def test_index_store_swap_updates_data():
    from rank_bm25 import BM25Okapi
    store = IndexStore()
    sections = [Section(heading="H", body="body", source_file="f.md", tokens=["body"])]
    bm25 = BM25Okapi([["body"]])
    store.swap(sections, bm25, file_count=1, last_indexed="2026-05-29T00:00:00Z")
    assert len(store.sections) == 1
    assert store.bm25 is not None
    assert store.file_count == 1


def test_build_index_indexes_md_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "test.md")
        with open(md_path, "w") as f:
            f.write("## Python\n\nPython is a programming language.\n")
        store = IndexStore()
        build_index(tmpdir, store)
        assert len(store.sections) > 0
        assert store.bm25 is not None
        assert store.file_count == 1


def test_build_index_handles_missing_directory():
    store = IndexStore()
    build_index("/nonexistent/path/xyz", store)
    assert store.sections == []
    assert store.bm25 is None


def test_build_index_skips_malformed_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = os.path.join(tmpdir, "bad.md")
        with open(bad_path, "wb") as f:
            f.write(b"\xff\xfe invalid utf-8 \x00")
        store = IndexStore()
        build_index(tmpdir, store)
        # Should not raise; bad file is skipped
        assert store.sections == []


def test_build_index_walks_subdirectories():
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = os.path.join(tmpdir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "nested.md"), "w") as f:
            f.write("## Nested\n\nNested content here.\n")
        store = IndexStore()
        build_index(tmpdir, store)
        assert len(store.sections) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_indexer.py::test_index_store_starts_empty -v
```

Expected: `ImportError: cannot import name 'IndexStore'`

- [ ] **Step 3: Add IndexStore and build_index to app/indexer.py**

Append to `app/indexer.py` (after the existing `parse_sections` function):

```python
class IndexStore:
    """Thread-safe container for the BM25 index. Supports atomic swap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sections: list[Section] = []
        self._bm25: BM25Okapi | None = None
        self._file_count: int = 0
        self._last_indexed: str | None = None

    def swap(
        self,
        sections: list[Section],
        bm25: BM25Okapi | None,
        file_count: int,
        last_indexed: str,
    ) -> None:
        with self._lock:
            self._sections = sections
            self._bm25 = bm25
            self._file_count = file_count
            self._last_indexed = last_indexed

    def snapshot(self) -> tuple[list[Section], BM25Okapi | None]:
        """Return sections + bm25 atomically to avoid TOCTOU between two reads."""
        with self._lock:
            return self._sections, self._bm25

    @property
    def sections(self) -> list[Section]:
        with self._lock:
            return self._sections

    @property
    def bm25(self) -> BM25Okapi | None:
        with self._lock:
            return self._bm25

    @property
    def file_count(self) -> int:
        with self._lock:
            return self._file_count

    @property
    def last_indexed(self) -> str | None:
        with self._lock:
            return self._last_indexed


def build_index(knowledge_dir: str, index_store: IndexStore) -> None:
    """Walk knowledge_dir, parse all .md files, build BM25 index, atomically swap."""
    knowledge_path = Path(knowledge_dir)
    if not knowledge_path.exists():
        logger.error("Knowledge directory not found: %s", knowledge_dir)
        return

    md_files = list(knowledge_path.rglob("*.md"))
    sections: list[Section] = []

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            rel_path = str(md_file.relative_to(Path.cwd()))
            sections.extend(parse_sections(content, rel_path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", md_file, exc)

    bm25 = BM25Okapi([s.tokens for s in sections]) if sections else None
    last_indexed = datetime.now(timezone.utc).isoformat()
    index_store.swap(sections, bm25, len(md_files), last_indexed)
    logger.info("Indexed %d sections from %d files", len(sections), len(md_files))
```

- [ ] **Step 4: Run all indexer tests to verify they pass**

```
pytest tests/test_indexer.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add app/indexer.py tests/test_indexer.py
git commit -m "feat: add IndexStore and build_index"
```

---

## Task 7: Search Engine

**Files:**
- Create: `app/searcher.py`
- Create: `tests/test_searcher.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_searcher.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_searcher.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.searcher'`

- [ ] **Step 3: Write implementation**

Create `app/searcher.py`:

```python
from app.config import CONFIDENCE_THRESHOLD, SCORE_SCALE, MAX_RESULTS, MAX_SYNONYMS_PER_TOKEN
from app.indexer import IndexStore
from app.models import AskResponse, SectionResult
from app.nlp import expand_synonyms, lemmatize_text_with_pos

_NO_MATCH = AskResponse(
    answer=None,
    section=None,
    source=None,
    confidence=0.0,
    alternatives=[],
    message="I don't have enough information to answer that.",
)


def search(question: str, index_store: IndexStore) -> AskResponse:
    sections, bm25 = index_store.snapshot()

    if not sections or bm25 is None:
        return _NO_MATCH

    tokens_with_pos = lemmatize_text_with_pos(question)
    query_tokens = expand_synonyms(tokens_with_pos, max_synonyms=MAX_SYNONYMS_PER_TOKEN)

    if not query_tokens:
        return _NO_MATCH

    scores = bm25.get_scores(query_tokens)

    if not scores.size or scores.max() < CONFIDENCE_THRESHOLD:
        return _NO_MATCH

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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_searcher.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add app/searcher.py tests/test_searcher.py
git commit -m "feat: add BM25 search engine with synonym expansion"
```

---

## Task 8: File Watcher

**Files:**
- Create: `app/watcher.py`
- Create: `tests/test_watcher.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_watcher.py`:

```python
import os
import tempfile
import time
from app.watcher import KnowledgeWatcher


def test_watcher_calls_callback_on_md_change():
    called = []
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = KnowledgeWatcher(tmpdir, lambda: called.append(1), debounce_ms=100)
        watcher.start()
        try:
            with open(os.path.join(tmpdir, "new.md"), "w") as f:
                f.write("## New\n\nContent.\n")
            time.sleep(0.4)
        finally:
            watcher.stop()
    assert len(called) >= 1


def test_watcher_ignores_non_md_files():
    called = []
    with tempfile.TemporaryDirectory() as tmpdir:
        watcher = KnowledgeWatcher(tmpdir, lambda: called.append(1), debounce_ms=100)
        watcher.start()
        try:
            with open(os.path.join(tmpdir, "notes.txt"), "w") as f:
                f.write("not markdown")
            time.sleep(0.4)
        finally:
            watcher.stop()
    assert len(called) == 0


def test_watcher_debounces_rapid_saves():
    called = []
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "test.md")
        watcher = KnowledgeWatcher(tmpdir, lambda: called.append(1), debounce_ms=300)
        watcher.start()
        try:
            for i in range(5):
                with open(md_path, "w") as f:
                    f.write(f"## Section {i}\n\nContent {i}.\n")
                time.sleep(0.05)
            time.sleep(0.6)
        finally:
            watcher.stop()
    # 5 rapid saves should produce 1 or 2 callbacks, not 5
    assert len(called) <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_watcher.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.watcher'`

- [ ] **Step 3: Write implementation**

Create `app/watcher.py`:

```python
import logging
import threading

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _DebounceHandler(FileSystemEventHandler):
    def __init__(self, callback, debounce_ms: int) -> None:
        super().__init__()
        self._callback = callback
        self._debounce_s = debounce_ms / 1000.0
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return
        if not str(event.src_path).endswith(".md"):
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_s, self._callback)
            self._timer.start()


class KnowledgeWatcher:
    def __init__(self, knowledge_dir: str, callback, debounce_ms: int = 500) -> None:
        self._handler = _DebounceHandler(callback, debounce_ms)
        self._observer = Observer()
        self._observer.schedule(self._handler, knowledge_dir, recursive=True)

    def start(self) -> None:
        self._observer.start()
        logger.info("Knowledge base watcher started.")

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
        logger.info("Knowledge base watcher stopped.")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_watcher.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add app/watcher.py tests/test_watcher.py
git commit -m "feat: add debounced file watcher"
```

---

## Task 9: FastAPI Application

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Uses the real knowledge/sample.md created in Task 1.
# app.config reads KNOWLEDGE_DIR at import time, so env-var tricks are fragile.
# Relying on the actual knowledge/ directory is simpler and tests the real integration.

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["indexed_sections"] > 0


def test_ask_returns_answer_for_known_topic(client):
    resp = client.post("/ask", json={"question": "what is python"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is not None
    assert "python" in data["answer"].lower() or "language" in data["answer"].lower()


def test_ask_no_match_returns_message(client):
    resp = client.post("/ask", json={"question": "quantum entanglement neutrino flux"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is None
    assert data["message"] is not None
    assert data["confidence"] == 0.0


def test_ask_empty_question_returns_422(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_missing_question_returns_422(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_sections_returns_list(client):
    resp = client.get("/sections")
    assert resp.status_code == 200
    data = resp.json()
    assert "sections" in data
    assert isinstance(data["sections"], list)
    assert len(data["sections"]) > 0


def test_reload_triggers_reindex(client):
    resp = client.post("/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reindexing"


def test_ask_synonym_finds_automobile(client):
    resp = client.post("/ask", json={"question": "what is an automobile"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is not None


def test_response_has_confidence_field(client):
    resp = client.post("/ask", json={"question": "what is python"})
    data = resp.json()
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Write implementation**

Create `app/main.py`:

```python
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import KNOWLEDGE_DIR, WATCHER_DEBOUNCE_MS
from app.indexer import IndexStore, build_index
from app.models import AskRequest, AskResponse, HealthResponse, ReloadResponse, SectionInfo, SectionsResponse
from app.searcher import search
from app.watcher import KnowledgeWatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_index_store = IndexStore()
_watcher: KnowledgeWatcher | None = None


def _trigger_reindex() -> None:
    threading.Thread(
        target=build_index, args=(KNOWLEDGE_DIR, _index_store), daemon=True
    ).start()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _watcher
    if not Path(KNOWLEDGE_DIR).exists():
        logger.error("Knowledge directory not found: %s — API will return no matches.", KNOWLEDGE_DIR)
    else:
        build_index(KNOWLEDGE_DIR, _index_store)
        _watcher = KnowledgeWatcher(KNOWLEDGE_DIR, _trigger_reindex, WATCHER_DEBOUNCE_MS)
        _watcher.start()
    yield
    if _watcher:
        _watcher.stop()


app = FastAPI(title="Knowledge Base API", version="1.0.0", lifespan=_lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return search(request.question, _index_store)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        indexed_sections=len(_index_store.sections),
        indexed_files=_index_store.file_count,
        last_indexed=_index_store.last_indexed,
    )


@app.post("/reload", response_model=ReloadResponse)
def reload() -> ReloadResponse:
    _trigger_reindex()
    return ReloadResponse(status="reindexing", message="Reindex triggered.")


@app.get("/sections", response_model=SectionsResponse)
def sections() -> SectionsResponse:
    return SectionsResponse(
        sections=[
            SectionInfo(heading=s.heading, source=s.source_file)
            for s in _index_store.sections
        ]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_api.py -v
```

Expected: all `PASSED`

- [ ] **Step 5: Run full test suite**

```
pytest -v
```

Expected: all tests across all files `PASSED`

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add FastAPI application with all routes"
```

---

## Task 10: End-to-End Verification

**Files:**
- No new files — run the live server and manually verify

- [ ] **Step 1: Start the server**

```
uvicorn app.main:app --reload
```

Expected output:
```
INFO:     Indexed N sections from M files
INFO:     Uvicorn running on http://127.0.0.1:8000
```

- [ ] **Step 2: Check health**

```
curl http://127.0.0.1:8000/health
```

Expected:
```json
{"status":"ok","indexed_sections":4,"indexed_files":1,"last_indexed":"..."}
```

- [ ] **Step 3: Ask a direct question**

```
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"what is python\"}"
```

Expected: `answer` field contains Python language description, `confidence > 0`.

- [ ] **Step 4: Test synonym expansion**

```
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"what is an automobile\"}"
```

Expected: returns the "What is a Car" section (synonym expansion working).

- [ ] **Step 5: Test "I don't know"**

```
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"quantum entanglement in photon spin\"}"
```

Expected: `answer: null`, `message: "I don't have enough information to answer that."`

- [ ] **Step 6: Test live reindex** — add a new .md file while server is running

```
echo "## New Topic\n\nThis is a brand new topic added at runtime." > knowledge/new_topic.md
```

Wait 1 second, then:

```
curl http://127.0.0.1:8000/health
```

Expected: `indexed_sections` count increased (watcher auto-triggered reindex).

- [ ] **Step 7: Open Swagger UI**

Navigate to `http://127.0.0.1:8000/docs` — verify all four endpoints are listed and testable via the browser UI.

- [ ] **Step 8: Final commit**

```bash
git add .
git commit -m "feat: knowledge base API complete and verified"
```

---

## Running Tests

```bash
# Full suite
pytest -v

# One file
pytest tests/test_api.py -v

# One test
pytest tests/test_searcher.py::test_search_synonym_expansion_finds_automobile -v

# With coverage
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

## Running the Server

```bash
uvicorn app.main:app --reload
# Custom knowledge dir:
KNOWLEDGE_DIR=my_docs/ uvicorn app.main:app --reload
# Tune thresholds:
CONFIDENCE_THRESHOLD=0.5 SCORE_SCALE=8.0 uvicorn app.main:app --reload
```
