import logging
import threading
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
    heading_seen = False

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
            heading_seen = True
        elif token.type == "heading_close":
            inside_heading = False
        elif token.type == "fence":
            pass  # fence token holds the entire code block; intentionally not appended to body
        elif token.type == "inline":
            if inside_heading:
                current_heading = token.content
            else:
                current_body_parts.append(token.content)

    flush()

    if not heading_seen and content.strip():
        # No headings found: discard any "Introduction" section and use filename stem
        sections.clear()
        filename = Path(source_file).stem
        body = content.strip()
        sections.append(Section(
            heading=filename,
            body=body,
            source_file=source_file,
            tokens=lemmatize_text(body),
        ))

    return sections


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
            rel_path = str(md_file.relative_to(knowledge_path))
            sections.extend(parse_sections(content, rel_path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", md_file, exc)

    bm25 = BM25Okapi([s.tokens for s in sections]) if sections else None
    last_indexed = datetime.now(timezone.utc).isoformat()
    index_store.swap(sections, bm25, len(md_files), last_indexed)
    logger.info("Indexed %d sections from %d files", len(sections), len(md_files))
