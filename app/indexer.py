import logging
from pathlib import Path

from markdown_it import MarkdownIt

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
