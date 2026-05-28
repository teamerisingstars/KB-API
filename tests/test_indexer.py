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
