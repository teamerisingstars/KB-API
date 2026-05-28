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
