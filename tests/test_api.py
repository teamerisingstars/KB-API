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
