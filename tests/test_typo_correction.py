"""End-to-end: a typo in a query should still find the right section, and
the response should signal that fuzzy correction fired."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_typo_in_query_finds_section(client):
    # "pyton" is a 1-edit typo of "python" — should still find the Python section
    resp = client.post("/ask", json={"question": "what is pyton"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is not None
    assert (
        "python" in (data["section"] or "").lower()
        or "python" in (data["answer"] or "").lower()
    )


def test_typo_response_signals_fuzzy_match(client):
    resp = client.post("/ask", json={"question": "what is pyton"})
    data = resp.json()
    # Either the message tells the caller fuzzy was used, or confidence is capped
    assert (data.get("message") is not None) or (data["confidence"] <= 0.6 + 1e-6)


def test_clean_query_does_not_trigger_fuzzy_message(client):
    resp = client.post("/ask", json={"question": "what is python"})
    data = resp.json()
    assert data.get("message") is None
