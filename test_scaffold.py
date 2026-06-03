"""Phase 1 tests for the receipt parser.

Run with:  pytest test_scaffold.py -v
The real-model test is skipped automatically if ANTHROPIC_API_KEY is unset.
"""
import os
from unittest.mock import patch

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

import scaffold
from scaffold import app

client = TestClient(app)

# The README example input/output, reused by both tests.
EXAMPLE_RECEIPT = (
    "Uber Eats       $34.20\n"
    "AWS invoice     $412.00\n"
    "Office Depot    $28.50\n"
    "Delta Airlines  $890.00"
)
VALID_CATEGORIES = {c.value for c in scaffold.Category}


def test_parse_with_mocked_model():
    """Endpoint returns validated, structured line items for well-formed model JSON.

    We patch call_model so the test is fast, deterministic, and free: it exercises
    our parsing/validation path, not the model."""
    fake_response = (
        '[{"item": "Uber Eats", "amount": 34.20, "category": "meals"},'
        ' {"item": "AWS invoice", "amount": 412.00, "category": "software"}]'
    )
    with patch.object(scaffold, "call_model", return_value=fake_response):
        resp = client.post("/parse", json={"receipt_text": EXAMPLE_RECEIPT})

    assert resp.status_code == 200
    items = resp.json()["result"]
    assert items == [
        {"item": "Uber Eats", "amount": 34.20, "category": "meals"},
        {"item": "AWS invoice", "amount": 412.00, "category": "software"},
    ]


# --- Phase 2: structured error handling ---------------------------------------

def test_model_api_error_returns_502():
    """A failure reaching the model becomes a structured 502, not an unhandled 500."""
    api_error = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
    with patch.object(scaffold, "call_model", side_effect=api_error):
        resp = client.post("/parse", json={"receipt_text": EXAMPLE_RECEIPT})

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "model_unavailable"
    assert "detail" in body


def test_unparseable_output_returns_422():
    """Model output that isn't valid JSON becomes a structured 422 with the raw text."""
    with patch.object(scaffold, "call_model", return_value="Sure! Here are your items..."):
        resp = client.post("/parse", json={"receipt_text": EXAMPLE_RECEIPT})

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "invalid_model_output"
    assert body["raw"] == "Sure! Here are your items..."


def test_invalid_category_returns_422():
    """Valid JSON but an out-of-schema category still fails validation (untrusted output)."""
    bad = '[{"item": "Coffee", "amount": 4.5, "category": "snacks"}]'
    with patch.object(scaffold, "call_model", return_value=bad):
        resp = client.post("/parse", json={"receipt_text": EXAMPLE_RECEIPT})

    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_model_output"


# --- Phase 3: corrective retry ------------------------------------------------

def test_retry_recovers_after_one_bad_response():
    """A malformed first response triggers exactly one retry; the well-formed
    second response is returned. Asserts call_model is called twice (initial +
    one retry) and the endpoint returns 200 with the corrected items."""
    bad = "Here are the items, not JSON"
    good = '[{"item": "Coffee", "amount": 4.5, "category": "meals"}]'
    with patch.object(scaffold, "call_model", side_effect=[bad, good]) as mock:
        resp = client.post("/parse", json={"receipt_text": EXAMPLE_RECEIPT})

    assert mock.call_count == 2  # initial attempt + exactly one corrective retry
    assert resp.status_code == 200
    assert resp.json()["result"] == [
        {"item": "Coffee", "amount": 4.5, "category": "meals"}
    ]


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live model call",
)
def test_parse_with_real_model():
    """End-to-end: hit the live model and confirm the output satisfies our schema.

    We assert on the contract (shape + valid categories), not exact category
    choices, since model judgment can vary."""
    resp = client.post("/parse", json={"receipt_text": EXAMPLE_RECEIPT})

    assert resp.status_code == 200, resp.text
    items = resp.json()["result"]
    assert len(items) == 4
    for entry in items:
        assert set(entry.keys()) == {"item", "amount", "category"}
        assert isinstance(entry["amount"], (int, float))
        assert entry["category"] in VALID_CATEGORIES
