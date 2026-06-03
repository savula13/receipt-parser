import json
import time
from datetime import datetime, timezone
from enum import Enum

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, RootModel, field_validator
import anthropic

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = anthropic.Anthropic()

# Model parameters live at module scope so the call site stays clean and the
# values are easy to find and tune in one place.
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

# Observability: we append human-readable markdown entries (raw responses, token
# usage, latency, failures) so a run can be reconstructed after the fact.
LOG_FILE = "agent_log.md"


def _write_log(block: str) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(block)


def log_model_call(raw: str, usage, latency_s: float) -> None:
    """Record one model round-trip: latency, token usage, and the raw response."""
    ts = datetime.now(timezone.utc).isoformat()
    _write_log(
        f"## {ts} — model call\n\n"
        f"- latency: {latency_s:.2f}s\n"
        f"- tokens: input={usage.input_tokens}, output={usage.output_tokens}\n\n"
        f"**Raw model response:**\n```\n{raw}\n```\n\n---\n\n"
    )


def log_failure(stage: str, detail: str, raw: str = "") -> None:
    """Record a failure (the stage, why, and the last raw output) for diagnosis."""
    ts = datetime.now(timezone.utc).isoformat()
    _write_log(
        f"## {ts} — FAILURE: {stage}\n\n"
        f"- detail: {detail}\n\n"
        f"**Last raw response:**\n```\n{raw}\n```\n\n---\n\n"
    )


# --- Schema: the contract we enforce on the model's output ---------------------

class Category(str, Enum):
    """The only categories we accept. Anything else is invalid output."""
    meals = "meals"
    travel = "travel"
    software = "software"
    office_supplies = "office_supplies"
    other = "other"


# Upper bounds: untrusted output shouldn't be able to hand downstream consumers
# absurd values (unbounded strings, runaway counts, or nonsensical amounts).
MAX_ITEM_LEN = 200
MAX_ITEMS = 200
MAX_AMOUNT = 1_000_000  # well above any realistic single line item


class LineItem(BaseModel):
    """A single validated expense line."""
    # min_length rejects empty names; max_length caps unbounded strings.
    item: str = Field(min_length=1, max_length=MAX_ITEM_LEN)
    # ge=0 rejects negatives; le caps absurd values; allow_inf_nan=False rejects
    # NaN/Infinity, which Python's json parser would otherwise accept.
    amount: float = Field(ge=0, le=MAX_AMOUNT, allow_inf_nan=False)
    category: Category

    @field_validator("item")
    @classmethod
    def clean_item(cls, v: str) -> str:
        # Strip control characters (e.g. NUL, escape sequences) that could break
        # logs/terminals or be used for injection downstream, then trim.
        cleaned = "".join(ch for ch in v if ch.isprintable() or ch == " ").strip()
        if not cleaned:
            raise ValueError("item is empty after removing control characters")
        return cleaned

    @field_validator("amount")
    @classmethod
    def round_amount(cls, v: float) -> float:
        # Money has 2 decimals; normalize so downstream doesn't see 34.199999.
        return round(v, 2)


class LineItems(RootModel[list[LineItem]]):
    """Wrapper so we can validate the whole list (the model returns a JSON array)."""
    # Cap list length so a malformed/huge response can't balloon the payload.
    root: list[LineItem] = Field(max_length=MAX_ITEMS)


# --- Request model -------------------------------------------------------------

class ReceiptRequest(BaseModel):
    # Bound the input too: limits token cost and is a basic abuse/DoS guard.
    receipt_text: str = Field(min_length=1, max_length=10_000)


# --- Errors --------------------------------------------------------------------

class ModelOutputError(Exception):
    """The model responded, but the content was unusable (empty / non-text).

    Distinct from anthropic.APIError (we couldn't reach the model at all) so the
    caller can tell *where* things went wrong."""


def error_response(status: int, stage: str, detail: str, raw: str | None = None):
    """Build a consistent, structured error body. `stage` says where it failed.

    Failures are logged here so every error path is captured in one place."""
    log_failure(stage, detail, raw or "")
    content = {"error": stage, "detail": detail}
    if raw is not None:
        content["raw"] = raw
    return JSONResponse(status_code=status, content=content)


# --- Prompt --------------------------------------------------------------------

# We constrain the output tightly: JSON array only, fixed keys, fixed category set.
# This is what turns free-form model text into something we can validate.
SYSTEM_PROMPT = (
    "You are a receipt parser. Extract each line item from the receipt and "
    "return ONLY a JSON array. Each element must be an object with exactly these keys:\n"
    '  "item": the item name (string)\n'
    '  "amount": the amount as a number (no currency symbol)\n'
    '  "category": one of "meals", "travel", "software", "office_supplies", "other"\n'
    "Choose the closest category; use \"other\" if none fit. "
    "Do not include any text, explanation, or markdown fences outside the JSON array."
)


# Corrective-retry prompt. We show the model exactly what it returned and why we
# rejected it, then ask for a clean correction. The fenced sections make the bad
# response unambiguous and separate from the instructions.
CORRECTION_TEMPLATE = (
    "Your previous response could not be used.\n\n"
    "=== ORIGINAL RECEIPT ===\n{receipt}\n\n"
    "=== YOUR PREVIOUS RESPONSE (REJECTED) ===\n{bad}\n\n"
    "=== WHY IT WAS REJECTED ===\n{error}\n\n"
    "Return a corrected response that is ONLY a JSON array matching the required "
    "schema (keys: item, amount as a number, category from the allowed set). "
    "Do not include any text outside the JSON array."
)


def call_model(user_content: str) -> str:
    """Send a user message to the model and return the raw text response.

    Takes arbitrary user content (the receipt on the first try, a correction
    prompt on retry) so the same call path is reused for both."""
    start = time.perf_counter()
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_content}
        ],
    )
    latency_s = time.perf_counter() - start

    # Don't assume the content list is non-empty or that the first block is text.
    content = message.content
    text = content[0].text if content and content[0].type == "text" else ""

    # Log every round-trip (including empty ones) with usage + latency.
    log_model_call(text, message.usage, latency_s)

    if not text:
        raise ModelOutputError("Model returned no text content")
    return text


def parse_and_validate(raw: str) -> list[LineItem]:
    """Parse raw model text as JSON and validate it against our schema.

    Raises json.JSONDecodeError (not JSON) or pydantic ValidationError, a
    ValueError subclass (missing fields, wrong types, bad category)."""
    # parse_constant rejects NaN/Infinity/-Infinity, which json.loads accepts by
    # default — we don't want those reaching the schema or downstream consumers.
    parsed = json.loads(raw, parse_constant=_reject_json_constant)
    return LineItems.model_validate(parsed).root


def _reject_json_constant(token: str):
    raise ValueError(f"Disallowed JSON constant in model output: {token}")


# Output problems that warrant one corrective retry: empty/non-text response,
# non-JSON text, or schema failures (missing fields, wrong types, bad category).
# API errors are NOT in here — a failure to reach the model isn't fixable by
# re-prompting, so it returns immediately.
OUTPUT_ERRORS = (ModelOutputError, json.JSONDecodeError, ValueError)


MAX_ATTEMPTS = 2  # initial attempt + one corrective retry


@app.post("/parse")
def parse_receipt(request: ReceiptRequest):
    # The user message for attempt 1 is the receipt; on retry it becomes a
    # correction prompt built from the previous bad output.
    user_content = request.receipt_text
    raw = ""        # last raw model output (empty if the model returned nothing)
    last_error = ""

    for attempt in range(MAX_ATTEMPTS):
        raw = ""  # reset so a ModelOutputError reports empty, not a stale value
        try:
            raw = call_model(user_content)
            items = parse_and_validate(raw)
            return {"result": [i.model_dump(mode="json") for i in items]}
        except anthropic.APIError as e:
            # Can't reach the model — re-prompting won't help, so return now.
            return error_response(502, "model_unavailable",
                                  f"Could not get a response from the model: {e}")
        except OUTPUT_ERRORS as e:
            # Output problem (empty / non-JSON / schema). Build a correction and
            # let the loop retry; if this was the last attempt, fall through.
            last_error = str(e)
            user_content = CORRECTION_TEMPLATE.format(
                receipt=request.receipt_text, bad=raw, error=last_error,
            )

    # All attempts exhausted — give up with a structured error for diagnosis.
    return error_response(422, "invalid_model_output", last_error, raw=raw)
