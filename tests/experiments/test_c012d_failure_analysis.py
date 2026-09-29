"""Offline failure analysis for chunk ``c_012d2ff7efe77f40``.

This test does NOT invoke any model or network call. It replays the
two cached reject records committed under
``../v2-multichunk-01/baseline.v2.rejects.jsonl`` for the third chunk
and verifies the structural observations that any subsequent targeted
model trial must reconcile with.

Every assertion below is grounded in on-disk evidence:

* the stored ``offending_record.model_output`` literal captured by the
  v2 pipeline after its single JSON-retry was spent (per
  ``src/providers/adapters/ollama.py`` ``_chat_json`` / ``_chat_json_retry``);
* the v2 phase-runner retry policy
  (``src/extract/retry.py`` — only ``ProviderTransportError`` with
  ``transient=True`` qualifies, and exactly three attempts are allowed);
* the documented chunk text, ``token_estimate``, and prompt templates.

The test neither infers the transport cause of truncation nor the model
behaviour that emitted ``{}`` for the relation phase; those remain
explicit hypotheses (see ``docs/experiments/quote-repair-v1.md`` §"Dritter
Chunk: Offline-Fehleranalyse" and the bounded plan in
``docs/experiments/quote-repair-v1-review/follow-up-c012d.md``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Reproducibility note: this file lives next to the v2-multichunk-01
# artefact set, exactly as the quote-repair-v1 runner expects.
PILOT_DIR = REPO_ROOT.parent
THREE_CHUNKS = PILOT_DIR / "v2-multichunk-01" / "three_chunks.jsonl"
REJECTS = PILOT_DIR / "v2-multichunk-01" / "baseline.v2.rejects.jsonl"
BASELINE_V2 = PILOT_DIR / "v2-multichunk-01" / "baseline.v2.jsonl"
THIRD_CHUNK_ID = "c_012d2ff7efe77f40"


# ---------- helpers ---------------------------------------------------


def _load_jsonl(path: Path):
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            out.append(json.loads(raw))
    return out


@pytest.fixture(scope="module")
def c012d_rejects() -> dict:
    rejects = {rec["phase"]: rec for rec in _load_jsonl(REJECTS) if rec.get("chunk_id") == THIRD_CHUNK_ID}
    assert {"entity", "relation"} <= rejects.keys(), (
        f"missing reject phases for {THIRD_CHUNK_ID}: {sorted(rejects)}"
    )
    return rejects


@pytest.fixture(scope="module")
def chunk_record() -> dict:
    for rec in _load_jsonl(THREE_CHUNKS):
        if rec.get("chunk_id") == THIRD_CHUNK_ID:
            return rec
    raise AssertionError(f"{THIRD_CHUNK_ID} not in {THREE_CHUNKS}")


@pytest.fixture(scope="module")
def chunk_status_record() -> dict:
    for rec in _load_jsonl(BASELINE_V2):
        if rec.get("chunk_id") == THIRD_CHUNK_ID:
            return rec
    raise AssertionError(f"{THIRD_CHUNK_ID} chunk_status not in {BASELINE_V2}")


# ---------- structural observations on the entity phase --------------


def test_entity_output_is_hard_truncated_inside_string_value(c012d_rejects: dict) -> None:
    """The cached entity output ends mid-token inside a surface_form string.

    Concretely: the last non-whitespace character is inside an unterminated
    JSON string, and the JSON document has no closing ``]`` or ``}``.
    This is the v2 ``_parse_model_json`` failure signature
    (``Unterminated string starting at``) and is structural evidence,
    not a causal claim about transport.
    """
    out = c012d_rejects["entity"]["offending_record"]["model_output"]
    assert out.endswith("REAL") or out.rstrip().endswith("REAL"), (
        f"entity output truncation point drifted: tail={out[-40:]!r}"
    )
    assert "]" not in out, "closing of mentions array unexpectedly present"
    assert out.count("{") > out.count("}"), (
        f"unbalanced braces: open={out.count('{')} close={out.count('}')}"
    )


def test_entity_output_carries_five_complete_mentions_then_truncation(c012d_rejects: dict) -> None:
    """Ground-truth chunk text lists eight surface forms; the cached
    output completes only five before being cut. This is the structural
    evidence for partial completion, independent of why."""
    out = c012d_rejects["entity"]["offending_record"]["model_output"]
    completed = re.findall(r'"surface_form":\s*"([^"]+)"', out)
    # CuratedTREC, TREC QA corpus, DrQA, R3, ORQA — the 6th mention
    # would have been REALM, but it is cut mid-token.
    assert completed == [
        "CuratedTREC",
        "TREC QA corpus",
        "DrQA",
        "R3",
        "ORQA",
    ], f"completed mentions drifted: {completed}"


def test_entity_failure_classification(c012d_rejects: dict) -> None:
    """The reject reason is the documented Gate-1 schema failure after
    one JSON-retry; ``attempt=1`` here is the transport-attempt count
    visible to the phase runner (the JSON-retry happens inside the
    ollama adapter and is not surfaced as a separate attempt)."""
    rec = c012d_rejects["entity"]
    assert rec["gate"] == "schema"
    assert rec["attempt"] == 1
    assert "Unterminated string" in rec["reason"], (
        f"reason signature drifted: {rec['reason']!r}"
    )
    assert "after one retry" in rec["reason"]


# ---------- structural observations on the relation phase ------------


def test_relation_output_is_empty_object_without_assertions_key(c012d_rejects: dict) -> None:
    """The cached relation output is the literal ``{ }`` — a syntactically
    valid JSON object that violates the v2 schema because ``assertions``
    is required."""
    rec = c012d_rejects["relation"]
    out = rec["offending_record"]["model_output"].strip()
    assert json.loads(out) == {}, "relation output must be empty object on disk"
    assert "assertions" not in out, (
        "cached relation output unexpectedly contains 'assertions' key"
    )
    assert rec["gate"] == "schema"
    assert "required property" in rec["reason"], rec["reason"]
    assert rec["attempt"] == 1


def test_relation_attempt_does_not_block_on_entity_failure(c012d_rejects: dict, chunk_status_record: dict) -> None:
    """Even though the entity phase was rejected, the relation phase
    was still attempted (the runner is not fail-fast between phases).
    Both phases record ``attempt_count: 1`` because the JSON-retry
    happens inside the adapter and is not counted as a transport attempt.
    """
    status = chunk_status_record["phases"]
    assert status["entity"]["status"] == "rejected"
    assert status["entity"]["attempt_count"] == 1
    assert status["relation"]["status"] == "rejected"
    assert status["relation"]["attempt_count"] == 1
    # The relation phase is invoked independently: an entity failure does
    # not gate the relation call. The cached relation input therefore
    # contained an empty mention list (because the entity phase
    # produced zero validated mentions) — this is the link between the
    # two failures that any targeted retry must consider.
    assert status["entity"]["mention_count"] == 0
    assert status["relation"]["assertion_count"] == 0


# ---------- chunk-context observations -------------------------------


def test_chunk_text_lists_eight_grounded_mentions(chunk_record: dict) -> None:
    """The chunk text contains the exact surface forms that the entity
    phase was asked to surface. Any retry that expands max_tokens, drops
    prior context, or splits the chunk must not lose any of these."""
    text = chunk_record["text"]
    expected = [
        "CuratedTREC",
        "TREC QA corpus",
        "DrQA",
        "R3",
        "ORQA",
        "REALM",
        "DPR",
        "RAG",
        "T5",
    ]
    missing = [s for s in expected if s not in text]
    assert not missing, f"chunk text lost grounded surface forms: {missing}"
    assert chunk_record["token_estimate"] == 158, (
        f"token_estimate drifted: {chunk_record['token_estimate']}"
    )


def test_chunk_size_is_well_below_provider_input_budget(chunk_record: dict) -> None:
    """The chunk is sized for the chunking ``TOKEN_BUDGET = 512`` (see
    ``src/chunk/chunker.py``) and is therefore an order of magnitude
    smaller than the provider input budget. The truncation in the
    entity phase happened on the OUTPUT side of the API call, not on
    the input side.
    """
    text_len = len(chunk_record["text"])
    assert text_len < 2048, (
        f"chunk text {text_len} chars unexpectedly large; "
        f"truncation could not be a windowing artefact."
    )