#!/usr/bin/env python3
"""Bounded diagnostic trial for chunk ``c_012d2ff7efe77f40``.

Scope (locked by ``docs/experiments/quote-repair-v1-review/follow-up-c012d.md``):

* Issue exactly two direct ``POST /api/chat`` requests to the locally
  configured Ollama endpoint.
* No retries, no fallback, no manifest calls, no ``make_llm()``.
* Reconstruct the relation request with the empty mention surface that
  the original baseline used (entity phase returned zero validated
  mentions).
* Write the verbatim HTTP JSON responses plus full Ollama metadata to
  a fresh, timestamped directory under ``~/migration-incoming/``.
* Abort (exit non-zero) if that directory already exists, to prevent
  overwriting prior trials.

Run ``--dry-run`` to exercise the script end-to-end against an
in-process mock httpx transport without touching the network. The
live mode is only meant to be invoked once.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Third-party deps only.
import httpx
import jsonschema
import yaml

# ---------------------------------------------------------------------------
# Constants pinned to the original baseline (v2-pilot-03 / v2-multichunk-01).
# ---------------------------------------------------------------------------

CHUNK_ID = "c_012d2ff7efe77f40"
DOC_ID = "W3202422640"  # chunk text confirms this doc_id

MODEL = "qwen3:8b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_PATH = "/api/chat"

# Identical to ``pipeline_v2.run_extraction_v2`` defaults (T3a, #53):
MAX_TOKENS = 2048
TEMPERATURE = 0.0

# Mirrors ``src/extract/pipeline_v2.py::_render_system_v2`` byte-for-byte:
SYSTEM_PROMPT = (
    "You are an information-extraction assistant. Respond only with "
    "the JSON object requested by the user message. Do not include "
    "prose, commentary, or code fences."
)

# Output root requested by the reviewer.
OUTPUT_ROOT = Path(os.environ.get(
    "QUOTE_REPAIR_TRIAL_OUT",
    str(Path.home() / "migration-incoming"),
)).resolve()

# Source paths inside this worktree and the sibling pilot/multichunk dirs.
REPO_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_PATH = REPO_ROOT / "ontology.yaml"
ENTITY_PROMPT_PATH = REPO_ROOT / "prompts" / "extract_entities_v2.txt"
# The relation prompt in this branch (cb4... vs pilot-03's 1132...) DIFFERS
# from the pilot-03 artifact that produced the original baseline rejection.
# Per the follow-up note we MUST use the *original* prompt for the
# relation request; otherwise we would silently change the cache key.
PILOT_03_DIR = Path("/home/immo/migration-incoming/v2-pilot-03")
RELATION_PROMPT_PATH = PILOT_03_DIR / "prompts" / "extract_relations_v2.txt"
CHUNKS_JSONL = Path("/home/immo/migration-incoming/v2-multichunk-01/three_chunks.jsonl")
BASELINE_REJECTS = Path("/home/immo/migration-incoming/v2-multichunk-01/baseline.v2.rejects.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_compact() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_chunk(path: Path, chunk_id: str) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("chunk_id") == chunk_id:
            return rec
    raise SystemExit(f"chunk_id={chunk_id} not found in {path}")


def _load_baseline_reject(path: Path, chunk_id: str, phase: str) -> dict | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("chunk_id") == chunk_id and rec.get("phase") == phase:
            return rec
    return None


def _build_mention_list_block(mentions: list[dict]) -> str:
    """Render the mention surface as ``render_mention_list`` would.

    Mirrors ``src/extract/prompts.py::render_mention_list`` byte-for-byte
    for the empty case (the only case the diagnostic cares about).
    """
    if not mentions:
        return "(no same-chunk mentions available)"
    lines = []
    for m in mentions:
        lines.append(
            "- {mid}  entity_type={et}  surface_form={sf!r}  "
            "char_range=[{cs},{ce})".format(
                mid=m["mention_id"],
                et=m["entity_type"],
                sf=m["surface_form"],
                cs=m["evidence"]["char_start"],
                ce=m["evidence"]["char_end"],
            )
        )
    return "\n".join(lines)


def _build_user_messages(
    *,
    chunk_text: str,
    chunk_id: str,
    doc_id: str,
    ontology: Mapping[str, Any],
    entity_prompt: str,
    relation_prompt: str,
) -> tuple[str, str, list[dict]]:
    """Render the two user messages and return the empty mention surface.

    Returns ``(entity_user, relation_user, empty_mention_records)``.
    """
    allowed_entity_types = sorted(
        et["name"]
        for et in ontology.get("entity_types", [])
        if et.get("population_mode") == "extracted"
    )
    allowed_relation_types = sorted(
        rt["name"]
        for rt in ontology.get("relation_types", [])
        if rt.get("population_mode") == "extracted"
    )

    # entity: populate ``{allowed_entity_types}``, ``{entity_type_definitions}``,
    # ``{verified_examples}``, ``{chunk_text}``. We reuse
    # ``schema.verified_examples_block`` for both blocks (matches the
    # pilot pipeline's behaviour where both placeholders receive the
    # verified-examples block — see ``_run_entity_phase_v2``).
    from src.extract import schema as schema_mod
    import yaml as _yaml

    examples_block = schema_mod.verified_examples_block(dict(ontology))

    allowed_et_str = ", ".join(allowed_entity_types) if allowed_entity_types else "(none)"
    entity_user = entity_prompt.format(
        allowed_entity_types=allowed_et_str,
        entity_type_definitions=examples_block,
        verified_examples=examples_block,
        chunk_text=chunk_text,
    )

    # relation: empty mention list, as used by the original baseline
    # after the entity phase yielded zero validated mentions.
    empty_mention_records: list[dict] = []
    mention_list_str = _build_mention_list_block(empty_mention_records)

    relations_def_block = schema_mod.relation_definitions_block(dict(ontology))
    allowed_rt_str = ", ".join(allowed_relation_types) if allowed_relation_types else "(none)"

    relation_user = relation_prompt.format(
        allowed_relation_types=allowed_rt_str,
        relation_definitions=relations_def_block,
        chunk_id=chunk_id,
        doc_id=doc_id,
        mention_list=mention_list_str,
        chunk_text=chunk_text,
    )

    return entity_user, relation_user, empty_mention_records


def _build_payload(*, user: str) -> dict:
    """Build the exact ``/api/chat`` request body (matches the adapter)."""
    return {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "format": "json",
        "options": {
            "num_predict": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    }


# ---------------------------------------------------------------------------
# Network layer
# ---------------------------------------------------------------------------


class _MockTransport(httpx.BaseTransport):
    """In-process httpx transport that records every request."""

    def __init__(self, scripted_responses: list[httpx.Response]) -> None:
        self._scripted = list(scripted_responses)
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if not self._scripted:
            raise AssertionError("mock transport received an extra request")
        return self._scripted.pop(0)


@dataclasses.dataclass
class CallResult:
    phase: str
    status_code: int
    elapsed_ms: float
    request_payload_sha256: str
    request_payload: dict
    response_sha256: str
    response_bytes: bytes
    response_body: Any  # parsed JSON or None
    response_parse_error: str | None
    schema_valid: bool | None
    schema_errors: list[str]
    ollama_meta: dict
    model_output_text: str | None
    transport_error: str | None
    diff_vs_baseline: dict


def _do_call(
    *,
    phase: str,
    client: httpx.Client,
    payload: dict,
    schema: dict,
    baseline_record: dict | None,
) -> CallResult:
    payload_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload_sha = _sha256_bytes(payload_bytes)

    started = time.perf_counter()
    try:
        http_resp = client.post(
            OLLAMA_BASE_URL + OLLAMA_CHAT_PATH,
            content=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return CallResult(
            phase=phase,
            status_code=0,
            elapsed_ms=elapsed_ms,
            request_payload_sha256=payload_sha,
            request_payload=payload,
            response_sha256="",
            response_bytes=b"",
            response_body=None,
            response_parse_error=None,
            schema_valid=None,
            schema_errors=[],
            ollama_meta={},
            model_output_text=None,
            transport_error=f"{type(exc).__name__}: {exc}",
            diff_vs_baseline={},
        )

    response_bytes = http_resp.content
    response_sha = _sha256_bytes(response_bytes)
    body: Any = None
    parse_error: str | None = None
    try:
        body = json.loads(response_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    schema_valid: bool | None = None
    schema_errors: list[str] = []
    model_output_text: str | None = None
    ollama_meta: dict = {}
    if isinstance(body, dict):
        ollama_meta = {
            k: body[k]
            for k in (
                "model",
                "created_at",
                "done",
                "done_reason",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
                "prompt_eval_count",
                "eval_count",
            )
            if k in body
        }
        message = body.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                model_output_text = content

        if model_output_text is not None and parse_error is None:
            # Try parsing the model text per the same strict rule the
            # adapter applies (``_parse_model_json``).
            stripped = model_output_text.strip()
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                schema_valid = False
                schema_errors.append(f"json_decode: {exc.msg}")
            else:
                if not isinstance(parsed, dict):
                    schema_valid = False
                    schema_errors.append(
                        f"top_level_not_object: got {type(parsed).__name__}"
                    )
                else:
                    validator = jsonschema.Draft202012Validator(schema)
                    errs = sorted(validator.iter_errors(parsed), key=lambda e: e.path)
                    schema_errors = [e.message for e in errs]
                    schema_valid = not errs

    # Diff against the cached reject: compare verbatim stored text, plus
    # observed meta signals.
    diff: dict = {}
    if baseline_record is not None:
        cached = baseline_record.get("offending_record", {}).get("model_output", "")
        diff["baseline_phase"] = baseline_record.get("phase")
        diff["baseline_gate"] = baseline_record.get("gate")
        diff["baseline_attempt"] = baseline_record.get("attempt")
        diff["baseline_reason"] = baseline_record.get("reason")
        diff["baseline_model_output_len"] = len(cached) if isinstance(cached, str) else None
        diff["baseline_model_output_tail"] = (
            cached[-120:] if isinstance(cached, str) and cached else None
        )
        diff["live_model_output_len"] = (
            len(model_output_text) if isinstance(model_output_text, str) else None
        )
        diff["verbatim_match"] = (
            isinstance(cached, str) and isinstance(model_output_text, str)
            and cached == model_output_text
        )
        diff["prefix_match_len"] = 0
        if (
            isinstance(cached, str)
            and isinstance(model_output_text, str)
        ):
            a, b = cached, model_output_text
            n = min(len(a), len(b))
            i = 0
            while i < n and a[i] == b[i]:
                i += 1
            diff["prefix_match_len"] = i

    return CallResult(
        phase=phase,
        status_code=http_resp.status_code,
        elapsed_ms=elapsed_ms,
        request_payload_sha256=payload_sha,
        request_payload=payload,
        response_sha256=response_sha,
        response_bytes=response_bytes,
        response_body=body,
        response_parse_error=parse_error,
        schema_valid=schema_valid,
        schema_errors=schema_errors,
        ollama_meta=ollama_meta,
        model_output_text=model_output_text,
        transport_error=None,
        diff_vs_baseline=diff,
    )


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _write_outputs(out_dir: Path, *, entity: CallResult, relation: CallResult,
                   inputs_summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)

    # Verbatim raw response bodies (the most important artefacts).
    (out_dir / "entity.response.raw.json").write_bytes(entity.response_bytes)
    (out_dir / "relation.response.raw.json").write_bytes(relation.response_bytes)

    # The two request payloads that we actually sent (so we can replay
    # or audit them without rebuilding the prompts from scratch).
    (out_dir / "entity.request.json").write_text(
        json.dumps(entity.request_payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (out_dir / "relation.request.json").write_text(
        json.dumps(relation.request_payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    # Structured summary that includes the verified model text, the
    # parsed JSON (if any), the V2 schema verdict, and the Ollama meta.
    summary = {
        "trial": inputs_summary,
        "calls": {
            "entity": _summary_for(entity),
            "relation": _summary_for(relation),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Stable SHA256 manifest for audit.
    manifest = {
        "entity": {
            "request_sha256": entity.request_payload_sha256,
            "response_sha256": entity.response_sha256,
        },
        "relation": {
            "request_sha256": relation.request_payload_sha256,
            "response_sha256": relation.response_sha256,
        },
    }
    (out_dir / "sha256.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _summary_for(res: CallResult) -> dict:
    parsed_mentions_or_assertions: Any = None
    if isinstance(res.response_body, dict):
        message = res.response_body.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                try:
                    obj = json.loads(content.strip())
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict):
                    parsed_mentions_or_assertions = obj
    return {
        "status_code": res.status_code,
        "elapsed_ms": round(res.elapsed_ms, 3),
        "ollama_meta": res.ollama_meta,
        "model_output_text": res.model_output_text,
        "model_output_len": (
            len(res.model_output_text) if isinstance(res.model_output_text, str) else None
        ),
        "response_parse_error": res.response_parse_error,
        "schema_valid": res.schema_valid,
        "schema_errors": res.schema_errors,
        "parsed_top_level": parsed_mentions_or_assertions,
        "transport_error": res.transport_error,
        "diff_vs_baseline": res.diff_vs_baseline,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _build_v2_schemas(ontology: Mapping[str, Any]) -> tuple[dict, dict]:
    """Construct the V2 schemas from the local ontology, byte-for-byte
    the same way the V2 pipeline would at run time."""
    from src.extract import schema as schema_mod

    entity_schema = schema_mod.entity_pass_v2_schema(dict(ontology))
    relation_schema = schema_mod.relation_pass_v2_schema(dict(ontology))
    return entity_schema, relation_schema


def _assert_dry_run_quiescence(mock: _MockTransport) -> None:
    """Sanity check: after the run, no extra scripted responses may remain."""
    if mock._scripted:
        raise SystemExit(
            "dry-run: too many scripted responses left over "
            f"(expected 2, got {2 - len(mock._scripted)} consumed and "
            f"{len(mock._scripted)} still pending)"
        )


def run_dry_run(ontology: Mapping[str, Any], entity_prompt: str,
                relation_prompt: str, chunk_text: str) -> None:
    """Exercise the script with synthetic responses; never touches network."""
    print("[dry-run] starting in-process mock", flush=True)

    # Mimic the truncated entity output the original baseline saw.
    truncated_entity = (
        "{\n\n\"mentions\": [\n"
        "    {\n        \"entity_type\": \"dataset\",\n"
        "        \"surface_form\": \"CuratedTREC\",\n"
        "        \"canonical_form\": \"CuratedTREC\",\n"
        "        \"confidence\": \"stated\"\n    },\n"
        "    {\n        \"entity_type\": \"method\",\n"
        "        \"surface_form\": \"REAL"
    )
    entity_response = {
        "model": MODEL,
        "created_at": "2026-09-29T00:00:00Z",
        "message": {"role": "assistant", "content": truncated_entity},
        "done": True,
        "done_reason": "stop",
        "total_duration": 1_000_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 50_000_000,
        "eval_duration": 800_000_000,
        "prompt_eval_count": 158,
        "eval_count": 200,
    }

    # Mimic the empty-object relation response the baseline saw.
    relation_response = {
        "model": MODEL,
        "created_at": "2026-09-29T00:00:01Z",
        "message": {"role": "assistant", "content": "{}"},
        "done": True,
        "done_reason": "stop",
        "total_duration": 1_100_000_000,
        "load_duration": 100_000_000,
        "prompt_eval_duration": 50_000_000,
        "eval_duration": 200_000_000,
        "prompt_eval_count": 162,
        "eval_count": 4,
    }

    mock = _MockTransport(
        scripted_responses=[
            httpx.Response(200, json=entity_response),
            httpx.Response(200, json=relation_response),
        ]
    )
    entity_schema, relation_schema = _build_v2_schemas(ontology)
    baseline_entity = _load_baseline_reject(BASELINE_REJECTS, CHUNK_ID, "entity")
    baseline_relation = _load_baseline_reject(BASELINE_REJECTS, CHUNK_ID, "relation")

    entity_user, relation_user, _ = _build_user_messages(
        chunk_text=chunk_text,
        chunk_id=CHUNK_ID,
        doc_id=DOC_ID,
        ontology=ontology,
        entity_prompt=entity_prompt,
        relation_prompt=relation_prompt,
    )
    entity_payload = _build_payload(user=entity_user)
    relation_payload = _build_payload(user=relation_user)

    with httpx.Client(transport=mock, timeout=10.0) as client:
        entity_res = _do_call(
            phase="entity",
            client=client,
            payload=entity_payload,
            schema=entity_schema,
            baseline_record=baseline_entity,
        )
        relation_res = _do_call(
            phase="relation",
            client=client,
            payload=relation_payload,
            schema=relation_schema,
            baseline_record=baseline_relation,
        )

    _assert_dry_run_quiescence(mock)

    # The two POSTs we made are visible in mock.calls.
    assert len(mock.calls) == 2, f"expected exactly 2 POSTs, got {len(mock.calls)}"
    for call, label in zip(mock.calls, ("entity", "relation")):
        assert call.method == "POST", f"{label} call method is {call.method}"
        assert call.url.path == OLLAMA_CHAT_PATH, f"{label} url path is {call.url.path}"

    # Schema validation must mark the truncated entity invalid and the
    # empty-object relation invalid (assertions missing).
    assert entity_res.schema_valid is False, "entity must be schema-invalid under mock"
    assert relation_res.schema_valid is False, "relation must be schema-invalid under mock"

    print(
        "[dry-run] OK — both POSTs emitted, schema verdict applied, "
        "diff_vs_baseline populated, no network access.",
        flush=True,
    )


def run_live(ontology: Mapping[str, Any], entity_prompt: str,
             relation_prompt: str, chunk_text: str, out_dir: Path) -> int:
    """Issue exactly two POSTs to the local Ollama endpoint."""
    print(f"[live] writing outputs under {out_dir}", flush=True)

    entity_schema, relation_schema = _build_v2_schemas(ontology)
    baseline_entity = _load_baseline_reject(BASELINE_REJECTS, CHUNK_ID, "entity")
    baseline_relation = _load_baseline_reject(BASELINE_REJECTS, CHUNK_ID, "relation")

    entity_user, relation_user, empty_mentions = _build_user_messages(
        chunk_text=chunk_text,
        chunk_id=CHUNK_ID,
        doc_id=DOC_ID,
        ontology=ontology,
        entity_prompt=entity_prompt,
        relation_prompt=relation_prompt,
    )
    entity_payload = _build_payload(user=entity_user)
    relation_payload = _build_payload(user=relation_user)

    entity_res: CallResult
    relation_res: CallResult

    # --- call 1: entity -------------------------------------------------
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        entity_res = _do_call(
            phase="entity",
            client=client,
            payload=entity_payload,
            schema=entity_schema,
            baseline_record=baseline_entity,
        )
        if entity_res.transport_error:
            print(f"[live] entity transport error: {entity_res.transport_error}", flush=True)

        # --- call 2: relation (INDEPENDENT reconstruction) -------------
        relation_res = _do_call(
            phase="relation",
            client=client,
            payload=relation_payload,
            schema=relation_schema,
            baseline_record=baseline_relation,
        )
        if relation_res.transport_error:
            print(f"[live] relation transport error: {relation_res.transport_error}", flush=True)

    inputs_summary = {
        "chunk_id": CHUNK_ID,
        "doc_id": DOC_ID,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "format": "json",
        "stream": False,
        "system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "entity_prompt_sha256": _sha256_bytes(entity_prompt.encode("utf-8")),
        "entity_prompt_path": str(ENTITY_PROMPT_PATH),
        "relation_prompt_sha256": _sha256_bytes(relation_prompt.encode("utf-8")),
        "relation_prompt_path": str(RELATION_PROMPT_PATH),
        "chunk_text_sha256": _sha256_bytes(chunk_text.encode("utf-8")),
        "chunks_jsonl_sha256": _sha256_bytes(CHUNKS_JSONL.read_bytes()),
        "ontology_sha256": _sha256_bytes(ONTOLOGY_PATH.read_bytes()),
        "ontology_path": str(ONTOLOGY_PATH),
        "baseline_rejects_sha256": _sha256_bytes(BASELINE_REJECTS.read_bytes()),
        "mention_surface_in_relation_prompt": _build_mention_list_block(empty_mentions),
        "empty_mention_count": len(empty_mentions),
        "ollama_base_url": OLLAMA_BASE_URL,
        "trial_started_utc": _utc_compact(),
    }

    try:
        _write_outputs(
            out_dir,
            entity=entity_res,
            relation=relation_res,
            inputs_summary=inputs_summary,
        )
    except FileExistsError:
        print(
            f"[live] FATAL: output dir {out_dir} already exists; aborting "
            "to avoid overwriting prior trial artefacts.",
            file=sys.stderr,
        )
        return 2

    print(
        f"[live] wrote {out_dir}/{{entity,relation}}.response.raw.json, "
        "summary.json, sha256.json",
        flush=True,
    )

    # Exit non-zero if either call lacked a transport response.
    if entity_res.transport_error or relation_res.transport_error:
        return 3
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise payload + parsing + validation logic with mock responses; "
             "no network access, no output dir is created.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="explicit output dir; default: timestamped subdir under "
             "QUOTE_REPAIR_TRIAL_OUT (default ~/migration-incoming).",
    )
    args = parser.parse_args(argv)

    # Read every input ourselves so we can hash them before any model call.
    ontology = yaml.safe_load(_read_text(ONTOLOGY_PATH))
    entity_prompt = _read_text(ENTITY_PROMPT_PATH)
    relation_prompt = _read_text(RELATION_PROMPT_PATH)
    chunk = _load_chunk(CHUNKS_JSONL, CHUNK_ID)
    chunk_text = chunk["text"]

    if args.dry_run:
        run_dry_run(
            ontology=ontology,
            entity_prompt=entity_prompt,
            relation_prompt=relation_prompt,
            chunk_text=chunk_text,
        )
        return 0

    out_dir = args.out_dir or (OUTPUT_ROOT / f"quote_repair_v1_c012d2ff_trial_{_utc_compact()}")
    if out_dir.exists():
        print(
            f"FATAL: output dir {out_dir} already exists; aborting to avoid "
            "overwriting prior trial artefacts.",
            file=sys.stderr,
        )
        return 2

    return run_live(
        ontology=ontology,
        entity_prompt=entity_prompt,
        relation_prompt=relation_prompt,
        chunk_text=chunk_text,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))