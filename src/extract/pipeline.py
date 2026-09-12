"""Extraction pipeline orchestrator (#23).

Implements ``docs/specs/extraction-contract.md`` §§2–§13 verbatim:

* ``run_extraction`` is the library entry point: builds the prompt
  artifacts, loads the chunks, drives entity + relation passes
  through ``LLMProvider.generate_json``, validates against the three
  gates, writes the canonical JSONL plus manifest.
* The CLI entry ``python -m src.extract`` lives in ``__main__.py``.
* Resumability is implemented as a per-chunk status table that the
  pipeline reads from disk before processing any chunk; chunks whose
  ``chunk_status.extraction_complete == true`` are skipped, others
  are re-run and their records replaced.
* Budget abort is caught at the chunk boundary; partial records for
  the in-progress chunk are kept and the run exits non-zero.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.extract import (
    ids as ids_module,
    prompts as prompts_module,
    schema as schema_module,
    validation as validation_module,
)
from src.extract.retry import with_transport_retry
from src.providers import BudgetExceeded, LLMProvider, Usage
from src.providers.cache import CACHE_FORMAT_VERSION, extraction_cache_scope
from src.providers.protocol import LLMValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "derived" / "chunks.jsonl"
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "ontology.yaml"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "data" / "extractions.jsonl"
DEFAULT_OUTPUT_REJECTS = REPO_ROOT / "data" / "extractions.rejects.jsonl"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "data" / "extractions.manifest.json"


# --- preflight types -------------------------------------------------


@dataclass(frozen=True)
class PreflightError(RuntimeError):
    """Raised at startup when a hard preflight gate fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class PreflightReport:
    chunks_path: Path
    chunks_sha256: str
    ontology_path: Path
    ontology_sha256: str
    ontology_version: int
    schema_version: int
    prompt_template_files: list
    prompt_template_sha256: list
    provider_name: str
    provider_model: str
    cache_format_version: str
    max_tokens: int
    temperature: float


# --- run identity / manifest -----------------------------------------


def _utc_now_compact() -> str:
    """Return a UTC timestamp suitable for the run_id suffix.

    Format: ``YYYYMMDDTHHMMSSZ``. Spec §13.1 expects the same
    timestamp shape the prior manifest carried so a resume can record
    ``resumed_from_run_id``; second precision is sufficient.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_fingerprint(*, prompt_sha, ontology_sha: str, chunks_sha: str) -> str:
    """A short fingerprint to disambiguate runs sharing a wall-clock second."""
    blob = "|".join(prompt_sha) + "|" + ontology_sha + "|" + chunks_sha
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _make_run_id(prompt_sha, ontology_sha: str, chunks_sha: str) -> str:
    return _utc_now_compact() + "-" + _run_fingerprint(
        prompt_sha=prompt_sha, ontology_sha=ontology_sha, chunks_sha=chunks_sha
    )


@dataclass
class RunState:
    """Mutable per-run state assembled during :func:`run_extraction`."""

    run_id: str
    resumed_from_run_id: str | None
    preflight: PreflightReport
    started_at: str
    ended_at: str | None = None
    cumulative_tokens: int = 0
    cumulative_cost_usd: float = 0.0
    chunk_status_counts: dict = field(default_factory=dict)
    reject_counts: dict = field(default_factory=dict)

    def record_usage(self, usage: Usage) -> None:
        if usage.input_tokens is not None:
            self.cumulative_tokens += usage.input_tokens
        if usage.output_tokens is not None:
            self.cumulative_tokens += usage.output_tokens
        if usage.cost_usd is not None:
            self.cumulative_cost_usd += usage.cost_usd

    def as_manifest(self) -> dict:
        return {
            "run_id": self.run_id,
            "resumed_from_run_id": self.resumed_from_run_id,
            "provider": self.preflight.provider_name,
            "model": self.preflight.provider_model,
            "ontology_version": self.preflight.ontology_version,
            "ontology_sha256": self.preflight.ontology_sha256,
            "prompt_template_files": list(self.preflight.prompt_template_files),
            "prompt_template_sha256": list(self.preflight.prompt_template_sha256),
            "schema_version": self.preflight.schema_version,
            "chunks_path": str(self.preflight.chunks_path),
            "chunks_sha256": self.preflight.chunks_sha256,
            "wall_clock_started_at": self.started_at,
            "wall_clock_ended_at": self.ended_at,
            "cumulative_tokens": self.cumulative_tokens,
            "cumulative_cost_usd": round(self.cumulative_cost_usd, 8),
            "max_tokens": self.preflight.max_tokens,
            "temperature": self.preflight.temperature,
            "cache_format_version": self.preflight.cache_format_version,
        }


# --- canonical writers -----------------------------------------------


def _atomic_rewrite_jsonl(path: Path, lines: Iterable[str]) -> None:
    """Rewrite ``path`` atomically with the supplied ``lines``.

    Each line is written UTF-8 with a trailing newline. Uses
    ``NamedTemporaryFile`` + ``os.replace`` so a crash or interrupt
    cannot leave a syntactically corrupted canonical file (spec
    §3, "atomic/rewrite-safe output handling").
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with io.open(fd, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically write a JSON manifest to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with io.open(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _read_existing_records(path: Path):
    """Load an existing ``extractions.jsonl`` into in-memory maps.

    Returns:

    * ``mentions_by_chunk``: ``{chunk_id: [mention_record, ...]}``
    * ``assertions_by_chunk``: ``{chunk_id: [assertion_record, ...]}``
    * ``chunk_status_by_chunk``: ``{chunk_id: chunk_status_record}``

    Records are sorted deterministically within each list per spec
    §3 ("records are sorted by ``(chunk_id, kind, mention_id or
    assertion_id)`` lexicographically").
    """
    mentions: dict = {}
    assertions: dict = {}
    statuses: dict = {}
    if not path.is_file():
        return mentions, assertions, statuses
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        cid = rec.get("chunk_id")
        if not isinstance(cid, str):
            continue
        if kind == "mention":
            mentions.setdefault(cid, []).append(rec)
        elif kind == "assertion":
            assertions.setdefault(cid, []).append(rec)
        elif kind == "chunk_status":
            statuses[cid] = rec
    for cid in mentions:
        mentions[cid].sort(key=lambda r: r.get("mention_id", ""))
    for cid in assertions:
        assertions[cid].sort(key=lambda r: r.get("assertion_id", ""))
    return mentions, assertions, statuses


def _is_complete(status: dict | None) -> bool:
    return bool(status) and status.get("extraction_complete") is True


# --- preflight -------------------------------------------------------


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_chunks(path: Path) -> list:
    if not path.is_file():
        raise PreflightError(f"chunks file not found: {path}")
    out: list = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        out.append(json.loads(line))
    if not out:
        raise PreflightError(f"chunks file is empty: {path}")
    return out


def preflight(
    *,
    chunks_path: Path,
    ontology_path: Path,
    prompts_dir: Path,
    provider: LLMProvider,
    max_tokens: int,
    temperature: float,
):
    """Read inputs, verify presence, build schemas, return state.

    Raises :class:`PreflightError` on any hard gate failure.
    """
    if not chunks_path.is_file():
        raise PreflightError(f"chunks file missing: {chunks_path}")
    if not ontology_path.is_file():
        raise PreflightError(f"ontology file missing: {ontology_path}")

    prompt_artifacts = prompts_module.verify_prompt_artifacts(prompts_dir)
    ontology = schema_module.load_ontology(ontology_path)
    if not isinstance(ontology, dict):
        raise PreflightError("ontology.yaml did not parse as a mapping")

    entity_types = schema_module.extracted_entity_types(ontology)
    relation_types = schema_module.extracted_relation_types(ontology)
    if not entity_types:
        raise PreflightError("ontology has no extracted entity types")
    if not relation_types:
        raise PreflightError("ontology has no extracted relation types")

    report = PreflightReport(
        chunks_path=chunks_path,
        chunks_sha256=_file_sha256(chunks_path),
        ontology_path=ontology_path,
        ontology_sha256=_file_sha256(ontology_path),
        ontology_version=int(ontology.get("version", 0)) or 0,
        schema_version=schema_module.SCHEMA_VERSION,
        prompt_template_files=[
            prompt_artifacts[prompts_module.ENTITY_PROMPT_FILE]["filename"],
            prompt_artifacts[prompts_module.RELATION_PROMPT_FILE]["filename"],
        ],
        prompt_template_sha256=[
            prompt_artifacts[prompts_module.ENTITY_PROMPT_FILE]["sha256"],
            prompt_artifacts[prompts_module.RELATION_PROMPT_FILE]["sha256"],
        ],
        provider_name=provider.name,
        provider_model=provider.model,
        cache_format_version=CACHE_FORMAT_VERSION,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return report, ontology, prompt_artifacts

# --- phase runners ---------------------------------------------------


@dataclass
class PhaseOutcome:
    """Result of one phase (entity or relation) for one chunk.

    Authoritative semantics (spec §3.1.3, §9):

    ``status == "validated"`` means the provider call completed and
    ``generate_json`` returned schema-valid output. A validated phase
    may still carry Gate 2 (ontology) or Gate 3 (evidence)
    per-record rejects, so ``failure_count`` may be > 0; a validated
    phase with ``failure_count > 0`` does NOT pass all three gates
    and therefore prevents ``extraction_complete`` (spec §9 rule 3).

    ``status == "rejected"`` means a phase-level Gate 1 (schema)
    failure, or another phase-level failure such as exhausted
    validation. When ``status == "rejected"``, ``failure_count`` is
    pinned to 0 and the failure is recorded in ``reject_records``
    only; this is because Gate 1 fails the entire phase rather than
    a per-record reject.

    ``failure_count`` counts ONLY Gate 2 (ontology) + Gate 3
    (evidence) per-record rejects; Gate 1 failures do not enter it.

    ``reject_records`` carries every reject entry the phase produced
    (schema-level + ontology-level + evidence-level) for the reject
    log.
    """

    status: str  # "validated" | "rejected"
    attempt_count: int
    mention_count: int = 0
    assertion_count: int = 0
    failure_count: int = 0
    reject_records: list = field(default_factory=list)


def _render_system() -> str:
    """A small system preamble shared by both phases.

    Spec §10.1: ``system_prompt`` is the rendered system message —
    extraction's two templates each carry a small system preamble
    explaining the role and the schema.
    """
    return (
        "You are a deterministic JSON extractor for a scientific-paper corpus. "
        "Respond ONLY with a single JSON object matching the supplied schema. "
        "Do not add prose, commentary, or markdown fences."
    )


def _call_with_retry(op, *, sleep):
    """Wrap ``op()`` in the spec §12.1 transport retry policy.

    ``LLMValidationError`` and other non-transient exceptions
    propagate (spec §12.2). Transient transport failures are retried
    up to MAX_ATTEMPTS with the (2s, 4s) schedule.

    ``sleep`` is the injectable sleep callable the spec §12.1 helper
    requires; production code passes ``time.sleep``, tests pass a
    recorder.

    After this helper returns (or raises), the caller sees the exact
    final exception object for retry-exhausted transient
    ``ProviderTransportError``, for non-transient
    ``ProviderTransportError``, and for arbitrary unexpected
    exceptions. The phase runners rely on that propagation to
    surface transport / internal errors verbatim per spec §12.1 /
    §12.2.
    """
    return with_transport_retry(op, sleep=sleep)


def _run_entity_phase(
    *,
    chunk: dict,
    ontology: dict,
    prompt_artifact: dict,
    provider: LLMProvider,
    max_tokens: int,
    temperature: float,
    schema: dict,
    run_state: RunState,
):
    """Drive the entity pass for one chunk.

    Returns ``(outcome, validated_mentions)``. The outcome's
    ``mention_count`` is the count of records that survived all three
    gates; reject records go to ``outcome.reject_records`` and are
    also written to the reject log by the caller.
    """
    chunk_id = chunk["chunk_id"]
    doc_id = chunk["doc_id"]
    text = chunk["text"]
    sha = validation_module.source_text_sha256(text)

    allowed_lists = schema_module.build_allowed_name_lists(ontology)
    entity_user = prompts_module.render_entity_prompt(
        template=prompt_artifact["text"],
        allowed_entity_types=allowed_lists["entity_types"],
        entity_type_definitions=schema_module.verified_examples_block(ontology),
        verified_examples=schema_module.verified_examples_block(ontology),
        chunk_text=text,
    )

    extraction_scope_kwargs = dict(
        chunk_id=chunk_id,
        source_text_sha256=sha,
        ontology_version=str(run_state.preflight.ontology_version),
        ontology_sha256=run_state.preflight.ontology_sha256,
        prompt_template_filename=prompt_artifact["filename"],
        prompt_template_content_sha256=prompt_artifact["sha256"],
    )

    attempts = 0
    parsed: dict | None = None
    usage = Usage()

    def _do_call():
        nonlocal attempts
        attempts += 1
        return provider.generate_json(
            _render_system(),
            entity_user,
            schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # Spec §5.5a: every generate_json call for extraction enters the
    # strict-superset identity scope so the provider-side cache key
    # is a strict superset of the canonical §5.1 key.
    #
    # Defect 1 (transport / internal error propagation): do NOT add a
    # broad ``except Exception`` around the provider call here. The
    # only exception we route as a phase-level reject is
    # ``LLMValidationError`` (Gate 1, schema). ``BudgetExceeded``,
    # ``ProviderTransportError`` (transient retry-exhausted and
    # non-transient), and arbitrary unexpected exceptions propagate
    # unchanged so the run aborts with the real failure.
    with extraction_cache_scope(**extraction_scope_kwargs):
        try:
            parsed, usage = _call_with_retry(_do_call, sleep=time.sleep)
        except LLMValidationError as exc:
            # Gate 1 phase-level reject per spec §8.1: the schema
            # retry owned by generate_json was already spent.
            msg = str(exc)
            reject = {
                "chunk_id": chunk_id,
                "phase": "entity",
                "attempt": attempts,
                "gate": "schema",
                "reason": msg,
                "offending_record": {"model_output": exc.model_output},
            }
            return PhaseOutcome(
                status="rejected",
                attempt_count=attempts,
                mention_count=0,
                failure_count=0,
                reject_records=[reject],
            ), []
        # BudgetExceeded / ProviderTransportError / arbitrary
        # exceptions propagate verbatim per spec §11 / §12.1 / §12.2.

    run_state.record_usage(usage)

    # Defect 5: if generate_json returned a payload whose shape is
    # impossible (e.g. ``mentions`` not an array) it cannot be a
    # validated phase. Treat that as a phase-level Gate 1 failure:
    # status="rejected", failure_count=0, one canonical schema reject.
    mentions_raw = parsed.get("mentions", [])
    if not isinstance(mentions_raw, list):
        reject = {
            "chunk_id": chunk_id,
            "phase": "entity",
            "attempt": attempts,
            "gate": "schema",
            "reason": "top-level 'mentions' is not an array",
            "offending_record": None,
        }
        return PhaseOutcome(
            status="rejected",
            attempt_count=attempts,
            mention_count=0,
            failure_count=0,
            reject_records=[reject],
        ), []

    validated: list[validation_module.ValidatedMention] = []
    rejects: list[dict] = []
    for raw in mentions_raw:
        # Per-record Gate 2 / Gate 3 rejects only (Defect 5). The
        # top-level shape was already validated by ``generate_json``
        # (Gate 1) and by the array-shape check above; an individual
        # entry that fails schema-shape should not be classified as
        # Gate 1 because the schema is owned by generate_json. Such
        # impossible-shape entries are surface as Gate 2 rejects
        # since the field-level ontology conformity check is the
        # first per-record check the validator runs.
        if not isinstance(raw, dict):
            rejects.append({
                "chunk_id": chunk_id,
                "phase": "entity",
                "attempt": attempts,
                "gate": "ontology",
                "reason": "mention entry is not an object",
                "offending_record": {"raw": raw},
            })
            continue
        try:
            validation_module.check_mention_ontology(mention=raw, ontology=ontology)
        except validation_module.OntologyError as exc:
            rejects.append({
                "chunk_id": chunk_id,
                "phase": "entity",
                "attempt": attempts,
                "gate": "ontology",
                "reason": str(exc),
                "offending_record": raw,
            })
            continue
        # Stamp the canonical source_text_sha256 before evidence check
        # so the stored value always matches the chunk text.
        # The model emits char_start / char_end at the top level; the
        # evidence gate reads them out of an ``evidence`` block, so we
        # mirror the offsets into a fresh ``evidence`` dict and let it
        # carry the SHA too.
        raw_with_sha = dict(raw)
        raw_with_sha["evidence"] = {
            "char_start": int(raw["char_start"]),
            "char_end": int(raw["char_end"]),
            "source_text_sha256": sha,
        }
        try:
            validation_module.check_mention_evidence(
                mention=raw_with_sha,
                chunk_text=text,
                expected_source_text_sha256=sha,
            )
        except validation_module.EvidenceError as exc:
            rejects.append({
                "chunk_id": chunk_id,
                "phase": "entity",
                "attempt": attempts,
                "gate": "evidence",
                "reason": str(exc),
                "offending_record": raw,
            })
            continue
        # All three gates passed; assign deterministic ID and build record.
        validated.append(
            validation_module.build_validated_mention(
                raw_mention=raw_with_sha,
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_text_sha256_hex=sha,
            )
        )

    return PhaseOutcome(
        status="validated",
        attempt_count=attempts,
        mention_count=len(validated),
        failure_count=len(rejects),
        reject_records=rejects,
    ), validated


def _run_relation_phase(
    *,
    chunk: dict,
    ontology: dict,
    prompt_artifact: dict,
    provider: LLMProvider,
    max_tokens: int,
    temperature: float,
    schema: dict,
    run_state: RunState,
    same_chunk_mentions: list,  # list[ValidatedMention]
    entity_type_for_mention: dict,
):
    """Drive the relation pass for one chunk.

    The validated mentions carry assigned mention IDs and are the
    only legal target surface for ``kind: "mention"`` endpoints.
    Document endpoints resolve to ``chunk.doc_id`` byte-for-byte
    (spec §5.3, §8.2). Cross-chunk mention refs fail.
    """
    chunk_id = chunk["chunk_id"]
    doc_id = chunk["doc_id"]
    text = chunk["text"]
    sha = validation_module.source_text_sha256(text)

    allowed_lists = schema_module.build_allowed_name_lists(ontology)
    mention_lines = [
        {
            "mention_id": m.mention_id,
            "entity_type": m.entity_type,
            "surface_form": m.surface_form,
            "evidence": {
                "char_start": m.evidence["char_start"],
                "char_end": m.evidence["char_end"],
            },
        }
        for m in same_chunk_mentions
    ]
    relation_user = prompts_module.render_relation_prompt(
        template=prompt_artifact["text"],
        allowed_relation_types=allowed_lists["relation_types"],
        relation_definitions=schema_module.relation_definitions_block(ontology),
        chunk_id=chunk_id,
        doc_id=doc_id,
        mention_list=prompts_module.render_mention_list(mention_lines),
        chunk_text=text,
    )

    extraction_scope_kwargs = dict(
        chunk_id=chunk_id,
        source_text_sha256=sha,
        ontology_version=str(run_state.preflight.ontology_version),
        ontology_sha256=run_state.preflight.ontology_sha256,
        prompt_template_filename=prompt_artifact["filename"],
        prompt_template_content_sha256=prompt_artifact["sha256"],
    )

    attempts = 0
    parsed: dict | None = None
    usage = Usage()

    def _do_call():
        nonlocal attempts
        attempts += 1
        return provider.generate_json(
            _render_system(),
            relation_user,
            schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # Spec §5.5a: same strict-superset identity scope as the entity
    # phase, with the same six dimensions.
    #
    # Defect 1 (transport / internal error propagation): do NOT add a
    # broad ``except Exception`` around the provider call here. The
    # only exception we route as a phase-level reject is
    # ``LLMValidationError`` (Gate 1, schema). ``BudgetExceeded``,
    # ``ProviderTransportError`` (transient retry-exhausted and
    # non-transient), and arbitrary unexpected exceptions propagate
    # unchanged so the run aborts with the real failure.
    with extraction_cache_scope(**extraction_scope_kwargs):
        try:
            parsed, usage = _call_with_retry(_do_call, sleep=time.sleep)
        except LLMValidationError as exc:
            msg = str(exc)
            reject = {
                "chunk_id": chunk_id,
                "phase": "relation",
                "attempt": attempts,
                "gate": "schema",
                "reason": msg,
                "offending_record": {"model_output": exc.model_output},
            }
            return PhaseOutcome(
                status="rejected",
                attempt_count=attempts,
                assertion_count=0,
                failure_count=0,
                reject_records=[reject],
            ), []
        # BudgetExceeded / ProviderTransportError / arbitrary
        # exceptions propagate verbatim per spec §11 / §12.1 / §12.2.

    run_state.record_usage(usage)

    # Defect 5: impossible shape after Gate 1 is a phase-level Gate 1
    # failure: status="rejected", failure_count=0.
    assertions_raw = parsed.get("assertions", [])
    if not isinstance(assertions_raw, list):
        reject = {
            "chunk_id": chunk_id,
            "phase": "relation",
            "attempt": attempts,
            "gate": "schema",
            "reason": "top-level 'assertions' is not an array",
            "offending_record": None,
        }
        return PhaseOutcome(
            status="rejected",
            attempt_count=attempts,
            assertion_count=0,
            failure_count=0,
            reject_records=[reject],
        ), []

    same_chunk_index = {m.mention_id: m.to_record() for m in same_chunk_mentions}
    validated: list[validation_module.ValidatedAssertion] = []
    rejects: list[dict] = []
    for raw in assertions_raw:
        # Per-record Gate 2 / Gate 3 rejects only (Defect 5).
        if not isinstance(raw, dict):
            rejects.append({
                "chunk_id": chunk_id,
                "phase": "relation",
                "attempt": attempts,
                "gate": "ontology",
                "reason": "assertion entry is not an object",
                "offending_record": {"raw": raw},
            })
            continue
        try:
            validation_module.check_assertion_ontology(
                assertion=raw,
                ontology=ontology,
                doc_id=doc_id,
                same_chunk_mentions=same_chunk_index,
                entity_type_for_mention=entity_type_for_mention,
            )
        except validation_module.OntologyError as exc:
            rejects.append({
                "chunk_id": chunk_id,
                "phase": "relation",
                "attempt": attempts,
                "gate": "ontology",
                "reason": str(exc),
                "offending_record": raw,
            })
            continue
        raw_with_sha = dict(raw)
        raw_with_sha["evidence"] = {
            "char_start": int(raw["char_start"]),
            "char_end": int(raw["char_end"]),
            "source_text_sha256": sha,
        }
        try:
            validation_module.check_assertion_evidence(
                assertion=raw_with_sha,
                chunk_text=text,
                expected_source_text_sha256=sha,
            )
        except validation_module.EvidenceError as exc:
            rejects.append({
                "chunk_id": chunk_id,
                "phase": "relation",
                "attempt": attempts,
                "gate": "evidence",
                "reason": str(exc),
                "offending_record": raw,
            })
            continue
        validated.append(
            validation_module.build_validated_assertion(
                raw_assertion=raw_with_sha,
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_text_sha256_hex=sha,
            )
        )

    return PhaseOutcome(
        status="validated",
        attempt_count=attempts,
        assertion_count=len(validated),
        failure_count=len(rejects),
        reject_records=rejects,
    ), validated


# --- main run --------------------------------------------------------


@dataclass
class RunResult:
    """Return value of :func:`run_extraction`.

    ``exit_code`` is the suggested process exit code (0 = clean,
    1 = budget abort, 2 = preflight failure). ``records_written``
    is the total number of JSONL records (mentions + assertions +
    chunk_statuses) the pipeline wrote to disk on this run.
    """

    exit_code: int
    run_id: str
    resumed_from_run_id: str | None
    records_written: int
    mention_count: int
    assertion_count: int
    complete_chunk_count: int
    reject_count: int
    budget_aborted: bool


def _entity_type_lookup(validated_mentions: list) -> dict:
    """Build ``mention_id -> entity_type`` for relation-pass validation."""
    return {m.mention_id: m.entity_type for m in validated_mentions}


def _format_records(
    *,
    mentions_by_chunk: dict,
    assertions_by_chunk: dict,
    statuses_by_chunk: dict,
) -> Iterable[str]:
    """Render the canonical JSONL bytes in spec §3 deterministic order.

    The contract key is a single lexicographic tuple applied to every
    record in the union of mentions, assertions, and chunk_statuses:

    ``(chunk_id, kind_rank, record_id)``

    where ``kind_rank`` is 0 for mentions, 1 for assertions, and 2
    for chunk_status records (per the spec §3 contract "records are
    sorted by (chunk_id, kind, mention_id or assertion_id)
    lexicographically" — a chunk_status record carries no
    mention/assertion id, so the kind slot is what fixes its
    position within its chunk bucket). For chunk_status, the
    third element is the chunk_id (the record's only identity
    field). The same contract is asserted by the test
    ``test_records_sorted_by_chunk_then_id`` and is the single
    ordering rule for the JSONL. There is no other ordering rule.
    """
    rows: list[tuple] = []
    for chunk_id, recs in mentions_by_chunk.items():
        for rec in recs:
            rows.append((chunk_id, 0, rec["mention_id"], rec))
    for chunk_id, recs in assertions_by_chunk.items():
        for rec in recs:
            rows.append((chunk_id, 1, rec["assertion_id"], rec))
    for chunk_id, status in statuses_by_chunk.items():
        rows.append((chunk_id, 2, chunk_id, status))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    for _cid, _rank, _id, rec in rows:
        yield json.dumps(rec, ensure_ascii=False, sort_keys=True)


def _build_chunk_status(
    *,
    chunk_id: str,
    doc_id: str,
    entity_outcome: PhaseOutcome,
    relation_outcome: PhaseOutcome,
) -> dict:
    """Build the §3.1.3 ``chunk_status`` record.

    ``extraction_complete == true`` only when BOTH phases are
    validated AND neither phase accumulated any Gate 2 / Gate 3
    per-record rejects. Spec §9: "A chunk is never extraction-complete
    on a single phase." Per-record Gate 2 + Gate 3 failures are
    reported via ``failure_count`` so the operator can see how many
    candidates were dropped while still treating the phase's overall
    status as ``validated`` (the LLM payload parsed; records that
    failed were dropped and the survivors are usable).
    """
    extraction_complete = (
        entity_outcome.status == "validated"
        and relation_outcome.status == "validated"
        and entity_outcome.failure_count == 0
        and relation_outcome.failure_count == 0
    )
    return {
        "kind": "chunk_status",
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "phases": {
            "entity": {
                "status": entity_outcome.status,
                "attempt_count": entity_outcome.attempt_count,
                "mention_count": entity_outcome.mention_count,
                "failure_count": entity_outcome.failure_count,
            },
            "relation": {
                "status": relation_outcome.status,
                "attempt_count": relation_outcome.attempt_count,
                "assertion_count": relation_outcome.assertion_count,
                "failure_count": relation_outcome.failure_count,
            },
        },
        "extraction_complete": extraction_complete,
    }


def run_extraction(
    *,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    prompts_dir: Path = prompts_module.PROMPTS_DIR,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    output_rejects: Path = DEFAULT_OUTPUT_REJECTS,
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST,
    provider: LLMProvider | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> RunResult:
    """Drive the full extraction pipeline.

    Public library entry point. The CLI wrapper in ``__main__.py``
    calls this with the same arguments and propagates ``exit_code``.
    """
    if provider is None:
        raise PreflightError(
            "run_extraction requires an injected LLMProvider; the CLI "
            "entry point reads env vars via src.providers.make_llm"
        )

    preflight_report, ontology, prompt_artifacts = preflight(
        chunks_path=chunks_path,
        ontology_path=ontology_path,
        prompts_dir=prompts_dir,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Load any prior state for resumability (spec §13.1).
    prev_mentions, prev_assertions, prev_statuses = _read_existing_records(output_jsonl)
    prev_manifest: dict | None = None
    if output_manifest.is_file():
        try:
            prev_manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_manifest = None
    resumed_from = prev_manifest.get("run_id") if prev_manifest else None

    chunks = _read_chunks(chunks_path)
    run_state = RunState(
        run_id=_make_run_id(
            prompt_sha=preflight_report.prompt_template_sha256,
            ontology_sha=preflight_report.ontology_sha256,
            chunks_sha=preflight_report.chunks_sha256,
        ),
        resumed_from_run_id=resumed_from,
        preflight=preflight_report,
        started_at=_utc_now_compact(),
    )

    schemas = {
        "entity": schema_module.entity_pass_schema(ontology),
        "relation": schema_module.relation_pass_schema(ontology),
    }
    entity_artifact = prompt_artifacts[prompts_module.ENTITY_PROMPT_FILE]
    relation_artifact = prompt_artifacts[prompts_module.RELATION_PROMPT_FILE]

    mentions_by_chunk: dict = {}
    assertions_by_chunk: dict = {}
    statuses_by_chunk: dict = {}

    # Carry over records for chunks whose status is already complete
    # (spec §13.1: "chunks whose final chunk_status says
    # extraction_complete=true are skipped entirely"). The previous
    # state is the source of truth for these.
    for cid, status in prev_statuses.items():
        if _is_complete(status):
            mentions_by_chunk[cid] = list(prev_mentions.get(cid, []))
            assertions_by_chunk[cid] = list(prev_assertions.get(cid, []))
            statuses_by_chunk[cid] = status

    budget_aborted = False
    total_mentions = sum(len(v) for v in mentions_by_chunk.values())
    total_assertions = sum(len(v) for v in assertions_by_chunk.values())

    try:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in statuses_by_chunk and _is_complete(statuses_by_chunk[chunk_id]):
                # Skipped on resume per spec §13.1.
                continue
            try:
                entity_outcome, validated_mentions = _run_entity_phase(
                    chunk=chunk,
                    ontology=ontology,
                    prompt_artifact=entity_artifact,
                    provider=provider,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schemas["entity"],
                    run_state=run_state,
                )
            except BudgetExceeded:
                # Spec §11: clean abort — leave in-progress records in
                # place (none yet for this chunk) and exit non-zero.
                budget_aborted = True
                break

            mentions_by_chunk[chunk_id] = [m.to_record() for m in validated_mentions]

            entity_type_lookup = _entity_type_lookup(validated_mentions)

            try:
                relation_outcome, validated_assertions = _run_relation_phase(
                    chunk=chunk,
                    ontology=ontology,
                    prompt_artifact=relation_artifact,
                    provider=provider,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schemas["relation"],
                    run_state=run_state,
                    same_chunk_mentions=validated_mentions,
                    entity_type_for_mention=entity_type_lookup,
                )
            except BudgetExceeded:
                # Keep this chunk's already-validated mentions; the
                # relation phase never started for it. The
                # chunk_status record (if we write it) will reflect
                # entity-only completion and ``extraction_complete``
                # stays False, so resume is well-defined.
                budget_aborted = True
                break

            assertions_by_chunk[chunk_id] = [a.to_record() for a in validated_assertions]

            statuses_by_chunk[chunk_id] = _build_chunk_status(
                chunk_id=chunk_id,
                doc_id=chunk["doc_id"],
                entity_outcome=entity_outcome,
                relation_outcome=relation_outcome,
            )

            # Emit rejects for this chunk to the reject log.
            new_rejects = entity_outcome.reject_records + relation_outcome.reject_records
            if new_rejects:
                _append_rejects(output_rejects, new_rejects)
                for r in new_rejects:
                    gate = r.get("gate", "unknown")
                    run_state.reject_counts[gate] = (
                        run_state.reject_counts.get(gate, 0) + 1
                    )

            run_state.chunk_status_counts["processed"] = (
                run_state.chunk_status_counts.get("processed", 0) + 1
            )

        # Final pass: write JSONL (mentions/assertions/chunk_statuses)
        # and the manifest. Done whether we exited cleanly or via
        # BudgetExceeded so the partial state is durable.
        run_state.ended_at = _utc_now_compact()

        # Sort each chunk's records deterministically for output.
        for cid in list(mentions_by_chunk):
            mentions_by_chunk[cid].sort(key=lambda r: r.get("mention_id", ""))
        for cid in list(assertions_by_chunk):
            assertions_by_chunk[cid].sort(key=lambda r: r.get("assertion_id", ""))

        _atomic_rewrite_jsonl(
            output_jsonl,
            _format_records(
                mentions_by_chunk=mentions_by_chunk,
                assertions_by_chunk=assertions_by_chunk,
                statuses_by_chunk=statuses_by_chunk,
            ),
        )
        _atomic_write_json(output_manifest, run_state.as_manifest())

        total_mentions = sum(len(v) for v in mentions_by_chunk.values())
        total_assertions = sum(len(v) for v in assertions_by_chunk.values())
        total_statuses = sum(1 for s in statuses_by_chunk.values() if _is_complete(s))

        return RunResult(
            exit_code=1 if budget_aborted else 0,
            run_id=run_state.run_id,
            resumed_from_run_id=run_state.resumed_from_run_id,
            records_written=total_mentions + total_assertions + len(statuses_by_chunk),
            mention_count=total_mentions,
            assertion_count=total_assertions,
            complete_chunk_count=total_statuses,
            reject_count=sum(run_state.reject_counts.values()),
            budget_aborted=budget_aborted,
        )
    except BudgetExceeded:
        # Catch-all in case the budget is hit at an unusual point
        # (e.g. between the entity and relation calls). The records
        # for the in-progress chunk are partial; we still write what
        # we have so the next resume can finish them.
        run_state.ended_at = _utc_now_compact()
        for cid in list(mentions_by_chunk):
            mentions_by_chunk[cid].sort(key=lambda r: r.get("mention_id", ""))
        for cid in list(assertions_by_chunk):
            assertions_by_chunk[cid].sort(key=lambda r: r.get("assertion_id", ""))
        _atomic_rewrite_jsonl(
            output_jsonl,
            _format_records(
                mentions_by_chunk=mentions_by_chunk,
                assertions_by_chunk=assertions_by_chunk,
                statuses_by_chunk=statuses_by_chunk,
            ),
        )
        _atomic_write_json(output_manifest, run_state.as_manifest())
        return RunResult(
            exit_code=1,
            run_id=run_state.run_id,
            resumed_from_run_id=run_state.resumed_from_run_id,
            records_written=sum(len(v) for v in mentions_by_chunk.values())
            + sum(len(v) for v in assertions_by_chunk.values())
            + len(statuses_by_chunk),
            mention_count=sum(len(v) for v in mentions_by_chunk.values()),
            assertion_count=sum(len(v) for v in assertions_by_chunk.values()),
            complete_chunk_count=sum(
                1 for s in statuses_by_chunk.values() if _is_complete(s)
            ),
            reject_count=sum(run_state.reject_counts.values()),
            budget_aborted=True,
        )


def _append_rejects(path: Path, records: list) -> None:
    """Append one JSONL record per reject entry to ``path``.

    Atomic per line: each reject is its own line. The reject log is
    intended to be append-only (spec §8.4) so a crash does not lose
    audit records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            f.write("\n")
