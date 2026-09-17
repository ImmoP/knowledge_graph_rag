"""Fresh-run v2 semantic-execution runner (#53).

Companion to :mod:`src.extract.pipeline` for the v2 extraction contract
(#47, #49, #51). The v1 ``pipeline.py`` stays frozen and runnable; this
module adds the v2 runner without modifying v1 symbols. The runner
exercises the v2 aligner (#49), v2 evidence gate (this module's
``check_mention_evidence_v2`` / ``check_assertion_evidence_v2`` in
``validation``), v2 ontology gate (the reused v1 helpers, which are
structurally agnostic to the schema_version discriminator), and v2
content-addressed IDs (``ids.mention_id_v2`` / ``ids.assertion_id_v2``).

Design notes (#53 acceptance gate):

* Fresh-run only — never reads prior ``output_jsonl`` and never
  honours prior ``chunk_status.extraction_complete == true``. The
  output JSONL is rewritten atomically on every call. There is no
  ``resumed_from_run_id`` and no v2 manifest sidecar in this slice.
* v2 schema-version discriminator (``SCHEMA_VERSION_V2 = 2``) is
  baked into every emitted mention / assertion / chunk_status record
  via the v2 dataclasses (``ValidatedV2Mention``, ``ValidatedV2Assertion``)
  and the explicit chunk-status builder below.
* Extraction cache identity is the v1 six-dimension ``extraction_cache_scope``
  with the v2 prompt filenames + content sha256 — strictly the same
  shape as v1; only the prompt filenames differ. The cache code in
  ``src/providers/cache.py`` is untouched.
* Relation phase receives only validated same-chunk v2 mentions
  filtered through the v2 aligner + v2 evidence gate + ontology gate
  (Gate 2 reused unchanged from v1).
* LLM still MUST NOT emit ``char_start`` / ``char_end``; Python
  derives them via the v2 aligner. The exact-equality evidence gate
  is structurally enforced by the aligner (#49 acceptance #20); the
  v2 ``check_mention_evidence_v2`` / ``check_assertion_evidence_v2``
  add a defense-in-depth assertion (byte equality on the slice + SHA
  drift rejection) per #53 acceptance #4 / #5.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extract import (
    align as align_module,
    ids as ids_module,
    prompts as prompts_module,
    schema as schema_module,
    validation as validation_module,
)
from src.extract.validation import (
    ValidatedV2Assertion,
    ValidatedV2Mention,
)
from src.providers.cache import CACHE_FORMAT_VERSION, extraction_cache_scope
from src.providers.protocol import (
    LLMProvider,
    LLMValidationError,
    Usage,
)
from src.providers.budget import BudgetExceeded


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "derived" / "chunks.jsonl"
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "ontology.yaml"

V2_ENTITY_PROMPT_FILENAME = prompts_module.ENTITY_PROMPT_FILE_V2
V2_RELATION_PROMPT_FILENAME = prompts_module.RELATION_PROMPT_FILE_V2


# ---------- atomic I/O (mirrors v1, inlined to keep v1 frozen) -------


def _atomic_rewrite_jsonl(path: Path, lines: Iterable[str]) -> None:
    """Rewrite ``path`` atomically with the supplied ``lines``.

    Inlined copy of the v1 helper so v1's ``pipeline.py`` can stay
    frozen. Writes to a sibling temp file under the same parent dir,
    fsyncs, then ``os.replace`` onto ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for line in lines:
                if not line.endswith("\n"):
                    line = line + "\n"
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ---------- preflight -------------------------------------------------


class PreflightErrorV2(RuntimeError):
    """Raised at startup when a hard v2 preflight gate fails."""


@dataclass(frozen=True)
class PreflightReportV2:
    chunks_path: Path
    chunks_sha256: str
    ontology_path: Path
    ontology_sha256: str
    ontology_version: int
    prompt_entity_sha256: str
    prompt_entity_filename: str
    prompt_relation_sha256: str
    prompt_relation_filename: str
    system_prompt: str | None = None
    system_prompt_sha256: str | None = None
    alignment_algorithm_version: str | None = None
    alignment_algorithm_sha256: str | None = None
    pipeline_version: str | None = None
    schema_version: int | None = None
    provider_name: str | None = None
    provider_model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    cache_format_version: str | None = None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_chunks(path: Path) -> list:
    if not path.is_file():
        raise PreflightErrorV2(f"chunks file not found: {path}")
    out: list = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.append(json.loads(line))
    return out


def _ontology_version(ontology: dict) -> int:
    v = ontology.get("version")
    if not isinstance(v, int) or isinstance(v, bool):
        raise PreflightErrorV2(
            "ontology.yaml: top-level 'version' must be int (not bool)"
        )
    return v


def _preflight_v2(
    *,
    chunks_path: Path,
    ontology_path: Path,
    prompts_dir: Path,
    provider: LLMProvider | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    require_pipeline_version: bool = False,
) -> tuple[dict, list, dict, PreflightReportV2]:
    """Load ontology + chunks + v2 prompt artifacts; fail closed.

    Returns ``(ontology, chunks, prompt_artifacts, report)``.

    The system prompt and aligner sha256 are computed here so the
    canonical run_id fingerprint has a single source of truth and
    every disk artifact carries the same value the runtime will
    use.
    """
    if not chunks_path.is_file():
        raise PreflightErrorV2(f"chunks file not found: {chunks_path}")
    if not ontology_path.is_file():
        raise PreflightErrorV2(f"ontology file not found: {ontology_path}")

    try:
        import yaml
    except ImportError as e:
        raise PreflightErrorV2(
            "PyYAML is required to load the ontology; install 'pyyaml'"
        ) from e

    ontology = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    if not isinstance(ontology, dict):
        raise PreflightErrorV2("ontology.yaml: top-level must be a mapping")

    chunks = _read_chunks(chunks_path)

    prompt_artifacts = prompts_module.verify_prompt_artifacts_v2(prompts_dir)

    system_prompt = _render_system_v2()
    align_path = Path(__file__).resolve().parent / "align.py"

    report = PreflightReportV2(
        chunks_path=chunks_path,
        chunks_sha256=_file_sha256(chunks_path),
        ontology_path=ontology_path,
        ontology_sha256=_file_sha256(ontology_path),
        ontology_version=_ontology_version(ontology),
        prompt_entity_sha256=prompt_artifacts[V2_ENTITY_PROMPT_FILENAME]["sha256"],
        prompt_entity_filename=V2_ENTITY_PROMPT_FILENAME,
        prompt_relation_sha256=prompt_artifacts[V2_RELATION_PROMPT_FILENAME]["sha256"],
        prompt_relation_filename=V2_RELATION_PROMPT_FILENAME,
        system_prompt=system_prompt,
        system_prompt_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        alignment_algorithm_version=align_module.ALIGNMENT_ALGORITHM_VERSION,
        alignment_algorithm_sha256=hashlib.sha256(align_path.read_bytes()).hexdigest(),
        pipeline_version=_pipeline_version_v2() if require_pipeline_version else None,
        schema_version=schema_module.SCHEMA_VERSION_V2,
        provider_name=provider.name,
        provider_model=provider.model,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_format_version=CACHE_FORMAT_VERSION,
    )
    return ontology, chunks, prompt_artifacts, report


# ---------- run state -------------------------------------------------


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%MSSZ")


def _pipeline_version_v2() -> str:
    """Return the current ``git rev-parse HEAD`` for manifest audit.

    Fail-closed: a non-zero exit / empty stdout / missing git
    binary raises :class:`PreflightErrorV2`. The canonical run-id
    fingerprint excludes ``pipeline_version``, but the manifest
    still carries it as an audit field so operators can trace
    which codebase produced a given artifact set.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise PreflightErrorV2(
            f"pipeline_version: git rev-parse HEAD failed ({exc!r})"
        ) from exc
    if proc.returncode != 0:
        raise PreflightErrorV2(
            "pipeline_version: git rev-parse HEAD exited "
            f"non-zero ({proc.returncode}); stderr={proc.stderr.strip()!r}"
        )
    sha = proc.stdout.strip()
    if not sha:
        raise PreflightErrorV2(
            "pipeline_version: git rev-parse HEAD returned empty stdout"
        )
    return sha


def _run_id_fingerprint_v2(*, preflight: PreflightReportV2) -> str:
    """Short fingerprint over the 11 resume-identity dimensions (#55).

    ``pipeline_version`` is intentionally excluded: an unchanged
    pipeline that runs against a freshly-checked-out commit must
    produce a resume-compatible ``run_id`` so the operator can
    resume after ``git pull`` without losing the carry-forward.
    """
    parts = [
        str(preflight.schema_version),
        preflight.chunks_sha256,
        preflight.ontology_sha256,
        preflight.prompt_entity_sha256,
        preflight.prompt_relation_sha256,
        preflight.system_prompt_sha256,
        preflight.provider_name,
        preflight.provider_model,
        str(preflight.max_tokens),
        str(preflight.temperature),
        preflight.alignment_algorithm_version,
        preflight.alignment_algorithm_sha256,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _make_run_id_v2(*, preflight: PreflightReportV2) -> str:
    return (
        _utc_now_compact()
        + "-"
        + _run_id_fingerprint_v2(preflight=preflight)
    )


# ---------- phase outcome --------------------------------------------


@dataclass
class PhaseOutcomeV2:
    """Result of one v2 phase (entity or relation) for one chunk.

    ``status`` is ``validated`` (LLM payload parsed; per-record
    rejects may still exist in ``reject_records``) or ``rejected``
    (phase-level Gate 1 fail).
    """

    status: str
    attempt_count: int
    mention_count: int = 0
    assertion_count: int = 0
    failure_count: int = 0
    reject_records: list = field(default_factory=list)


# ---------- main run result ------------------------------------------


@dataclass
class RunResultV2:
    """Return value of :func:`run_extraction_v2`."""

    exit_code: int
    run_id: str
    records_written: int
    mention_count: int
    assertion_count: int
    complete_chunk_count: int
    reject_count: int
    budget_aborted: bool = False


# ---------- chunk_status ---------------------------------------------


def _build_chunk_status_v2(
    *,
    chunk_id: str,
    doc_id: str,
    entity_outcome: PhaseOutcomeV2,
    relation_outcome: PhaseOutcomeV2,
) -> dict:
    """Build the v2 ``chunk_status`` record (#53).

    ``extraction_complete == true`` only when BOTH phases are
    validated AND both ``failure_count`` fields are zero (same rule
    as v1). The record carries ``schema_version`` so re-derivation
    helpers know which contract the record obeys.
    """
    extraction_complete = (
        entity_outcome.status == "validated"
        and relation_outcome.status == "validated"
        and entity_outcome.failure_count == 0
        and relation_outcome.failure_count == 0
    )
    return {
        "kind": "chunk_status",
        "schema_version": schema_module.SCHEMA_VERSION_V2,
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


# ---------- record serialization ------------------------------------


def _format_records_v2(
    *,
    mentions_by_chunk: dict,
    assertions_by_chunk: dict,
    statuses_by_chunk: dict,
) -> Iterable[str]:
    """Yield canonical JSONL bytes sorted by ``(chunk_id, kind_rank, id)``.

    Mirrors v1 ``_format_records`` with no semantic changes. ``kind_rank``
    is 0 for mentions, 1 for assertions, 2 for chunk_status.
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


# ---------- phase runners --------------------------------------------


def _render_system_v2() -> str:
    """Small system preamble shared by both v2 phases."""
    return (
        "You are an information-extraction assistant. Respond only with "
        "the JSON object requested by the user message. Do not include "
        "prose, commentary, or code fences."
    )


def _call_with_retry_v2(op, *, sleep) -> tuple[Any, Usage]:
    """Wrap ``op()`` in the v1 transport-retry policy (spec §12.1)."""
    from src.extract.retry import with_transport_retry

    return with_transport_retry(op, sleep=sleep)


def _run_entity_phase_v2(
    *,
    chunk: dict,
    ontology: dict,
    prompt_artifact: dict,
    provider: LLMProvider,
    max_tokens: int,
    temperature: float,
    schema: dict,
    preflight: PreflightReportV2,
    system_prompt: str | None = None,
) -> tuple[PhaseOutcomeV2, list[ValidatedV2Mention], Usage]:
    """v2 entity pass: schema gate, aligner, evidence gate, ontology gate.

    Returns ``(outcome, validated_mentions, usage)``.
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

    scope_kwargs = dict(
        chunk_id=chunk_id,
        source_text_sha256=sha,
        ontology_version=str(preflight.ontology_version),
        ontology_sha256=preflight.ontology_sha256,
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
            system_prompt if system_prompt is not None else _render_system_v2(),
            entity_user,
            schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    with extraction_cache_scope(**scope_kwargs):
        try:
            parsed, usage = _call_with_retry_v2(_do_call, sleep=time.sleep)
        except LLMValidationError as exc:
            return (
                PhaseOutcomeV2(
                    status="rejected",
                    attempt_count=attempts,
                    mention_count=0,
                    failure_count=0,
                    reject_records=[{
                        "chunk_id": chunk_id,
                        "phase": "entity",
                        "attempt": attempts,
                        "gate": "schema",
                        "reason": str(exc),
                        "offending_record": {"model_output": exc.model_output},
                    }],
                ),
                [],
                usage,
            )
        # BudgetExceeded / ProviderTransportError / arbitrary
        # exceptions propagate verbatim.

    mentions_raw = parsed.get("mentions", []) if isinstance(parsed, dict) else []
    if not isinstance(mentions_raw, list):
        return (
            PhaseOutcomeV2(
                status="rejected",
                attempt_count=attempts,
                mention_count=0,
                failure_count=0,
                reject_records=[{
                    "chunk_id": chunk_id,
                    "phase": "entity",
                    "attempt": attempts,
                    "gate": "schema",
                    "reason": "top-level 'mentions' is not an array",
                    "offending_record": None,
                }],
            ),
            [],
            usage,
        )

    valid_dicts = [m for m in mentions_raw if isinstance(m, dict)]
    accepted, align_rejects = align_module.align_mentions(
        text, valid_dicts, chunk_id=chunk_id,
    )

    rejects: list[dict] = []
    for r in align_rejects:
        rejects.append({
            "chunk_id": chunk_id,
            "phase": "entity",
            "attempt": attempts,
            "gate": "alignment",
            "reason": r.reason.value,
            "offending_record": r.to_record(),
            "aligner_reject": True,
            "aligner_reason": r.reason.value,
        })

    validated: list[ValidatedV2Mention] = []
    for am in accepted:
        am_dict = {
            "entity_type": am.entity_type,
            "surface_form": am.surface_form,
            "canonical_form": am.canonical_form,
            "confidence": am.confidence,
            "evidence": {
                "char_start": am.char_start,
                "char_end": am.char_end,
                "source_text_sha256": sha,
            },
        }
        try:
            validation_module.check_mention_ontology(
                mention=am_dict, ontology=ontology,
            )
        except validation_module.OntologyError as exc:
            rejects.append({
                "chunk_id": chunk_id,
                "phase": "entity",
                "attempt": attempts,
                "gate": "ontology",
                "reason": str(exc),
                "offending_record": {
                    "entity_type": am.entity_type,
                    "surface_form": am.surface_form,
                },
            })
            continue
        try:
            validation_module.check_mention_evidence_v2(
                mention=am_dict,
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
                "offending_record": {
                    "entity_type": am.entity_type,
                    "surface_form": am.surface_form,
                },
            })
            continue
        mid = ids_module.mention_id_v2(
            chunk_id=chunk_id,
            source_text_sha256=sha,
            entity_type=am.entity_type,
            char_start=am.char_start,
            char_end=am.char_end,
        )
        validated.append(ValidatedV2Mention(
            mention_id=mid,
            chunk_id=chunk_id,
            doc_id=doc_id,
            entity_type=am.entity_type,
            surface_form=am.surface_form,
            canonical_form=am.canonical_form,
            evidence=am_dict["evidence"],
            confidence=am.confidence,
        ))

    return (
        PhaseOutcomeV2(
            status="validated",
            attempt_count=attempts,
            mention_count=len(validated),
            failure_count=len(rejects),
            reject_records=rejects,
        ),
        validated,
        usage,
    )


def _run_relation_phase_v2(
    *,
    chunk: dict,
    ontology: dict,
    prompt_artifact: dict,
    provider: LLMProvider,
    max_tokens: int,
    temperature: float,
    schema: dict,
    preflight: PreflightReportV2,
    same_chunk_mentions: list,
    entity_type_for_mention: dict,
    system_prompt: str | None = None,
) -> tuple[PhaseOutcomeV2, list[ValidatedV2Assertion], Usage]:
    """v2 relation pass: schema gate, aligner + containment, evidence gate, ontology gate.

    Returns ``(outcome, validated_assertions, usage)``.
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

    scope_kwargs = dict(
        chunk_id=chunk_id,
        source_text_sha256=sha,
        ontology_version=str(preflight.ontology_version),
        ontology_sha256=preflight.ontology_sha256,
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
            system_prompt if system_prompt is not None else _render_system_v2(),
            relation_user,
            schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    with extraction_cache_scope(**scope_kwargs):
        try:
            parsed, usage = _call_with_retry_v2(_do_call, sleep=time.sleep)
        except LLMValidationError as exc:
            return (
                PhaseOutcomeV2(
                    status="rejected",
                    attempt_count=attempts,
                    assertion_count=0,
                    failure_count=0,
                    reject_records=[{
                        "chunk_id": chunk_id,
                        "phase": "relation",
                        "attempt": attempts,
                        "gate": "schema",
                        "reason": str(exc),
                        "offending_record": {"model_output": exc.model_output},
                    }],
                ),
                [],
                usage,
            )
        # BudgetExceeded / ProviderTransportError / arbitrary
        # exceptions propagate verbatim.

    assertions_raw = parsed.get("assertions", []) if isinstance(parsed, dict) else []
    if not isinstance(assertions_raw, list):
        return (
            PhaseOutcomeV2(
                status="rejected",
                attempt_count=attempts,
                assertion_count=0,
                failure_count=0,
                reject_records=[{
                    "chunk_id": chunk_id,
                    "phase": "relation",
                    "attempt": attempts,
                    "gate": "schema",
                    "reason": "top-level 'assertions' is not an array",
                    "offending_record": None,
                }],
            ),
            [],
            usage,
        )

    aligned_mentions_by_id = {
        m.mention_id: (m.evidence["char_start"], m.evidence["char_end"])
        for m in same_chunk_mentions
    }

    valid_dicts = [a for a in assertions_raw if isinstance(a, dict)]
    accepted, align_rejects = align_module.align_relations(
        text,
        valid_dicts,
        aligned_mentions_by_id,
        chunk_id=chunk_id,
    )

    rejects: list[dict] = []
    for r in align_rejects:
        rejects.append({
            "chunk_id": chunk_id,
            "phase": "relation",
            "attempt": attempts,
            "gate": "alignment",
            "reason": r.reason.value,
            "offending_record": r.to_record(),
            "aligner_reject": True,
            "aligner_reason": r.reason.value,
        })

    same_chunk_index = {m.mention_id: m.to_record() for m in same_chunk_mentions}
    validated: list[ValidatedV2Assertion] = []
    for aa in accepted:
        aa_dict = {
            "relation_type": aa.relation_type,
            "subject_ref": dict(aa.subject_ref),
            "object_ref": dict(aa.object_ref),
            "evidence_quote": aa.evidence_quote,
            "confidence": aa.confidence,
            "evidence": {
                "char_start": aa.char_start,
                "char_end": aa.char_end,
                "source_text_sha256": sha,
            },
        }
        try:
            validation_module.check_assertion_ontology(
                assertion=aa_dict,
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
                "offending_record": {
                    "relation_type": aa.relation_type,
                    "evidence_quote": aa.evidence_quote,
                },
            })
            continue
        try:
            validation_module.check_assertion_evidence_v2(
                assertion=aa_dict,
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
                "offending_record": {
                    "relation_type": aa.relation_type,
                    "evidence_quote": aa.evidence_quote,
                },
            })
            continue
        aid = ids_module.assertion_id_v2(
            chunk_id=chunk_id,
            source_text_sha256=sha,
            relation_type=aa.relation_type,
            subject_ref=dict(aa.subject_ref),
            object_ref=dict(aa.object_ref),
            char_start=aa.char_start,
            char_end=aa.char_end,
        )
        validated.append(ValidatedV2Assertion(
            assertion_id=aid,
            chunk_id=chunk_id,
            doc_id=doc_id,
            relation_type=aa.relation_type,
            subject_ref=dict(aa.subject_ref),
            object_ref=dict(aa.object_ref),
            evidence_quote=aa.evidence_quote,
            evidence=aa_dict["evidence"],
            confidence=aa.confidence,
        ))

    return (
        PhaseOutcomeV2(
            status="validated",
            attempt_count=attempts,
            assertion_count=len(validated),
            failure_count=len(rejects),
            reject_records=rejects,
        ),
        validated,
        usage,
    )


# ---------- public entry points --------------------------------------


def preflight_v2(
    *,
    chunks_path: Path,
    ontology_path: Path,
    prompts_dir: Path = prompts_module.PROMPTS_DIR,
    provider: LLMProvider,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> PreflightReportV2:
    """Load ontology + chunks + v2 prompt artifacts; fail closed.

    Returns a :class:`PreflightReportV2`. Used by callers that want
    to fail early on missing artifacts before invoking
    :func:`run_extraction_v2` or :func:`run_extraction_v2_canonical`.
    """
    _, _, _, report = _preflight_v2(
        chunks_path=chunks_path,
        ontology_path=ontology_path,
        prompts_dir=prompts_dir,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return report


def run_extraction_v2(
    *,
    provider: LLMProvider,
    output_jsonl: Path,
    output_rejects: Path,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    prompts_dir: Path = prompts_module.PROMPTS_DIR,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> RunResultV2:
    """Drive the v2 extraction runner (#53).

    Public library entry point. No CLI wrapper (T3a is library-callable
    only). Fresh-run semantics: never reads prior ``output_jsonl``,
    never honours prior ``chunk_status.extraction_complete == true``,
    always rewrites the JSONL atomically.

    Returns a :class:`RunResultV2`. ``exit_code`` is ``0`` on clean
    completion and ``1`` on ``BudgetExceeded`` abort.
    """
    ontology, chunks, prompt_artifacts, report = _preflight_v2(
        chunks_path=chunks_path,
        ontology_path=ontology_path,
        prompts_dir=prompts_dir,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    schemas = {
        "entity": schema_module.entity_pass_v2_schema(ontology),
        "relation": schema_module.relation_pass_v2_schema(ontology),
    }
    entity_artifact = prompt_artifacts[V2_ENTITY_PROMPT_FILENAME]
    relation_artifact = prompt_artifacts[V2_RELATION_PROMPT_FILENAME]

    run_id = _make_run_id_v2(preflight=report)
    mentions_by_chunk: dict = {}
    assertions_by_chunk: dict = {}
    statuses_by_chunk: dict = {}
    budget_aborted = False
    collected_rejects: list = []

    try:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            try:
                entity_outcome, validated_mentions, entity_usage = _run_entity_phase_v2(
                    chunk=chunk,
                    ontology=ontology,
                    prompt_artifact=entity_artifact,
                    provider=provider,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schemas["entity"],
                    preflight=report,
                )
            except BudgetExceeded:
                budget_aborted = True
                break

            mentions_by_chunk[chunk_id] = [m.to_record() for m in validated_mentions]

            entity_type_lookup = {
                m.mention_id: m.entity_type for m in validated_mentions
            }

            try:
                (
                    relation_outcome,
                    validated_assertions,
                    relation_usage,
                ) = _run_relation_phase_v2(
                    chunk=chunk,
                    ontology=ontology,
                    prompt_artifact=relation_artifact,
                    provider=provider,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schemas["relation"],
                    preflight=report,
                    same_chunk_mentions=validated_mentions,
                    entity_type_for_mention=entity_type_lookup,
                )
            except BudgetExceeded:
                budget_aborted = True
                break

            assertions_by_chunk[chunk_id] = [a.to_record() for a in validated_assertions]
            statuses_by_chunk[chunk_id] = _build_chunk_status_v2(
                chunk_id=chunk_id,
                doc_id=chunk["doc_id"],
                entity_outcome=entity_outcome,
                relation_outcome=relation_outcome,
            )

            collected_rejects.extend(
                entity_outcome.reject_records + relation_outcome.reject_records
            )

        # Final pass: atomic write of the canonical JSONL and the
        # reject log. The reject log is rewritten (truncates any
        # stale content) on every run so reused output paths
        # never carry over rejects from a previous run.
        for cid in list(mentions_by_chunk):
            mentions_by_chunk[cid].sort(key=lambda r: r.get("mention_id", ""))
        for cid in list(assertions_by_chunk):
            assertions_by_chunk[cid].sort(key=lambda r: r.get("assertion_id", ""))

        _atomic_rewrite_jsonl(
            output_jsonl,
            _format_records_v2(
                mentions_by_chunk=mentions_by_chunk,
                assertions_by_chunk=assertions_by_chunk,
                statuses_by_chunk=statuses_by_chunk,
            ),
        )
        _atomic_rewrite_jsonl(
            output_rejects,
            (json.dumps(r, ensure_ascii=False, sort_keys=True) for r in collected_rejects),
        )

        total_mentions = sum(len(v) for v in mentions_by_chunk.values())
        total_assertions = sum(len(v) for v in assertions_by_chunk.values())
        total_statuses = len(statuses_by_chunk)
        total_complete = sum(
            1 for s in statuses_by_chunk.values() if s.get("extraction_complete")
        )

        return RunResultV2(
            exit_code=1 if budget_aborted else 0,
            run_id=run_id,
            records_written=total_mentions + total_assertions + total_statuses,
            mention_count=total_mentions,
            assertion_count=total_assertions,
            complete_chunk_count=total_complete,
            reject_count=len(collected_rejects),
            budget_aborted=budget_aborted,
        )
    except BudgetExceeded:
        # Partial state still durable; the next clean run will overwrite.
        for cid in list(mentions_by_chunk):
            mentions_by_chunk[cid].sort(key=lambda r: r.get("mention_id", ""))
        for cid in list(assertions_by_chunk):
            assertions_by_chunk[cid].sort(key=lambda r: r.get("assertion_id", ""))
        _atomic_rewrite_jsonl(
            output_jsonl,
            _format_records_v2(
                mentions_by_chunk=mentions_by_chunk,
                assertions_by_chunk=assertions_by_chunk,
                statuses_by_chunk=statuses_by_chunk,
            ),
        )
        _atomic_rewrite_jsonl(
            output_rejects,
            (json.dumps(r, ensure_ascii=False, sort_keys=True) for r in collected_rejects),
        )
        return RunResultV2(
            exit_code=1,
            run_id=run_id,
            records_written=sum(len(v) for v in mentions_by_chunk.values())
            + sum(len(v) for v in assertions_by_chunk.values())
            + len(statuses_by_chunk),
            mention_count=sum(len(v) for v in mentions_by_chunk.values()),
            assertion_count=sum(len(v) for v in assertions_by_chunk.values()),
            complete_chunk_count=sum(
                1 for s in statuses_by_chunk.values() if s.get("extraction_complete")
            ),
            reject_count=len(collected_rejects),
            budget_aborted=True,
        )


def _merge_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=(a.input_tokens or 0) + (b.input_tokens or 0) or None,
        output_tokens=(a.output_tokens or 0) + (b.output_tokens or 0) or None,
        cost_usd=(
            (a.cost_usd or 0.0) + (b.cost_usd or 0.0) or None
        ),
    )


# =====================================================================
# T3b — Canonical v2 artifact set, manifest, and semantic resume (#55)
# =====================================================================
#
# Companion entry ``run_extraction_v2_canonical`` adds the canonical
# v2 artifact set (manifest + current-state rejects + per-chunk
# carry-forward on resume) alongside the frozen T3a fresh-run
# semantics of ``run_extraction_v2``. Crash-safe ordering is:
# precompute → unlink prior manifest → atomic JSONL → atomic rejects
# → atomic manifest LAST.


DEFAULT_OUTPUT_V2_JSONL = REPO_ROOT / "data" / "extractions.v2.jsonl"
DEFAULT_OUTPUT_V2_REJECTS = REPO_ROOT / "data" / "extractions.v2.rejects.jsonl"
DEFAULT_OUTPUT_V2_MANIFEST = REPO_ROOT / "data" / "extractions.v2.manifest.json"


@dataclass(frozen=True)
class RunResultV2Canonical:
    """Return value of :func:`run_extraction_v2_canonical`.

    Mirrors :class:`RunResultV2` plus the T3b audit fields:
    ``resumed_from_run_id``, ``cumulative_tokens``,
    ``cumulative_cost_usd`` (with zeros on fresh run / mismatch).
    ``reject_count`` is the current-state canonical reject count.
    """

    exit_code: int
    run_id: str
    pipeline_version: str
    resumed_from_run_id: str | None
    records_written: int
    mention_count: int
    assertion_count: int
    complete_chunk_count: int
    reject_count: int
    cumulative_tokens: int
    cumulative_cost_usd: float
    budget_aborted: bool = False


@dataclass(frozen=True)
class ManifestV2:
    """Canonical v2 manifest written by ``run_extraction_v2_canonical``.

    11 fields participate in resume identity (also see
    ``_run_id_fingerprint_v2`` / ``_identity_matches_v2``). The rest
    are audit-only.
    """

    schema_version: int
    run_id: str
    resumed_from_run_id: str | None
    chunks_sha256: str
    ontology_sha256: str
    prompt_template_sha256: tuple
    system_prompt_sha256: str
    provider_name: str
    provider_model: str
    max_tokens: int
    temperature: float
    alignment_algorithm_version: str
    alignment_algorithm_sha256: str

    ontology_version: int
    prompt_template_files: tuple
    chunks_path: str
    ontology_path: str
    pipeline_version: str
    cache_format_version: str
    wall_clock_started_at: str
    wall_clock_ended_at: str
    cumulative_tokens: int
    cumulative_cost_usd: float
    reject_count: int
    chunk_count_total: int
    chunk_count_complete: int

    def as_manifest_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "resumed_from_run_id": self.resumed_from_run_id,
            "chunks_sha256": self.chunks_sha256,
            "ontology_sha256": self.ontology_sha256,
            "prompt_template_sha256": list(self.prompt_template_sha256),
            "system_prompt_sha256": self.system_prompt_sha256,
            "provider": self.provider_name,
            "model": self.provider_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "alignment_algorithm_version": self.alignment_algorithm_version,
            "alignment_algorithm_sha256": self.alignment_algorithm_sha256,
            "ontology_version": self.ontology_version,
            "prompt_template_files": list(self.prompt_template_files),
            "chunks_path": self.chunks_path,
            "ontology_path": self.ontology_path,
            "pipeline_version": self.pipeline_version,
            "cache_format_version": self.cache_format_version,
            "wall_clock_started_at": self.wall_clock_started_at,
            "wall_clock_ended_at": self.wall_clock_ended_at,
            "cumulative_tokens": self.cumulative_tokens,
            "cumulative_cost_usd": round(self.cumulative_cost_usd, 8),
            "reject_count": self.reject_count,
            "chunk_count_total": self.chunk_count_total,
            "chunk_count_complete": self.chunk_count_complete,
        }


def _atomic_write_json_canonical(path: Path, payload: dict) -> None:
    """Atomic JSON write with fsync. The manifest is the durability
    marker — its atomicity matters because every other artifact is
    invalid until it appears."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with io.open(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _invalidate_prior_manifest(path: Path) -> None:
    """Unlink ``path`` if it exists; idempotent.

    Manifest invalidation is a strict unlink per #55 — never a
    sentinel/empty file. Missing manifest means no prior identity.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _read_existing_v2_records(path: Path) -> tuple[list, list, dict]:
    """Load v2 ``extractions.v2.jsonl`` into ``(mentions, assertions, statuses)``.

    Fail-closed: every record must carry
    ``schema_version == SCHEMA_VERSION_V2``. Malformed JSON, missing
    ``chunk_id``, or unknown ``kind`` raise :class:`PreflightErrorV2`
    with the offending line number. Returns empty maps when ``path``
    is absent (no prior state → fresh run).
    """
    mentions: list = []
    assertions: list = []
    statuses: dict = {}
    if not path.is_file():
        return mentions, assertions, statuses
    for lineno, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreflightErrorV2(
                f"canonical: malformed JSONL at {path}:{lineno}: {exc}"
            ) from exc
        if not isinstance(rec, dict):
            raise PreflightErrorV2(
                f"canonical: record at {path}:{lineno} is not an object"
            )
        sv = rec.get("schema_version")
        if type(sv) is not int or sv != schema_module.SCHEMA_VERSION_V2:
            raise PreflightErrorV2(
                f"canonical: record at {path}:{lineno} has schema_version="
                f"{sv!r}; expected SCHEMA_VERSION_V2="
                f"{schema_module.SCHEMA_VERSION_V2}"
            )
        kind = rec.get("kind")
        cid = rec.get("chunk_id")
        if not isinstance(cid, str):
            raise PreflightErrorV2(
                f"canonical: record at {path}:{lineno} missing string chunk_id"
            )
        if kind == "mention":
            mentions.append(rec)
        elif kind == "assertion":
            assertions.append(rec)
        elif kind == "chunk_status":
            statuses[cid] = rec
        else:
            raise PreflightErrorV2(
                f"canonical: record at {path}:{lineno} has unknown kind={kind!r}"
            )
    mentions.sort(key=lambda r: r.get("mention_id", ""))
    assertions.sort(key=lambda r: r.get("assertion_id", ""))
    return mentions, assertions, statuses


def _read_existing_v2_rejects(path: Path) -> list:
    """Load strictly versioned canonical reject records."""
    rejects: list = []
    if not path.is_file():
        return rejects
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PreflightErrorV2(f"canonical: malformed rejects JSONL at {path}:{lineno}: {exc}") from exc
        if not isinstance(rec, dict):
            raise PreflightErrorV2(f"canonical: reject at {path}:{lineno} is not an object")
        sv = rec.get("schema_version")
        if type(sv) is not int or sv != schema_module.SCHEMA_VERSION_V2:
            raise PreflightErrorV2(f"canonical: reject at {path}:{lineno} has invalid schema_version={sv!r}")
        rejects.append(rec)
    return rejects


def _canonical_reject_records(rejects: list) -> Iterable[dict]:
    for reject in rejects:
        yield {"schema_version": schema_module.SCHEMA_VERSION_V2, **reject}


def _read_existing_v2_manifest(path: Path) -> dict | None:
    """Load a closed, structurally valid canonical manifest."""
    if not path.is_file(): return None
    try: rec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise PreflightErrorV2(f"canonical: manifest at {path} is not valid JSON: {exc}") from exc
    if not isinstance(rec, dict): raise PreflightErrorV2(f"canonical: manifest at {path} is not a JSON object")
    keys = {"schema_version","run_id","resumed_from_run_id","chunks_sha256","ontology_sha256","prompt_template_sha256","system_prompt_sha256","provider","model","max_tokens","temperature","alignment_algorithm_version","alignment_algorithm_sha256","ontology_version","prompt_template_files","chunks_path","ontology_path","pipeline_version","cache_format_version","wall_clock_started_at","wall_clock_ended_at","cumulative_tokens","cumulative_cost_usd","reject_count","chunk_count_total","chunk_count_complete"}
    if set(rec) != keys: raise PreflightErrorV2(f"canonical: manifest at {path} must have exactly required keys")
    ints=("schema_version","ontology_version","max_tokens","cumulative_tokens","reject_count","chunk_count_total","chunk_count_complete")
    strings=("run_id","chunks_sha256","ontology_sha256","system_prompt_sha256","provider","model","alignment_algorithm_version","alignment_algorithm_sha256","chunks_path","ontology_path","pipeline_version","cache_format_version","wall_clock_started_at","wall_clock_ended_at")
    if type(rec["schema_version"]) is not int or rec["schema_version"] != schema_module.SCHEMA_VERSION_V2: raise PreflightErrorV2("canonical: manifest has invalid schema_version")
    if any(type(rec[x]) is not int for x in ints) or any(type(rec[x]) is not str for x in strings): raise PreflightErrorV2("canonical: manifest has invalid scalar types")
    if rec["resumed_from_run_id"] is not None and type(rec["resumed_from_run_id"]) is not str: raise PreflightErrorV2("canonical: manifest has invalid resumed_from_run_id")
    if type(rec["temperature"]) is not float or type(rec["cumulative_cost_usd"]) is not float: raise PreflightErrorV2("canonical: manifest has invalid float types")
    for key in ("prompt_template_sha256","prompt_template_files"):
        if type(rec[key]) is not list or len(rec[key]) != 2 or any(type(x) is not str for x in rec[key]): raise PreflightErrorV2(f"canonical: manifest has invalid {key}")
    return rec


def _identity_matches_v2(
    *,
    prev_manifest: dict | None,
    preflight: PreflightReportV2,
) -> bool:
    """True iff every one of the 11 resume-identity fields matches.

    All 11 must match; any missing field or type mismatch fails
    closed. ``prev_manifest is None`` returns False (no prior
    identity → no carry-forward). The list fields
    (``prompt_template_sha256``) are compared as ordered tuples
    so a swap invalidates identity.
    """
    if not isinstance(prev_manifest, dict):
        return False
    fields = (
        ("schema_version", preflight.schema_version),
        ("chunks_sha256", preflight.chunks_sha256),
        ("ontology_sha256", preflight.ontology_sha256),
        ("system_prompt_sha256", preflight.system_prompt_sha256),
        ("provider", preflight.provider_name),
        ("model", preflight.provider_model),
        ("max_tokens", preflight.max_tokens),
        ("temperature", preflight.temperature),
        (
            "alignment_algorithm_version",
            preflight.alignment_algorithm_version,
        ),
        (
            "alignment_algorithm_sha256",
            preflight.alignment_algorithm_sha256,
        ),
    )
    for key, expected in fields:
        actual = prev_manifest.get(key)
        if type(actual) is not type(expected) or actual != expected:
            return False
    prev_prompts = prev_manifest.get("prompt_template_sha256")
    cur_prompts = [
        preflight.prompt_entity_sha256,
        preflight.prompt_relation_sha256,
    ]
    if (type(prev_prompts) is not list or len(prev_prompts) != 2
            or any(type(x) is not str for x in prev_prompts)
            or prev_prompts != cur_prompts):
        return False
    return True


def _carry_forward_for_chunk(
    *,
    chunk: dict,
    status: dict,
    mentions: list,
    assertions: list,
    ontology: dict,
) -> bool:
    """True iff a prior chunk is safe to carry forward (#55 §"Carry-forward").

    Enforces every #55 acceptance condition for a single chunk:

    * chunk_status is a v2 chunk_status, ``extraction_complete``
      is True, both phase statuses are ``validated``, both
      ``failure_count == 0``, both ``mention_count`` /
      ``assertion_count`` equal the actual carried counts.
    * Every carried mention re-derives its v2 ID; its evidence
      char range matches the chunk text byte-for-byte; its
      source_text_sha256 equals ``sha256(chunk.text)``.
    * Every carried assertion re-derives its v2 ID; its
      source_text_sha256 equals the chunk sha; every mention-kind
      endpoint resolves to a carried same-chunk mention and is
      fully contained in the assertion's evidence span (#49
      ``ENDPOINT_NOT_CONTAINED``); every document-kind endpoint
      equals ``chunk.doc_id``.

    Failure on any condition returns ``False``; the caller
    silently narrows the resume for that chunk only.
    """
    from src.extract import ids as ids_module_inner

    chunk_id = chunk["chunk_id"]
    doc_id = chunk["doc_id"]
    text = chunk["text"]
    chunk_sha = validation_module.source_text_sha256(text)

    if not isinstance(status, dict):
        return False
    if status.get("kind") != "chunk_status":
        return False
    if status.get("schema_version") != schema_module.SCHEMA_VERSION_V2:
        return False
    if status.get("doc_id") != doc_id:
        return False
    if status.get("chunk_id") != chunk_id:
        return False
    if status.get("extraction_complete") is not True:
        return False
    phases = status.get("phases") or {}
    ent = phases.get("entity") or {}
    rel = phases.get("relation") or {}
    if ent.get("status") != "validated" or ent.get("failure_count", 0) != 0:
        return False
    if rel.get("status") != "validated" or rel.get("failure_count", 0) != 0:
        return False
    if ent.get("mention_count", -1) != len(mentions):
        return False
    if rel.get("assertion_count", -1) != len(assertions):
        return False

    mention_index = {m["mention_id"]: m for m in mentions}
    entity_types = {m.get("mention_id"): m.get("entity_type") for m in mentions}

    for m in mentions:
        if m.get("schema_version") != schema_module.SCHEMA_VERSION_V2:
            return False
        if m.get("kind") != "mention":
            return False
        if m.get("chunk_id") != chunk_id:
            return False
        if m.get("doc_id") != doc_id:
            return False
        ev = m.get("evidence") or {}
        try:
            if ids_module_inner.rederive_v2_mention_id(m) != m["mention_id"]:
                return False
            cs = int(ev["char_start"])
            ce = int(ev["char_end"])
            if not (0 <= cs < ce <= len(text)):
                return False
            validation_module.check_mention_evidence_v2(mention=m, chunk_text=text, expected_source_text_sha256=chunk_sha)
            validation_module.check_mention_ontology(mention=m, ontology=ontology)
        except (KeyError, TypeError, ValueError, validation_module.EvidenceError, validation_module.OntologyError):
            return False

    for a in assertions:
        if a.get("schema_version") != schema_module.SCHEMA_VERSION_V2:
            return False
        if a.get("kind") != "assertion":
            return False
        if a.get("chunk_id") != chunk_id:
            return False
        if a.get("doc_id") != doc_id:
            return False
        ev = a.get("evidence") or {}
        try:
            if ids_module_inner.rederive_v2_assertion_id(a) != a["assertion_id"]:
                return False
            acs = int(ev["char_start"])
            ace = int(ev["char_end"])
            if not (0 <= acs < ace <= len(text)):
                return False
            validation_module.check_assertion_evidence_v2(assertion=a, chunk_text=text, expected_source_text_sha256=chunk_sha)
            validation_module.check_assertion_ontology(assertion=a, ontology=ontology, doc_id=doc_id, same_chunk_mentions=mention_index, entity_type_for_mention=entity_types)
        except (KeyError, TypeError, ValueError, validation_module.EvidenceError, validation_module.OntologyError):
            return False
        for label, ref in (
            ("subject_ref", a.get("subject_ref")),
            ("object_ref", a.get("object_ref")),
        ):
            if not isinstance(ref, dict):
                return False
            kind = ref.get("kind")
            ref_id = ref.get("id")
            if kind == "mention":
                if not isinstance(ref_id, str):
                    return False
                # #49 ENDPOINT_NOT_CONTAINED: defer the
                # containment check to the aligner helper
                # rather than recompute it locally.
                target = mention_index.get(ref_id)
                if target is None:
                    return False
                mention_spans = {mid: (m["evidence"]["char_start"], m["evidence"]["char_end"]) for mid, m in mention_index.items()}
                if not align_module._endpoint_span_contained(
                    relation_start=acs,
                    relation_end=ace,
                    endpoint=ref,
                    aligned_mentions_by_id=mention_spans,
                ):
                    return False
            elif kind == "document":
                if ref_id != doc_id:
                    return False
            else:
                return False
    return True


def run_extraction_v2_canonical(
    *,
    provider: LLMProvider,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    prompts_dir: Path = prompts_module.PROMPTS_DIR,
    output_jsonl: Path = DEFAULT_OUTPUT_V2_JSONL,
    output_rejects: Path = DEFAULT_OUTPUT_V2_REJECTS,
    output_manifest: Path = DEFAULT_OUTPUT_V2_MANIFEST,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> RunResultV2Canonical:
    """Canonical v2 entry: manifest + semantic resume (#55).

    Semantics:

    * Every fresh run starts with ``run_id`` derived from the
      current preflight identity (all 11 resume-identity
      dimensions). On a resume-compatible prior manifest, the
      cumulative tokens/cost seed from the prior manifest and
      ``resumed_from_run_id`` is set; on any mismatch the resume
      silently narrows to a fresh run.
    * Chunks whose prior ``chunk_status.extraction_complete``
      passes the carry-forward validator are reused verbatim;
      chunks that fail the validator are silently re-run, as are
      chunks with no prior record.
    * The canonical artifact set is crash-safe: prior manifest is
      unlinked before any new disk write, then JSONL + rejects +
      manifest are rewritten in that order. A crash before the
      final manifest write leaves the manifest absent ⇒ next
      invocation treats the JSONL/rejects as untrusted.

    Always-rewritten artifacts (``output_jsonl``,
    ``output_rejects``, ``output_manifest``) are pre-truncated by
    the atomic-replace path, so a successful run always produces
    exactly the artifact set the manifest advertises.
    """
    ontology, chunks, prompt_artifacts, report = _preflight_v2(
        chunks_path=chunks_path,
        ontology_path=ontology_path,
        prompts_dir=prompts_dir,
        provider=provider,
        max_tokens=max_tokens,
        temperature=temperature,
        require_pipeline_version=True,
    )

    schemas = {
        "entity": schema_module.entity_pass_v2_schema(ontology),
        "relation": schema_module.relation_pass_v2_schema(ontology),
    }
    entity_artifact = prompt_artifacts[V2_ENTITY_PROMPT_FILENAME]
    relation_artifact = prompt_artifacts[V2_RELATION_PROMPT_FILENAME]

    # Pre-compute manifest-driven state. All in-memory work happens
    # before any disk mutation so a crash before the first write
    # leaves the prior on-disk set untouched.
    prev_manifest = _read_existing_v2_manifest(output_manifest)
    # Operator-facing invariant: manifest present ⇒ every other
    # artifact in the canonical set is consistent with it. A
    # manifest without its sibling JSONL or rejects is unrecoverable
    # (we cannot know which chunks had been completed), so it is a
    # hard preflight error rather than a silent re-run.
    if prev_manifest is not None:
        if not output_jsonl.is_file():
            raise PreflightErrorV2(
                f"canonical: manifest present at {output_manifest} "
                f"but canonical JSONL is missing at {output_jsonl}"
            )
        if not output_rejects.is_file():
            raise PreflightErrorV2(
                f"canonical: manifest present at {output_manifest} "
                f"but canonical rejects are missing at {output_rejects}"
            )
    prev_mentions: list = []
    prev_assertions: list = []
    prev_statuses: dict = {}
    if prev_manifest is not None:
        prev_mentions, prev_assertions, prev_statuses = _read_existing_v2_records(output_jsonl)
        _read_existing_v2_rejects(output_rejects)
    identity_ok = _identity_matches_v2(
        prev_manifest=prev_manifest, preflight=report,
    )
    resumed_from: str | None = None
    cumulative_tokens = 0
    cumulative_cost = 0.0
    carry_mentions_by_chunk: dict = {}
    carry_assertions_by_chunk: dict = {}
    carry_statuses_by_chunk: dict = {}
    if identity_ok and prev_manifest is not None:
        resumed_from = prev_manifest.get("run_id")
        rft = prev_manifest.get("cumulative_tokens", 0)
        if isinstance(rft, int):
            cumulative_tokens = rft
        rfc = prev_manifest.get("cumulative_cost_usd", 0.0)
        if isinstance(rfc, (int, float)):
            cumulative_cost = float(rfc)
        prev_mentions_by_chunk: dict = {}
        for m in prev_mentions:
            prev_mentions_by_chunk.setdefault(m["chunk_id"], []).append(m)
        prev_assertions_by_chunk: dict = {}
        for a in prev_assertions:
            prev_assertions_by_chunk.setdefault(a["chunk_id"], []).append(a)
        for chunk in chunks:
            cid = chunk["chunk_id"]
            status = prev_statuses.get(cid)
            if status is None:
                continue
            ms = prev_mentions_by_chunk.get(cid, [])
            as_ = prev_assertions_by_chunk.get(cid, [])
            if _carry_forward_for_chunk(
                chunk=chunk, status=status, mentions=ms, assertions=as_, ontology=ontology,
            ):
                carry_mentions_by_chunk[cid] = ms
                carry_assertions_by_chunk[cid] = as_
                carry_statuses_by_chunk[cid] = status

    started_at = _utc_now_compact()
    run_id = _make_run_id_v2(preflight=report)
    budget_aborted = False
    mentions_by_chunk = {k: list(v) for k, v in carry_mentions_by_chunk.items()}
    assertions_by_chunk = {
        k: list(v) for k, v in carry_assertions_by_chunk.items()
    }
    statuses_by_chunk = dict(carry_statuses_by_chunk)
    collected_rejects: list = []

    try:
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in statuses_by_chunk:
                # Resumed from prior valid chunk; skip re-run.
                continue
            try:
                (
                    entity_outcome,
                    validated_mentions,
                    entity_usage,
                ) = _run_entity_phase_v2(
                    chunk=chunk,
                    ontology=ontology,
                    prompt_artifact=entity_artifact,
                    provider=provider,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schemas["entity"],
                    preflight=report,
                    system_prompt=report.system_prompt,
                )
            except BudgetExceeded:
                budget_aborted = True
                break

            cumulative_tokens, cumulative_cost = _accumulate_usage(
                cumulative_tokens=cumulative_tokens,
                cumulative_cost=cumulative_cost,
                usage=entity_usage,
            )

            mentions_by_chunk[chunk_id] = [m.to_record() for m in validated_mentions]

            entity_type_lookup = {
                m.mention_id: m.entity_type for m in validated_mentions
            }

            try:
                (
                    relation_outcome,
                    validated_assertions,
                    relation_usage,
                ) = _run_relation_phase_v2(
                    chunk=chunk,
                    ontology=ontology,
                    prompt_artifact=relation_artifact,
                    provider=provider,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    schema=schemas["relation"],
                    preflight=report,
                    same_chunk_mentions=validated_mentions,
                    entity_type_for_mention=entity_type_lookup,
                    system_prompt=report.system_prompt,
                )
            except BudgetExceeded:
                budget_aborted = True
                break

            cumulative_tokens, cumulative_cost = _accumulate_usage(
                cumulative_tokens=cumulative_tokens,
                cumulative_cost=cumulative_cost,
                usage=relation_usage,
            )

            assertions_by_chunk[chunk_id] = [
                a.to_record() for a in validated_assertions
            ]
            statuses_by_chunk[chunk_id] = _build_chunk_status_v2(
                chunk_id=chunk_id,
                doc_id=chunk["doc_id"],
                entity_outcome=entity_outcome,
                relation_outcome=relation_outcome,
            )

            collected_rejects.extend(
                entity_outcome.reject_records + relation_outcome.reject_records
            )

        # Crash-safe ordering: invalidate manifest FIRST so a crash
        # never leaves a stale manifest pointing at partial JSONL.
        # Then atomic-replace the JSONL and the rejects. Finally
        # write the manifest — its presence advertises consistency.
        _invalidate_prior_manifest(output_manifest)

        for cid in list(mentions_by_chunk):
            mentions_by_chunk[cid].sort(key=lambda r: r.get("mention_id", ""))
        for cid in list(assertions_by_chunk):
            assertions_by_chunk[cid].sort(key=lambda r: r.get("assertion_id", ""))

        _atomic_rewrite_jsonl(
            output_jsonl,
            _format_records_v2(
                mentions_by_chunk=mentions_by_chunk,
                assertions_by_chunk=assertions_by_chunk,
                statuses_by_chunk=statuses_by_chunk,
            ),
        )
        _atomic_rewrite_jsonl(
            output_rejects,
            (
                json.dumps(r, ensure_ascii=False, sort_keys=True)
                for r in _canonical_reject_records(collected_rejects)
            ),
        )

        ended_at = _utc_now_compact()
        total_complete = sum(
            1 for s in statuses_by_chunk.values()
            if isinstance(s, dict) and s.get("extraction_complete")
        )
        manifest = ManifestV2(
            schema_version=report.schema_version,
            run_id=run_id,
            pipeline_version=report.pipeline_version,
            resumed_from_run_id=resumed_from,
            chunks_sha256=report.chunks_sha256,
            ontology_sha256=report.ontology_sha256,
            prompt_template_sha256=(
                report.prompt_entity_sha256,
                report.prompt_relation_sha256,
            ),
            system_prompt_sha256=report.system_prompt_sha256,
            provider_name=report.provider_name,
            provider_model=report.provider_model,
            max_tokens=report.max_tokens,
            temperature=report.temperature,
            alignment_algorithm_version=report.alignment_algorithm_version,
            alignment_algorithm_sha256=report.alignment_algorithm_sha256,
            ontology_version=report.ontology_version,
            prompt_template_files=(
                report.prompt_entity_filename,
                report.prompt_relation_filename,
            ),
            chunks_path=str(report.chunks_path),
            ontology_path=str(report.ontology_path),
            cache_format_version=report.cache_format_version,
            wall_clock_started_at=started_at,
            wall_clock_ended_at=ended_at,
            cumulative_tokens=cumulative_tokens,
            cumulative_cost_usd=cumulative_cost,
            reject_count=len(collected_rejects),
            chunk_count_total=len(chunks),
            chunk_count_complete=total_complete,
        )
        _atomic_write_json_canonical(output_manifest, manifest.as_manifest_dict())

        total_mentions = sum(len(v) for v in mentions_by_chunk.values())
        total_assertions = sum(len(v) for v in assertions_by_chunk.values())

        return RunResultV2Canonical(
            exit_code=1 if budget_aborted else 0,
            run_id=run_id,
            pipeline_version=report.pipeline_version,
            resumed_from_run_id=resumed_from,
            records_written=total_mentions + total_assertions + len(statuses_by_chunk),
            mention_count=total_mentions,
            assertion_count=total_assertions,
            complete_chunk_count=total_complete,
            reject_count=len(collected_rejects),
            cumulative_tokens=cumulative_tokens,
            cumulative_cost_usd=cumulative_cost,
            budget_aborted=budget_aborted,
        )
    except BudgetExceeded:
        # Partial state still durable: write whatever we have so
        # the next invocation can resume the remaining chunks.
        _invalidate_prior_manifest(output_manifest)

        for cid in list(mentions_by_chunk):
            mentions_by_chunk[cid].sort(key=lambda r: r.get("mention_id", ""))
        for cid in list(assertions_by_chunk):
            assertions_by_chunk[cid].sort(key=lambda r: r.get("assertion_id", ""))

        _atomic_rewrite_jsonl(
            output_jsonl,
            _format_records_v2(
                mentions_by_chunk=mentions_by_chunk,
                assertions_by_chunk=assertions_by_chunk,
                statuses_by_chunk=statuses_by_chunk,
            ),
        )
        _atomic_rewrite_jsonl(
            output_rejects,
            (
                json.dumps(r, ensure_ascii=False, sort_keys=True)
                for r in _canonical_reject_records(collected_rejects)
            ),
        )
        ended_at = _utc_now_compact()
        total_complete = sum(
            1 for s in statuses_by_chunk.values()
            if isinstance(s, dict) and s.get("extraction_complete")
        )
        manifest = ManifestV2(
            schema_version=report.schema_version,
            run_id=run_id,
            pipeline_version=report.pipeline_version,
            resumed_from_run_id=resumed_from,
            chunks_sha256=report.chunks_sha256,
            ontology_sha256=report.ontology_sha256,
            prompt_template_sha256=(
                report.prompt_entity_sha256,
                report.prompt_relation_sha256,
            ),
            system_prompt_sha256=report.system_prompt_sha256,
            provider_name=report.provider_name,
            provider_model=report.provider_model,
            max_tokens=report.max_tokens,
            temperature=report.temperature,
            alignment_algorithm_version=report.alignment_algorithm_version,
            alignment_algorithm_sha256=report.alignment_algorithm_sha256,
            ontology_version=report.ontology_version,
            prompt_template_files=(
                report.prompt_entity_filename,
                report.prompt_relation_filename,
            ),
            chunks_path=str(report.chunks_path),
            ontology_path=str(report.ontology_path),
            cache_format_version=report.cache_format_version,
            wall_clock_started_at=started_at,
            wall_clock_ended_at=ended_at,
            cumulative_tokens=cumulative_tokens,
            cumulative_cost_usd=cumulative_cost,
            reject_count=len(collected_rejects),
            chunk_count_total=len(chunks),
            chunk_count_complete=total_complete,
        )
        _atomic_write_json_canonical(output_manifest, manifest.as_manifest_dict())
        total_mentions = sum(len(v) for v in mentions_by_chunk.values())
        total_assertions = sum(len(v) for v in assertions_by_chunk.values())
        return RunResultV2Canonical(
            exit_code=1,
            run_id=run_id,
            pipeline_version=report.pipeline_version,
            resumed_from_run_id=resumed_from,
            records_written=total_mentions + total_assertions + len(statuses_by_chunk),
            mention_count=total_mentions,
            assertion_count=total_assertions,
            complete_chunk_count=total_complete,
            reject_count=len(collected_rejects),
            cumulative_tokens=cumulative_tokens,
            cumulative_cost_usd=cumulative_cost,
            budget_aborted=True,
        )


def _accumulate_usage(
    *,
    cumulative_tokens: int,
    cumulative_cost: float,
    usage: Usage,
) -> tuple[int, float]:
    """Add ``usage`` (which may have ``None`` fields) to the running totals."""
    if usage.input_tokens is not None:
        cumulative_tokens += usage.input_tokens
    if usage.output_tokens is not None:
        cumulative_tokens += usage.output_tokens
    if usage.cost_usd is not None:
        cumulative_cost += usage.cost_usd
    return cumulative_tokens, cumulative_cost



