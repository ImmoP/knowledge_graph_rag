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
import json
import os
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
from src.providers.cache import extraction_cache_scope
from src.providers.protocol import (
    LLMProvider,
    LLMValidationError,
    Usage,
)
from src.providers.budget import BudgetExceeded


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "derived" / "chunks.jsonl"
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "ontology.yaml"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "data" / "extractions_v2.jsonl"
DEFAULT_OUTPUT_REJECTS = REPO_ROOT / "data" / "extractions_v2.rejects.jsonl"

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


def _append_rejects(path: Path, records: list) -> None:
    """Append one JSONL line per reject record (inlined v1 helper)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


# ---------- preflight -------------------------------------------------


@dataclass(frozen=True)
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
) -> tuple[dict, list, dict, PreflightReportV2]:
    """Load ontology + chunks + v2 prompt artifacts; fail closed.

    Returns ``(ontology, chunks, prompt_artifacts, report)``.
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
    )
    return ontology, chunks, prompt_artifacts, report


# ---------- run state -------------------------------------------------


@dataclass
class RunStateV2:
    """Mutable per-run state for the v2 runner (#53)."""

    preflight: PreflightReportV2
    started_at: str
    usage: Usage = field(default_factory=Usage)
    budget_aborted: bool = False


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_fingerprint(*, preflight: PreflightReportV2) -> str:
    blob = "|".join((
        preflight.chunks_sha256,
        preflight.ontology_sha256,
        preflight.prompt_entity_sha256,
        preflight.prompt_relation_sha256,
    ))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _make_run_id(*, preflight: PreflightReportV2) -> str:
    return _utc_now_compact() + "-" + _run_fingerprint(preflight=preflight)


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
            _render_system_v2(),
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
            "gate": "aligner",
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
            _render_system_v2(),
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
            "gate": "aligner",
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
) -> PreflightReportV2:
    """Load ontology + chunks + v2 prompt artifacts; fail closed.

    Returns a :class:`PreflightReportV2`. Used by callers that want
    to fail early on missing artifacts before invoking
    :func:`run_extraction_v2`.
    """
    _, _, _, report = _preflight_v2(
        chunks_path=chunks_path,
        ontology_path=ontology_path,
        prompts_dir=prompts_dir,
    )
    return report


def run_extraction_v2(
    *,
    provider: LLMProvider,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    prompts_dir: Path = prompts_module.PROMPTS_DIR,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    output_rejects: Path = DEFAULT_OUTPUT_REJECTS,
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
    )

    schemas = {
        "entity": schema_module.entity_pass_v2_schema(ontology),
        "relation": schema_module.relation_pass_v2_schema(ontology),
    }
    entity_artifact = prompt_artifacts[V2_ENTITY_PROMPT_FILENAME]
    relation_artifact = prompt_artifacts[V2_RELATION_PROMPT_FILENAME]

    run_state = RunStateV2(
        preflight=report,
        started_at=_utc_now_compact(),
    )
    run_id = _make_run_id(preflight=report)

    mentions_by_chunk: dict = {}
    assertions_by_chunk: dict = {}
    statuses_by_chunk: dict = {}
    budget_aborted = False

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
                run_state.budget_aborted = True
                budget_aborted = True
                break

            run_state.usage = _merge_usage(run_state.usage, entity_usage)

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
                run_state.budget_aborted = True
                budget_aborted = True
                break

            run_state.usage = _merge_usage(run_state.usage, relation_usage)

            assertions_by_chunk[chunk_id] = [a.to_record() for a in validated_assertions]
            statuses_by_chunk[chunk_id] = _build_chunk_status_v2(
                chunk_id=chunk_id,
                doc_id=chunk["doc_id"],
                entity_outcome=entity_outcome,
                relation_outcome=relation_outcome,
            )

            new_rejects = entity_outcome.reject_records + relation_outcome.reject_records
            if new_rejects:
                _append_rejects(output_rejects, new_rejects)

        # Final pass: atomic write of the canonical JSONL.
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
            reject_count=_count_rejects(statuses_by_chunk),
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
            reject_count=_count_rejects(statuses_by_chunk),
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


def _count_rejects(statuses_by_chunk: dict) -> int:
    """Approximate total reject count from per-chunk phase outcomes.

    The reject log (jsonl append-only) is the source of truth; this
    helper only counts per-chunk phase-level rejects already recorded
    on the chunk_status record.
    """
    n = 0
    for status in statuses_by_chunk.values():
        phases = status.get("phases", {})
        n += phases.get("entity", {}).get("failure_count", 0)
        n += phases.get("relation", {}).get("failure_count", 0)
    return n



