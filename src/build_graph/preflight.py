"""§2 hard preflight for Neo4j ingestion (#25, #9).

Implements every numbered rule from
``docs/specs/neo4j-ingestion.md`` §2 verbatim. The preflight is a pure
function over on-disk canonical artifacts; it never opens a Neo4j
session. A :class:`PreflightError` is raised on any failure with a
precise reason; the existing graph and prior ingestion manifest are
left untouched.

The preflight reuses #23's ``src/extract/validation`` helpers for
evidence and ontology-endpoint checks, and the #8 constants from
``src/graph/entity_resolution`` for resolution contract checks.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.build_graph.schema import (
    derive_chunk_id,
    is_chunk_id,
    is_entity_id,
    is_openalex_id,
    normalize_openalex_id,
)
from src.extract import ids as extraction_ids
from src.extract import schema as extraction_schema
from src.extract import prompts as extraction_prompts
from src.extract.validation import (
    EvidenceError,
    OntologyError,
    check_assertion_evidence,
    check_assertion_ontology,
    check_mention_evidence,
    check_mention_ontology,
    source_text_sha256,
)
from src.graph import entity_resolution as er_module


# ---------------------------------------------------------------------------
# Public error type.
# ---------------------------------------------------------------------------


class PreflightError(RuntimeError):
    """Raised when the §2 hard preflight fails. The existing Neo4j
    graph and prior ingestion manifest are left untouched."""


# ---------------------------------------------------------------------------
# Path defaults.
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "derived" / "chunks.jsonl"
DEFAULT_ONTOLOGY_PATH = REPO_ROOT / "ontology.yaml"
DEFAULT_EXTRACTIONS_PATH = REPO_ROOT / "data" / "extractions.jsonl"
DEFAULT_EXTRACTIONS_MANIFEST_PATH = REPO_ROOT / "data" / "extractions.manifest.json"
DEFAULT_ENTITIES_PATH = REPO_ROOT / "data" / "entities.jsonl"
DEFAULT_ENTITY_RESOLUTIONS_PATH = REPO_ROOT / "data" / "entity_resolutions.jsonl"
DEFAULT_ENTITY_RESOLUTION_MANIFEST_PATH = (
    REPO_ROOT / "data" / "entity_resolution.manifest.json"
)
DEFAULT_ALIASES_PATH = REPO_ROOT / "data" / "aliases.json"
DEFAULT_OPENALEX_DIR = REPO_ROOT / "data" / "raw" / "openalex"
DEFAULT_OUTPUT_MANIFEST_PATH = REPO_ROOT / "data" / "graph_ingestion.manifest.json"


# Entity-resolution threshold contract per entity-resolution.md §4.1.
_ALLOWED_RESOLUTION_METHODS = frozenset({"exact_normalized", "fuzzy"})

# ---------------------------------------------------------------------------
# Record dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DocumentRecord:
    doc_id: str
    title: str
    raw_openalex_sha256: str  # per §2.1 rule 9


@dataclass(frozen=True)
class _ChunkRecord:
    chunk_id: str
    doc_id: str
    section_path: list
    chunk_index: int
    text: str


@dataclass(frozen=True)
class _EntityRecord:
    entity_id: str
    entity_type: str
    canonical_name: str
    mention_ids: list
    aliases: list
    resolution_confidence: float


@dataclass(frozen=True)
class _MentionRecord:
    mention_id: str
    chunk_id: str
    doc_id: str
    entity_type: str
    evidence: dict  # full §3.1.1 evidence
    confidence: str
    resolution_method: str  # populated from entity_resolutions.jsonl
    resolution_score: float  # populated from entity_resolutions.jsonl

    def to_record(self) -> dict:
        return {
            "kind": "mention",
            "mention_id": self.mention_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "entity_type": self.entity_type,
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class _AssertionRecord:
    assertion_id: str
    chunk_id: str
    doc_id: str
    relation_type: str
    subject_ref: dict
    object_ref: dict
    evidence: dict  # full §3.1.2 evidence
    confidence: str

    def to_record_proxy(self) -> dict:
        return {
            "kind": "assertion",
            "assertion_id": self.assertion_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "relation_type": self.relation_type,
            "subject_ref": dict(self.subject_ref),
            "object_ref": dict(self.object_ref),
            "evidence": dict(self.evidence),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PreflightBundle:
    """Loaded inputs needed by :mod:`src.build_graph.ingestion`."""

    documents: dict
    chunks: dict
    entities: dict
    mentions: dict
    mention_to_entity: dict
    assertions: dict
    ontology: dict
    ontology_version: int
    entity_subtypes: dict
    citation_pairs: list
    citation_source_bytes: bytes
    aliases: dict
    inputs: dict
    ontology_sha: str
    manifest_sha: str
    chunks_sha: str
    extractions_sha: str
    extractions_manifest_sha: str
    entities_sha: str
    entity_resolutions_sha: str
    entity_resolution_manifest_sha: str
    aliases_sha: str
    openalex_dir: Path
    er_algorithm_version: str
    er_normalization_version: str
    er_similarity_metric: str
    er_threshold: float


# ---------------------------------------------------------------------------
# Generic helpers.
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise PreflightError(f"input file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreflightError(
                    f"{path} line {line_number} is not valid JSON: {exc}"
                ) from exc


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise PreflightError(f"input file missing: {path}")
    try:
        out = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PreflightError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(out, dict):
        raise PreflightError(f"{path}: top-level must be a JSON object")
    return out


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_repo_relative(path: Path) -> str:
    """§7.3 / §8.2: manifest path strings are repository-relative POSIX.

    A path outside the repository fails closed — the spec forbids
    absolute paths in the published manifest.
    """
    rel = path.resolve().relative_to(REPO_ROOT)
    return rel.as_posix()


def _load_ontology(path: Path) -> dict:
    if not path.is_file():
        raise PreflightError(f"ontology file missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _check_exact(record: dict, required: frozenset, *, where: str) -> None:
    """§3 / §4 producer-shape compliance: the record's field set must
    exactly match ``required``."""
    extra = set(record) - required
    missing = required - set(record)
    if extra or missing:
        raise PreflightError(
            f"{where}: fields mismatch (extra={sorted(extra) or 'set()'}, "
            f"missing={sorted(missing) or 'set()'})"
        )


def _pascal_case_safe(entity_type: str) -> str | None:
    if entity_type in {"document", "chunk"}:
        return None
    return "".join(p[:1].upper() + p[1:] for p in entity_type.split("_"))


def _validate_ontology(ontology: dict) -> tuple:
    """§2.1 rule 8: every ontology relation endpoint maps to a node
    type admitted by §3.1 / §3.3.
    """
    if not isinstance(ontology, dict):
        raise PreflightError(
            f"ontology must be a YAML mapping, got {type(ontology).__name__}"
        )
    version = ontology.get("version")
    if not isinstance(version, int):
        raise PreflightError(
            f"ontology 'version' must be an int, got {type(version).__name__}"
        )
    entity_types = ontology.get("entity_types", [])
    if not isinstance(entity_types, list) or not entity_types:
        raise PreflightError("ontology.entity_types must be a non-empty list")
    relation_types = ontology.get("relation_types", [])
    if not isinstance(relation_types, list):
        raise PreflightError("ontology.relation_types must be a list")
    subtypes: dict = {}
    extracted_seen: set = set()
    structural_seen: set = set()
    for et in entity_types:
        if not isinstance(et, dict):
            raise PreflightError("entity_type entry must be a mapping")
        name = et.get("name")
        if not isinstance(name, str) or not name:
            raise PreflightError("entity_type.name must be a non-empty string")
        mode = et.get("population_mode")
        if mode not in {"structural", "extracted", "derived"}:
            raise PreflightError(
                f"entity_type {name!r}: invalid population_mode {mode!r}"
            )
        if mode == "extracted":
            subtype = _pascal_case_safe(name)
            if subtype is None:
                raise PreflightError(
                    f"entity_type {name!r}: extracted type maps to a structural label"
                )
            subtypes[name] = subtype
            extracted_seen.add(name)
        elif mode == "structural":
            if name not in {"document", "chunk"}:
                raise PreflightError(
                    f"entity_type {name!r}: structural label not admitted by §3.1"
                )
            structural_seen.add(name)
    if "document" not in structural_seen or "chunk" not in structural_seen:
        raise PreflightError(
            "ontology must declare structural entity_types 'document' and 'chunk'"
        )
    if not extracted_seen:
        raise PreflightError("ontology must declare at least one extracted entity_type")
    valid_entity_names = set(subtypes) | structural_seen
    for rt in relation_types:
        if not isinstance(rt, dict):
            raise PreflightError("relation_type entry must be a mapping")
        name = rt.get("name")
        if not isinstance(name, str) or not name:
            raise PreflightError("relation_type.name must be a non-empty string")
        mode = rt.get("population_mode")
        if mode not in {"structural", "extracted", "derived"}:
            raise PreflightError(
                f"relation_type {name!r}: invalid population_mode {mode!r}"
            )
        for slot in ("from", "to"):
            endpoint = rt.get(slot)
            if endpoint not in valid_entity_names:
                raise PreflightError(
                    f"relation_type {name!r}.{slot}: endpoint type {endpoint!r} "
                    f"not in ontology"
                )
    return subtypes, version


# ---------------------------------------------------------------------------
# #23 extraction-manifest producer contract.
# ---------------------------------------------------------------------------


_EXTRACTION_MANIFEST_FIELDS = frozenset(
    {
        "run_id",
        "resumed_from_run_id",
        "provider",
        "model",
        "ontology_version",
        "ontology_sha256",
        "prompt_template_files",
        "prompt_template_sha256",
        "schema_version",
        "chunks_path",
        "chunks_sha256",
        "wall_clock_started_at",
        "wall_clock_ended_at",
        "cumulative_tokens",
        "cumulative_cost_usd",
        "max_tokens",
        "temperature",
        "cache_format_version",
    }
)


def _validate_extractions_manifest_shape(manifest: dict) -> None:
    """Validate exactly the fields emitted by
    ``src.extract.pipeline.RunState.as_manifest``.

    Downstream ingestion consumes this immutable producer contract; it
    neither invents ``extractions_path`` nor accepts unknown fields.
    """
    _check_exact(
        manifest,
        _EXTRACTION_MANIFEST_FIELDS,
        where="extractions manifest",
    )
    string_fields = (
        "run_id",
        "provider",
        "model",
        "ontology_sha256",
        "chunks_path",
        "chunks_sha256",
        "wall_clock_started_at",
        "wall_clock_ended_at",
        "cache_format_version",
    )
    for field in string_fields:
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise PreflightError(
                f"extractions manifest field {field!r} must be a non-empty string"
            )
    resumed = manifest["resumed_from_run_id"]
    if resumed is not None and (not isinstance(resumed, str) or not resumed):
        raise PreflightError(
            "extractions manifest resumed_from_run_id must be null or non-empty string"
        )
    if isinstance(manifest["ontology_version"], bool) or not isinstance(
        manifest["ontology_version"], int
    ):
        raise PreflightError("extractions manifest ontology_version must be int")
    if manifest["schema_version"] != extraction_schema.SCHEMA_VERSION:
        raise PreflightError(
            "extractions manifest schema_version disagrees with the #23 contract"
        )
    expected_prompts = [
        extraction_prompts.ENTITY_PROMPT_FILE,
        extraction_prompts.RELATION_PROMPT_FILE,
    ]
    if manifest["prompt_template_files"] != expected_prompts:
        raise PreflightError(
            "extractions manifest prompt_template_files disagrees with #23"
        )
    prompt_shas = manifest["prompt_template_sha256"]
    if (
        not isinstance(prompt_shas, list)
        or len(prompt_shas) != 2
        or any(
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in prompt_shas
        )
    ):
        raise PreflightError(
            "extractions manifest prompt_template_sha256 must contain "
            "two lowercase SHA256 values"
        )
    for field in ("ontology_sha256", "chunks_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", manifest[field]):
            raise PreflightError(
                f"extractions manifest field {field!r} must be lowercase SHA256"
            )
    for field in ("cumulative_tokens", "max_tokens"):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PreflightError(
                f"extractions manifest field {field!r} must be non-negative int"
            )
    for field in ("cumulative_cost_usd", "temperature"):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PreflightError(
                f"extractions manifest field {field!r} must be numeric"
            )


# ---------------------------------------------------------------------------
# Document validation (§2.1 rule 9).
# ---------------------------------------------------------------------------


def _validate_documents(path: Path, openalex_dir: Path) -> dict:
    """§2.1 rule 9: every manifest record has its raw OpenAlex record,
    AND that record's bytes match the ``sha256`` recorded for the
    document. A present raw record whose ``referenced_works`` field is
    ``null`` or ``[]`` is valid zero-citation input. A missing raw
    record is an input-integrity failure.
    """
    if not openalex_dir.is_dir():
        raise PreflightError(f"raw OpenAlex dir missing: {openalex_dir}")
    docs: dict = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise PreflightError("manifest record must be a JSON object")
        doc_id = row.get("openalex_id")
        title = row.get("title")
        declared_sha = row.get("sha256")
        if not is_openalex_id(doc_id):
            raise PreflightError(
                f"manifest doc_id {doc_id!r} does not match OpenAlex W pattern"
            )
        if not isinstance(title, str) or not title:
            raise PreflightError(
                f"manifest title must be a non-empty string: {doc_id!r}"
            )
        if not isinstance(declared_sha, str) or not re.fullmatch(
            r"[0-9a-f]{64}", declared_sha
        ):
            raise PreflightError(
                f"manifest sha256 for {doc_id!r} must be a lowercase SHA256 hex"
            )
        if doc_id in docs:
            raise PreflightError(f"duplicate doc_id in manifest: {doc_id!r}")
        raw_path = openalex_dir / f"{doc_id}.json"
        if not raw_path.is_file():
            raise PreflightError(
                f"raw OpenAlex record missing for canonical doc {doc_id!r}"
            )
        actual_sha = _file_sha256(raw_path)
        if actual_sha != declared_sha:
            raise PreflightError(
                f"raw OpenAlex SHA drift for {doc_id!r}: manifest {declared_sha!r} "
                f"!= raw file {actual_sha!r}"
            )
        docs[doc_id] = _DocumentRecord(
            doc_id=doc_id,
            title=title,
            raw_openalex_sha256=actual_sha,
        )
    return docs


# ---------------------------------------------------------------------------
# Chunk validation (§2.1 rule 2 + chunk-record §2 + §3).
# ---------------------------------------------------------------------------


_CHUNK_RECORD_FIELDS = frozenset(
    {
        "chunk_id",
        "doc_id",
        "section_path",
        "chunk_index",
        "text",
        "token_estimate",
        "char_start",
        "char_end",
    }
)


def _validate_chunks(path: Path, documents: dict) -> dict:
    chunks: dict = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise PreflightError("chunk record must be a JSON object")
        _check_exact(row, _CHUNK_RECORD_FIELDS, where="chunk record")
        cid = row["chunk_id"]
        if not is_chunk_id(cid):
            raise PreflightError(f"chunk_id {cid!r} does not match pattern c_<16 hex>")
        if cid in chunks:
            raise PreflightError(f"duplicate chunk_id in chunks input: {cid!r}")
        doc_id = row["doc_id"]
        if doc_id not in documents:
            raise PreflightError(
                f"chunk {cid!r}: doc_id {doc_id!r} not in manifest"
            )
        section_path = row["section_path"]
        if not isinstance(section_path, list) or not all(
            isinstance(component, str) and bool(component)
            for component in section_path
        ):
            raise PreflightError(
                f"chunk {cid!r}: section_path must be list[str] with "
                "non-empty components"
            )
        chunk_index = row["chunk_index"]
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            raise PreflightError(
                f"chunk {cid!r}: chunk_index must be a non-negative int"
            )
        text = row["text"]
        if not isinstance(text, str) or not text:
            raise PreflightError(
                f"chunk {cid!r}: text must be a non-empty string"
            )
        token_estimate = row["token_estimate"]
        expected_token_estimate = len(text) // 4
        if (
            isinstance(token_estimate, bool)
            or not isinstance(token_estimate, int)
            or token_estimate != expected_token_estimate
        ):
            raise PreflightError(
                f"chunk {cid!r}: token_estimate must equal len(text) // 4 "
                f"({expected_token_estimate}), got {token_estimate!r}"
            )
        char_start = row["char_start"]
        char_end = row["char_end"]
        if (
            isinstance(char_start, bool)
            or isinstance(char_end, bool)
            or not isinstance(char_start, int)
            or not isinstance(char_end, int)
        ):
            raise PreflightError(
                f"chunk {cid!r}: char_start and char_end must be integers"
            )
        if char_start == -1:
            if char_end != -1:
                raise PreflightError(
                    f"chunk {cid!r}: char_end must be -1 when char_start "
                    "uses the -1 escape hatch"
                )
        elif not (char_end > char_start >= 0):
            raise PreflightError(
                f"chunk {cid!r}: offsets must satisfy char_end > "
                "char_start >= 0 or both be -1"
            )
        derived = derive_chunk_id(
            doc_id=doc_id, section_path=section_path, chunk_index=chunk_index
        )
        if derived != cid:
            raise PreflightError(
                f"chunk {cid!r}: chunk_id does not re-derive under the chunk "
                f"contract (expected {derived!r})"
            )
        chunks[cid] = _ChunkRecord(
            chunk_id=cid,
            doc_id=doc_id,
            section_path=list(section_path),
            chunk_index=chunk_index,
            text=text,
        )
    return chunks


# ---------------------------------------------------------------------------
# Entity validation.
# ---------------------------------------------------------------------------


_ENTITY_RECORD_FIELDS = frozenset(
    {
        "entity_id",
        "entity_type",
        "canonical_name",
        "mention_ids",
        "aliases",
        "resolution_confidence",
    }
)


def _validate_entities(path: Path, entity_subtypes: dict) -> dict:
    entities: dict = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise PreflightError("entity record must be a JSON object")
        _check_exact(row, _ENTITY_RECORD_FIELDS, where="entity record")
        eid = row["entity_id"]
        if not is_entity_id(eid):
            raise PreflightError(
                f"entity_id {eid!r} does not match pattern e_<16 hex>"
            )
        if eid in entities:
            raise PreflightError(f"duplicate entity_id: {eid!r}")
        etype = row["entity_type"]
        if not isinstance(etype, str) or not etype:
            raise PreflightError(
                f"entity {eid!r}: entity_type must be a non-empty string"
            )
        if etype not in entity_subtypes:
            raise PreflightError(
                f"entity {eid!r}: entity_type {etype!r} not in ontology"
            )
        canonical_name = row["canonical_name"]
        if not isinstance(canonical_name, str) or not canonical_name:
            raise PreflightError(
                f"entity {eid!r}: canonical_name must be a non-empty string"
            )
        derived_eid = er_module.entity_id(
            canonical_name=canonical_name,
            entity_type=etype,
        )
        if derived_eid != eid:
            raise PreflightError(
                f"entity {eid!r}: entity_id does not re-derive under #8 "
                f"contract (expected {derived_eid!r})"
            )
        mention_ids = row["mention_ids"]
        if (
            not isinstance(mention_ids, list)
            or not mention_ids
            or not all(isinstance(mid, str) and re.fullmatch(r"m_[0-9a-f]{16}", mid) for mid in mention_ids)
            or mention_ids != sorted(set(mention_ids))
        ):
            raise PreflightError(
                f"entity {eid!r}: mention_ids must be a non-empty sorted unique "
                "list of valid mention IDs"
            )
        aliases = row["aliases"]
        if not isinstance(aliases, list) or not all(
            isinstance(a, str) and a for a in aliases
        ):
            raise PreflightError(f"entity {eid!r}: aliases must be list[str]")
        confidence = row["resolution_confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise PreflightError(
                f"entity {eid!r}: resolution_confidence must be numeric"
            )
        entities[eid] = _EntityRecord(
            entity_id=eid,
            entity_type=etype,
            canonical_name=canonical_name,
            mention_ids=list(mention_ids),
            aliases=list(aliases),
            resolution_confidence=float(confidence),
        )
    return entities


# ---------------------------------------------------------------------------
# Mention + assertion + resolution validation (§2.1 rules 5–7).
# ---------------------------------------------------------------------------


_MENTION_RECORD_FIELDS = frozenset(
    {
        "kind",
        "mention_id",
        "chunk_id",
        "doc_id",
        "entity_type",
        "surface_form",
        "canonical_form",
        "evidence",
        "confidence",
    }
)
_ASSERTION_RECORD_FIELDS = frozenset(
    {
        "kind",
        "assertion_id",
        "chunk_id",
        "doc_id",
        "relation_type",
        "subject_ref",
        "object_ref",
        "evidence",
        "confidence",
    }
)
_EVIDENCE_FIELDS = frozenset({"char_start", "char_end", "source_text_sha256"})


def _validate_evidence_shape(evidence, *, where: str) -> None:
    if not isinstance(evidence, dict):
        raise PreflightError(f"{where}: evidence must be a JSON object")
    if set(evidence) != _EVIDENCE_FIELDS:
        raise PreflightError(
            f"{where}.evidence: fields mismatch (got {sorted(evidence)})"
        )
    cs, ce, sha = evidence["char_start"], evidence["char_end"], evidence["source_text_sha256"]
    if (
        isinstance(cs, bool)
        or isinstance(ce, bool)
        or not isinstance(cs, int)
        or not isinstance(ce, int)
    ):
        raise PreflightError(f"{where}.evidence: char offsets must be integers")
    if cs < 0 or ce <= cs:
        raise PreflightError(f"{where}.evidence: invalid offsets [{cs},{ce})")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise PreflightError(
            f"{where}.evidence.source_text_sha256 must be a lowercase SHA256"
        )


def _validate_mention_record(
    record: dict, *, chunks: dict, entity_subtypes: dict, ontology: dict
) -> _MentionRecord:
    _check_exact(record, _MENTION_RECORD_FIELDS, where="mention record")
    mid = record["mention_id"]
    cid = record["chunk_id"]
    if cid not in chunks:
        raise PreflightError(f"mention {mid!r}: chunk_id {cid!r} not in chunks")
    doc_id = record["doc_id"]
    if doc_id != chunks[cid].doc_id:
        raise PreflightError(
            f"mention {mid!r}: doc_id {doc_id!r} disagrees with chunk "
            f"doc_id {chunks[cid].doc_id!r}"
        )
    if record["confidence"] not in extraction_schema.CONFIDENCE_VALUES:
        raise PreflightError(f"mention {mid!r}: invalid confidence")
    chunk_text = chunks[cid].text
    expected_sha = source_text_sha256(chunk_text)
    _validate_evidence_shape(record["evidence"], where=f"mention {mid!r}")
    try:
        check_mention_ontology(mention=record, ontology=ontology)
        check_mention_evidence(
            mention=record,
            chunk_text=chunk_text,
            expected_source_text_sha256=expected_sha,
        )
    except (OntologyError, EvidenceError) as exc:
        raise PreflightError(str(exc)) from exc
    try:
        derived_mid = extraction_ids.rederive_mention_id(record)
    except Exception as exc:  # noqa: BLE001 — propagate as preflight failure
        raise PreflightError(
            f"mention {mid!r}: ID re-derivation failed: {exc}"
        ) from exc
    if derived_mid != mid:
        raise PreflightError(f"mention {mid!r}: ID does not re-derive from #23 contract")
    etype = record["entity_type"]
    if etype not in entity_subtypes:
        raise PreflightError(
            f"mention {mid!r}: entity_type {etype!r} not extracted"
        )
    return _MentionRecord(
        mention_id=mid,
        chunk_id=cid,
        doc_id=doc_id,
        entity_type=etype,
        evidence=dict(record["evidence"]),
        confidence=record["confidence"],
        resolution_method="",
        resolution_score=-1.0,
    )


def _validate_assertion_record(
    record: dict, *, chunks: dict, ontology: dict
) -> _AssertionRecord:
    _check_exact(record, _ASSERTION_RECORD_FIELDS, where="assertion record")
    aid = record["assertion_id"]
    cid = record["chunk_id"]
    if cid not in chunks:
        raise PreflightError(f"assertion {aid!r}: chunk_id {cid!r} not in chunks")
    doc_id = record["doc_id"]
    if doc_id != chunks[cid].doc_id:
        raise PreflightError(
            f"assertion {aid!r}: doc_id disagrees with chunk doc_id"
        )
    if record["confidence"] not in extraction_schema.CONFIDENCE_VALUES:
        raise PreflightError(f"assertion {aid!r}: invalid confidence")
    chunk_text = chunks[cid].text
    expected_sha = source_text_sha256(chunk_text)
    _validate_evidence_shape(record["evidence"], where=f"assertion {aid!r}")
    # The ontology gate is applied later when same-chunk mentions are
    # known; we still validate the relation_type is in the ontology.
    rt_map = extraction_schema.relation_endpoints(ontology)
    if record["relation_type"] not in rt_map:
        raise PreflightError(
            f"assertion {aid!r}: relation_type {record['relation_type']!r} "
            f"not in extracted ontology"
        )
    try:
        check_assertion_evidence(
            assertion=record,
            chunk_text=chunk_text,
            expected_source_text_sha256=expected_sha,
        )
    except EvidenceError as exc:
        raise PreflightError(str(exc)) from exc
    try:
        derived_aid = extraction_ids.rederive_assertion_id(record)
    except Exception as exc:  # noqa: BLE001 — propagate as preflight failure
        raise PreflightError(
            f"assertion {aid!r}: ID re-derivation failed: {exc}"
        ) from exc
    if derived_aid != aid:
        raise PreflightError(
            f"assertion {aid!r}: ID does not re-derive from #23 contract"
        )
    return _AssertionRecord(
        assertion_id=aid,
        chunk_id=cid,
        doc_id=doc_id,
        relation_type=record["relation_type"],
        subject_ref=dict(record["subject_ref"]),
        object_ref=dict(record["object_ref"]),
        evidence=dict(record["evidence"]),
        confidence=record["confidence"],
    )


_CHUNK_STATUS_FIELDS = frozenset(
    {"kind", "chunk_id", "doc_id", "phases", "extraction_complete"}
)
_PHASE_FIELDS = frozenset({"status", "attempt_count", "failure_count"})


def _validate_chunk_status_record(record: dict, *, chunks: dict) -> None:
    """Validate the exact #23 chunk_status persisted shape and require
    a completed, failure-free extraction generation."""
    _check_exact(record, _CHUNK_STATUS_FIELDS, where="chunk_status record")
    cid = record["chunk_id"]
    if cid not in chunks:
        raise PreflightError(
            f"chunk_status record references unknown chunk_id {cid!r}"
        )
    if record["doc_id"] != chunks[cid].doc_id:
        raise PreflightError(
            f"chunk_status doc_id {record['doc_id']!r} != chunk "
            f"doc_id {chunks[cid].doc_id!r}"
        )
    phases = record["phases"]
    if not isinstance(phases, dict) or set(phases) != {"entity", "relation"}:
        raise PreflightError(
            "chunk_status.phases must contain exactly entity and relation"
        )
    for phase, count_field in (
        ("entity", "mention_count"),
        ("relation", "assertion_count"),
    ):
        value = phases[phase]
        if not isinstance(value, dict):
            raise PreflightError(
                f"chunk_status.phases.{phase} must be an object"
            )
        _check_exact(
            value,
            _PHASE_FIELDS | {count_field},
            where=f"chunk_status.phases.{phase}",
        )
        if value["status"] not in {"validated", "rejected"}:
            raise PreflightError(
                f"chunk_status.phases.{phase}.status is invalid"
            )
        for field in ("attempt_count", "failure_count", count_field):
            count = value[field]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise PreflightError(
                    f"chunk_status.phases.{phase}.{field} must be non-negative int"
                )
    complete = (
        phases["entity"]["status"] == "validated"
        and phases["relation"]["status"] == "validated"
        and phases["entity"]["failure_count"] == 0
        and phases["relation"]["failure_count"] == 0
    )
    if (
        not isinstance(record["extraction_complete"], bool)
        or record["extraction_complete"] is not complete
        or record["extraction_complete"] is not True
    ):
        raise PreflightError(
            f"chunk {cid!r}: extraction_complete != true or disagrees "
            "with phase state"
        )


# Entity-resolution record shape (entity-resolution.md §7.2):
# exactly four persisted fields.
_RESOLUTION_RECORD_FIELDS = frozenset(
    {"mention_id", "entity_id", "resolution_method", "resolution_score"}
)


def _validate_resolution_record(record: dict, *, mentions: dict, entities: dict) -> tuple:
    """§2.1 rule 5 / rule 7.

    Returns ``(mention_id, entity_id)`` for the mention→entity map.
    """
    _check_exact(record, _RESOLUTION_RECORD_FIELDS, where="entity_resolutions record")
    mid = record["mention_id"]
    eid = record["entity_id"]
    method = record["resolution_method"]
    score = record["resolution_score"]
    if mid not in mentions:
        raise PreflightError(
            f"entity_resolutions: mention_id {mid!r} not in eligible mentions"
        )
    if eid not in entities:
        raise PreflightError(
            f"entity_resolutions: entity_id {eid!r} not in canonical entities"
        )
    if method not in _ALLOWED_RESOLUTION_METHODS:
        raise PreflightError(
            f"entity_resolutions: resolution_method {method!r} for {mid!r} "
            f"not in {sorted(_ALLOWED_RESOLUTION_METHODS)}"
        )
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise PreflightError(
            f"entity_resolutions: resolution_score for {mid!r} must be numeric"
        )
    score = float(score)
    if method == "exact_normalized" and score != 1.0:
        raise PreflightError(
            f"entity_resolutions: exact_normalized resolution for {mid!r} "
            f"must have score 1.0, got {score!r}"
        )
    if method == "fuzzy":
        threshold = er_module.NORMALIZED_LEVENSHTEIN_THRESHOLD
        if not (threshold <= score <= 1.0):
            raise PreflightError(
                f"entity_resolutions: fuzzy resolution for {mid!r} score "
                f"{score!r} outside [{threshold}, 1.0]"
            )
    mention = mentions[mid]
    entity = entities[eid]
    if entity.entity_type != mention.entity_type:
        raise PreflightError(
            f"entity_resolutions: mention {mid!r} entity_type "
            f"{mention.entity_type!r} != entity {eid!r} type "
            f"{entity.entity_type!r}"
        )
    return mid, eid


# ---------------------------------------------------------------------------
# Citation-source validation (§4.4 + §8.2 citation_source_sha256).
# ---------------------------------------------------------------------------


def _resolve_alias(openalex_id: str, aliases: dict[str, str]) -> str:
    """Follow an alias chain to its terminal OpenAlex ID.

    The helper is deliberately small: aliases are already validated as
    OpenAlex IDs, and a repeated node is the only invalid state.
    """
    current = openalex_id
    seen: set[str] = set()
    while current in aliases:
        if current in seen:
            raise PreflightError(f"aliases.json: cycle through {openalex_id!r}")
        seen.add(current)
        current = aliases[current]
    return current


def _load_aliases(path: Path) -> dict:
    """Load the required aliases file and validate an acyclic mapping.

    A present ``{}`` is valid. File presence is enforced by the global
    required-input gate before this function runs.
    """
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PreflightError(f"aliases.json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PreflightError("aliases.json must be a JSON object")
    for key, value in raw.items():
        if not is_openalex_id(key) or not is_openalex_id(value):
            raise PreflightError(
                f"aliases.json: invalid id {key!r}->{value!r} (must be W...)"
            )
    for key in raw:
        _resolve_alias(key, raw)
    return dict(raw)


def _canonicalize_referenced_works(values) -> list:
    if values is None:
        return []
    if not isinstance(values, list):
        raise PreflightError(
            "raw OpenAlex referenced_works must be a list of strings"
        )
    canonical: list = []
    for entry in values:
        cid = normalize_openalex_id(entry)
        if cid is None:
            continue
        canonical.append(cid)
    canonical = sorted(set(canonical))
    return canonical


def _build_citation_source(
    documents: dict, openalex_dir: Path, aliases: dict
) -> tuple:
    """§4.4 + §8.2: for every manifest doc, read its raw OpenAlex
    ``referenced_works`` field, normalize to canonical short IDs, sort
    + de-duplicate per record, and serialize to the canonical
    ``citation_source_sha256`` digest bytes.

    Serialization happens BEFORE alias remapping and in-corpus
    filtering. The combination of ``citation_source_sha256``,
    ``aliases_sha256``, and ``manifest_sha256`` fully identifies the
    deterministic input to ``CITES`` construction.
    """
    from src.build_graph.schema import canonical_json_bytes

    records: list = []
    pairs: set = set()
    for doc_id in sorted(documents):
        raw_path = openalex_dir / f"{doc_id}.json"
        if not raw_path.is_file():
            raise PreflightError(
                f"raw OpenAlex record missing for canonical doc {doc_id!r}"
            )
        try:
            raw = json.loads(raw_path.read_text())
        except json.JSONDecodeError as exc:
            raise PreflightError(
                f"raw OpenAlex {doc_id}.jsonl is not valid JSON: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise PreflightError(
                f"raw OpenAlex {doc_id}.jsonl must be a JSON object"
            )
        raw_ids = _canonicalize_referenced_works(raw.get("referenced_works"))
        records.append({"doc_id": doc_id, "referenced_work_ids": raw_ids})
        for cited in raw_ids:
            target = _resolve_alias(cited, aliases)
            if target == doc_id:
                continue
            if target not in documents:
                continue
            pairs.add((doc_id, target))
    citation_source_bytes = canonical_json_bytes(records)
    sorted_pairs = sorted(pairs)
    return sorted_pairs, citation_source_bytes


# ---------------------------------------------------------------------------
# Top-level orchestrator (§2.1 rule-by-rule).
# ---------------------------------------------------------------------------


def run_preflight(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    extractions_path: Path = DEFAULT_EXTRACTIONS_PATH,
    extractions_manifest_path: Path = DEFAULT_EXTRACTIONS_MANIFEST_PATH,
    entities_path: Path = DEFAULT_ENTITIES_PATH,
    entity_resolutions_path: Path = DEFAULT_ENTITY_RESOLUTIONS_PATH,
    entity_resolution_manifest_path: Path = DEFAULT_ENTITY_RESOLUTION_MANIFEST_PATH,
    aliases_path: Path = DEFAULT_ALIASES_PATH,
    openalex_dir: Path = DEFAULT_OPENALEX_DIR,
) -> PreflightBundle:
    """§2 hard preflight. Returns a :class:`PreflightBundle` on success;
    raises :class:`PreflightError` on any rule failure.

    The preflight reads inputs only; no Neo4j session is opened.
    """
    # ---- §2.1 rule 1: file presence ---------------------------------
    for path in (
        manifest_path,
        chunks_path,
        ontology_path,
        extractions_path,
        extractions_manifest_path,
        entities_path,
        entity_resolutions_path,
        entity_resolution_manifest_path,
        aliases_path,
    ):
        if not path.is_file():
            raise PreflightError(f"required input missing: {path}")

    # ---- §2.1 rule 8: ontology shape + relation endpoint types ------
    ontology = _load_ontology(ontology_path)
    entity_subtypes, ontology_version = _validate_ontology(ontology)
    ontology_sha = _file_sha256(ontology_path)

    # ---- §2.1 rule 9: manifest shape + raw OpenAlex SHA drift ------
    documents = _validate_documents(manifest_path, openalex_dir)

    # ---- chunks: §2.1 rule 2 + chunk-record §2 + §3 re-derivation --
    chunks = _validate_chunks(chunks_path, documents)

    # ---- entities: #8 producer shape -------------------------------
    entities = _validate_entities(entities_path, entity_subtypes)

    # ---- §2.1 rule 3: extractions manifest + completeness ---------
    extractions_manifest = _read_json(extractions_manifest_path)
    _validate_extractions_manifest_shape(extractions_manifest)
    chunks_sha = _file_sha256(chunks_path)
    if extractions_manifest.get("chunks_sha256") != chunks_sha:
        raise PreflightError(
            "extractions manifest chunks_sha256 disagrees with current "
            f"chunks input (got {extractions_manifest.get('chunks_sha256')!r}, "
            f"expected {chunks_sha!r})"
        )
    if extractions_manifest.get("ontology_sha256") != ontology_sha:
        raise PreflightError(
            "extractions manifest ontology_sha256 disagrees with current "
            f"ontology.yaml (got {extractions_manifest.get('ontology_sha256')!r}, "
            f"expected {ontology_sha!r})"
        )
    if extractions_manifest.get("ontology_version") != ontology_version:
        raise PreflightError(
            "extractions manifest ontology_version disagrees with current "
            "ontology.yaml"
        )

    # ---- aliases + raw OpenAlex SHA-aware citation digest ----------
    aliases = _load_aliases(aliases_path)
    citation_pairs, citation_source_bytes = _build_citation_source(
        documents, openalex_dir, aliases
    )

    # ---- mentions + assertions (typed-ref contract) ---------------
    relation_endpoint_kinds = extraction_schema.entity_type_endpoint_table(ontology)
    mentions: dict = {}
    assertions: dict = {}
    chunk_statuses: dict[str, dict] = {}

    for row in _read_jsonl(extractions_path):
        if not isinstance(row, dict):
            raise PreflightError("extraction record must be a JSON object")
        kind = row.get("kind")
        if kind == "mention":
            mention = _validate_mention_record(
                row,
                chunks=chunks,
                entity_subtypes=entity_subtypes,
                ontology=ontology,
            )
            if mention.mention_id in mentions:
                raise PreflightError(
                    f"duplicate mention_id in extractions: {mention.mention_id!r}"
                )
            mentions[mention.mention_id] = mention
        elif kind == "assertion":
            assertion = _validate_assertion_record(
                row, chunks=chunks, ontology=ontology
            )
            if assertion.assertion_id in assertions:
                raise PreflightError(
                    f"duplicate assertion_id in extractions: {assertion.assertion_id!r}"
                )
            assertions[assertion.assertion_id] = assertion
        elif kind == "chunk_status":
            _validate_chunk_status_record(row, chunks=chunks)
            cid = row["chunk_id"]
            if cid in chunk_statuses:
                raise PreflightError(
                    f"duplicate chunk_status records for chunk_id {cid!r}"
                )
            chunk_statuses[cid] = row
        else:
            raise PreflightError(
                f"unknown extraction record kind: {kind!r}"
            )
    missing_chunks = set(chunks) - set(chunk_statuses)
    if missing_chunks:
        raise PreflightError(
            f"missing chunk_status records for {len(missing_chunks)} chunk(s) "
            f"(first: {sorted(missing_chunks)[0]!r})"
        )
    extra_statuses = set(chunk_statuses) - set(chunks)
    if extra_statuses:
        raise PreflightError(
            f"chunk_status records reference unknown chunks; first: "
            f"{sorted(extra_statuses)[0]!r}"
        )
    mentions_per_chunk: dict[str, int] = {}
    assertions_per_chunk: dict[str, int] = {}
    for mention in mentions.values():
        mentions_per_chunk[mention.chunk_id] = mentions_per_chunk.get(mention.chunk_id, 0) + 1
    for assertion in assertions.values():
        assertions_per_chunk[assertion.chunk_id] = assertions_per_chunk.get(assertion.chunk_id, 0) + 1
    for cid, status in chunk_statuses.items():
        if status["phases"]["entity"]["mention_count"] != mentions_per_chunk.get(cid, 0):
            raise PreflightError(
                f"chunk_status mention_count disagrees for {cid!r}"
            )
        if status["phases"]["relation"]["assertion_count"] != assertions_per_chunk.get(cid, 0):
            raise PreflightError(
                f"chunk_status assertion_count disagrees for {cid!r}"
            )

    # ---- §2.1 rule 6 typed-ref: ontology endpoint type compat -------
    same_chunk_mentions: dict = {}
    for cid, chunk in chunks.items():
        same_chunk_mentions[cid] = {
            m.mention_id: {"entity_type": m.entity_type}
            for m in mentions.values()
            if m.chunk_id == cid
        }
    for assertion in assertions.values():
        try:
            check_assertion_ontology(
                assertion=assertion.to_record_proxy(),
                ontology=ontology,
                doc_id=assertion.doc_id,
                same_chunk_mentions=same_chunk_mentions.get(assertion.chunk_id, {}),
                entity_type_for_mention={
                    mid: m.entity_type
                    for mid, m in mentions.items()
                    if m.chunk_id == assertion.chunk_id
                },
            )
        except OntologyError as exc:
            raise PreflightError(
                f"assertion {assertion.assertion_id!r}: {exc}"
            ) from exc

    # ---- entity_resolutions.jsonl: §2.1 rule 5 + rule 7 ------------
    mention_to_entity: dict = {}
    resolution_method_by_mid: dict = {}
    resolution_score_by_mid: dict = {}
    seen_mids: set = set()
    for row in _read_jsonl(entity_resolutions_path):
        if not isinstance(row, dict):
            raise PreflightError("entity_resolutions record must be a JSON object")
        mid, eid = _validate_resolution_record(
            row, mentions=mentions, entities=entities
        )
        if mid in seen_mids:
            raise PreflightError(
                f"entity_resolutions: duplicate mapping for mention_id {mid!r}"
            )
        seen_mids.add(mid)
        mention_to_entity[mid] = eid
        resolution_method_by_mid[mid] = row["resolution_method"]
        resolution_score_by_mid[mid] = float(row["resolution_score"])
    missing_mentions = set(mentions) - set(mention_to_entity)
    if missing_mentions:
        raise PreflightError(
            f"entity_resolutions: {len(missing_mentions)} eligible mention(s) "
            f"have no resolution mapping (first: "
            f"{sorted(missing_mentions)[0]!r})"
        )
    member_to_entity: dict[str, str] = {}
    for eid, entity in entities.items():
        for mid in entity.mention_ids:
            if mid not in mentions:
                raise PreflightError(
                    f"entity {eid!r}: mention_ids contains ineligible mention {mid!r}"
                )
            if mid in member_to_entity:
                raise PreflightError(
                    f"eligible mention {mid!r} occurs in multiple entity membership lists"
                )
            if mentions[mid].entity_type != entity.entity_type:
                raise PreflightError(
                    f"entity {eid!r}: member mention {mid!r} has entity_type "
                    f"{mentions[mid].entity_type!r}, expected {entity.entity_type!r}"
                )
            member_to_entity[mid] = eid
    if set(member_to_entity) != set(mentions):
        missing_members = sorted(set(mentions) - set(member_to_entity))
        raise PreflightError(
            f"{len(missing_members)} eligible mention(s) missing from entity "
            f"membership; first: {missing_members[0]!r}"
        )
    for mid, eid in mention_to_entity.items():
        if member_to_entity[mid] != eid:
            raise PreflightError(
                f"entity membership for mention {mid!r} maps to "
                f"{member_to_entity[mid]!r}, resolution maps to {eid!r}"
            )

    # Thread resolution method/score back onto each mention record.
    mentions_threaded: dict = {}
    for mid, mention in mentions.items():
        mentions_threaded[mid] = _MentionRecord(
            mention_id=mention.mention_id,
            chunk_id=mention.chunk_id,
            doc_id=mention.doc_id,
            entity_type=mention.entity_type,
            evidence=mention.evidence,
            confidence=mention.confidence,
            resolution_method=resolution_method_by_mid[mid],
            resolution_score=resolution_score_by_mid[mid],
        )

    # ---- §2.1 rule 4: entity-resolution manifest hash agreement -----
    er_manifest = _read_json(entity_resolution_manifest_path)
    _validate_entity_resolution_manifest(
        er_manifest,
        chunks_sha=chunks_sha,
        ontology_sha=ontology_sha,
        ontology_version=ontology_version,
        extractions_sha=_file_sha256(extractions_path),
        extractions_manifest_sha=_file_sha256(extractions_manifest_path),
        entities_sha=_file_sha256(entities_path),
        entity_resolutions_sha=_file_sha256(entity_resolutions_path),
    )

    # ---- Compute all named input SHAs. -------------------------------
    extractions_sha = _file_sha256(extractions_path)
    extractions_manifest_sha = _file_sha256(extractions_manifest_path)
    entities_sha = _file_sha256(entities_path)
    entity_resolutions_sha = _file_sha256(entity_resolutions_path)
    entity_resolution_manifest_sha = _file_sha256(entity_resolution_manifest_path)
    manifest_sha = _file_sha256(manifest_path)
    aliases_sha = _file_sha256(aliases_path)

    inputs = {
        "manifest_path": _canonical_repo_relative(manifest_path),
        "manifest_sha256": manifest_sha,
        "chunks_path": _canonical_repo_relative(chunks_path),
        "chunks_sha256": chunks_sha,
        "ontology_path": _canonical_repo_relative(ontology_path),
        "ontology_sha256": ontology_sha,
        "ontology_version": ontology_version,
        "extractions_path": _canonical_repo_relative(extractions_path),
        "extractions_sha256": extractions_sha,
        "extractions_manifest_path": _canonical_repo_relative(extractions_manifest_path),
        "extractions_manifest_sha256": extractions_manifest_sha,
        "entities_path": _canonical_repo_relative(entities_path),
        "entities_sha256": entities_sha,
        "entity_resolutions_path": _canonical_repo_relative(entity_resolutions_path),
        "entity_resolutions_sha256": entity_resolutions_sha,
        "entity_resolution_manifest_path": _canonical_repo_relative(entity_resolution_manifest_path),
        "entity_resolution_manifest_sha256": entity_resolution_manifest_sha,
        "aliases_path": _canonical_repo_relative(aliases_path),
        "aliases_sha256": aliases_sha,
    }

    return PreflightBundle(
        documents=documents,
        chunks=chunks,
        entities=entities,
        mentions=mentions_threaded,
        mention_to_entity=mention_to_entity,
        assertions=assertions,
        ontology=ontology,
        ontology_version=ontology_version,
        entity_subtypes=entity_subtypes,
        citation_pairs=citation_pairs,
        citation_source_bytes=citation_source_bytes,
        aliases=aliases,
        inputs=inputs,
        ontology_sha=ontology_sha,
        manifest_sha=manifest_sha,
        chunks_sha=chunks_sha,
        extractions_sha=extractions_sha,
        extractions_manifest_sha=extractions_manifest_sha,
        entities_sha=entities_sha,
        entity_resolutions_sha=entity_resolutions_sha,
        entity_resolution_manifest_sha=entity_resolution_manifest_sha,
        aliases_sha=aliases_sha,
        openalex_dir=openalex_dir,
        er_algorithm_version=str(
            er_manifest.get("algorithm", {}).get("algorithm_version", "")
        ),
        er_normalization_version=str(
            er_manifest.get("algorithm", {}).get("normalization_version", "")
        ),
        er_similarity_metric=str(
            er_manifest.get("algorithm", {}).get("similarity_metric", "")
        ),
        er_threshold=float(er_manifest.get("algorithm", {}).get("threshold", 0.0)),
    )


def _validate_entity_resolution_manifest(
    manifest: dict,
    *,
    chunks_sha: str,
    ontology_sha: str,
    ontology_version: int,
    extractions_sha: str,
    extractions_manifest_sha: str,
    entities_sha: str,
    entity_resolutions_sha: str,
) -> None:
    """§2.1 rule 4: validate exactly the #8 producer manifest shape,
    algorithm identity, input hashes, output hashes, and record counts.
    """
    _check_exact(
        manifest,
        frozenset({"algorithm", "counts", "inputs", "outputs", "schema_version"}),
        where="entity_resolution manifest",
    )
    if manifest["schema_version"] != 1:
        raise PreflightError("entity_resolution manifest schema_version must be 1")
    inputs = manifest["inputs"]
    outputs = manifest["outputs"]
    algorithm = manifest["algorithm"]
    counts = manifest["counts"]
    if not all(isinstance(v, dict) for v in (inputs, outputs, algorithm, counts)):
        raise PreflightError(
            "entity_resolution manifest algorithm/counts/inputs/outputs must be objects"
        )
    _check_exact(
        algorithm,
        frozenset(
            {
                "algorithm_version",
                "normalization_version",
                "similarity_metric",
                "threshold",
            }
        ),
        where="entity_resolution manifest.algorithm",
    )
    expected_algorithm = {
        "algorithm_version": er_module.ENTITY_RESOLUTION_ALGORITHM_VERSION,
        "normalization_version": er_module.NORMALIZATION_VERSION,
        "similarity_metric": er_module.SIMILARITY_METRIC,
        "threshold": er_module.NORMALIZED_LEVENSHTEIN_THRESHOLD,
    }
    if algorithm != expected_algorithm:
        raise PreflightError(
            "entity_resolution manifest algorithm identity disagrees with #8 "
            f"(got {algorithm!r}, expected {expected_algorithm!r})"
        )
    _check_exact(
        inputs,
        frozenset(
            {
                "chunks_path",
                "chunks_sha256",
                "extractions_manifest_path",
                "extractions_manifest_sha256",
                "extractions_path",
                "extractions_sha256",
                "ontology_path",
                "ontology_sha256",
                "ontology_version",
            }
        ),
        where="entity_resolution manifest.inputs",
    )
    _check_exact(
        outputs,
        frozenset(
            {
                "entities_path",
                "entities_records",
                "entities_sha256",
                "entity_resolutions_path",
                "entity_resolutions_records",
                "entity_resolutions_sha256",
            }
        ),
        where="entity_resolution manifest.outputs",
    )
    _check_exact(
        counts,
        frozenset(
            {
                "eligible_mentions",
                "entities",
                "extraction_complete_chunks",
                "identical_duplicate_mentions",
            }
        ),
        where="entity_resolution manifest.counts",
    )
    expected_inputs = {
        "chunks_sha256": chunks_sha,
        "ontology_sha256": ontology_sha,
        "ontology_version": ontology_version,
        "extractions_sha256": extractions_sha,
        "extractions_manifest_sha256": extractions_manifest_sha,
    }
    for key, want in expected_inputs.items():
        if inputs.get(key) != want:
            raise PreflightError(
                f"entity_resolution manifest inputs.{key} disagrees "
                f"(got {inputs.get(key)!r}, expected {want!r})"
            )
    if outputs.get("entities_sha256") != entities_sha:
        raise PreflightError(
            "entity_resolution manifest outputs.entities_sha256 disagrees "
            f"with current entities.jsonl (got {outputs.get('entities_sha256')!r}, "
            f"expected {entities_sha!r})"
        )
    if outputs.get("entity_resolutions_sha256") != entity_resolutions_sha:
        raise PreflightError(
            "entity_resolution manifest outputs.entity_resolutions_sha256 "
            "disagrees with current entity_resolutions.jsonl "
            f"(got {outputs.get('entity_resolutions_sha256')!r}, "
            f"expected {entity_resolutions_sha!r})"
        )
