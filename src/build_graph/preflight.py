"""§2 hard preflight for Neo4j ingestion (#25, #9).

Implements every numbered rule from
``docs/specs/neo4j-ingestion.md`` §2. The preflight is a pure function
over on-disk canonical artifacts; it never opens a Neo4j session. A
:class:`PreflightError` is raised on any failure with a precise reason;
the existing graph and prior ingestion manifest are left untouched.

Returns a :class:`PreflightBundle` containing the loaded inputs needed by
the orchestrator and the ingestion transaction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.build_graph.schema import (
    entity_subtype_label,
    is_chunk_id,
    is_entity_id,
    is_openalex_id,
    normalize_openalex_id,
)


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


# Population modes per ontology spec §3.
_STRUCTURAL_POPULATION_MODE = "structural"
_EXTRACTED_POPULATION_MODE = "extracted"
_DERIVED_POPULATION_MODE = "derived"


# ---------------------------------------------------------------------------
# Record dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DocumentRecord:
    doc_id: str
    title: str


@dataclass(frozen=True)
class _ChunkRecord:
    chunk_id: str
    doc_id: str
    section_path: list
    chunk_index: int


@dataclass(frozen=True)
class _EntityRecord:
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list
    resolution_confidence: float


@dataclass(frozen=True)
class _MentionRecord:
    mention_id: str
    chunk_id: str
    doc_id: str
    entity_type: str
    resolution_method: str
    resolution_score: float


@dataclass(frozen=True)
class _AssertionRecord:
    assertion_id: str
    chunk_id: str
    doc_id: str
    relation_type: str
    subject_ref: dict
    object_ref: dict


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
    inputs: dict  # canonical input paths + shas
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_repo_relative(path: Path) -> str:
    """Manifest path strings are repository-relative POSIX (§7.3 mirror).

    Paths outside the repository (e.g. in test fixtures) fall back to
    their absolute POSIX form so the manifest is still well-formed.
    """
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
        return rel.as_posix()
    except ValueError:
        return path.resolve().as_posix()

def _check_exact(record: dict, required: frozenset, *, where: str) -> None:
    extra = set(record) - required
    missing = required - set(record)
    if extra or missing:
        raise PreflightError(
            f"{where}: fields mismatch (extra={sorted(extra) or '∅'}, "
            f"missing={sorted(missing) or '∅'})"
        )


def _load_ontology(path: Path) -> dict:
    if not path.is_file():
        raise PreflightError(f"ontology file missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _validate_ontology(ontology: dict) -> tuple:
    """Return ``(entity_subtypes, version)``."""
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
    subtypes: dict = {}
    for et in entity_types:
        if not isinstance(et, dict):
            raise PreflightError("entity_type entry must be a mapping")
        name = et.get("name")
        if not isinstance(name, str) or not name:
            raise PreflightError("entity_type.name must be a non-empty string")
        mode = et.get("population_mode")
        if mode not in {
            _STRUCTURAL_POPULATION_MODE,
            _EXTRACTED_POPULATION_MODE,
            _DERIVED_POPULATION_MODE,
        }:
            raise PreflightError(
                f"entity_type {name!r}: invalid population_mode {mode!r}"
            )
        if mode == _EXTRACTED_POPULATION_MODE:
            subtype = entity_subtype_label(name)
            if subtype is None:
                raise PreflightError(
                    f"entity_type {name!r}: extracted type maps to a structural label"
                )
            subtypes[name] = subtype
        else:
            subtypes[name] = ""
    return subtypes, version

# ---------------------------------------------------------------------------
# Document validation (§2.1 rule 9).
# ---------------------------------------------------------------------------


def _validate_documents(path: Path) -> dict:
    """§2.1 rule 9 input integrity: every manifest record has a
    matching raw OpenAlex file. Returns documents keyed by doc_id."""
    docs: dict = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise PreflightError("manifest record must be a JSON object")
        doc_id = row.get("openalex_id")
        title = row.get("title")
        if not is_openalex_id(doc_id):
            raise PreflightError(
                f"manifest doc_id {doc_id!r} does not match OpenAlex W pattern"
            )
        if not isinstance(title, str) or not title:
            raise PreflightError(f"manifest title must be a non-empty string: {doc_id!r}")
        if doc_id in docs:
            raise PreflightError(f"duplicate doc_id in manifest: {doc_id!r}")
        docs[doc_id] = _DocumentRecord(doc_id=doc_id, title=title)
    return docs


# ---------------------------------------------------------------------------
# Chunk validation.
# ---------------------------------------------------------------------------


_REQUIRED_CHUNK_FIELDS = frozenset(
    {"chunk_id", "doc_id", "section_path", "chunk_index"}
)


def _validate_chunks(path: Path, documents: dict) -> dict:
    chunks: dict = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise PreflightError("chunk record must be a JSON object")
        # §3.4: these four properties are required; other fields
        # (text, token_estimate, char_start, char_end) live on the
        # canonical artifact but are not part of the graph projection.
        for field in _REQUIRED_CHUNK_FIELDS:
            if field not in row:
                raise PreflightError(
                    f"chunk record missing required field: {field!r}"
                )
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
            isinstance(p, str) for p in section_path
        ):
            raise PreflightError(
                f"chunk {cid!r}: section_path must be list[str]"
            )
        chunk_index = row["chunk_index"]
        if not isinstance(chunk_index, int):
            raise PreflightError(
                f"chunk {cid!r}: chunk_index must be int, got "
                f"{type(chunk_index).__name__}"
            )
        chunks[cid] = _ChunkRecord(
            chunk_id=cid,
            doc_id=doc_id,
            section_path=list(section_path),
            chunk_index=chunk_index,
        )
    return chunks
# ---------------------------------------------------------------------------
# Entity validation.
# ---------------------------------------------------------------------------


_REQUIRED_ENTITY_FIELDS = frozenset(
    {"entity_id", "entity_type", "canonical_name", "aliases", "resolution_confidence"}
)


def _validate_entities(path: Path, entity_subtypes: dict) -> dict:
    entities: dict = {}
    for row in _read_jsonl(path):
        if not isinstance(row, dict):
            raise PreflightError("entity record must be a JSON object")
        # §3.4 requires these five properties on Entity nodes; the
        # source ``entities.jsonl`` may also carry ``mention_ids`` (an
        # audit field) which is not duplicated on the graph node.
        for field in _REQUIRED_ENTITY_FIELDS:
            if field not in row:
                raise PreflightError(
                    f"entity record missing required field: {field!r}"
                )
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
        aliases = row["aliases"]
        if not isinstance(aliases, list) or not all(
            isinstance(a, str) for a in aliases
        ):
            raise PreflightError(
                f"entity {eid!r}: aliases must be list[str]"
            )
        confidence = row["resolution_confidence"]
        if not isinstance(confidence, (int, float)):
            raise PreflightError(
                f"entity {eid!r}: resolution_confidence must be numeric"
            )
        entities[eid] = _EntityRecord(
            entity_id=eid,
            entity_type=etype,
            canonical_name=canonical_name,
            aliases=list(aliases),
            resolution_confidence=float(confidence),
        )
    return entities


# ---------------------------------------------------------------------------
# Mention + assertion + resolution validation (§2.1 rules 3–7).
# ---------------------------------------------------------------------------


def _validate_mention_record(
    record: dict, *, chunks: dict, entity_subtypes: dict
) -> _MentionRecord:
    required = frozenset(
        {
            "kind",
            "mention_id",
            "chunk_id",
            "doc_id",
            "entity_type",
            "evidence",
            "surface_form",
            "canonical_form",
            "confidence",
        }
    )
    _check_exact(record, required, where="mention record")
    cid = record["chunk_id"]
    if cid not in chunks:
        raise PreflightError(
            f"mention {record['mention_id']!r}: chunk_id {cid!r} not in chunks"
        )
    doc_id = record["doc_id"]
    if doc_id != chunks[cid].doc_id:
        raise PreflightError(
            f"mention {record['mention_id']!r}: doc_id {doc_id!r} disagrees "
            f"with chunk doc_id {chunks[cid].doc_id!r}"
        )
    etype = record["entity_type"]
    if etype not in entity_subtypes:
        raise PreflightError(
            f"mention {record['mention_id']!r}: entity_type {etype!r} not extracted"
        )
    return _MentionRecord(
        mention_id=record["mention_id"],
        chunk_id=cid,
        doc_id=doc_id,
        entity_type=etype,
        resolution_method=record.get("_resolution_method", "exact_normalized"),
        resolution_score=float(record.get("_resolution_score", 1.0)),
    )


def _validate_assertion_record(
    record: dict, *, chunks: dict, relation_endpoint_kinds: dict
) -> _AssertionRecord:
    required = frozenset(
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
    _check_exact(record, required, where="assertion record")
    cid = record["chunk_id"]
    if cid not in chunks:
        raise PreflightError(
            f"assertion {record['assertion_id']!r}: chunk_id {cid!r} not in chunks"
        )
    doc_id = record["doc_id"]
    if doc_id != chunks[cid].doc_id:
        raise PreflightError(
            f"assertion {record['assertion_id']!r}: doc_id disagrees with chunk"
        )
    relation_type = record["relation_type"]
    if relation_type not in relation_endpoint_kinds:
        raise PreflightError(
            f"assertion {record['assertion_id']!r}: relation_type "
            f"{relation_type!r} not in ontology"
        )
    subject = record["subject_ref"]
    obj = record["object_ref"]
    for ref, where in ((subject, "subject_ref"), (obj, "object_ref")):
        if not isinstance(ref, dict):
            raise PreflightError(
                f"assertion {record['assertion_id']!r}: {where} must be a dict"
            )
        if set(ref) != {"kind", "id"}:
            raise PreflightError(
                f"assertion {record['assertion_id']!r}: {where} keys mismatch"
            )
        if ref["kind"] not in {"document", "mention"}:
            raise PreflightError(
                f"assertion {record['assertion_id']!r}: {where}.kind invalid"
            )
    return _AssertionRecord(
        assertion_id=record["assertion_id"],
        chunk_id=cid,
        doc_id=doc_id,
        relation_type=relation_type,
        subject_ref=dict(subject),
        object_ref=dict(obj),
    )


def _validate_resolution_record(
    record: dict, *, mentions: dict, entities: dict, mention_to_entity: dict
) -> None:
    """§2.1 rule 7: mention_id maps to exactly one entity_id."""
    mid = record.get("mention_id")
    eid = record.get("entity_id")
    if not isinstance(mid, str) or mid not in mentions:
        raise PreflightError(
            f"entity_resolution: mention_id {mid!r} not in mentions"
        )
    if not isinstance(eid, str) or eid not in entities:
        raise PreflightError(
            f"entity_resolution: entity_id {eid!r} not in entities"
        )
    if mid in mention_to_entity:
        if mention_to_entity[mid] != eid:
            raise PreflightError(
                f"entity_resolution: mention_id {mid!r} maps to multiple entities"
            )
    else:
        mention_to_entity[mid] = eid
    mention = mentions[mid]
    entity = entities[eid]
    if entity.entity_type != mention.entity_type:
        raise PreflightError(
            f"entity_resolution: mention {mid!r} entity_type "
            f"{mention.entity_type!r} != entity {eid!r} type {entity.entity_type!r}"
        )


# ---------------------------------------------------------------------------
# Citation-source validation (§4.4 + §8.2 citation_source_sha256).
# ---------------------------------------------------------------------------


def _load_aliases(path: Path) -> dict:
    """Load ``data/aliases.json`` (§4.4) and validate that every key and
    every value is a canonical OpenAlex ID and the alias graph is acyclic.
    """
    if not path.is_file():
        # Aliases are optional; absence is allowed.
        return {}
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
    # Acyclic alias graph: every key's transitive closure must terminate
    # without revisiting itself.
    for key in raw:
        seen: set = set()
        current = key
        while current in raw:
            if current in seen:
                raise PreflightError(
                    f"aliases.json: cycle through {key!r}"
                )
            seen.add(current)
            current = raw[current]
    return dict(raw)


def _canonicalize_referenced_works(values) -> list:
    """Map every entry of a raw OpenAlex ``referenced_works`` field to its
    canonical ``W...`` short form. Unknown forms are silently dropped
    (per §4.4, only canonical manifest IDs survive in-corpus filtering).
    """
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
    documents: dict,
    openalex_dir: Path,
    aliases: dict,
) -> tuple:
    """§4.4 + §8.2: for every manifest doc, read its raw OpenAlex
    ``referenced_works`` field, normalize to canonical short IDs,
    sort + de-duplicate per record, and serialize to the canonical
    ``citation_source_sha256`` digest bytes.

    The serialization happens BEFORE alias remapping and in-corpus
    filtering (§8.2 last paragraph).

    Returns (citation_pairs, citation_source_bytes). ``citation_pairs`` is
    the de-duplicated, in-corpus, self-loop-removed (citing, cited) pairs
    after alias remapping.
    """
    if not openalex_dir.is_dir():
        raise PreflightError(f"raw OpenAlex dir missing: {openalex_dir}")

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
                f"raw OpenAlex {doc_id}.json is not valid JSON: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise PreflightError(
                f"raw OpenAlex {doc_id}.json must be a JSON object"
            )
        raw_ids = _canonicalize_referenced_works(raw.get("referenced_works"))
        records.append({"doc_id": doc_id, "referenced_work_ids": raw_ids})
        for cited in raw_ids:
            target = aliases.get(cited, cited)
            if target == doc_id:
                # Self-loop from alias canonicalisation; drop per §4.4.
                continue
            if target not in documents:
                # External; not in corpus. Drop per §4.4.
                continue
            pairs.add((doc_id, target))

    # §8.2: serialize sorted + dedup, before alias remapping/in-corpus filter.
    from src.build_graph.schema import canonical_json_bytes

    citation_source_bytes = canonical_json_bytes(records)
    sorted_pairs = sorted(pairs)
    return sorted_pairs, citation_source_bytes


# ---------------------------------------------------------------------------
# Top-level orchestrator (§2.1, rule-by-rule).
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
    ):
        if not path.is_file():
            raise PreflightError(f"required input missing: {path}")

    # ---- §2.1 rule 1/2: ontology shape + version -------------------
    ontology = _load_ontology(ontology_path)
    entity_subtypes, ontology_version = _validate_ontology(ontology)
    ontology_sha = _file_sha256(ontology_path)

    # ---- §2.1 rule 1/9: manifest shape -----------------------------
    documents = _validate_documents(manifest_path)

    # ---- chunks: §2.1 rule 2/3 ---------------------------------------
    chunks = _validate_chunks(chunks_path, documents)

    # ---- entities: §2.1 rule 1/4 -----------------------------------
    entities = _validate_entities(entities_path, entity_subtypes)

    # ---- extractions manifest + record validation ------------------
    if not extractions_manifest_path.is_file():
        raise PreflightError(
            f"extractions manifest missing: {extractions_manifest_path}"
        )
    try:
        extractions_manifest = json.loads(extractions_manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise PreflightError(
            f"extractions manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(extractions_manifest, dict):
        raise PreflightError("extractions manifest must be a JSON object")

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
            f"ontology.yaml"
        )

    # ---- aliases + raw OpenAlex -------------------------------------
    aliases = _load_aliases(aliases_path)
    citation_pairs, citation_source_bytes = _build_citation_source(
        documents, openalex_dir, aliases
    )

    # ---- mentions + assertions + resolutions ------------------------
    relation_endpoint_kinds: dict = {}
    for rt in ontology.get("relation_types", []):
        if isinstance(rt, dict) and rt.get("population_mode") == _EXTRACTED_POPULATION_MODE:
            relation_endpoint_kinds[rt["name"]] = (rt["from"], rt["to"])

    mentions: dict = {}
    assertions: dict = {}
    mention_to_entity: dict = {}
    extraction_chunks_seen: set = set()

    for row in _read_jsonl(extractions_path):
        if not isinstance(row, dict):
            raise PreflightError("extraction record must be a JSON object")
        kind = row.get("kind")
        if kind == "mention":
            mention = _validate_mention_record(row, chunks=chunks, entity_subtypes=entity_subtypes)
            if mention.mention_id in mentions:
                raise PreflightError(
                    f"duplicate mention_id in extractions: {mention.mention_id!r}"
                )
            mentions[mention.mention_id] = mention
            extraction_chunks_seen.add(mention.chunk_id)
        elif kind == "assertion":
            assertion = _validate_assertion_record(
                row, chunks=chunks, relation_endpoint_kinds=relation_endpoint_kinds
            )
            if assertion.assertion_id in assertions:
                raise PreflightError(
                    f"duplicate assertion_id in extractions: {assertion.assertion_id!r}"
                )
            assertions[assertion.assertion_id] = assertion
            extraction_chunks_seen.add(assertion.chunk_id)
        elif kind == "chunk_status":
            if not row.get("extraction_complete"):
                raise PreflightError(
                    f"chunk {row.get('chunk_id')!r}: extraction_complete != true"
                )
            extraction_chunks_seen.add(row["chunk_id"])
    missing_chunks = set(chunks) - extraction_chunks_seen
    if missing_chunks:
        raise PreflightError(
            f"{len(missing_chunks)} chunk(s) missing any extraction record "
            f"(first: {sorted(missing_chunks)[0]!r})"
        )

    for row in _read_jsonl(entity_resolutions_path):
        if not isinstance(row, dict):
            raise PreflightError("entity_resolutions record must be a JSON object")
        _validate_resolution_record(
            row, mentions=mentions, entities=entities, mention_to_entity=mention_to_entity
        )

    if not mention_to_entity:
        raise PreflightError("entity_resolutions produced no mention → entity mapping")

    # Final structural checks: every assertion's mention endpoint
    # resolves, document endpoints match the chunk's doc.
    for assertion in assertions.values():
        for ref in (assertion.subject_ref, assertion.object_ref):
            if ref["kind"] == "document":
                if ref["id"] != assertion.doc_id:
                    raise PreflightError(
                        f"assertion {assertion.assertion_id!r}: document endpoint "
                        f"{ref['id']!r} != chunk doc {assertion.doc_id!r}"
                    )
            else:
                mid = ref["id"]
                if mid not in mention_to_entity:
                    raise PreflightError(
                        f"assertion {assertion.assertion_id!r}: mention ref {mid!r} "
                        f"not resolved"
                    )

    # Compute all named input SHAs.
    extractions_sha = _file_sha256(extractions_path)
    extractions_manifest_sha = _file_sha256(extractions_manifest_path)
    entities_sha = _file_sha256(entities_path)
    entity_resolutions_sha = _file_sha256(entity_resolutions_path)
    entity_resolution_manifest_sha = _file_sha256(entity_resolution_manifest_path)
    manifest_sha = _file_sha256(manifest_path)
    aliases_sha = _file_sha256(aliases_path) if aliases_path.is_file() else ""

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
        "aliases_path": _canonical_repo_relative(aliases_path) if aliases_path.is_file() else None,
        "aliases_sha256": aliases_sha or None,
    }

    return PreflightBundle(
        documents=documents,
        chunks=chunks,
        entities=entities,
        mentions=mentions,
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
    )
