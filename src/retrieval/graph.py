"""Deterministic, read-only graph retriever for Issue #28.

The retriever lifts canonical seeds, validates the #25 projection, traverses
ontology relationships in Python, collects path-specific provenance, then
reduces one scored hit per chunk deterministically.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Protocol, runtime_checkable

from src.graph.entity_resolution import normalized_resolution_name

GRAPH_HOP_BUDGET = 3

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTITIES_PATH = _REPO_ROOT / "data" / "entities.jsonl"
DEFAULT_MANIFEST_PATH = _REPO_ROOT / "data" / "manifest.jsonl"
from src.build_graph.preflight import (
    DEFAULT_OPENALEX_DIR as _DEFAULT_OPENALEX_DIR,
    DEFAULT_OUTPUT_MANIFEST_PATH,
)

DEFAULT_GRAPH_INGESTION_MANIFEST_PATH = DEFAULT_OUTPUT_MANIFEST_PATH
DEFAULT_OPENALEX_DIR = _DEFAULT_OPENALEX_DIR
DEFAULT_GRAPH_DATABASE = "neo4j"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CANONICAL_INPUT_KEYS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "chunks_path",
        "chunks_sha256",
        "ontology_path",
        "ontology_sha256",
        "ontology_version",
        "extractions_path",
        "extractions_sha256",
        "extractions_manifest_path",
        "extractions_manifest_sha256",
        "entities_path",
        "entities_sha256",
        "entity_resolutions_path",
        "entity_resolutions_sha256",
        "entity_resolution_manifest_path",
        "entity_resolution_manifest_sha256",
        "aliases_path",
        "aliases_sha256",
    }
)
_CANONICAL_INPUT_HASH_NAMES = tuple(
    key for key in sorted(_CANONICAL_INPUT_KEYS) if key.endswith("_sha256")
)


@dataclass(frozen=True)
class GraphEvidenceStep:
    step: int
    from_id: str
    relation: str
    to_id: str
    direction: Literal["forward", "reverse"]


@dataclass(frozen=True)
class GraphRetrievalHit:
    chunk_id: str
    score: float
    evidence_path: tuple[GraphEvidenceStep, ...]


@runtime_checkable
class GraphRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int,
        hop_budget: int = GRAPH_HOP_BUDGET,
    ) -> list[GraphRetrievalHit]: ...


class GraphRetrievalError(RuntimeError):
    """Raised when graph retrieval cannot satisfy its deterministic contract."""




class GraphFreshnessError(GraphRetrievalError):
    """Raised when the Neo4j projection no longer matches canonical inputs."""


GraphContentSha256Reader = Callable[[object, str], str]
CitationSourceSha256Reader = Callable[[Path, Path, Path], str]


def _canonical_file_sha256(path: Path) -> str:
    """Reuse #25's raw-byte hashing contract without importing Neo4j."""
    from src.build_graph.preflight import _file_sha256

    return _file_sha256(path)


def _default_graph_content_sha256_reader(driver: object, database: str) -> str:
    """Lazy #25 graph readback seam; importing this module needs no Neo4j."""
    from src.build_graph.ingestion import readback_graph_content_sha256

    return readback_graph_content_sha256(driver, database=database)


def _default_citation_source_sha256_reader(
    manifest_path: Path,
    aliases_path: Path,
    openalex_dir: Path,
) -> str:
    """Recompute #25's pre-alias canonical citation source digest."""
    from src.build_graph.preflight import (
        _build_citation_source,
        _load_aliases,
        _validate_documents,
    )

    documents = _validate_documents(manifest_path, openalex_dir)
    aliases = _load_aliases(aliases_path)
    _, citation_source_bytes = _build_citation_source(documents, openalex_dir, aliases)
    return _file_sha256_bytes(citation_source_bytes)


def _file_sha256_bytes(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def _require_sha256(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise GraphFreshnessError(f"{where} must be a lowercase SHA-256 hex string")
    return value


def _load_ingestion_manifest(manifest_path: Path) -> dict:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GraphFreshnessError(f"ingestion manifest missing: {manifest_path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphFreshnessError(f"ingestion manifest is unreadable: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise GraphFreshnessError("ingestion manifest must be a JSON object")
    return payload


def validate_graph_freshness(
    driver: object,
    *,
    manifest_path: Path = DEFAULT_GRAPH_INGESTION_MANIFEST_PATH,
    repository_root: Path = _REPO_ROOT,
    openalex_dir: Path = DEFAULT_OPENALEX_DIR,
    database: str = DEFAULT_GRAPH_DATABASE,
    graph_content_sha256_reader: GraphContentSha256Reader | None = None,
    citation_source_sha256_reader: CitationSourceSha256Reader | None = None,
) -> None:
    """Fail closed unless #25's inputs and current graph match its manifest."""
    manifest = _load_ingestion_manifest(manifest_path)
    expected_graph_sha = _require_sha256(
        manifest.get("graph_content_sha256"), where="graph_content_sha256"
    )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != _CANONICAL_INPUT_KEYS:
        raise GraphFreshnessError("ingestion manifest inputs are malformed")

    resolved_root = repository_root.resolve()
    for hash_key in _CANONICAL_INPUT_HASH_NAMES:
        expected_hash = _require_sha256(inputs.get(hash_key), where=f"inputs.{hash_key}")
        path_key = hash_key.removesuffix("_sha256") + "_path"
        recorded_path = inputs.get(path_key)
        if (
            not isinstance(recorded_path, str)
            or not recorded_path
            or Path(recorded_path).is_absolute()
            or any(part in {".", ".."} for part in Path(recorded_path).parts)
        ):
            raise GraphFreshnessError(f"inputs.{path_key} must be a canonical relative path")
        path = repository_root / recorded_path
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError as exc:
            raise GraphFreshnessError(f"canonical input escapes repository root: {recorded_path}") from exc
        if not path.is_file():
            raise GraphFreshnessError(f"canonical input missing: {recorded_path}")
        try:
            actual_hash = _canonical_file_sha256(path)
        except OSError as exc:
            raise GraphFreshnessError(f"canonical input unreadable: {recorded_path}") from exc
        if actual_hash != expected_hash:
            raise GraphFreshnessError(f"canonical input hash mismatch: {recorded_path}")

    ontology = manifest.get("ontology")
    if not isinstance(ontology, dict) or not isinstance(inputs["ontology_version"], int):
        raise GraphFreshnessError("ingestion manifest ontology record is malformed")
    if (
        ontology.get("ontology_path") != inputs["ontology_path"]
        or ontology.get("ontology_sha256") != inputs["ontology_sha256"]
        or not isinstance(ontology.get("ontology_version"), int)
        or ontology["ontology_version"] != inputs["ontology_version"]
    ):
        raise GraphFreshnessError("ingestion manifest ontology metadata disagrees with inputs")

    expected_citation_sha = _require_sha256(
        manifest.get("citation_source_sha256"), where="citation_source_sha256"
    )
    citation_reader = citation_source_sha256_reader or _default_citation_source_sha256_reader
    try:
        actual_citation_sha = citation_reader(
            repository_root / inputs["manifest_path"],
            repository_root / inputs["aliases_path"],
            openalex_dir,
        )
    except Exception as exc:
        raise GraphFreshnessError("cannot recompute citation source hash") from exc
    if _require_sha256(actual_citation_sha, where="current citation source hash") != expected_citation_sha:
        raise GraphFreshnessError("citation source hash mismatch")

    reader = graph_content_sha256_reader or _default_graph_content_sha256_reader
    try:
        actual_graph_sha = reader(driver, database)
    except Exception as exc:
        raise GraphFreshnessError("cannot read current graph content hash") from exc
    if _require_sha256(actual_graph_sha, where="current graph content hash") != expected_graph_sha:
        raise GraphFreshnessError("graph content hash mismatch")

class GraphDriver(Protocol):
    """Traversal port for deterministic semantic graph traversal.

    ``semantic_adjacency`` returns the ontology-declared semantic edges
    incident to the node identified by ``node_id``. Plumbing edges
    (``HAS_CHUNK`` and ``MENTIONS``) must not appear in the returned
    sequence per ``graph-retrieval.md`` §3. Production drivers may be
    thin Neo4j projections; unit tests inject in-memory mappings.
    """

    def semantic_adjacency(self, node_id: str) -> object: ...


@dataclass(frozen=True)
class GraphSeed:
    """A canonical graph node selected by deterministic query seed lifting."""

    node_kind: Literal["Document", "Entity"]
    canonical_id: str


@dataclass(frozen=True)
class _Candidate:
    seed: GraphSeed
    normalized_text: str


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    seed: GraphSeed


def canonical_evidence_path_json(path: tuple[GraphEvidenceStep, ...]) -> str:
    """Return the canonical JSON identity of an ordered evidence path."""
    return json.dumps(
        [
            {
                "step": step.step,
                "from": step.from_id,
                "relation": step.relation,
                "to": step.to_id,
                "direction": step.direction,
            }
            for step in path
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def normalize_query(query: str) -> str:
    """Normalize a whole query without changing its punctuation."""
    if not isinstance(query, str):
        raise TypeError("query must be a str")
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an int")
    if k <= 0:
        raise ValueError("k must be greater than zero")


def _validate_hop_budget(hop_budget: int) -> None:
    if isinstance(hop_budget, bool) or not isinstance(hop_budget, int):
        raise TypeError("hop_budget must be an int")
    if hop_budget < 1:
        raise ValueError("hop_budget must be at least one")


def _candidate(seed: GraphSeed, text: object) -> _Candidate | None:
    if not isinstance(text, str):
        return None
    normalized = normalized_resolution_name(text)
    return _Candidate(seed, normalized) if normalized else None


def entity_candidates(records: list[dict]) -> list[_Candidate]:
    """Build Entity candidates from entity IDs, names, and aliases."""
    candidates: list[_Candidate] = []
    for record in records:
        entity_id = record.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise GraphSeedCatalogueError("entity record requires a non-empty entity_id")
        seed = GraphSeed("Entity", entity_id)
        for text in (entity_id, record.get("canonical_name"), *record.get("aliases", [])):
            candidate = _candidate(seed, text)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def document_candidates(records: list[dict]) -> list[_Candidate]:
    """Build Document candidates from document IDs and titles.

    Existing canonical manifests use ``openalex_id`` as the document ID;
    ``doc_id`` is accepted for isolated catalogue fixtures and future inputs.
    """
    candidates: list[_Candidate] = []
    for record in records:
        doc_id = record.get("doc_id", record.get("openalex_id"))
        if not isinstance(doc_id, str) or not doc_id:
            raise GraphSeedCatalogueError("document record requires a non-empty doc_id")
        seed = GraphSeed("Document", doc_id)
        for text in (doc_id, record.get("title")):
            candidate = _candidate(seed, text)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def matched_seeds(query: str, candidates: list[_Candidate]) -> list[GraphSeed]:
    """Lift all maximal, boundary-valid candidate spans from ``query``."""
    normalized_query = normalize_query(query)
    matches: list[_Match] = []
    for candidate in candidates:
        start = normalized_query.find(candidate.normalized_text)
        while start != -1:
            end = start + len(candidate.normalized_text)
            before_ok = start == 0 or not normalized_query[start - 1].isalnum()
            after_ok = end == len(normalized_query) or not normalized_query[end].isalnum()
            if before_ok and after_ok:
                matches.append(_Match(start, end, candidate.seed))
            start = normalized_query.find(candidate.normalized_text, start + 1)

    maximal = [
        match
        for match in matches
        if not any(
            other.start <= match.start
            and other.end >= match.end
            and (other.start < match.start or other.end > match.end)
            for other in matches
        )
    ]
    return sorted(
        {match.seed for match in maximal},
        key=lambda seed: (0 if seed.node_kind == "Document" else 1, seed.canonical_id),
    )


def _ordered_unique_seeds(*catalogue_seeds: list[GraphSeed]) -> list[GraphSeed]:
    """Union independently retained catalogue seeds in canonical order."""
    return sorted(
        {seed for seeds in catalogue_seeds for seed in seeds},
        key=lambda seed: (0 if seed.node_kind == "Document" else 1, seed.canonical_id),
    )


class Neo4jGraphRetriever(GraphRetriever):
    """Graph-only retriever with injected freshness and traversal seams."""

    def __init__(
        self,
        driver: GraphDriver,
        *,
        entities: list[dict] | None = None,
        documents: list[dict] | None = None,
        relation_set: frozenset[str] | None = None,
        manifest_path: Path = DEFAULT_GRAPH_INGESTION_MANIFEST_PATH,
        repository_root: Path = _REPO_ROOT,
        openalex_dir: Path = DEFAULT_OPENALEX_DIR,
        database: str = DEFAULT_GRAPH_DATABASE,
        graph_content_sha256_reader: GraphContentSha256Reader | None = None,
        citation_source_sha256_reader: CitationSourceSha256Reader | None = None,
        assertion_resolver: ExtractedAssertionResolver | None = None,
        chunk_children: ChunkChildrenProvider | None = None,
    ) -> None:
        self._driver = driver
        self._entity_candidates = entity_candidates(entities or [])
        self._document_candidates = document_candidates(documents or [])
        self._relation_set = relation_set
        self._manifest_path = manifest_path
        self._repository_root = repository_root
        self._openalex_dir = openalex_dir
        self._database = database
        self._graph_content_sha256_reader = graph_content_sha256_reader
        self._citation_source_sha256_reader = citation_source_sha256_reader
        self._assertion_resolver = assertion_resolver
        self._chunk_children = chunk_children

    def _effective_relation_set(self) -> frozenset[str]:
        if self._relation_set is not None:
            return self._relation_set
        try:
            import yaml

            ontology = yaml.safe_load((self._repository_root / "ontology.yaml").read_text())
        except Exception as exc:
            raise GraphTraversalError("cannot load canonical ontology") from exc
        relations = semantic_relation_types(ontology)
        if not relations:
            raise GraphTraversalError("canonical ontology has no semantic relation types")
        return relations

    def retrieve(
        self, query: str, k: int, hop_budget: int = GRAPH_HOP_BUDGET
    ) -> list[GraphRetrievalHit]:
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        _validate_k(k)
        _validate_hop_budget(hop_budget)
        seeds = self.lift_seeds(query)
        if not seeds:
            return []
        validate_graph_freshness(
            self._driver, manifest_path=self._manifest_path,
            repository_root=self._repository_root, openalex_dir=self._openalex_dir,
            database=self._database,
            graph_content_sha256_reader=self._graph_content_sha256_reader,
            citation_source_sha256_reader=self._citation_source_sha256_reader,
        )
        relation_set = self._effective_relation_set()
        graph_driver = _as_graph_driver(self._driver, self._database, relation_set)
        adjacency = _collect_driver_adjacency(graph_driver, seeds, hop_budget)
        paths = traverse_semantic_paths(
            seeds, adjacency=adjacency, relation_set=relation_set, hop_budget=hop_budget
        )
        evidence = collect_path_evidence(
            paths, adjacency=adjacency,
            assertion_resolver=self._assertion_resolver or _as_assertion_resolver(self._driver, self._database),
            chunk_children=self._chunk_children or _as_chunk_children_provider(self._driver, self._database),
        )
        return _reduce_evidence_candidates(evidence)[:k]

    def lift_seeds(self, query: str) -> list[GraphSeed]:
        """Expose Phase-1 seed lifting for later traversal and unit tests."""
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        if not normalize_query(query):
            return []
        return _ordered_unique_seeds(
            matched_seeds(query, self._document_candidates),
            matched_seeds(query, self._entity_candidates),
        )

    def traverse_semantic_paths(
        self,
        query: str,
        *,
        hop_budget: int = GRAPH_HOP_BUDGET,
        relation_set: frozenset[str] | None = None,
        adjacency: dict[str, list[_SemanticEdge]] | None = None,
    ) -> list[_Path]:
        """Return valid semantic paths after the required freshness check."""
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        _validate_hop_budget(hop_budget)
        seeds = self.lift_seeds(query)
        if not seeds:
            return []
        validate_graph_freshness(
            self._driver,
            manifest_path=self._manifest_path,
            repository_root=self._repository_root,
            openalex_dir=self._openalex_dir,
            database=self._database,
            graph_content_sha256_reader=self._graph_content_sha256_reader,
            citation_source_sha256_reader=self._citation_source_sha256_reader,
        )
        effective_relation_set = relation_set if relation_set is not None else self._effective_relation_set()
        if adjacency is None:
            adjacency = _collect_driver_adjacency(self._driver, seeds, hop_budget)
        return traverse_semantic_paths(
            seeds,
            adjacency=adjacency,
            relation_set=effective_relation_set,
            hop_budget=hop_budget,
        )
# ---------------------------------------------------------------------------
# Phase 2 — semantic traversal.
#
# The retriever walks the ontology-declared semantic relationship surface only
# (`graph-retrieval.md` §3). ``HAS_CHUNK`` and ``MENTIONS`` are #9
# infrastructure edges and are not part of the semantic surface.
# ---------------------------------------------------------------------------

# Ontology relationship population modes whose edges are part of the semantic
# traversal surface. ``HAS_CHUNK`` / ``MENTIONS`` are intentionally absent
# from the union because they are plumbing.
_SEMANTIC_POPULATION_MODES = frozenset({"extracted", "structural", "derived"})
# Infrastructure relationship types excluded from semantic traversal.
_PLUMBING_RELATION_TYPES = frozenset({"HAS_CHUNK", "MENTIONS"})


def semantic_relation_types(ontology: dict | None) -> frozenset[str]:
    """Return the ontology-declared semantic relationship types.

    The set is derived from the loaded ontology's ``relation_types``
    entries whose ``population_mode`` is ``extracted``, ``structural`` or
    ``derived`` (per ``graph-retrieval.md`` §3 and ``neo4j-ingestion.md``
    §4). The two plumbing types ``HAS_CHUNK`` and ``MENTIONS`` are always
    excluded. A ``None`` ontology yields the empty set so callers cannot
    silently traverse plumbing relationships.
    """
    if not isinstance(ontology, dict):
        return frozenset()
    out: set[str] = set()
    for entry in ontology.get("relation_types", []) or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        mode = entry.get("population_mode")
        if not isinstance(name, str) or not name:
            continue
        if mode in _SEMANTIC_POPULATION_MODES and name not in _PLUMBING_RELATION_TYPES:
            out.add(name)
    return frozenset(out)


@dataclass(frozen=True)
class _SemanticEdge:
    """Immutable canonical semantic-relationship representation.

    ``source_id`` and ``target_id`` are the canonical endpoint IDs in the
    ontology's stored direction and never flip. Traversal direction is
    tracked on :class:`_SemanticStep` only; canonical logical
    relationship identity is always computed from this stored
    orientation so reverse traversal does not mirror the edge.

    ``assertion_id`` and ``chunk_id`` are the canonical properties carried
    by extracted ontology assertions (see ``neo4j-ingestion.md`` §4.3).
    ``supporting_assertion_ids`` carries the derived ``EVALUATED_ON``
    provenance per ``neo4j-ingestion.md`` §4.5. Structural ``CITES``
    relationships leave all three provenance fields empty.
    """

    relation: str
    source_id: str
    target_id: str
    assertion_id: str = ""
    chunk_id: str = ""
    supporting_assertion_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SemanticStep:
    """One semantic hop within a traversal path.

    ``traversal_direction`` records whether this hop followed the stored
    ontology edge (``"forward"``) or the opposite direction
    (``"reverse"``). The connected-path invariant
    ``previous.to_id == current.from_id`` is preserved by construction.
    """

    from_id: str
    relation: str
    to_id: str
    traversal_direction: Literal["forward", "reverse"]


@dataclass(frozen=True)
class _Path:
    """Immutable internal traversal path.

    ``directed_identities`` is the ordered tuple of directed logical
    relationship identities traversed along this path. It is the source
    of truth for the repeated-relationship rejection rule
    (``graph-retrieval.md`` §3.2).
    """

    start_id: str
    steps: tuple[_SemanticStep, ...]
    directed_identities: tuple[tuple, ...]

    def to_evidence_path(self) -> tuple[GraphEvidenceStep, ...]:
        """Convert this path into the canonical wire ``GraphEvidenceStep`` tuple."""
        return tuple(
            GraphEvidenceStep(
                step=index + 1,
                from_id=step.from_id,
                relation=step.relation,
                to_id=step.to_id,
                direction=step.traversal_direction,
            )
            for index, step in enumerate(self.steps)
        )




_LOGICAL_ID_CASE = (
    "CASE WHEN 'Chunk' IN labels({node}) THEN {node}.chunk_id "
    "WHEN 'Entity' IN labels({node}) THEN {node}.entity_id "
    "ELSE {node}.doc_id END"
)


class _OfficialGraphReadDriver:
    """Read-only projection over the official Neo4j driver."""

    def __init__(self, driver: object, database: str, relations: frozenset[str]) -> None:
        self._driver, self._database, self._relations = driver, database, relations

    def semantic_adjacency(self, node_id: str) -> Iterable[_SemanticEdge]:
        query = (
            "MATCH (a)-[r]->(b) WHERE type(r) IN $relations AND ("
            + _LOGICAL_ID_CASE.format(node="a") + " = $node_id OR "
            + _LOGICAL_ID_CASE.format(node="b") + " = $node_id) "
            "RETURN type(r) AS relation, " + _LOGICAL_ID_CASE.format(node="a")
            + " AS source_id, " + _LOGICAL_ID_CASE.format(node="b")
            + " AS target_id, r.assertion_id AS assertion_id, r.chunk_id AS chunk_id, "
            "r.supporting_assertion_ids AS supporting_assertion_ids"
        )
        return _read_edges(self._driver, self._database, query, {"relations": sorted(self._relations), "node_id": node_id})


def _as_graph_driver(driver: object, database: str, relations: frozenset[str]) -> GraphDriver:
    return driver if hasattr(driver, "semantic_adjacency") else _OfficialGraphReadDriver(driver, database, relations)  # type: ignore[return-value]


def _read_edges(driver: object, database: str, query: str, parameters: dict) -> list[_SemanticEdge]:
    try:
        with driver.session(  # type: ignore[attr-defined]
            database=database, default_access_mode="READ"
        ) as session:
            rows = session.run(query, **parameters)
            edges = [_SemanticEdge(
                relation=row["relation"], source_id=row["source_id"], target_id=row["target_id"],
                assertion_id=row["assertion_id"] or "", chunk_id=row["chunk_id"] or "",
                supporting_assertion_ids=tuple(sorted(row["supporting_assertion_ids"] or ())),
            ) for row in rows]
    except Exception as exc:
        raise GraphTraversalError("cannot read semantic graph state") from exc
    return _ordered_adjacency(edges)


def _as_assertion_resolver(driver: object, database: str) -> ExtractedAssertionResolver:
    query = (
        "MATCH (a)-[r]->(b) WHERE r.assertion_id = $assertion_id "
        "RETURN type(r) AS relation, " + _LOGICAL_ID_CASE.format(node="a") + " AS source_id, "
        + _LOGICAL_ID_CASE.format(node="b") + " AS target_id, r.assertion_id AS assertion_id, "
        "r.chunk_id AS chunk_id, r.supporting_assertion_ids AS supporting_assertion_ids"
    )
    def resolve(assertion_id: str) -> _SemanticEdge | None:
        edges = _read_edges(driver, database, query, {"assertion_id": assertion_id})
        if len(edges) > 1:
            raise StaleDerivedProvenanceError(f"assertion_id {assertion_id!r} is not unique")
        return edges[0] if edges else None
    return resolve


def _as_chunk_children_provider(driver: object, database: str) -> ChunkChildrenProvider:
    query = "MATCH (:Document {doc_id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) RETURN c.chunk_id AS chunk_id"
    def children(doc_id: str) -> Iterable[str]:
        try:
            with driver.session(  # type: ignore[attr-defined]
                database=database, default_access_mode="READ"
            ) as session:
                return tuple(sorted({row["chunk_id"] for row in session.run(query, doc_id=doc_id)}))
        except Exception as exc:
            raise GraphEvidenceError("cannot read terminal document chunks") from exc
    return children
def _canonical_relationship_identity(edge: _SemanticEdge) -> tuple:
    """Return the canonical base identity for a semantic edge.

    Per ``graph-retrieval.md`` §3.2 / Phase-2 §6:

    - extracted ontology assertions (``PROPOSES``, ``INTRODUCES``,
      ``EVALUATES_ON``, ``USES_METHOD``, ``REPORTS_METRIC``, ``EXTENDS``,
      ``USED_FOR_TASK``, ``MEASURES``) are identified by
      ``assertion_id`` alone — ``relation`` is not part of the
      identity so two extracted edges with the same ``assertion_id``
      collapse to the same logical relationship regardless of how the
      driver labels them;
    - structural ``CITES`` is identified by the canonical
      ``(citing_doc_id, cited_doc_id)`` endpoint pair — the same pair is
      used regardless of traversal direction;
    - derived ``EVALUATED_ON`` is identified by the canonical
      ``(method_entity_id, dataset_entity_id)`` endpoint pair.

    Extracted relationships without a non-empty ``assertion_id`` cannot
    satisfy the canonical identity contract and fail closed: the
    traversal layer raises :class:`GraphTraversalError` rather than
    inventing an identity from ``source_id``/``target_id``.
    """
    if edge.relation == "CITES":
        return ("CITES", edge.source_id, edge.target_id)
    if edge.relation == "EVALUATED_ON":
        return ("EVALUATED_ON", edge.source_id, edge.target_id)
    if edge.assertion_id:
        return ("ASSERTION", edge.assertion_id)
    raise GraphTraversalError(
        f"extracted semantic relation {edge.relation!r} is missing "
        "assertion_id; canonical identity cannot be derived"
    )


def _directed_identity(
    edge: _SemanticEdge, traversal_direction: Literal["forward", "reverse"]
) -> tuple:
    """Return the directed identity used for repeated-relationship rejection.

    The directed identity always appends the traversal direction so that
    traversing the same canonical relationship forward and reverse counts
    as two distinct directed identities per Phase-2 §7. Reverse
    traversal does not flip the canonical base endpoints: the directed
    identity for a reverse step always uses the stored
    ``source_id``/``target_id`` pair.
    """
    return (*_canonical_relationship_identity(edge), traversal_direction)


class GraphTraversalError(GraphRetrievalError):
    """Raised when semantic traversal cannot proceed (e.g. plumbing edge in input)."""


def _ordered_adjacency(edges: list[_SemanticEdge]) -> list[_SemanticEdge]:
    """Sort adjacency canonically so the traversal is driver-order independent.
    Canonical sort key:
    ``(relation, source_id, target_id, assertion_id, chunk_id,
    supporting_assertion_ids)``. The provenance fields are tuples of
    strings and sort lexicographically as required.
    """
    return sorted(
        edges,
        key=lambda edge: (
            edge.relation,
            edge.source_id,
            edge.target_id,
            edge.assertion_id,
            edge.chunk_id,
            edge.supporting_assertion_ids,
        ),
    )


def _is_semantic_relation_type(relation: str, relation_set: frozenset[str]) -> bool:
    return relation in relation_set


def _expand_path(
    path: _Path,
    *,
    relation_set: frozenset[str],
    adjacency: dict[str, list[_SemanticEdge]],
) -> list[_Path]:
    """Return every valid one-hop extension of ``path``.

    Extensions whose directed logical relationship identity is already
    present on ``path`` are rejected per Phase-2 §7. Plumbing edges
    (``HAS_CHUNK`` / ``MENTIONS``) cannot enter semantic traversal and
    therefore cannot be expanded into, even if a stray driver returned
    them; the relation-set membership check rejects them before identity
    comparison.

    Forward traversal is valid when the terminal sits at the stored
    source endpoint; reverse traversal is valid when the terminal sits
    at the stored target endpoint. A canonical self-loop
    (``source_id == target_id``) is traversable in both directions
    because each step uses a distinct directed identity; the repeated
    identity check still prevents either direction from being used
    twice in the same path.
    """
    terminal = path.steps[-1].to_id if path.steps else path.start_id
    seen = path.directed_identities
    out: list[_Path] = []
    for edge in adjacency.get(terminal, ()):  # type: ignore[arg-type]
        if not _is_semantic_relation_type(edge.relation, relation_set):
            continue
        orientations: list[tuple[Literal["forward", "reverse"], str, str]] = []
        if terminal == edge.source_id:
            orientations.append(("forward", terminal, edge.target_id))
        if terminal == edge.target_id:
            orientations.append(("reverse", terminal, edge.source_id))
        for direction, from_id, to_id in orientations:
            directed_id = _directed_identity(edge, direction)
            if directed_id in seen:
                continue
            out.append(
                _Path(
                    start_id=path.start_id,
                    steps=path.steps + (
                        _SemanticStep(
                            from_id=from_id,
                            relation=edge.relation,
                            to_id=to_id,
                            traversal_direction=direction,
                        ),
                    ),
                    directed_identities=seen + (directed_id,),
                )
            )
    return out

def _collect_driver_adjacency(
    driver: GraphDriver, seeds: list[GraphSeed], hop_budget: int
) -> dict[str, list[_SemanticEdge]]:
    """Materialise every adjacency frontier required by ``hop_budget``."""
    seen: set[str] = {seed.canonical_id for seed in seeds}
    adjacency: dict[str, list[_SemanticEdge]] = {node: [] for node in seen}
    frontier: set[str] = set(seen)
    for _ in range(hop_budget):
        next_frontier: set[str] = set()
        for node in sorted(frontier):
            edges = list(driver.semantic_adjacency(node))
            adjacency[node] = _ordered_adjacency(edges)
            for edge in edges:
                other = edge.target_id if edge.source_id == node else edge.source_id
                if other not in seen:
                    seen.add(other)
                    next_frontier.add(other)
        frontier = next_frontier
        if not frontier:
            break
    return adjacency

def traverse_semantic_paths(
    seeds: list[GraphSeed],
    *,
    adjacency: dict[str, list[_SemanticEdge]],
    relation_set: frozenset[str],
    hop_budget: int = GRAPH_HOP_BUDGET,
) -> list[_Path]:
    """Enumerate semantic paths of lengths 1..``hop_budget`` from every seed.

    The enumeration is deterministic regardless of how ``adjacency``
    orders its edges — each ``adjacency[node]`` list is sorted
    canonically before expansion. Path-uniqueness is enforced on the
    directed logical relationship identity (Phase-2 §7) so node revisits
    are permitted as long as no directed identity repeats. The function
    returns every valid path; later phases select among them.

    ``seeds`` carries the Phase-1 ``(node_kind, canonical_id)`` graph-node
    identity; only the ``canonical_id`` is used here.
    """
    _validate_hop_budget(hop_budget)
    frontier: list[_Path] = []
    for seed in seeds:
        frontier.append(
            _Path(
                start_id=seed.canonical_id,
                steps=(),
                directed_identities=(),
            )
        )
    completed: list[_Path] = []
    for _ in range(hop_budget):
        next_frontier: list[_Path] = []
        for path in frontier:
            next_frontier.extend(
                _expand_path(
                    path,
                    relation_set=relation_set,
                    adjacency={
                        node: _ordered_adjacency(list(edges))
                        for node, edges in adjacency.items()
                    },
                )
            )
        completed.extend(next_frontier)
        if not next_frontier:
            break
        frontier = next_frontier
    # Sort completed paths deterministically: by start_id then by their
    # canonical step sequence so identical graph states yield byte-equal
    # output regardless of the adjacency ordering fed in.
    completed.sort(
        key=lambda path: (
            path.start_id,
            tuple(
                (
                    step.from_id,
                    step.relation,
                    step.to_id,
                    step.traversal_direction,
                )
                for step in path.steps
            ),
        )
    )
    return completed
# ---------------------------------------------------------------------------
# Phase 3 — path-specific evidence collection.
#
# Evidence is collected per traversed path. Only the rules below contribute
# ``chunk_id`` values:
#
# * every extracted ontology relationship on the path contributes the
#   canonical ``chunk_id`` stored on the EXACT traversed assertion
#   relationship;
# * every derived ``EVALUATED_ON`` on the path contributes the canonical
#   ``chunk_id`` of every participating extracted assertion resolved from
#   its ``supporting_assertion_ids``;
# * a pure-CITES path (every semantic step is ``CITES``) is eligible for the
#   single structural fallback: the canonical ``HAS_CHUNK`` children of the
#   terminal ``Document`` in traversal order.
#
# Mixed paths (containing any extracted or derived step) collect exactly the
# sorted unique set of their assertion-evidence chunks. ``HAS_CHUNK`` and
# ``MENTIONS`` plumbing edges never appear in ``evidence_path`` and never
# contribute evidence except for the pure-CITES terminal fallback above.
#
# The assertion-resolution and chunk-children lookups are injectable seams so
# synthetic unit tests can exercise this phase without a live Neo4j driver.
# ---------------------------------------------------------------------------


class StaleDerivedProvenanceError(GraphRetrievalError):
    """Raised when a derived ``EVALUATED_ON`` references participating
    assertions that do not exist, are not extracted ontology assertions, or
    are missing the canonical ``chunk_id``. Such a path is stale graph state
    and retrieval fails closed rather than silently skipping the bad
    support.
    """


class GraphEvidenceError(GraphRetrievalError):
    """Raised when path-specific evidence cannot be satisfied (e.g. an
    extracted assertion edge without its required canonical ``chunk_id``).
    """


# Extracted ontology relationship names whose canonical identity is the
# ``assertion_id`` carried on the relationship (see ``neo4j-ingestion.md``
# §4.3 and ``graph-retrieval.md`` §3.2). ``CITES`` is structural; the rest
# in this set participate in derived ``EVALUATED_ON``.
_EXTRACTED_RELATION_TYPES = frozenset(
    {
        "PROPOSES",
        "INTRODUCES",
        "EVALUATES_ON",
        "USES_METHOD",
        "REPORTS_METRIC",
        "EXTENDS",
        "USED_FOR_TASK",
        "MEASURES",
    }
)
# Derived relation whose evidence is resolved through supporting assertions.
_DERIVED_RELATION_TYPES = frozenset({"EVALUATED_ON"})

_DERIVED_SUPPORT_RELATIONS = frozenset({"PROPOSES", "EVALUATES_ON"})


@dataclass(frozen=True)
class _PathEvidence:
    """Immutable internal record of path-specific evidence.

    The Phase-3 representation preserves the exact semantic ``_Path``
    together with sorted unique canonical ``chunk_id`` values and a small
    ``origin`` tag sufficient to distinguish the three permitted sources
    when later phases score and reduce candidates.
    """

    path: _Path
    chunk_ids: tuple[str, ...]
    origin: Literal["assertion", "structural_cites", "empty"]


# Type aliases for the injectable read seams. Both are pure-Python callables
# so unit tests can construct synthetic graphs without a Neo4j driver.
ExtractedAssertionResolver = Callable[[str], _SemanticEdge | None]
ChunkChildrenProvider = Callable[[str], Iterable[str]]


def _default_assertion_resolver(
    edges_by_assertion_id: dict[str, _SemanticEdge],
) -> ExtractedAssertionResolver:
    """Return an ``assertion_id -> _SemanticEdge`` resolver closure seeded
    from a static mapping. Used by the synthetic-graph unit-test seam and
    by callers that already hold the adjacency in-process.
    """

    def _resolve(assertion_id: str) -> _SemanticEdge | None:
        return edges_by_assertion_id.get(assertion_id)

    return _resolve


def _default_chunk_children_provider(
    children_by_doc_id: dict[str, Iterable[str]],
) -> ChunkChildrenProvider:
    """Return a ``doc_id -> iterable[chunk_id]`` closure seeded from a
    static mapping. The closure materialises each iterable into a sorted
    tuple of unique strings so the fallback is deterministic.
    """

    prepared: dict[str, tuple[str, ...]] = {
        doc_id: tuple(sorted({chunk_id for chunk_id in ids}))
        for doc_id, ids in children_by_doc_id.items()
    }

    def _children(doc_id: str) -> Iterable[str]:
        return prepared.get(doc_id, ())

    return _children


def _relation_class(relation: str) -> Literal["extracted", "derived", "cites"]:
    """Return the Phase-3 evidence class for ``relation``."""
    if relation == "CITES":
        return "cites"
    if relation in _DERIVED_RELATION_TYPES:
        return "derived"
    if relation in _EXTRACTED_RELATION_TYPES:
        return "extracted"
    # Defensive: traversal already rejects plumbing edges, so anything
    # reaching this branch violates the semantic surface. Surface a
    # fail-closed error rather than silently misclassifying.
    raise GraphEvidenceError(
        f"relation {relation!r} is not part of the ontology semantic surface"
    )


def _path_step_edge(
    path: _Path, adjacency: dict[str, list[_SemanticEdge]]
) -> dict[tuple[str, ...], _SemanticEdge]:
    """Return a mapping from each path step's directed identity to the
    traversed ``_SemanticEdge`` object. The traversal layer is the sole
    source of truth for which edge produced which step; this lookup keeps
    Phase-3 evidence reads confined to the exact traversed object.
    """
    terminal = path.start_id
    out: dict[tuple[str, ...], _SemanticEdge] = {}
    for index, step in enumerate(path.steps):
        directed_id = path.directed_identities[index]
        for edge in adjacency.get(terminal, ()):
            if step.relation != edge.relation:
                continue
            matches = False
            if step.traversal_direction == "forward":
                matches = (
                    terminal == edge.source_id
                    and step.to_id == edge.target_id
                )
            elif step.traversal_direction == "reverse":
                matches = (
                    terminal == edge.target_id
                    and step.to_id == edge.source_id
                )
            if not matches:
                continue
            # Extracted-relation identity is keyed on ``assertion_id``;
            # without this guard the function would match any
            # parallel edge sharing endpoints and read the wrong
            # canonical ``chunk_id`` from a sibling assertion.
            if (
                edge.relation in _EXTRACTED_RELATION_TYPES
                and (
                    "ASSERTION",
                    edge.assertion_id,
                    step.traversal_direction,
                ) != directed_id
            ):
                continue
            out[directed_id] = edge
            break
        terminal = step.to_id
    return out


def _step_directed_identity(
    path: _Path, step_index: int
) -> tuple[str, ...]:
    """Return the directed identity of the step at ``step_index``."""
    # ``_Path`` stores its directed identities in traversal order so the
    # index is 0-based against ``path.steps`` and ``path.directed_identities``.
    return path.directed_identities[step_index]


def _collect_extracted_chunk_ids(
    path: _Path,
    step_edges: dict[tuple[str, ...], _SemanticEdge],
) -> list[str]:
    """Collect the canonical ``chunk_id`` of every extracted ontology step
    on ``path``. Reads the value off the exact traversed ``_SemanticEdge``
    object — never through ``MENTIONS`` or arbitrary document context.
    Fails closed if an extracted step lacks its canonical ``chunk_id``.
    """
    out: list[str] = []
    for index, step in enumerate(path.steps):
        if _relation_class(step.relation) != "extracted":
            continue
        directed_id = _step_directed_identity(path, index)
        edge = step_edges.get(directed_id)
        if edge is None:
            # Traversal invariant guarantees the step's edge is in the
            # adjacency; defensive error rather than silent skip.
            raise GraphEvidenceError(
                f"extracted step {index + 1} ({step.relation}) has no "
                "matching traversed edge in adjacency"
            )
        if not edge.chunk_id:
            raise GraphEvidenceError(
                f"extracted assertion {edge.relation!r} "
                f"assertion_id={edge.assertion_id!r} is missing its canonical "
                "chunk_id; fail-closed evidence rule"
            )
        out.append(edge.chunk_id)
    return out


def _resolve_supporting_assertion(
    support_id: str,
    *,
    derived_edge: _SemanticEdge,
    assertion_resolver: ExtractedAssertionResolver,
) -> _SemanticEdge:
    """Resolve and validate one derived ``EVALUATED_ON`` support.

    A support is valid only when it is the exact referenced ``PROPOSES`` or
    ``EVALUATES_ON`` assertion for the derived method/dataset endpoints.
    """
    resolved = assertion_resolver(support_id)
    if resolved is None:
        raise StaleDerivedProvenanceError(
            f"supporting_assertion_id {support_id!r} on derived "
            f"EVALUATED_ON ({derived_edge.source_id!r}->{derived_edge.target_id!r}) "
            "does not resolve to a participating assertion"
        )
    if resolved.assertion_id != support_id:
        raise StaleDerivedProvenanceError(
            f"supporting_assertion_id {support_id!r} resolved to assertion_id "
            f"{resolved.assertion_id!r}"
        )
    if resolved.relation not in _DERIVED_SUPPORT_RELATIONS:
        raise StaleDerivedProvenanceError(
            f"supporting_assertion_id {support_id!r} resolved to invalid "
            f"derived provenance relation {resolved.relation!r}"
        )
    if not resolved.chunk_id:
        raise StaleDerivedProvenanceError(
            f"participating assertion {support_id!r} has no canonical chunk_id"
        )
    if not resolved.source_id.startswith("W"):
        raise StaleDerivedProvenanceError(
            f"participating assertion {support_id!r} source {resolved.source_id!r} "
            "is not a canonical Document"
        )
    expected_target = (
        derived_edge.source_id
        if resolved.relation == "PROPOSES"
        else derived_edge.target_id
    )
    if resolved.target_id != expected_target:
        raise StaleDerivedProvenanceError(
            f"participating {resolved.relation} assertion {support_id!r} targets "
            f"{resolved.target_id!r}, not {expected_target!r}"
        )
    return resolved


def _collect_derived_chunk_ids(
    path: _Path,
    step_edges: dict[tuple[str, ...], _SemanticEdge],
    *,
    assertion_resolver: ExtractedAssertionResolver,
) -> list[str]:
    """Collect evidence from validated same-Document derived witnesses."""
    out: list[str] = []
    for index, step in enumerate(path.steps):
        if _relation_class(step.relation) != "derived":
            continue
        directed_id = _step_directed_identity(path, index)
        edge = step_edges.get(directed_id)
        if edge is None:
            raise GraphEvidenceError(
                f"derived step {index + 1} (EVALUATED_ON) has no matching "
                "traversed edge in adjacency"
            )
        if not edge.supporting_assertion_ids:
            raise StaleDerivedProvenanceError(
                f"derived EVALUATED_ON ({edge.source_id!r}->{edge.target_id!r}) "
                "has no supporting_assertion_ids"
            )

        witnesses: dict[str, set[str]] = {}
        for support_id in edge.supporting_assertion_ids:
            resolved = _resolve_supporting_assertion(
                support_id,
                derived_edge=edge,
                assertion_resolver=assertion_resolver,
            )
            witnesses.setdefault(resolved.source_id, set()).add(resolved.relation)
            out.append(resolved.chunk_id)

        for document_id, relations in witnesses.items():
            if "PROPOSES" not in relations or "EVALUATES_ON" not in relations:
                raise StaleDerivedProvenanceError(
                    f"derived EVALUATED_ON ({edge.source_id!r}->{edge.target_id!r}) "
                    f"has incomplete same-Document witness {document_id!r}: "
                    f"{sorted(relations)!r}"
                )
    return out


def _is_pure_cites_path(path: _Path) -> bool:
    """Return ``True`` exactly when every semantic step on ``path`` is
    ``CITES`` and the path contains at least one step. The structural
    fallback is reserved for this single configuration.
    """
    return bool(path.steps) and all(
        step.relation == "CITES" for step in path.steps
    )


def collect_path_evidence(
    paths: list[_Path],
    *,
    adjacency: dict[str, list[_SemanticEdge]],
    assertion_resolver: ExtractedAssertionResolver,
    chunk_children: ChunkChildrenProvider,
) -> list[_PathEvidence]:
    """Phase-3 evidence collection seam.

    For every path in ``paths``:

    * if the path carries at least one extracted or derived assertion step,
      its evidence is the sorted unique union of the canonical ``chunk_id``
      values of those steps (including the supporting-assertion chunks
      resolved from any derived ``EVALUATED_ON``). No terminal
      ``HAS_CHUNK`` / ``MENTIONS`` context is added;
    * otherwise (pure-CITES path) the path uses the structural fallback:
      the sorted unique ``HAS_CHUNK`` children of the terminal ``Document``
      in traversal order.

    ``assertion_resolver`` is the unit-testable read seam that maps an
    ``assertion_id`` to its participating extracted ``_SemanticEdge``. It
    must be injectable so this layer can be exercised without a live
    Neo4j driver. ``chunk_children`` is the analogous seam for the
    terminal-document ``HAS_CHUNK`` lookup.

    The returned list is in the same order as ``paths``.
    """
    out: list[_PathEvidence] = []
    for path in paths:
        step_edges = _path_step_edge(path, adjacency)
        assertion_chunks = _collect_extracted_chunk_ids(path, step_edges)
        assertion_chunks.extend(
            _collect_derived_chunk_ids(
                path, step_edges, assertion_resolver=assertion_resolver
            )
        )
        if assertion_chunks:
            out.append(
                _PathEvidence(
                    path=path,
                    chunk_ids=tuple(sorted(set(assertion_chunks))),
                    origin="assertion",
                )
            )
            continue
        if not _is_pure_cites_path(path):
            # Non-CITES path with no extracted/derived evidence chunk is
            # forbidden by spec; the traversal layer should already have
            # rejected plumbing-only paths, but be defensive.
            raise GraphEvidenceError(
                "path contains no extracted/derived assertion step and is "
                "not a pure-CITES path; structural fallback is forbidden"
            )
        # Pure-CITES fallback: only the TERMINAL Document contributes.
        terminal_doc = path.steps[-1].to_id
        if not terminal_doc.startswith("W"):
            # The terminal must be a Document for the structural fallback
            # to apply. A pure-CITES path whose terminal is not a Document
            # cannot use the fallback; surface the rule rather than
            # fabricate evidence.
            raise GraphEvidenceError(
                "pure-CITES path terminal is not a Document; terminal "
                "HAS_CHUNK fallback requires a Document terminal"
            )
        chunks = sorted({chunk_id for chunk_id in chunk_children(terminal_doc)})
        out.append(
            _PathEvidence(
                path=path,
                chunk_ids=tuple(chunks),
                origin="structural_cites",
            )
        )
    return out


def _reduce_evidence_candidates(evidence: list[_PathEvidence]) -> list[GraphRetrievalHit]:
    """Reduce path evidence globally to one deterministic hit per chunk."""
    selected: dict[str, GraphRetrievalHit] = {}
    for item in evidence:
        path = item.path.to_evidence_path()
        score = 1.0 / len(path)
        for chunk_id in item.chunk_ids:
            candidate = GraphRetrievalHit(chunk_id, score, path)
            current = selected.get(chunk_id)
            if current is None or _candidate_key(candidate) < _candidate_key(current):
                selected[chunk_id] = candidate
    return sorted(selected.values(), key=lambda hit: (-hit.score, hit.chunk_id))


def _candidate_key(hit: GraphRetrievalHit) -> tuple[float, bytes]:
    return (-hit.score, canonical_evidence_path_json(hit.evidence_path).encode("utf-8"))
__all__ = [
    "DEFAULT_ENTITIES_PATH",
    "CitationSourceSha256Reader",
    "DEFAULT_OPENALEX_DIR",
    "DEFAULT_GRAPH_INGESTION_MANIFEST_PATH",
    "GraphFreshnessError",
    "DEFAULT_MANIFEST_PATH",
    "GRAPH_HOP_BUDGET",
    "ChunkChildrenProvider",
    "ExtractedAssertionResolver",
    "GraphDriver",
    "GraphEvidenceError",
    "GraphEvidenceStep",
    "GraphRetrievalError",
    "GraphRetrievalHit",
    "GraphRetriever",
    "GraphSeed",
    "GraphSeedCatalogueError",
    "Neo4jGraphRetriever",
    "StaleDerivedProvenanceError",
    "canonical_evidence_path_json",
    "collect_path_evidence",
    "document_candidates",
    "entity_candidates",
    "matched_seeds",
    "normalize_query",
    "semantic_relation_types",
    "traverse_semantic_paths",
    "validate_graph_freshness",
]
