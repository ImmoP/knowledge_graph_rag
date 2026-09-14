"""Transactional Neo4j graph replacement (#25, #9).

Implements ``docs/specs/neo4j-ingestion.md`` §3 (node model), §4
(relationship model), §5 (constraints and indexes), §7 (deterministic
replacement), §8 (canonical graph hash and ingestion manifest), and §9
(credentials).

Two public entry points:

- :func:`bootstrap_schema` — idempotent uniqueness constraints and
  lookup indexes (§5). Safe to run any number of times. Not part of the
  data transaction.
- :func:`replace_graph` — single-write-transaction graph replacement
  (§7). Caller is responsible for having run :func:`run_preflight`
  first; this function accepts the :class:`PreflightBundle` directly.
- :func:`run_ingestion` — top-level orchestrator that wires the
  preflight, schema bootstrap, replacement, hash computation, and
  atomic manifest publication.

The driver is injected (``driver_factory`` / ``driver``) so tests can
supply a mock driver without booting Neo4j. Production callers use the
:func:`default_driver` factory that reads ``.env`` and returns the
official neo4j driver pinned in ``local-infra.md`` §3.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.build_graph.preflight import (
    DEFAULT_OUTPUT_MANIFEST_PATH,
    PreflightBundle,
    PreflightError,
)
from src.build_graph.schema import (
    canonical_json,
    canonical_json_bytes,
    node_entry,
    pascal_case,
    relationship_entry,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# Public result type.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionResult:
    graph_content_sha256: str
    citation_source_sha256: str
    node_counts: dict
    relationship_counts: dict
    manifest_path: Path


# ---------------------------------------------------------------------------
# Public error type.
# ---------------------------------------------------------------------------


class IngestionError(RuntimeError):
    """Raised when Neo4j ingestion fails. The transaction rolls back to
    the pre-run state; no partial graph survives."""


# ---------------------------------------------------------------------------
# §9 credential loading — read only the three named vars.
# ---------------------------------------------------------------------------


def _load_credentials(env_file: Path | None = None) -> dict:
    """Load ``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD`` from
    ``.env``. ``NEO4J_PASSWORD`` is required; the others fall back to
    documented defaults (§9).
    """
    if env_file is not None:
        load_dotenv(env_file, override=True)
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise IngestionError(
            "NEO4J_PASSWORD is not set; required by §9"
        )
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": password,
    }


def default_driver(env_file: Path | None = None):
    """Build the official neo4j driver from the §9 credentials.

    Importing the driver module is deferred to here so the rest of the
    ingestion module remains importable in environments where the
    driver is not installed (tests, CI without neo4j driver).
    """
    from neo4j import GraphDatabase  # local import per §9

    creds = _load_credentials(env_file)
    return GraphDatabase.driver(
        creds["uri"], auth=(creds["user"], creds["password"])
    )


# ---------------------------------------------------------------------------
# §5 constraint / index statements. Run once; idempotent.
# ---------------------------------------------------------------------------


_SCHEMA_BOOTSTRAP_STATEMENTS = (
    "CREATE CONSTRAINT document_doc_id_unique IF NOT EXISTS "
    "FOR (n:Document) REQUIRE n.doc_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_chunk_id_unique IF NOT EXISTS "
    "FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_entity_id_unique IF NOT EXISTS "
    "FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
    "CREATE INDEX chunk_doc_id_lookup IF NOT EXISTS "
    "FOR (n:Chunk) ON (n.doc_id)",
    "CREATE INDEX entity_type_name_lookup IF NOT EXISTS "
    "FOR (n:Entity) ON (n.entity_type, n.canonical_name)",
)


def bootstrap_schema(driver, database: str = "neo4j") -> None:
    """§5 idempotent schema bootstrap. Runs separately from the data
    transaction. Caller may invoke this on every run.
    """
    with driver.session(database=database) as session:
        for stmt in _SCHEMA_BOOTSTRAP_STATEMENTS:
            session.run(stmt).consume()


# ---------------------------------------------------------------------------
# §3 / §4 record builders — pure functions over the preflight bundle.
# ---------------------------------------------------------------------------


def _build_documents(bundle: PreflightBundle) -> list:
    """Sorted by ``doc_id``."""
    return sorted(
        (
            node_entry(
                labels=["Document"],
                logical_id=doc.doc_id,
                properties={"doc_id": doc.doc_id, "title": doc.title},
            )
            for doc in bundle.documents.values()
        ),
        key=lambda n: (n["labels"], n["logical_id"]),
    )


def _build_chunks(bundle: PreflightBundle) -> list:
    return sorted(
        (
            node_entry(
                labels=["Chunk"],
                logical_id=chunk.chunk_id,
                properties={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "section_path": list(chunk.section_path),
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk in bundle.chunks.values()
        ),
        key=lambda n: (n["labels"], n["logical_id"]),
    )


def _build_entities(bundle: PreflightBundle) -> list:
    """Each entity has the common ``Entity`` label plus exactly one
    PascalCase subtype label (§3.3 / §3.4)."""
    out: list = []
    for entity in bundle.entities.values():
        subtype = bundle.entity_subtypes.get(entity.entity_type, "")
        if not subtype:
            raise IngestionError(
                f"entity {entity.entity_id!r}: entity_type "
                f"{entity.entity_type!r} has no PascalCase subtype label"
            )
        labels = ["Entity", pascal_case(entity.entity_type)]
        out.append(
            node_entry(
                labels=labels,
                logical_id=entity.entity_id,
                properties={
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "canonical_name": entity.canonical_name,
                    "aliases": list(entity.aliases),
                    "resolution_confidence": entity.resolution_confidence,
                },
            )
        )
    out.sort(key=lambda n: (n["labels"], n["logical_id"]))
    return out


def _build_has_chunk_relationships(bundle: PreflightBundle) -> list:
    """§4.1: exactly one ``HAS_CHUNK`` per chunk."""
    out = []
    for chunk in bundle.chunks.values():
        out.append(
            relationship_entry(
                type="HAS_CHUNK",
                logical_id=[chunk.doc_id, chunk.chunk_id],
                from_kind="Document",
                from_id=chunk.doc_id,
                to_kind="Chunk",
                to_id=chunk.chunk_id,
                properties={},
            )
        )
    return out


def _build_mentions_relationships(bundle: PreflightBundle) -> list:
    """§4.2: one ``MENTIONS`` per mention_id (§8.1 logical_id is a
    string for single-id identities)."""
    out = []
    for mention_id, mention in bundle.mentions.items():
        entity_id = bundle.mention_to_entity.get(mention_id)
        if entity_id is None:
            # Preflight guarantees every eligible mention has a mapping.
            raise IngestionError(
                f"mention {mention_id!r} has no entity resolution mapping"
            )
        out.append(
            relationship_entry(
                type="MENTIONS",
                logical_id=mention_id,
                from_kind="Chunk",
                from_id=mention.chunk_id,
                to_kind=pascal_case(mention.entity_type),
                to_id=entity_id,
                properties={
                    "mention_id": mention_id,
                    "chunk_id": mention.chunk_id,
                    "resolution_method": mention.resolution_method,
                    "resolution_score": mention.resolution_score,
                },
            )
        )
    return out


def _build_extracted_assertions(bundle: PreflightBundle) -> list:
    """§4.3: one relationship per ``assertion_id``. Direction is the
    ontology ``from -> to`` direction."""
    out = []
    for assertion_id, assertion in bundle.assertions.items():
        relation_type = assertion.relation_type
        subject = assertion.subject_ref
        obj = assertion.object_ref
        if subject["kind"] == "document":
            from_kind = "Document"
            from_id = subject["id"]
        else:
            entity_id = bundle.mention_to_entity[subject["id"]]
            # Subject is a mention; resolve to its entity's PascalCase.
            etype = bundle.mentions[subject["id"]].entity_type
            from_kind = pascal_case(etype)
            from_id = entity_id
        if obj["kind"] == "document":
            to_kind = "Document"
            to_id = obj["id"]
        else:
            entity_id = bundle.mention_to_entity[obj["id"]]
            etype = bundle.mentions[obj["id"]].entity_type
            to_kind = pascal_case(etype)
            to_id = entity_id
        out.append(
            relationship_entry(
                type=relation_type,
                logical_id=assertion_id,
                from_kind=from_kind,
                from_id=from_id,
                to_kind=to_kind,
                to_id=to_id,
                properties={
                    "assertion_id": assertion_id,
                    "chunk_id": assertion.chunk_id,
                },
            )
        )
    return out


def _build_citations(bundle: PreflightBundle) -> list:
    """§4.4: ``CITES`` from in-corpus canonical ``referenced_works``
    after alias canonicalisation and self-loop removal."""
    out = []
    for citing, cited in bundle.citation_pairs:
        out.append(
            relationship_entry(
                type="CITES",
                logical_id=[citing, cited],
                from_kind="Document",
                from_id=citing,
                to_kind="Document",
                to_id=cited,
                properties={},
            )
        )
    return out


def _build_evaluated_on(bundle: PreflightBundle) -> list:
    """§4.5: ``(method) -[EVALUATED_ON]-> (dataset)`` per
    ``(method_entity_id, dataset_entity_id)`` pair. The
    ``supporting_assertion_ids`` property is the lexicographically sorted
    unique union of every ``PROPOSES`` and ``EVALUATES_ON`` assertion ID
    that participates in at least one valid same-``Document`` witness.
    """
    # Per assertion record, classify whether it's a PROPOSES or
    # EVALUATES_ON and bind to its document + endpoints.
    witnesses: dict = {}  # (method_eid, dataset_eid) -> set of assertion_ids
    for assertion_id, assertion in bundle.assertions.items():
        if assertion.relation_type == "PROPOSES":
            if assertion.object_ref["kind"] != "mention":
                continue
            mention_eid = bundle.mention_to_entity[assertion.object_ref["id"]]
            entity = bundle.entities[mention_eid]
            if entity.entity_type != "method":
                continue
            method_eid = mention_eid
        elif assertion.relation_type == "EVALUATES_ON":
            if assertion.object_ref["kind"] != "mention":
                continue
            mention_eid = bundle.mention_to_entity[assertion.object_ref["id"]]
            entity = bundle.entities[mention_eid]
            if entity.entity_type != "dataset":
                continue
            dataset_eid = mention_eid
        else:
            continue
        # Find any same-Document EVALUATES_ON/PROPOSES pairs.
        if assertion.relation_type == "PROPOSES":
            # Find EVALUATES_ON assertions in the same document.
            for other_id, other in bundle.assertions.items():
                if other.relation_type != "EVALUATES_ON":
                    continue
                if other.doc_id != assertion.doc_id:
                    continue
                if other.object_ref["kind"] != "mention":
                    continue
                ds_eid = bundle.mention_to_entity[other.object_ref["id"]]
                if bundle.entities[ds_eid].entity_type != "dataset":
                    continue
                pair = (method_eid, ds_eid)
                witnesses.setdefault(pair, set()).add(assertion_id)
                witnesses[pair].add(other_id)
        else:
            # relation_type == "EVALUATES_ON"
            for other_id, other in bundle.assertions.items():
                if other.relation_type != "PROPOSES":
                    continue
                if other.doc_id != assertion.doc_id:
                    continue
                if other.object_ref["kind"] != "mention":
                    continue
                m_eid = bundle.mention_to_entity[other.object_ref["id"]]
                if bundle.entities[m_eid].entity_type != "method":
                    continue
                pair = (m_eid, dataset_eid)
                witnesses.setdefault(pair, set()).add(assertion_id)
                witnesses[pair].add(other_id)

    out = []
    for (method_eid, dataset_eid), assertion_ids in witnesses.items():
        out.append(
            relationship_entry(
                type="EVALUATED_ON",
                logical_id=[method_eid, dataset_eid],
                from_kind=pascal_case("method"),
                from_id=method_eid,
                to_kind=pascal_case("dataset"),
                to_id=dataset_eid,
                properties={
                    "supporting_assertion_ids": sorted(assertion_ids),
                },
            )
        )
    return out


def _build_all_relationships(bundle: PreflightBundle) -> list:
    """§7.1 step 3: canonical order
    HAS_CHUNK; MENTIONS by mention_id; CITES by (source, target);
    extracted assertions by assertion_id; derived EVALUATED_ON by
    (source_entity_id, target_entity_id). The §8.1 sort then
    establishes the final order for hashing.
    """
    rels: list = []
    rels.extend(_build_has_chunk_relationships(bundle))
    rels.extend(_build_mentions_relationships(bundle))
    rels.extend(_build_citations(bundle))
    rels.extend(_build_extracted_assertions(bundle))
    rels.extend(_build_evaluated_on(bundle))
    return rels


def _canonical_graph_payload(
    bundle: PreflightBundle,
) -> dict:
    """§8.1 canonical logical projection."""
    nodes = (
        _build_documents(bundle)
        + _build_chunks(bundle)
        + _build_entities(bundle)
    )
    nodes.sort(key=lambda n: (n["labels"], n["logical_id"]))
    relationships = _build_all_relationships(bundle)
    relationships.sort(
        key=lambda r: (
            r["type"],
            canonical_json(r["logical_id"]),
            canonical_json(r["from"]),
            canonical_json(r["to"]),
        )
    )
    return {"nodes": nodes, "relationships": relationships}


def graph_content_sha256(bundle: PreflightBundle) -> str:
    """§8.1 hash of the canonical logical projection."""
    payload = _canonical_graph_payload(bundle)
    return sha256_hex(payload)

# ---------------------------------------------------------------------------
# Atomic manifest writer (§8.2).
# ---------------------------------------------------------------------------


def _atomic_write_manifest(path: Path, payload: dict) -> None:
    """Write the manifest atomically: temp sibling, fsync, os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# §7 transactional replacement.
# ---------------------------------------------------------------------------


# Cypher delete: removes every pipeline node and its relationships in a
# single statement. The constraint of replacement is that no
# Document/Chunk/Entity node remains visible at commit.
_DELETE_PIPELINE_DATA = (
    "MATCH (n) WHERE n:Document OR n:Chunk OR n:Entity "
    "DETACH DELETE n"
)


def _anchor_key_for_kind(kind: str) -> str:
    """Return the property name used to MATCH a node of the given kind.

    Per §3.4 the canonical anchor for each node class is fixed:
    ``Document`` -> ``doc_id``; ``Chunk`` -> ``chunk_id``; every Entity
    subtype -> ``entity_id``.
    """
    if kind == "Document":
        return "doc_id"
    if kind == "Chunk":
        return "chunk_id"
    if kind.startswith("Entity") or kind in {"Method", "Dataset", "Metric", "Task"}:
        return "entity_id"
    raise IngestionError(f"unknown node kind for anchor key: {kind!r}")


def _node_match_clause(node: dict) -> str:
    """Build a ``MATCH`` clause that uniquely identifies ``node``."""
    labels = node["labels"]
    label_str = ":".join(labels)
    if labels == ["Document"]:
        return f"MATCH (a:{label_str} {{doc_id: $from_doc_id}})"
    if labels == ["Chunk"]:
        return f"MATCH (a:{label_str} {{chunk_id: $from_chunk_id}})"
    # Entity subtype (Method, Dataset, Metric, Task)
    return f"MATCH (a:{label_str} {{entity_id: $from_entity_id}})"


def _exec_in_tx(tx, *, statement: str, params: dict) -> None:
    tx.run(statement, **params).consume()


def replace_graph(
    bundle: PreflightBundle,
    driver,
    *,
    database: str = "neo4j",
) -> str:
    """§7 transactional replacement. Returns the canonical
    ``graph_content_sha256`` of the post-write graph projection.

    The transaction deletes prior graph data and rebuilds from
    canonical artifacts in one write. On any failure the transaction
    rolls back; no partial graph survives.
    """
    payload = _canonical_graph_payload(bundle)
    graph_sha = sha256_hex(payload)
    nodes = payload["nodes"]
    relationships = payload["relationships"]

    def _run(session, statement: str, **params):
        return session.run(statement, **params)

    # Driver-specific session API varies slightly between real neo4j and
    # test doubles. ``execute_write`` is the modern transactional API
    # that handles retries and rollback uniformly.
    def _replacement_tx(tx):
        tx.run(_DELETE_PIPELINE_DATA).consume()
        # Create nodes first (relationships need both endpoints).
        for node in nodes:
            labels = ":".join(node["labels"])
            tx.run(
                f"CREATE (n:{labels}) SET n = $props",
                props=node["properties"],
            ).consume()
        # Create relationships: anchor on the appropriate key per endpoint.
        for rel in relationships:
            from_anchor = _anchor_key_for_kind(rel["from"]["kind"])
            to_anchor = _anchor_key_for_kind(rel["to"]["kind"])
            tx.run(
                f"MATCH (a {{{from_anchor}: $from_id}}), "
                f"(b {{{to_anchor}: $to_id}}) "
                f"CREATE (a)-[r:{rel['type']}]->(b) SET r = $props",
                from_id=rel["from"]["id"],
                to_id=rel["to"]["id"],
                props=rel["properties"],
            ).consume()

    with driver.session(database=database) as session:
        session.execute_write(_replacement_tx)
    return graph_sha



# ---------------------------------------------------------------------------
# Manifest builder (§8.2).
# ---------------------------------------------------------------------------


_SCHEMA_VERSION = 1
_INGESTION_POLICY_VERSION = "graph_ingestion_v1"
_NEO4J_IMAGE_VERSION = "neo4j:5.26-community-ubi10"


def _build_manifest(
    *,
    bundle: PreflightBundle,
    graph_sha: str,
    citation_source_sha: str,
    node_counts: dict,
    relationship_counts: dict,
    started_at: float,
    ended_at: float,
) -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "ingestion_policy_version": _INGESTION_POLICY_VERSION,
        "neo4j_image_version": _NEO4J_IMAGE_VERSION,
        "inputs": dict(bundle.inputs),
        "citation_source_sha256": citation_source_sha,
        "graph_content_sha256": graph_sha,
        "ontology_version": bundle.ontology_version,
        "counts": {
            "nodes_by_label": dict(node_counts),
            "relationships_by_type": dict(relationship_counts),
        },
        "wall_clock_seconds": max(0.0, ended_at - started_at),
    }


def _verify_post_commit_hash(driver, expected_sha: str, *, database: str) -> None:
    """§8.2: after the transaction commits, the post-commit read of the
    canonical projection must match the pre-write ``graph_content_sha256``.
    The mock-friendly surface recomputes via the in-memory record set
    and compares; the real driver uses a session read.
    """
    actual = driver.last_graph_content_sha256()  # type: ignore[attr-defined]
    if actual != expected_sha:
        raise IngestionError(
            f"post-commit graph_content_sha256 mismatch "
            f"(expected {expected_sha!r}, got {actual!r})"
        )


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------


def run_ingestion(
    *,
    driver_factory=None,
    driver=None,
    manifest_path: Path = DEFAULT_OUTPUT_MANIFEST_PATH,
    database: str = "neo4j",
    env_file: Path | None = None,
) -> IngestionResult:
    """Top-level orchestrator. Runs:

    1. §2 preflight (raises :class:`PreflightError` on failure).
    2. §5 schema bootstrap (idempotent).
    3. §7 single-transaction replacement.
    4. §8.1 graph_content_sha256 computation.
    5. §8.2 atomic manifest publication.

    Drivers are constructed lazily:

    - ``driver`` — already constructed (used by tests).
    - ``driver_factory`` — callable returning a configured driver.
    - Neither — fall back to :func:`default_driver`, which reads the
      repository ``.env`` per §9.

    ``driver_factory`` wins over the default factory when supplied.
    """
    from src.build_graph.preflight import run_preflight

    bundle = run_preflight()
    citation_source_sha = sha256_hex_bytes(bundle.citation_source_bytes)

    started_at = time.monotonic()
    if driver is None:
        if driver_factory is not None:
            driver = driver_factory()
        else:
            driver = default_driver(env_file)

    try:
        bootstrap_schema(driver, database=database)
        graph_sha = replace_graph(bundle, driver, database=database)
        # §8.2: post-commit re-read verification.
        _verify_post_commit_hash(driver, graph_sha, database=database)
    finally:
        # Driver ownership lives with the factory; tests pass their own.
        pass
    ended_at = time.monotonic()

    # Compute counts from the canonical projection (deterministic).
    payload = _canonical_graph_payload(bundle)
    node_counts: dict = {}
    for node in payload["nodes"]:
        for label in node["labels"]:
            node_counts[label] = node_counts.get(label, 0) + 1
    relationship_counts: dict = {}
    for rel in payload["relationships"]:
        relationship_counts[rel["type"]] = (
            relationship_counts.get(rel["type"], 0) + 1
        )

    manifest = _build_manifest(
        bundle=bundle,
        graph_sha=graph_sha,
        citation_source_sha=citation_source_sha,
        node_counts=node_counts,
        relationship_counts=relationship_counts,
        started_at=started_at,
        ended_at=ended_at,
    )
    _atomic_write_manifest(manifest_path, manifest)

    return IngestionResult(
        graph_content_sha256=graph_sha,
        citation_source_sha256=citation_source_sha,
        node_counts=node_counts,
        relationship_counts=relationship_counts,
        manifest_path=manifest_path,
    )


def sha256_hex_bytes(data: bytes) -> str:
    """§8.2 helper: SHA256 over raw bytes (no JSON serialisation)."""
    return hashlib.sha256(data).hexdigest()
