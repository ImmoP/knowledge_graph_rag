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
  (§7) WITH §7 step 4 in-transaction referential-integrity checks
  (§11 acceptance gate #1, #2, #4, #5, #6). On any failure the
  transaction rolls back; no partial graph survives.
- :func:`run_ingestion` — top-level orchestrator that wires the
  preflight, schema bootstrap, replacement, hash computation,
  in-transaction integrity, post-commit readback verification (§8.2),
  and atomic manifest publication.

The driver is injected (``driver_factory`` / ``driver``) so tests can
supply a mock driver without booting Neo4j. Production callers use the
:func:`default_driver` factory that reads ``.env`` and returns the
official neo4j driver pinned in ``local-infra.md`` §3.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.build_graph.preflight import (
    DEFAULT_OUTPUT_MANIFEST_PATH,
    PreflightBundle,
    PreflightError,
)
from src.build_graph.schema import (
    canonical_json,
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
    if env_file is not None:
        load_dotenv(env_file, override=True)
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise IngestionError("NEO4J_PASSWORD is not set; required by §9")
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": password,
    }


def default_driver(env_file: Path | None = None):
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
    """§4.2: MENTIONS carries the full §4.6 property set."""
    out = []
    for mention_id, mention in bundle.mentions.items():
        entity_id = bundle.mention_to_entity.get(mention_id)
        if entity_id is None:
            raise IngestionError(
                f"mention {mention_id!r} has no entity resolution mapping"
            )
        evidence = mention.evidence
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
                    "char_start": int(evidence["char_start"]),
                    "char_end": int(evidence["char_end"]),
                    "source_text_sha256": str(evidence["source_text_sha256"]),
                    "confidence": mention.confidence,
                    "resolution_method": mention.resolution_method,
                    "resolution_score": mention.resolution_score,
                },
            )
        )
    return out


def _build_extracted_assertions(bundle: PreflightBundle) -> list:
    """§4.3: extracted ontology relationships carry the full §4.6 property set."""
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
        evidence = assertion.evidence
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
                    "char_start": int(evidence["char_start"]),
                    "char_end": int(evidence["char_end"]),
                    "source_text_sha256": str(evidence["source_text_sha256"]),
                    "confidence": assertion.confidence,
                },
            )
        )
    return out


def _build_citations(bundle: PreflightBundle) -> list:
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
    """§4.5: derived EVALUATED_ON closure from PROPOSES × EVALUATES_ON.

    Pairs: ``(d:Document)-[:PROPOSES]->(m:Entity:Method)`` AND
    ``(d)-[:EVALUATES_ON]->(ds:Entity:Dataset)`` produce
    ``(m)-[:EVALUATED_ON]->(ds)`` per `(method_eid, dataset_eid)` pair.
    `supporting_assertion_ids` is the sorted unique union of every
    participating PROPOSES + EVALUATES_ON assertion ID across all
    same-document witnesses.
    """
    witnesses: dict = {}
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

# ---------------------------------------------------------------------------
# Canonical projection (§8.1).
# ---------------------------------------------------------------------------


def _build_all_relationships(bundle: PreflightBundle) -> list:
    rels: list = []
    rels.extend(_build_has_chunk_relationships(bundle))
    rels.extend(_build_mentions_relationships(bundle))
    rels.extend(_build_citations(bundle))
    rels.extend(_build_extracted_assertions(bundle))
    rels.extend(_build_evaluated_on(bundle))
    return rels


def _canonical_graph_payload(bundle: PreflightBundle) -> dict:
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


def sha256_hex_bytes(data: bytes) -> str:
    """§8.2 helper: SHA256 over raw bytes (no JSON serialisation)."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Atomic manifest writer (§8.2).
# ---------------------------------------------------------------------------


def _atomic_write_manifest(path: Path, payload: dict) -> None:
    """Write the manifest atomically: temp sibling, fsync, os.replace.

    The body is JSON with sorted keys and a single trailing newline,
    matching the §8.2 publication contract.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# §7 transactional replacement WITH §7 step 4 in-transaction integrity.
# ---------------------------------------------------------------------------


_DELETE_PIPELINE_DATA = (
    "MATCH (n) WHERE n:Document OR n:Chunk OR n:Entity DETACH DELETE n"
)


def _anchor_key_for_kind(kind: str) -> str:
    """Return the property name used to MATCH a node of the given kind."""
    if kind == "Document":
        return "doc_id"
    if kind == "Chunk":
        return "chunk_id"
    if kind.startswith("Entity") or kind in {"Method", "Dataset", "Metric", "Task"}:
        return "entity_id"
    raise IngestionError(f"unknown node kind for anchor key: {kind!r}")


def _match_labels_for_kind(kind: str) -> str:
    """Return the label expression used in endpoint MATCH clauses.

    Labels are mandatory: Chunk nodes intentionally carry ``doc_id``,
    so matching only on ``{doc_id: ...}`` would also match chunks and
    duplicate structural/assertion relationships.
    """
    if kind in {"Document", "Chunk"}:
        return kind
    if not kind or not kind[0].isupper() or not kind.isalnum():
        raise IngestionError(f"invalid entity subtype label {kind!r}")
    return f"Entity:{kind}"


def _expected_counts(bundle: PreflightBundle) -> tuple:
    """Return ``(nodes_by_label, relationships_by_type)`` expected
    counts after a successful replacement.

    These are the post-write counts the in-transaction integrity
    checker enforces (§7 step 4).
    """
    payload = _canonical_graph_payload(bundle)
    node_counts: dict = {}
    for node in payload["nodes"]:
        for label in node["labels"]:
            node_counts[label] = node_counts.get(label, 0) + 1
    rel_counts: dict = {}
    for rel in payload["relationships"]:
        rel_counts[rel["type"]] = rel_counts.get(rel["type"], 0) + 1
    return node_counts, rel_counts


def _check_in_transaction_integrity(tx, bundle: PreflightBundle) -> None:
    """§7 step 4 / acceptance gate §11 #1-#6 / #7.

    Runs inside the same write transaction as DELETE + CREATE.
    Raises :class:`IngestionError` on any mismatch; the surrounding
    transaction rolls back. Verified invariants:

    - pipeline nodes counts by label match the canonical expectation;
    - pipeline relationship counts by type match the canonical
      expectation;
    - every Chunk has exactly one incoming HAS_CHUNK;
    - every mention_id has exactly one MENTIONS;
    - every assertion_id has exactly one extracted ontology relationship;
    - relationship endpoints exist (no dangling references);
    - ontology relationships have the required endpoint labels and
      direction (enforced at write time by label construction; the
      readback just confirms counts).
    """
    exp_nodes, exp_rels = _expected_counts(bundle)

    def _run(query: str, **params: Any) -> int:
        result = tx.run(query, **params).single(strict=True)
        return int(result[0]) if result is not None else 0

    actual_doc = _run("MATCH (n:Document) RETURN count(n)")
    if actual_doc != exp_nodes.get("Document", 0):
        raise IngestionError(
            f"in-transaction: Document count mismatch "
            f"({actual_doc} != {exp_nodes.get('Document', 0)})"
        )
    actual_chunk = _run("MATCH (n:Chunk) RETURN count(n)")
    if actual_chunk != exp_nodes.get("Chunk", 0):
        raise IngestionError(
            f"in-transaction: Chunk count mismatch "
            f"({actual_chunk} != {exp_nodes.get('Chunk', 0)})"
        )
    actual_entity = _run("MATCH (n:Entity) RETURN count(n)")
    if actual_entity != exp_nodes.get("Entity", 0):
        raise IngestionError(
            f"in-transaction: Entity count mismatch "
            f"({actual_entity} != {exp_nodes.get('Entity', 0)})"
        )

    for rel_type, expected in exp_rels.items():
        actual = _run(
            "MATCH ()-[r:%s]->() RETURN count(r)" % rel_type,
        )
        if actual != expected:
            raise IngestionError(
                f"in-transaction: relationship type {rel_type!r} count "
                f"mismatch ({actual} != {expected})"
            )

    # HAS_CHUNK: every Chunk must have exactly one incoming HAS_CHUNK.
    bad_chunks = _run(
        "MATCH (c:Chunk) "
        "OPTIONAL MATCH (d:Document)-[r:HAS_CHUNK]->(c) "
        "WITH c, count(r) AS incoming "
        "WHERE incoming <> 1 "
        "RETURN count(c)"
    )
    if bad_chunks != 0:
        raise IngestionError(
            f"in-transaction: {bad_chunks} Chunk(s) lack incoming HAS_CHUNK"
        )
    # MENTIONS: every mention_id must have exactly one MENTIONS.
    bad_mentions = _run(
        "MATCH ()-[r:MENTIONS]->() "
        "WITH r.mention_id AS mid, count(r) AS multiplicity "
        "WHERE multiplicity <> 1 "
        "RETURN count(mid)"
    )
    if bad_mentions != 0:
        raise IngestionError(
            f"in-transaction: {bad_mentions} mention_id(s) lack unique MENTIONS"
        )
    # Extracted ontology relationships: every assertion_id exactly one.
    bad_assertions = _run(
        "MATCH ()-[r]->() "
        "WHERE type(r) IN ['PROPOSES','INTRODUCES','EVALUATES_ON',"
        "'USES_METHOD','REPORTS_METRIC','EXTENDS',"
        "'USED_FOR_TASK','MEASURES'] "
        "WITH r.assertion_id AS aid, count(r) AS multiplicity "
        "WHERE multiplicity <> 1 "
        "RETURN count(aid)"
    )
    if bad_assertions != 0:
        raise IngestionError(
            f"in-transaction: {bad_assertions} assertion_id(s) lack unique "
            "extracted ontology relationship"
        )
    # Exact projection equality covers endpoint existence, labels,
    # direction, multiplicity, and persisted property sets.
    if _read_graph_payload(tx) != _canonical_graph_payload(bundle):
        raise IngestionError(
            "in-transaction: canonical graph projection mismatch after replacement"
        )


def replace_graph(bundle: PreflightBundle, driver, *, database: str = "neo4j") -> str:
    """§7 transactional replacement with §7 step 4 in-transaction
    integrity checks. Returns the canonical ``graph_content_sha256``
    of the post-write graph projection.

    The transaction deletes prior graph data, rebuilds from canonical
    artifacts, runs referential-integrity checks INSIDE the same
    transaction, and commits only if every check passes; otherwise
    rolls back to the prior graph snapshot.
    """
    payload = _canonical_graph_payload(bundle)
    graph_sha = sha256_hex(payload)
    nodes = payload["nodes"]
    relationships = payload["relationships"]

    def _replacement_tx(tx):
        tx.run(_DELETE_PIPELINE_DATA).consume()
        # Nodes first (relationships need both endpoints).
        for node in nodes:
            labels = ":".join(node["labels"])
            tx.run(
                f"CREATE (n:{labels}) SET n = $props",
                props=node["properties"],
            ).consume()
        for rel in relationships:
            from_kind = rel["from"]["kind"]
            to_kind = rel["to"]["kind"]
            from_anchor = _anchor_key_for_kind(from_kind)
            to_anchor = _anchor_key_for_kind(to_kind)
            from_labels = _match_labels_for_kind(from_kind)
            to_labels = _match_labels_for_kind(to_kind)
            tx.run(
                f"MATCH (a:{from_labels} {{{from_anchor}: $from_id}}), "
                f"(b:{to_labels} {{{to_anchor}: $to_id}}) "
                f"CREATE (a)-[r:{rel['type']}]->(b) SET r = $props",
                from_id=rel["from"]["id"],
                to_id=rel["to"]["id"],
                props=rel["properties"],
            ).consume()
        _check_in_transaction_integrity(tx, bundle)

    with driver.session(database=database) as session:
        session.execute_write(_replacement_tx)
    return graph_sha


# ---------------------------------------------------------------------------
# §8.2 post-commit readback via the standard driver API.
# ---------------------------------------------------------------------------


#: §8.1: ``logical_id`` is label-determined, NOT a coalesce over the
#: three id properties — a ``Chunk`` node also carries ``doc_id``, so a
#: coalesce would report the document id for every chunk.
_LOGICAL_ID_CASE = (
    "CASE WHEN 'Chunk' IN labels({var}) THEN {var}.chunk_id "
    "WHEN 'Entity' IN labels({var}) THEN {var}.entity_id "
    "ELSE {var}.doc_id END"
)


def _read_graph_payload(tx) -> dict:
    """Read the ACTUAL committed pipeline projection through a normal
    Neo4j transaction and reconstruct the exact §8.1 payload."""
    nodes: list = []
    relationships: list = []
    for record in tx.run(
        "MATCH (n) WHERE n:Document OR n:Chunk OR n:Entity "
        "RETURN labels(n) AS labels, "
        + _LOGICAL_ID_CASE.format(var="n") + " AS logical_id, "
        "properties(n) AS props"
    ):
        nodes.append(
            node_entry(
                labels=list(record["labels"]),
                logical_id=record["logical_id"],
                properties=_read_props(record["props"]),
            )
        )
    for record in tx.run(
        "MATCH (a)-[r]->(b) "
        "WHERE (a:Document OR a:Chunk OR a:Entity) "
        "AND (b:Document OR b:Chunk OR b:Entity) "
        "RETURN type(r) AS rel_type, "
        + _LOGICAL_ID_CASE.format(var="a") + " AS from_id, "
        "labels(a) AS from_labels, "
        + _LOGICAL_ID_CASE.format(var="b") + " AS to_id, "
        "labels(b) AS to_labels, "
        "properties(r) AS props"
    ):
        relationships.append(
            _read_relationship_entry(
                rel_type=record["rel_type"],
                from_id=record["from_id"],
                from_labels=record["from_labels"],
                to_id=record["to_id"],
                to_labels=record["to_labels"],
                props=record["props"],
            )
        )
    nodes.sort(key=lambda n: (n["labels"], n["logical_id"]))
    relationships.sort(
        key=lambda r: (
            r["type"],
            canonical_json(r["logical_id"]),
            canonical_json(r["from"]),
            canonical_json(r["to"]),
        )
    )
    return {"nodes": nodes, "relationships": relationships}


def readback_graph_content_sha256(driver, *, database: str = "neo4j") -> str:
    """§8.2 post-commit readback through the standard driver read API."""
    payload: dict = {}

    with driver.session(database=database) as session:
        def _read_tx(tx):
            nonlocal payload
            payload = _read_graph_payload(tx)

        session.execute_read(_read_tx)
    return sha256_hex(payload)


def _read_props(props: dict) -> dict:
    out = dict(props)
    # Convert lists back to lists (Neo4j returns list values as native
    # Python lists already; this is a no-op safety pass).
    for key, value in list(out.items()):
        if isinstance(value, list):
            out[key] = list(value)
    return out


def _read_relationship_entry(
    *,
    rel_type: str,
    from_id: str,
    from_labels: list,
    to_id: str,
    to_labels: list,
    props: dict,
) -> dict:
    return {
        "type": rel_type,
        "logical_id": _read_logical_id(rel_type, props, from_id, to_id),
        "from": {"kind": _pick_kind_label(from_labels), "id": from_id},
        "to": {"kind": _pick_kind_label(to_labels), "id": to_id},
        "properties": _read_props(props),
    }


def _pick_kind_label(labels: list) -> str:
    """Pick the canonical kind for a node readback.

    Structural labels win (``Document`` / ``Chunk``); for entity nodes
    the chosen kind is the Entity subtype label (e.g. ``Method``),
    matching what the projection writes.
    """
    for candidate in labels:
        if candidate in {"Document", "Chunk"}:
            return candidate
    for candidate in labels:
        if candidate != "Entity":
            return candidate
    return "Entity"


def _read_logical_id(
    rel_type: str, props: dict, from_id: str, to_id: str
) -> Any:
    """Recover the §8.1 logical_id for a relationship.

    Single-id relationship identities are persisted on the
    relationship. Pair identities are reconstructed from the canonical
    logical endpoint IDs read in the same query; those properties are
    intentionally absent from HAS_CHUNK / CITES / EVALUATED_ON.
    """
    if rel_type == "MENTIONS":
        return props.get("mention_id", "")
    if rel_type in {
        "PROPOSES",
        "INTRODUCES",
        "EVALUATES_ON",
        "USES_METHOD",
        "REPORTS_METRIC",
        "EXTENDS",
        "USED_FOR_TASK",
        "MEASURES",
    }:
        return props.get("assertion_id", "")
    if rel_type in {"HAS_CHUNK", "CITES", "EVALUATED_ON"}:
        return [from_id, to_id]
    return props.get("assertion_id", props.get("mention_id", ""))


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
) -> dict:
    """§8.2 manifest.

    Deterministic across two runs with identical canonical inputs.
    No wall-clock timestamps, host names, Neo4j internal IDs, or
    credentials. Path strings are repository-relative POSIX.
    """
    inputs = dict(bundle.inputs)
    return {
        "schema_version": _SCHEMA_VERSION,
        "ingestion_policy_version": _INGESTION_POLICY_VERSION,
        "neo4j_image_version": _NEO4J_IMAGE_VERSION,
        "inputs": inputs,
        "citation_source_sha256": citation_source_sha,
        "graph_content_sha256": graph_sha,
        "ontology": {
            "ontology_path": inputs.get("ontology_path"),
            "ontology_sha256": bundle.ontology_sha,
            "ontology_version": bundle.ontology_version,
        },
        "entity_resolution_algorithm": {
            "algorithm_version": bundle.er_algorithm_version,
            "normalization_version": bundle.er_normalization_version,
            "similarity_metric": bundle.er_similarity_metric,
            "threshold": bundle.er_threshold,
        },
        "counts": {
            "nodes_by_label": dict(node_counts),
            "relationships_by_type": dict(relationship_counts),
        },
    }


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
    3. §7 single-transaction replacement WITH §7 step 4 in-transaction
       referential-integrity checks.
    4. §8.1 graph_content_sha256 computation.
    5. §8.2 post-commit readback verification via the standard
       driver read API; the readback hash must equal the pre-write
       hash.
    6. §8.2 atomic manifest publication.

    Drivers are constructed lazily: ``driver`` (already constructed,
    used by tests); ``driver_factory`` (callable returning a configured
    driver); otherwise :func:`default_driver` reads the repository
    ``.env`` per §9.
    """
    from src.build_graph.preflight import run_preflight

    bundle = run_preflight()
    citation_source_sha = sha256_hex_bytes(bundle.citation_source_bytes)

    if driver is None:
        driver = driver_factory() if driver_factory is not None else default_driver(env_file)
    bootstrap_schema(driver, database=database)
    graph_sha = replace_graph(bundle, driver, database=database)
    # §8.2: post-commit re-read verification via the standard driver API.
    readback_sha = readback_graph_content_sha256(driver, database=database)
    if readback_sha != graph_sha:
        raise IngestionError(
            "post-commit readback hash mismatch: "
            f"expected {graph_sha!r}, got {readback_sha!r}"
        )

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
    )
    _atomic_write_manifest(manifest_path, manifest)

    return IngestionResult(
        graph_content_sha256=graph_sha,
        citation_source_sha256=citation_source_sha,
        node_counts=node_counts,
        relationship_counts=relationship_counts,
        manifest_path=manifest_path,
    )
