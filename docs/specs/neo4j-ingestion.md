# Neo4j Graph Schema and Ingestion Contract

Draft resolution of [#9 — Decide the Neo4j graph schema and ingestion contract](https://github.com/ImmoP/knowledge_graph_rag/issues/9). This document fixes the graph shape, provenance mapping, constraints, and deterministic replacement contract. It does not implement ingestion.

The Neo4j database is a derived projection of canonical on-disk artifacts. It is disposable and rebuildable; it is never the authority for documents, chunks, extraction records, entity resolution, or ontology vocabulary.

## 1. Inputs and ownership

The ingestion reads:

- `data/manifest.jsonl` for canonical corpus documents.
- `data/derived/chunks.jsonl` for canonical chunks.
- `ontology.yaml` for entity labels, relation types, endpoint types, population modes, and derivations.
- `data/extractions.jsonl` and `data/extractions.manifest.json` for validated mentions, extracted assertions, and extraction completeness.
- `data/entities.jsonl`, `data/entity_resolutions.jsonl`, and `data/entity_resolution.manifest.json` for canonical extracted entities and mention resolution.
- `data/raw/openalex/<doc_id>.json` plus `data/aliases.json` for structural in-corpus `CITES` edges.

It connects only to the local Neo4j service fixed by the [local-infrastructure spec](local-infra.md). It does not read pgvector, call an LLM or embedding provider, or mutate any input artifact.

The implementing ticket writes `data/graph_ingestion.manifest.json` after a successful graph replacement. That file is the reproducibility record for the mutable database projection.

## 2. Global preflight

All validation completes before the write transaction begins. Any failure leaves the existing graph and prior ingestion manifest untouched.

The preflight verifies:

1. Every input exists, is schema-valid, and has a unique primary ID.
2. Every manifest `openalex_id` matches `^W[0-9]+$`; every chunk `doc_id` exists in the manifest; every chunk ID matches `^c_[0-9a-f]{16}$` and re-derives under the chunk contract.
3. The extraction manifest's ontology and chunk hashes match the current files, and every required chunk has exactly one `chunk_status` with `extraction_complete: true`.
4. The resolution manifest's extraction, chunk, and ontology hashes match the current inputs; its output hashes match the current entity and resolution files.
5. Every eligible mention occurs in exactly one entity membership list and exactly one resolution record. Resolution method/score and entity type are valid under #8.
6. Every assertion ID re-derives under the repaired #7 typed-ref contract. Document refs equal the assertion/chunk `doc_id`; mention refs resolve to validated same-chunk mentions of the ontology-required endpoint type.
7. Every assertion mention ref has exactly one mapping to an existing canonical entity whose `entity_type` matches the ontology endpoint type. Missing, duplicate, dangling, or type-incompatible mappings fail the run.
8. Every ontology relation endpoint resolves to a node type admitted by this schema. Vocabulary or population-mode drift without an ontology version/hash change fails the run.
9. Every canonical document in `data/manifest.jsonl` has its required raw OpenAlex metadata record, and that record matches the `sha256` recorded for the document. A present record whose `referenced_works` field is `null` or an empty list validly represents zero outgoing citations; a missing raw record is an input-integrity failure and is never treated as zero citations. Alias resolution is acyclic and ends at a canonical manifest document or outside the corpus.

There is no partial-ingestion mode. Invalid records are not skipped, quarantined, or converted into placeholder nodes.

## 3. Node model

There are no `Mention`, `Assertion`, `Relation`, `OntologyType`, or ingestion-run nodes in v1. Provenance that belongs to a mention or assertion is stored on its relationship. This keeps graph traversal direct while the JSONL inputs remain the complete audit source.

### 3.1 `Document`

One node for every record in `data/manifest.jsonl`:

```text
(:Document {
  doc_id,
  title
})
```

Rules:

- `doc_id` equals manifest `openalex_id` byte-for-byte and is the node's canonical ID.
- `title` is copied byte-for-byte and must be a non-empty string.
- No other manifest metadata is duplicated on the node. DOI, year, acquisition metadata, hashes, URLs, popularity counts, and merged IDs remain in the canonical manifest.

### 3.2 `Chunk`

One node for every record in `data/derived/chunks.jsonl`:

```text
(:Chunk {
  chunk_id,
  doc_id,
  section_path,
  chunk_index
})
```

Rules:

- These four properties are copied without normalization; `section_path` remains a list of strings.
- `doc_id` is retained as a denormalized cross-store join key and must equal the adjacent `Document.doc_id`.
- No other chunk property is stored. In particular, text, token estimate, and character offsets remain in `data/derived/chunks.jsonl`; canonical text is also the vector-side retrieval corpus.

### 3.3 Extracted entities

One node for every record in `data/entities.jsonl`:

```text
(:Entity:<TypeLabel> {
  entity_id,
  entity_type,
  canonical_name,
  aliases,
  resolution_confidence
})
```

Every extracted entity has the common `Entity` label plus exactly one ontology-derived subtype label. Ontology v1 maps:

| Ontology type | Neo4j labels |
|---|---|
| `method` | `:Entity:Method` |
| `dataset` | `:Entity:Dataset` |
| `metric` | `:Entity:Metric` |
| `task` | `:Entity:Task` |

The general mapping is deterministic PascalCase from the validated snake-case ontology type. `document` and `chunk` map to their structural `Document` and `Chunk` nodes; they do not also receive `Entity`. A label mapping collision or a type that cannot produce a valid Neo4j label fails preflight.

The exact persisted Entity properties are selected from `entities.jsonl` without recomputation. `mention_ids` remains in the canonical JSONL audit record and is not duplicated on the node. `entity_id` equals the #8 `e_*` ID byte-for-byte.

### 3.4 Exact node-property table

The property sets are exact; ingestion does not add convenience, timestamp, hash, or run-state properties.

| Node | Labels | Logical identity | Persisted properties |
|---|---|---|---|
| Document | `:Document` | `doc_id` | `doc_id`, `title` |
| Chunk | `:Chunk` | `chunk_id` | `chunk_id`, `doc_id`, `section_path`, `chunk_index` |
| Method | `:Entity:Method` | `entity_id` | `entity_id`, `entity_type`, `canonical_name`, `aliases`, `resolution_confidence` |
| Dataset | `:Entity:Dataset` | `entity_id` | same Entity property set |
| Metric | `:Entity:Metric` | `entity_id` | same Entity property set |
| Task | `:Entity:Task` | `entity_id` | same Entity property set |

## 4. Relationship model

Relationship type spellings are uppercase `SCREAMING_SNAKE_CASE`. Ontology relation names are used byte-for-byte after validation; ingestion does not maintain a parallel vocabulary.

Relationship classes are explicit:

- **Infrastructure / graph plumbing:** `HAS_CHUNK`, `MENTIONS`.
- **Ontology / semantic traversal:** `CITES`, `PROPOSES`, `INTRODUCES`, `EVALUATES_ON`, `USES_METHOD`, `REPORTS_METRIC`, `EXTENDS`, `EVALUATED_ON`, `USED_FOR_TASK`, `MEASURES`.

`HAS_CHUNK` and `MENTIONS` are not ontology relations. They do not count toward benchmark hop count and must never appear as `eval.evidence_path[].relation`. #11 may traverse them internally to locate supporting chunks, but only ontology relationships form the semantic traversal surface measured by the evaluation contract.

### 4.1 Structural join relationships

```text
(:Document)-[:HAS_CHUNK]->(:Chunk)
```

There is exactly one `HAS_CHUNK` relationship per chunk, directed from its manifest document to the chunk. It has no properties.

This direction is already fixed by local-infrastructure #3 and supersedes #9's older ticket-body example `(:Chunk)-[:FROM_DOC]->(:Document)`. The graph does not create both directions; Cypher can traverse `HAS_CHUNK` in reverse when needed.

### 4.2 Mention provenance

For each eligible mention:

```text
(:Chunk)-[:MENTIONS {
  mention_id,
  chunk_id,
  char_start,
  char_end,
  source_text_sha256,
  confidence,
  resolution_method,
  resolution_score
}]->(:Entity)
```

There is one `MENTIONS` relationship per `mention_id`, including when the same chunk mentions the same entity more than once. `mention_id` is its logical identity; `chunk_id` equals the source node's ID byte-for-byte. The relationship combines immutable mention/evidence fields from #7 with method/score from the matching #8 resolution record. Its target is the entity named by that mapping. Parallel `MENTIONS` relationships are intentional and remain distinct because their `mention_id` values differ. Neo4j internal relationship IDs are never provenance.

### 4.3 Extracted ontology relationships

For every validated `kind: "assertion"` record, create one relationship whose type is `assertion.relation_type` and whose direction is the ontology's `from -> to` direction.

Endpoint projection is fixed:

```text
{"kind":"document","id":doc_id}
    -> (:Document {doc_id: doc_id})

{"kind":"mention","id":mention_id}
    -> entity_resolutions.jsonl
    -> (:Entity {entity_id: entity_id})
```

The relationship carries:

```text
{
  assertion_id,
  chunk_id,
  char_start,
  char_end,
  source_text_sha256,
  confidence
}
```

There is one relationship per assertion ID. Parallel relationships of the same ontology type between the same nodes are intentional when separate assertions provide distinct evidence. Ingestion does not collapse, average, or select among them.

The eight extracted v1 relationship types and directions are:

| Relationship | Direction |
|---|---|
| `PROPOSES` | `Document -> Method` |
| `INTRODUCES` | `Document -> Dataset` |
| `EVALUATES_ON` | `Document -> Dataset` |
| `USES_METHOD` | `Document -> Method` |
| `REPORTS_METRIC` | `Document -> Metric` |
| `EXTENDS` | `Method -> Method` |
| `USED_FOR_TASK` | `Dataset -> Task` |
| `MEASURES` | `Metric -> Task` |

This table documents ontology v1 for readability; ingestion derives and validates it from `ontology.yaml` rather than hard-coding it.

### 4.4 Structural `CITES`

```text
(:Document)-[:CITES]->(:Document)
```

Direction is citing document to cited document. For each canonical manifest document, read its raw OpenAlex `referenced_works`, normalize OpenAlex URLs to short `W...` IDs, resolve IDs through `data/aliases.json`, retain only targets present in the canonical corpus manifest, de-duplicate `(citing_doc_id, cited_doc_id)`, and sort pairs lexicographically before creation.

External cited works do not create placeholder `Document` nodes in v1. Self-loops produced by alias canonicalisation are discarded. `CITES` has no LLM provenance and no relationship properties. Its complete v1 source is the deterministic combination of raw OpenAlex `referenced_works`, `data/aliases.json`, and canonical IDs in `data/manifest.jsonl`; bibliography parsing is not a second citation source.

### 4.5 Derived `EVALUATED_ON`

After extracted relationships exist, apply the ontology v1 derivation:

```text
(d:Document)-[:PROPOSES]->(m:Entity:Method)
(d)-[:EVALUATES_ON]->(ds:Entity:Dataset)
=> (m)-[:EVALUATED_ON]->(ds)
```

There is one `EVALUATED_ON` relationship per `(method_entity_id, dataset_entity_id)` pair. Its property `supporting_assertion_ids` is the lexicographically sorted, unique union of every `PROPOSES` assertion ID and every `EVALUATES_ON` assertion ID that participates in at least one valid same-`Document` witness producing that exact pair. If several documents independently witness the pair, all of their participating assertion IDs aggregate onto this single edge; ingestion does not create one derived edge per witness document. It contains no generated timestamp, random ID, or Neo4j internal ID.

The edge is recomputed from scratch on every replacement. It is never read from extraction and never emitted when either supporting extracted relationship is absent.

### 4.6 Exact relationship-property and identity table

| Relationship class/type | Logical identity | Persisted properties |
|---|---|---|
| `HAS_CHUNK` | `(doc_id, chunk_id)` | none |
| `MENTIONS` | `mention_id` | `mention_id`, `chunk_id`, `char_start`, `char_end`, `source_text_sha256`, `confidence`, `resolution_method`, `resolution_score` |
| Each extracted ontology assertion (`PROPOSES`, `INTRODUCES`, `EVALUATES_ON`, `USES_METHOD`, `REPORTS_METRIC`, `EXTENDS`, `USED_FOR_TASK`, `MEASURES`) | `assertion_id` | `assertion_id`, `chunk_id`, `char_start`, `char_end`, `source_text_sha256`, `confidence` |
| `CITES` | `(citing_doc_id, cited_doc_id)` | none |
| Derived `EVALUATED_ON` | `(method_entity_id, dataset_entity_id)` | `supporting_assertion_ids` |

These identities, not Neo4j internal relationship IDs or insertion position, drive de-duplication, post-write validation, canonical export, and graph hashing.

## 5. Constraints and indexes

One-time/idempotent schema bootstrap creates these named constraints before data replacement:

```cypher
CREATE CONSTRAINT document_doc_id_unique IF NOT EXISTS
FOR (n:Document) REQUIRE n.doc_id IS UNIQUE;

CREATE CONSTRAINT chunk_chunk_id_unique IF NOT EXISTS
FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE;

CREATE CONSTRAINT entity_entity_id_unique IF NOT EXISTS
FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE;
```

The preflight separately enforces presence, type, and regex rules because v1 targets Neo4j Community and does not depend on enterprise-only property-existence constraints.

Create these lookup indexes:

```cypher
CREATE INDEX chunk_doc_id_lookup IF NOT EXISTS
FOR (n:Chunk) ON (n.doc_id);

CREATE INDEX entity_type_name_lookup IF NOT EXISTS
FOR (n:Entity) ON (n.entity_type, n.canonical_name);
```

The uniqueness constraints already provide ID lookup indexes. No full-text, vector, relationship-property, APOC, or additional per-subtype indexes are part of v1.

## 6. Cross-store ID contract

- `Document.doc_id` equals `data/manifest.jsonl.openalex_id` and every chunk/pgvector `doc_id` byte-for-byte.
- `Chunk.chunk_id` equals `data/derived/chunks.jsonl.chunk_id` and pgvector `chunks.chunk_id` byte-for-byte.
- `Entity.entity_id` equals `data/entities.jsonl.entity_id` byte-for-byte.
- `MENTIONS.mention_id` equals the immutable #7 mention ID and its #8 mapping key byte-for-byte.
- Extracted ontology relationships retain `assertion_id` and `chunk_id` byte-for-byte from #7.

Ingestion never lowercases, URL-expands, prefixes, rehashes, or database-generates these IDs. Neo4j internal node and relationship IDs are explicitly outside the contract and must not appear in application or benchmark output.

## 7. Deterministic replacement and idempotency

The v1 default and only benchmark-producing mode is **replace**, not incremental upsert.

Schema bootstrap is separate from benchmark data replacement and may safely run repeatedly with `IF NOT EXISTS`. It is not claimed to be part of the data transaction.

After successful preflight and schema bootstrap, one Neo4j write transaction:

1. Deletes the previous pipeline graph data (`Document`, `Chunk`, and `Entity` nodes and their relationships).
2. Recreates nodes in canonical order: documents by `doc_id`, chunks by `chunk_id`, entities by `entity_id`.
3. Recreates relationships in canonical order: `HAS_CHUNK`; `MENTIONS` by `mention_id`; structural `CITES` by `(source_doc_id, target_doc_id)`; extracted assertions by `assertion_id`; derived `EVALUATED_ON` by `(source_entity_id, target_entity_id)`.
4. Runs post-write counts and referential-integrity checks inside the transaction.
5. Commits only if every check passes; otherwise rolls back to the prior graph snapshot.

Replace is selected because a pure `MERGE` upsert does not remove nodes or relationships that disappeared from changed inputs. The corpus is small enough for one transaction, and the graph database is dedicated to this reconstructible projection.

Constraints and indexes survive node replacement. No ingestion timestamp, run UUID, random value, host name, or provider identity is stored in graph content. Database-internal record order and internal IDs are not deterministic and are not compared.

Two ingestions from identical input bytes and ontology/resolver versions are equivalent when canonical export rows of all node labels, node properties, relationship types, endpoints, and relationship properties are byte-identical after deterministic sorting. Re-running identical inputs leaves the same logical graph and the same ingestion-manifest bytes.

## 8. Canonical graph hash and ingestion manifest

### 8.1 Canonical logical projection

`graph_content_sha256` is SHA256 over a canonical logical projection, never over Neo4j store files, internal IDs, query return order, or creation order.

Construct one top-level object with exactly `nodes` and `relationships` arrays:

```json
{"nodes":[...],"relationships":[...]}
```

Each node entry contains exactly:

```json
{"labels":["Entity","Method"],"logical_id":"e_...","properties":{...}}
```

`labels` is lexicographically sorted. `logical_id` is `doc_id`, `chunk_id`, or `entity_id` according to the node table. `properties` contains every and only persisted property from §3.4, with property-list values retained in their already-pinned deterministic order.

Each relationship entry contains exactly:

```json
{
  "from":{"kind":"Document","id":"W..."},
  "logical_id":["W...","c_..."],
  "properties":{},
  "to":{"kind":"Chunk","id":"c_..."},
  "type":"HAS_CHUNK"
}
```

`from` and `to` use logical node kind (`Document`, `Chunk`, or the Entity subtype label) and canonical node ID. `logical_id` encodes the exact identity from §4.6: a string for `mention_id`/`assertion_id`, or an ordered JSON array for pair identities. `properties` contains every and only the persisted properties in §4.6.

Sort node entries by `(labels, logical_id)` and relationship entries by `(type, canonical_json(logical_id), canonical_json(from), canonical_json(to))`. Serialise the top-level object with `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, and UTF-8; hash those exact bytes. JSON numbers remain numbers and no incidental whitespace or trailing newline enters the hash.

The same validated input bytes and graph-policy version must yield the same `graph_content_sha256` regardless of Neo4j internal IDs, insertion order, filesystem order, or query-plan order.

### 8.2 Ingestion manifest

`data/graph_ingestion.manifest.json` records:

- `schema_version` and ingestion-policy version.
- Neo4j image/version expected by the local-infrastructure contract.
- Input paths plus the separately named digests `manifest_sha256`, `aliases_sha256`, `ontology_sha256`, `chunks_sha256`, `extractions_sha256`, `extractions_manifest_sha256`, `entities_sha256`, `entity_resolutions_sha256`, and `entity_resolution_manifest_sha256`.
- `citation_source_sha256`, computed as specified below from the raw OpenAlex citation metadata before alias remapping and in-corpus filtering.
- Ontology version and entity-resolution algorithm/normalisation/similarity/threshold identity.
- Counts by node label and relationship type.
- `graph_content_sha256` computed over §8.1's canonical logical projection.

It contains no wall-clock timestamp, host name, database-internal IDs, or credentials. It is written to a temporary sibling and atomically replaces the prior manifest only after the Neo4j transaction commits and a post-commit read verifies the same content hash. If the database commit succeeds but manifest publication fails, the next consumer rejects the graph because no matching manifest exists; rerunning replace safely repairs the state.

#### Citation-source digest

Build a JSON array containing exactly one record for every canonical document in `data/manifest.jsonl`:

```json
[{"doc_id":"W...","referenced_work_ids":["W...","W..."]}]
```

For each record, `doc_id` is the canonical short OpenAlex ID and `referenced_work_ids` comes from that document's required raw OpenAlex metadata record. Normalize every OpenAlex URL or ID to its short `W...` form, de-duplicate it, and sort the resulting IDs lexicographically. Treat a present `null` or empty `referenced_works` value as an empty array. Sort records lexicographically by `doc_id`.

Serialize the complete array with `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, and UTF-8, with no trailing newline, and SHA256 those exact bytes. This calculation occurs before alias remapping, self-loop removal, and in-corpus filtering. The combination of `citation_source_sha256`, `aliases_sha256`, and `manifest_sha256` fully identifies the deterministic input to `CITES` construction.

## 9. Credentials and provider boundary

Ingestion loads the repository `.env` and reads exactly:

- `NEO4J_URI`, default `bolt://localhost:7687`.
- `NEO4J_USER`, default `neo4j`.
- `NEO4J_PASSWORD`, required with no default.

It uses the official `neo4j` Python driver fixed by #3. It reads no `LLM_*`, `EMBEDDING_*`, OpenAI, Anthropic, Ollama, pgvector, or other provider credentials. No credential is written to logs, graph properties, or the ingestion manifest.

## 10. Downstream contract

- #11 graph retrieval queries only this schema and returns canonical `chunk_id` values.
- #12 hybrid retrieval joins graph results to pgvector through byte-identical `chunk_id` values.
- #14 records and validates the graph-ingestion manifest path, SHA256, ontology version/hash, extraction hash, resolution hash, and graph-content hash before scoring.
- Eval evidence-path `W...`, `c_*`, and `e_*` IDs resolve respectively to `Document.doc_id`, `Chunk.chunk_id`, and `Entity.entity_id`.

Consumers fail on a missing/stale ingestion manifest or an ID absent from the current canonical files. They do not query by Neo4j internal IDs.

## 11. Implementing-ticket acceptance gate

The later implementing ticket closes only when:

1. Global preflight detects stale hashes, incomplete extraction, invalid typed refs, missing/duplicate mappings, type mismatches, a missing canonical document's raw OpenAlex record, citation-source drift, and ID collisions before any graph write; a present null/empty `referenced_works` value remains valid.
2. All node/relationship counts equal their canonical input-derived expectations.
3. Every chunk has exactly one incoming `HAS_CHUNK`; every mention ID has exactly one `MENTIONS`; every assertion ID has exactly one extracted ontology relationship.
4. Every ontology relationship's node labels match its declared endpoint types and direction.
5. `CITES` contains exactly the de-duplicated in-corpus canonical citation pairs; no external placeholder documents exist.
6. `EVALUATED_ON` equals the specified closure and carries the complete sorted supporting assertion-ID union across all same-document witnesses for each logical method/dataset pair.
7. A forced mid-transaction failure leaves the previous graph snapshot unchanged.
8. Two replacement ingestions from identical inputs produce equal canonical graph hashes and byte-identical ingestion manifests.
9. Removing an input entity/assertion and re-running replacement removes its graph projection, proving stale records cannot survive.
10. Cross-store checks confirm every graph `chunk_id` equals its canonical chunk-artifact ID byte-for-byte; #10's vector-store validation independently enforces the same equality for pgvector rows.
11. Static inspection finds no provider SDK, embedding, LLM, APOC, ORM, UUID, timestamp, or incremental-state dependency in ingestion.

## 12. Explicitly out of scope

- Incremental or streaming graph updates.
- Persistent mention/assertion nodes.
- External-corpus placeholder documents.
- Full-text or vector search inside Neo4j.
- APOC, Graph Data Science, triggers, procedures, or a third service.
- Database-generated identity, UUIDs, and ingestion timestamps.
- Resolver, extraction, citation-resolution, retrieval, or benchmark implementation.

---

Once approved and committed, this spec resolves #9. It supersedes the stale `FROM_DOC` and `data/chunks.jsonl` examples in the original ticket body with the already-resolved `HAS_CHUNK` direction and `data/derived/chunks.jsonl` path.
