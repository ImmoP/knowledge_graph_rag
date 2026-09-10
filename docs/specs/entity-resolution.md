# Entity Resolution Policy

Resolution of [#8 — Decide the entity resolution policy](https://github.com/ImmoP/knowledge_graph_rag/issues/8). This document is the **decision contract** for canonicalising extracted entity mentions. It fixes the production algorithm, determinism rules, output schemas, and threshold governance; it does not implement the resolver.

This spec consumes the immutable provenance emitted by the [extraction contract](extraction-contract.md). It never edits or replaces `data/extractions.jsonl`. Neo4j ingestion (#9) leaves assertion `document` refs as `doc_id` and resolves only assertion `mention` refs through `data/entity_resolutions.jsonl` to canonical `e_*` IDs.

## 1. Role and artifacts

```text
data/derived/chunks.jsonl ──┐
data/extractions.jsonl    ──┤
                           ├──► ENTITY RESOLUTION ──┬─► data/entities.jsonl
ontology.yaml             ──┘                          ├─► data/entity_resolutions.jsonl
                                                        └─► data/entity_resolution.manifest.json
```

The resolver is deterministic and string-only. It does not call `LLMProvider`, `EmbeddingProvider`, or any provider SDK. It does not build an embedding or similarity index. The `EmbeddingProvider` contract in [#16](provider-abstractions.md) is therefore outside this stage.

The three outputs are derived artifacts:

- `data/entities.jsonl`: one canonical entity per final cluster.
- `data/entity_resolutions.jsonl`: the complete `mention_id` → `entity_id` mapping for eligible mentions.
- `data/entity_resolution.manifest.json`: input identity, policy identity, counts, and output hashes.

All three are regenerated together. `data/extractions.jsonl` remains byte-for-byte untouched.

## 2. Eligible input mentions

The resolver reads `data/derived/chunks.jsonl`, `data/extractions.jsonl`, `data/extractions.manifest.json`, and `ontology.yaml`.

### 2.1 Global extraction preflight

Resolution must not run against a partial extraction. Before constructing an eligible mention set, it validates the complete #7 generation as one global gate:

1. `data/extractions.manifest.json` exists, conforms to the extraction-manifest schema, and describes a successfully completed run.
2. The manifest's ontology version and SHA256 match the current `ontology.yaml`.
3. The manifest's chunk-input path and SHA256 match the current `data/derived/chunks.jsonl`.
4. Every required `chunk_id` in `data/derived/chunks.jsonl` has exactly one schema-valid `chunk_status` record with `extraction_complete: true` in `data/extractions.jsonl`.
5. No extraction record refers to an unknown or stale `chunk_id`, and all extraction input identities required by #7 are internally consistent.

Any missing, incomplete, conflicting, unknown, or stale state aborts resolution before clustering. The failed attempt creates or replaces none of the three #8 output artifacts. If a prior valid resolution generation exists, it is left untouched but does not describe the failed input attempt.

### 2.2 Mention eligibility after preflight

A mention is eligible if and only if all of the following hold:

1. It is a schema-valid `kind: "mention"` record under the extraction contract.
2. Its `chunk_id` has exactly one schema-valid `kind: "chunk_status"` record with `extraction_complete: true`.
3. Its `entity_type` is an entity type whose `population_mode` is `extracted` in the current `ontology.yaml`.
4. Its `mention_id` can be re-derived from its evidence fields under extraction contract §6.1.

The global preflight means there is no partial-resolution mode: incomplete-chunk mentions are never silently ignored while other chunks are resolved. Conflicting duplicate `mention_id` records, conflicting duplicate chunk-status records, or an invalid extraction record fail the run before any output is replaced. Identical duplicate mention lines may be de-duplicated by `mention_id`, but the manifest records their count.

Only eligible extracted mentions can become cluster members or cause an entity to exist. Verified examples in `ontology.yaml` are not production seeds, aliases, or synthetic mentions. They may be copied into a separate human-labelled development fixture for threshold evaluation, but they never enter a production cluster unless extraction independently emitted a validated eligible mention.

Assertions do not participate in clustering. Their typed refs remain immutable extraction provenance: #8 neither resolves nor rewrites `document` refs, and #9 joins only `mention` refs through `data/entity_resolutions.jsonl`.

## 3. Resolution name normalisation

Each eligible mention's resolution input is its required, non-empty `canonical_form`. `surface_form` remains provenance and contributes to the final alias list, but it is not scored. There is no fallback from an invalid or empty `canonical_form`; that condition is an upstream validation defect and fails resolution.

`normalized_resolution_name` is produced in this exact order:

1. Apply Unicode NFKC normalisation.
2. Apply Unicode `casefold()`.
3. Map every Unicode dash-punctuation character (general category `Pd`) to ASCII `-`.
4. Replace each maximal run of Unicode whitespace with one ASCII space.
5. Remove leading and trailing whitespace.
6. Repeatedly remove one matched pair of surrounding wrapper quotation marks from this whitelist, trimming surrounding whitespace after each removal: ASCII double quotes (`"..."`), ASCII single quotes (`'...'`), grave accents (`` `...` ``), curly double quotes (`“...”`), curly single quotes (`‘...’`), guillemets (`«...»`), and single guillemets (`‹...›`). A pair is removed only when it encloses a non-empty value.

No other transformation is allowed. In particular, the resolver does not stem, lemmatise, expand acronyms, transliterate, strip accents, remove stop words, reorder tokens, or generally strip punctuation. Only the explicitly whitelisted matched wrapper quotes are removed. Semantically meaningful internal punctuation and symbols are preserved, including digits, decimal/version separators, `@k`, internal hyphens, slashes, and plus signs. Surrounding parentheses, brackets, hyphens, or other non-whitelisted characters are also preserved because they may be part of a technical name. Thus `"DPR"` and `DPR` become exact-match eligible, while `BERT v1` / `BERT v2`, `Recall@5` / `Recall@10`, and `T5-base` / `T5-large` remain distinct normalised inputs.

An empty normalised result is invalid and fails the run.

## 4. Similarity and pinned threshold

For two non-empty normalised strings `a` and `b`, similarity is normalized Levenshtein similarity:

```text
lev_similarity(a, b) = 1 - levenshtein_distance(a, b) / max(len(a), len(b))
```

Lengths and edit operations are measured over Unicode code points after §3 normalisation. Insertions, deletions, and substitutions each cost 1; transpositions are not a primitive operation. The score lies in `[0, 1]`.

The v1 production threshold is:

```text
NORMALIZED_LEVENSHTEIN_THRESHOLD = 0.92
```

A comparison qualifies when its unrounded score is `>= 0.92`. Implementations compare the exact integer ratio (or an equivalent representation that cannot change the boundary through premature rounding). Scores are rounded to six decimal places only when written to output.

The implementing ticket defines this number once, at module scope in `src/graph/entity_resolution.py`, alongside:

```text
ENTITY_RESOLUTION_ALGORITHM_VERSION = "entity_resolution_v1"
NORMALIZATION_VERSION = "resolution_name_v1"
```

No environment-variable or CLI override is allowed for a benchmark-producing run. This prevents two runs both labelled v1 from silently using different merge policies.

### 4.1 Threshold governance

The threshold is precision-first: a missed merge leaves two inspectable nodes, while a false merge corrupts every assertion routed to the combined node. `0.92` is the frozen initial v1 value and is treated as a conservative a-priori choice unless separate manually labelled resolver-development evidence exists. It is frozen before benchmark question authoring and before any benchmark evaluation.

Threshold evaluation, when performed, uses a separate, manually labelled development artifact such as `eval/entity_resolution_dev.jsonl`, containing positive and negative name-pair judgements. The canonical 45 records in `eval/questions.jsonl` — including their questions, expected answers, aliases, evidence paths, and supporting IDs — must never be used to select, validate, adjust, or justify the threshold.

Changing the threshold requires all of the following:

1. Change the single named constant.
2. Bump `ENTITY_RESOLUTION_ALGORITHM_VERSION`.
3. Record the development artifact path and SHA256 in the decision or implementation change.
4. Regenerate all three resolution outputs.
5. Re-ingest downstream graph artifacts before comparing benchmark runs.

The threshold is not changed or retrospectively defended in response to results on the 45 held-out questions. A run influenced by those questions is invalid under the [evaluation policy](eval-schema.md#6-held-out-policy-no-split--selected-operating-policy).

## 5. Two-stage clustering algorithm

Resolution is partitioned by `entity_type` before any comparison. Names belonging to different entity types can never merge, even when their normalised strings are identical.

### 5.1 Stage A: exact normalised groups

Group all eligible mentions by:

```text
(entity_type, normalized_resolution_name)
```

Each group is a Stage A cluster. Sort the `mention_id` values inside each cluster lexicographically. The cluster key used for ordering is its smallest `mention_id`.

Stage A exact grouping has confidence `1.0`. It operates on all eligible mentions, not just unique display strings or singleton mentions.

### 5.2 Stage B: deterministic greedy complete-link

Process Stage A clusters globally in ascending order of their smallest `mention_id`. Different types remain isolated by the hard partition.

For the current candidate Stage A cluster `C`, compare it with every already-established target cluster `T` of the same `entity_type`. The cross-cluster score is:

```text
cross_score(C, T) = min(
    lev_similarity(normalized_resolution_name(c), normalized_resolution_name(t))
    for c in C
    for t in T
)
```

`T` means its full current membership, including every earlier Stage A cluster already merged into it. Duplicate normalised strings do not change the result and may be cached, but the semantic rule is the all-member formula above.

A target qualifies only when `cross_score(C, T) >= 0.92`. If no target qualifies, `C` becomes a new target cluster. If one or more targets qualify, merge the whole Stage A cluster `C` into exactly one target:

1. Choose the target with the highest `cross_score(C, T)`.
2. If scores tie exactly, choose the target whose smallest `mention_id` is lexicographically smallest.

The candidate is never split. Already-established targets are never merged with one another as a separate operation. This makes the procedure a deterministic greedy pass and prevents single-link transitive chaining: every new member must remain above threshold against every existing member of its target.

O(n²) all-pairs comparison within an entity type is accepted for the v1 corpus. ANN, vector indexes, canopy clustering, blocking heuristics beyond the hard entity-type partition, and other scaling machinery are out of scope.

### 5.3 Resolution confidence

For every accepted Stage B merge, retain the unrounded `cross_score(C, T)` calculated immediately before that merge. A final entity's `resolution_confidence` is:

```text
1.0                                      if it contains one Stage A cluster
min(score for each accepted fuzzy merge) otherwise
```

The stored value is rounded to six decimal places. It therefore exposes the weakest fuzzy link admitted into the final complete-link cluster; scores are never averaged. Because all accepted merges meet the threshold, every output confidence is in `[0.92, 1.0]`.

The same accepted score is also attached to every mention in candidate cluster `C` as its mapping-level `resolution_score`, with `resolution_method: "fuzzy"`. Mentions in a Stage A cluster when it first becomes an established target receive `resolution_method: "exact_normalized"` and `resolution_score: 1.0`; those mapping values remain unchanged if later candidate clusters merge into the target. This records the decision that admitted each mention without rewriting history after subsequent merges.

## 6. Canonical IDs, names, and aliases

### 6.1 Canonical entity ID

After Stage B finishes and §6.3 selects the deterministic `canonical_name`, derive:

```text
entity_id = "e_" + sha256(canonical_entity_payload).hexdigest()[:16]
```

The canonical payload contains exactly:

```json
{
  "canonical_name": "dense passage retrieval",
  "entity_type": "method"
}
```

The payload's `canonical_name` value is the selected display `canonical_name` passed through the exact normalizer in §3. The payload contains only the fixed fields `canonical_name` and `entity_type`. Serialise with `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, and UTF-8, matching the extraction contract's canonical-JSON rules. The ID format is `^e_[0-9a-f]{16}$`. A hash-prefix collision between unequal payloads fails the run; the implementation must not append a nondeterministic suffix.

Resolver algorithm version, normalisation version, threshold, input hashes, and cluster membership do not enter this payload; they belong in `data/entity_resolution.manifest.json`. This fact does not create a cross-input stability guarantee: changed eligible input can change cluster evidence, which can cause a different canonical name to win and therefore produce a different `entity_id`. That is acceptable in v1.

### 6.2 Display normalisation

For canonical names and aliases only, a display value is formed by replacing Unicode whitespace runs with one ASCII space and trimming leading/trailing whitespace. Case, punctuation, digits, and Unicode characters are otherwise preserved. Empty display values are invalid.

### 6.3 Canonical name

Canonical-name candidates are the display-normalised `canonical_form` values of all member mentions. Select exactly one by these keys, in order:

1. Highest frequency under this pinned policy: the number of member mentions carrying that exact display-normalised candidate.
2. If tied, the candidate supported by the lexicographically smallest `mention_id`.
3. If still tied, the lexicographically smallest normalized candidate by Unicode code point, where the normalized candidate is the display value passed through §3.

This uses extraction's canonical suggestion without allowing input order or hash-map iteration order to choose the label.

### 6.4 Aliases

Aliases are the distinct display-normalised values from both `canonical_form` and `surface_form` across the final cluster, excluding the selected `canonical_name`. Sort them by `(normalized_resolution_name(alias), alias)` ascending. Case or typography variants remain auditable aliases when their display values differ.

## 7. Output schemas

All JSONL is UTF-8 with one canonical compact JSON object per line and a trailing newline. Object keys are sorted lexicographically. `entities.jsonl` is sorted by `(entity_type, entity_id)`; `entity_resolutions.jsonl` is sorted by `mention_id`.

### 7.1 `data/entities.jsonl`

```json
{
  "aliases": ["DPR", "dense passage retriever"],
  "canonical_name": "Dense Passage Retrieval",
  "entity_id": "e_4d2f3a1b0c9e8d7f",
  "entity_type": "method",
  "mention_ids": ["m_0123456789abcdef", "m_fedcba9876543210"],
  "resolution_confidence": 0.923077
}
```

Field rules:

- `entity_id`: deterministic ID from §6.1; unique in the file.
- `entity_type`: one extracted ontology entity type; identical for every member mention.
- `canonical_name`: deterministic display label from §6.3.
- `aliases`: deterministic, de-duplicated observed alias list from §6.4; may be empty.
- `mention_ids`: non-empty, unique, lexicographically sorted list containing every and only cluster member.
- `resolution_confidence`: weakest accepted Stage B merge from §5.3.

Every eligible `mention_id` appears in exactly one entity record. No ineligible mention appears.

### 7.2 `data/entity_resolutions.jsonl`

```json
{"entity_id":"e_4d2f3a1b0c9e8d7f","mention_id":"m_0123456789abcdef","resolution_method":"fuzzy","resolution_score":0.923077}
```

There is exactly one record for every eligible mention and exactly these four fields:

- `mention_id`: unique eligible extraction mention ID.
- `entity_id`: refers to exactly one record in `data/entities.jsonl`.
- `resolution_method`: exactly `"exact_normalized"` or `"fuzzy"`.
- `resolution_score`: `1.0` for `exact_normalized`; for `fuzzy`, the accepted unrounded `cross_score(C, T)` that admitted the mention's Stage A cluster, rounded to six decimal places for storage.

This file is the only join #9 uses to translate assertion refs whose `kind` is `mention`. Assertion refs whose `kind` is `document` retain their `doc_id` and bypass #8. Assertions whose mention ref lacks a mapping are rejected by ingestion rather than silently dropped or given a new entity. Records are deterministically sorted by `mention_id`.

### 7.3 `data/entity_resolution.manifest.json`

```json
{
  "algorithm": {
    "algorithm_version": "entity_resolution_v1",
    "normalization_version": "resolution_name_v1",
    "similarity_metric": "normalized_levenshtein",
    "threshold": 0.92
  },
  "counts": {
    "eligible_mentions": 1200,
    "entities": 340,
    "extraction_complete_chunks": 900,
    "identical_duplicate_mentions": 0
  },
  "inputs": {
    "chunks_path": "data/derived/chunks.jsonl",
    "chunks_sha256": "<64 lowercase hex>",
    "extractions_manifest_path": "data/extractions.manifest.json",
    "extractions_manifest_sha256": "<64 lowercase hex>",
    "extractions_path": "data/extractions.jsonl",
    "extractions_sha256": "<64 lowercase hex>",
    "ontology_path": "ontology.yaml",
    "ontology_sha256": "<64 lowercase hex>",
    "ontology_version": 1
  },
  "outputs": {
    "entities_path": "data/entities.jsonl",
    "entities_records": 340,
    "entities_sha256": "<64 lowercase hex>",
    "entity_resolutions_path": "data/entity_resolutions.jsonl",
    "entity_resolutions_records": 1200,
    "entity_resolutions_sha256": "<64 lowercase hex>"
  },
  "schema_version": 1
}
```

The actual manifest contains only the fields shown above. It is UTF-8 JSON written with sorted keys, two-space indentation, LF line endings, and one trailing newline. It contains no wall-clock timestamp, host name, or run UUID, so identical inputs and policy produce byte-identical outputs and manifest. Each JSONL is written to a temporary sibling and atomically replaced only after validation; the manifest is replaced last as the generation's commit marker. Consumers reject a generation when either output hash disagrees with the manifest, so a crash cannot make a mixed generation appear complete.

## 8. Determinism contract

The same input set means all of the following are identical:

- The eligible #7 mention set produced after the global preflight.
- SHA256 of `data/extractions.jsonl`.
- SHA256 of `data/extractions.manifest.json`.
- SHA256 and `version` of `ontology.yaml`.
- `ENTITY_RESOLUTION_ALGORITHM_VERSION`.
- `NORMALIZATION_VERSION`.
- normalized-Levenshtein definition and threshold.

For the same input set, the eligible mention set, Stage A groups, Stage B merge decisions, entity memberships, canonical names, aliases, IDs, confidence values, line ordering, and all three output bytes must be identical. Filesystem enumeration order, process hash seed, locale, and concurrency cannot affect the result. A logically equivalent extraction file with its lines reordered has a different input SHA256: it must produce byte-identical `entities.jsonl` and `entity_resolutions.jsonl`, while its manifest differs only where it records the changed extraction SHA256.

There is no entity-ID stability guarantee across different eligible extraction inputs or policy identities. If an extraction input changes, the deterministic frequency/tie-break calculation can select a different `canonical_name`; because normalized canonical name is an ID input, the resulting `entity_id` may change. The manifest's extraction/input hashes, algorithm version, normalisation version, similarity metric, and threshold identify the exact generation under which the IDs are reproducible.

## 9. Failure and boundary rules

- Resolution never mutates extraction records, rewrites typed assertion refs, resolves document refs, or replaces mention-ref `m_*` values with `e_*` values in provenance.
- Resolution never merges across `entity_type`.
- Resolution never invents a mention, alias, or entity from ontology examples.
- Resolution never uses assertions, graph topology, document co-occurrence, an LLM, embeddings, or held-out questions as merge evidence.
- Resolution does not decide Neo4j labels, constraints, or relationship ingestion; those belong to #9.
- Resolution provides no persistent identity across changed inputs or policies. It does not introduce an entity registry, UUIDs, database-backed identity state, redirects, or any other cross-run identity infrastructure.

## 10. Implementing-ticket acceptance gate

The later implementing ticket closes only when all of the following pass:

1. The global extraction preflight in §2.1 passes before clustering; every failure mode writes or replaces none of the three #8 outputs.
2. The three outputs conform to §7 and are written without changing the SHA256 of `data/extractions.jsonl`.
3. Every eligible mention appears exactly once in both the entity membership lists and the resolution mapping; no ineligible mention appears.
4. Every mapping carries the correct `resolution_method` and the exact accepted score rounded only for storage.
5. Reversing and randomly shuffling input-line order produces byte-identical JSONL outputs; the manifest changes only in the extraction input SHA256.
6. A clean run and a resumed/repeated run over the same inputs produce byte-identical outputs.
7. Fixtures cover exact normalisation, wrapper-quote removal, preserved non-whitelisted/internal punctuation, case/whitespace/dash variants, preserved digits/versions/`@k`, hard entity-type separation, Stage A clusters with multiple mentions, complete-link rejection of a single-link bridge, multi-target score selection, and smallest-`mention_id` tie-breaking.
8. A fixture verifies that `resolution_confidence` equals the weakest accepted fuzzy merge and is not an average.
9. Every `e_*` ID is re-derived from `entity_type` and normalized canonical name; a changed-input fixture is allowed to produce a changed ID when its deterministic canonical-name winner changes.
10. Hash mismatches, conflicting duplicate records, missing endpoint mappings, and simulated ID collisions fail loudly.
11. Static inspection confirms no LLM, embedding provider, provider SDK, ANN, or vector-index dependency in the resolver.
12. Threshold-development records are separate from the canonical 45 held-out questions, and their SHA256 is recorded when used to justify a future policy change.

---

This spec resolves ticket #8. It hands #9 a deterministic `mention_id` → `entity_id` join for assertion mention refs while document refs remain `doc_id`, preserving all #7 extraction records as immutable provenance.
