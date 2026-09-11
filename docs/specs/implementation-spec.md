# Knowledge-Graph RAG: Final Implementation Specification

> **Status:** Map (#1) cleared. All 15 child tickets closed (#2–#16).
> **Source of truth:** the 15 per-ticket specs in `docs/specs/*.md`, the
> canonical artifacts (`data/manifest.jsonl`, `data/derived/chunks.jsonl`,
> `ontology.yaml`, `eval/questions.jsonl`, `prompts/grounded_answer_v1.txt`),
> and the resolved map body on issue #1.
> **This document:** synthesizes the contracts above into one buildable spec.
> Where this document conflicts with a per-ticket spec, **the per-ticket spec
> wins** and the conflict is recorded in §15 (Cross-Spec Consistency Audit).
> **Out of scope:** `README.md` (user-owned) and implementation tickets
> (those go to `/to-tickets` after `/to-spec` lands).

## 0. Map of stages

```
                                    ┌─────────────────┐
data/raw/latex/*.tar.gz  ───────►   │   ingest.py     │
data/raw/openalex/*.json ──────►    │   (parse_latex  │  ──► data/manifest.jsonl
data/manifest.jsonl  (input)        │    + oa_import) │     (canonical corpus)
                                    └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   chunker.py    │  ──► data/derived/chunks.jsonl
                                    │   (strategy #6, │     (canonical chunks, gitignored)
                                    │    record #2)   │
                                    └─────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
    ┌──────────────────┐         ┌──────────────────────┐         ┌──────────────────┐
    │   extract.py     │         │  entity_resolve.py   │         │  build_graph.py  │
    │   (contract #7,  │ ──────► │  (policy #8)         │ ──────► │  (ingest #9)     │
    │    provider #16) │ extract │  ─► entities.jsonl   │ resolve │  ─► Neo4j        │
    └──────────────────┘         │     entity_resolu-   │         │     (graph store)│
              │                  │     tions.jsonl     │         └──────────────────┘
              ▼                  └──────────────────────┘                   │
    ┌──────────────────┐                                                   ▼
    │ build_vector.py  │         ┌─────────────────────────────────────────────────────┐
    │ (#10)            │         │             retrieval (serving-time)                │
    │ ─► pgvector      │         │  vector (#10)  graph (#11)  hybrid (#12)            │
    │   (vector store) │         │  VectorRetriever  GraphRetriever  HybridRetriever  │
    └──────────────────┘         └─────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                                  ┌─────────────────────┐
                                                  │  grounded_gen.py    │
                                                  │  (#13, prompt v1,   │
                                                  │   provider #16)     │  ──► GenerationResult
                                                  └─────────────────────┘
                                                               │
                                                               ▼
                                                  ┌─────────────────────┐
                                                  │  benchmark.py       │
                                                  │  (#14)              │  ──► data/runs/<run_id>/
                                                  │  on 45 held-out     │     report.json
                                                  └─────────────────────┘
                                                  ┌─────────────────────┐
                                                  │  api.py             │
                                                  │  (#15, POST /ask)   │
                                                  └─────────────────────┘
```

Five canonical artifacts (§2) and three live services (§3). Everything else
is derived.

## 1. Standing constraints (carry across every stage)

Source: map body on issue #1, "Standing preferences for this effort".

| # | Constraint | Where it lives |
|---|---|---|
| 1 | **No LangChain.** Reuse stdlib and existing deps (`httpx`, `python-dotenv`, plus the additions pinned in §10). | every ticket |
| 2 | **`data/manifest.jsonl` defines the corpus.** Every later stage reads from it; no stage defines the corpus independently. | ingest, chunking, extraction, graph, vector, retrieval, generation, benchmark, serving |
| 3 | **`data/raw/` is read-only.** Never mutate arXiv sources or raw OpenAlex JSON in place. | ingest |
| 4 | **Provider-agnostic.** Application code depends on `LLMProvider` / `EmbeddingProvider` Protocols (§7). Provider SDKs live in adapter modules only. Switching providers is env-var + adapter, not code change. | extraction, generation, vector |
| 5 | **Cache and budget are provider-aware.** Cache keys include provider and model identity (§7.5). Cost accounting tolerates `cost_usd=None`. | extraction, generation, embedding |
| 6 | **Graph and vector stores share stable IDs.** `chunk_id` and `doc_id` flow through both stores byte-for-byte (chunk-record §7). | graph ingest, vector ingest |
| 7 | **No tuning on held-out.** All 45 questions are held out. Retrieval parameters (model revision, k values, hop budget, hybrid policy constants, lexical proxy `n`, etc.) are finalized before benchmark numbers are recorded. Dev tuning lives on `eval/dev_questions.jsonl`, strictly out-of-band. | benchmark, retrieval, generation |
| 8 | **Smallest valid implementation.** Prefer the minimum code that can validly test the research question. | every ticket |
| 9 | **`README.md` is user-owned.** No ticket touches it. | every ticket |
| 10 | **No author-entity type.** Ontology v1 excludes `Author` and the other junk-drawer types. | ontology, extraction, graph |

## 2. Canonical artifacts

| Path | Format | Producer | Reader(s) |
|---|---|---|---|
| `data/manifest.jsonl` | JSONL, sorted by `openalex_id` | ingest | every later stage |
| `data/derived/chunks.jsonl` | JSONL, sorted by `chunk_id` | chunker (#6, schema #2) | extraction, graph, vector, retrieval, generation, benchmark |
| `ontology.yaml` | YAML | curated (#5) | extraction (#7), graph ingest (#9), prompt templates (auto-rendered from `entity_types` and `relation_types` lists) |
| `eval/questions.jsonl` | JSONL, 45 records, `split="held_out"` | post-Wayfinder authoring (#4 schema) | benchmark (#14) only |
| `prompts/grounded_answer_v1.txt` | text, versioned suffix | curated (#13) | grounded generator (#13) |

Five downstream artifacts produced by extraction / entity resolution / graph
ingestion (all gitignored):

- `data/extractions.jsonl` + `data/extractions.manifest.json` (#7)
- `data/entities.jsonl` + `data/entity_resolutions.jsonl` + `data/entity_resolution.manifest.json` (#8)
- `data/graph_ingestion.manifest.json` (#9)
- `data/runs/<run_id>/{records.jsonl, report.json}` (#14, benchmark output)

Two runtime stores, populated deterministically from canonical artifacts:

- Neo4j graph (local Docker, §3)
- pgvector table `chunks` (local Docker, §3)

## 3. Local infrastructure

Source: `docs/specs/local-infra.md`.

```
docker compose up    # neo4j + pgvector
```

Two services only:

| Service | Image | Named volumes | Env vars |
|---|---|---|---|
| neo4j | `neo4j:5.26-community-ubi10` | `neo4j_data`, `neo4j_logs` | `NEO4J_PASSWORD` |
| postgres+pgvector | `pgvector/pgvector:0.8.6-pg17-trixie` | `pgvector_data` | `PGVECTOR_USER`, `PGVECTOR_PASSWORD`, `PGVECTOR_DB` |

Single bridge network `rag-net`. Credentials live in `.env`, never in
`docker-compose.yml`. New Python dependencies added by #3: `neo4j`,
`psycopg[binary]`.

**Implementing-ticket scope:** an `infra/up.sh` (or equivalent) that
verifies both containers are healthy, neo4j Bolt is reachable on
`localhost:7687`, and pgvector Postgres on `localhost:5432`.

## 4. Ingest stage — corpus preflight + parser regression

**Note.** `data/manifest.jsonl` is a **curated, committed canonical
artifact** (per the map; it lives at repo root and is tracked by Git).
It is NOT rebuilt from scratch by the implementing ticket. The
implementing ticket repurposes the ingest stage into a **corpus
preflight + parser-regression harness** that:

1. re-validates the committed `data/manifest.jsonl` against the
   available `data/raw/latex/` and `data/raw/openalex/` artifacts
   (every committed `openalex_id` and `arxiv_id` has a backing raw
   artifact; every `referenced_works_canonical` either resolves
   through `data/aliases.json` or is a raw OpenAlex ID);
2. re-runs `parse_latex.py` against every tarball and verifies that
   the parser still emits `[[CITE:...]]` markers and the structural
   anchors downstream stages depend on (preserves existing user work
   per AGENTS.md);
3. ships a small **parser-regression fixture**: a frozen subset of
   tarballs with pinned parser-output SHA256s so a future LaTeX
   parser change is detected at preflight time, before any downstream
   stage reads the manifest.

`data/manifest.jsonl` itself is **not modified** by this ticket. The
ticket does NOT emit a new `data/manifest.jsonl`; it validates the
existing committed one and produces a preflight verdict (pass/fail +
SHA256 report on stderr and as a sidecar JSON next to the manifest's
own metadata).

`data/raw/` is never mutated; the parser only reads from it.

## 5. Chunking stage

Source: `docs/specs/chunking-strategy.md` + `docs/specs/chunk-record.md`.

**Input:** `data/manifest.jsonl`, `data/raw/latex/<id>.tar.gz` (via
`parse_latex.py`).
**Output:** `data/derived/chunks.jsonl`.

- Granularity: `\subsection` default; cascading fallbacks (whole-section,
  subsubsection-equivalent, paragraph subdivision of overlong subsections);
  pre-section content is `chunk_index=0` with `section_path=[]`.
- Section regex pinned: `\\(section|subsection|subsubsection)\*?\s*\{([^}]*)\}`.
- Citation-span rule: chunks **never** split inside `[[CITE:...]]`.
- `chunk_id = "c_" + sha1(f"{doc_id}|{json.dumps(section_path)}|{chunk_index}")[:16]`.
- Records sorted by `chunk_id` ascending, written atomically (tempfile +
  rename).
- Idempotency gate: two consecutive full-corpus runs produce byte-identical
  `data/derived/chunks.jsonl`; SHA256 of the file is the gate.

Record shape (every field required):

```json
{
  "chunk_id": "c_a1b2c3d4e5f60718",
  "doc_id": "W3015883388",
  "section_path": ["Introduction", "Background"],
  "chunk_index": 2,
  "text": "Stochastic gradient descent...",
  "token_estimate": 96,
  "char_start": 918,
  "char_end": 1302
}
```

`doc_id` is exactly the `openalex_id` from `data/manifest.jsonl` (regex
`^W[0-9]+$`).

**Token budget:** every chunk's `token_estimate` must fit the embedding
model's 512-token input limit (vector-retrieval §1.3). The chunker is the
right place to enforce this; the vector loader refuses to embed oversized
chunks if any slip through.

## 6. Extraction stage

Source: `docs/specs/extraction-contract.md`.

**Input:** `data/derived/chunks.jsonl`, `ontology.yaml`, `LLMProvider`
adapter (§7).
**Output:** `data/extractions.jsonl` + `data/extractions.manifest.json`.

Two-phase pipeline:

1. **Entity pass.** For each chunk, call `LLMProvider.generate_json` with a
   schema derived from `ontology.yaml`'s `entity_types` list of `extracted`
   types — `method`, `dataset`, `metric`, `task` — and produce `mention`
   records.
2. **Relation pass.** For each chunk, call `generate_json` with a schema
   derived from `relation_types` where `population_mode == "extracted"` and
   produce `assertion` records. Endpoints are typed `{kind, id}`:
   - Structural `document` endpoints: `kind="document"`, `id=<current chunk.doc_id>`
     (pipeline-injected, byte-for-byte, never trusted from model output).
   - Extracted endpoints: `kind="mention"`, `id=<validated same-chunk mention_id>`.

Content-addressed IDs assigned by pipeline code after validation:

```
m_<16 hex> = sha256(canonical_json({
    chunk_id, source_text_sha256, entity_type, char_start, char_end
}))[:16]
a_<16 hex> = sha256(canonical_json({
    chunk_id, source_text_sha256, relation_type, subject_ref, object_ref,
    char_start, char_end
}))[:16]
```

`canonical_json` uses `sort_keys=True`, `separators=(",",":")`, UTF-8.
IDs are stable across clean full runs and resumed runs; processing order
cannot change them.

Three gates per record (schema, ontology conformity, evidence integrity).
Failed records go to `data/extractions.rejects.jsonl` and bump the phase's
failure counter in `chunk_status`. The pipeline never crashes on a single
record failure.

`chunk_status.extraction_complete == true` is the per-chunk gate. The
Neo4j ingester (#9) consumes only those.

Cache key is a strict superset of #16 §5.1 (§7.5 below): additionally mixes
schema hash, generation parameters, `chunk_id`, `source_text_sha256`,
ontology `version` + SHA256, prompt-template filename + file content hash.
Invalidation is automatic on any input change.

Budget: `LLM_BUDGET_USD` and `LLM_BUDGET_TOKENS` (#16 §6). Clean abort on
cap; partial run is left on disk and resumable.

Retry: bounded exponential backoff (3 attempts, 2 s / 4 s) on transient
transport only; no retry on schema failures (the single validation retry is
owned by `generate_json` per #16 §2.2–2.3).

Resumability: on startup, load existing `data/extractions.jsonl` into an
in-memory map keyed by `(chunk_id, kind)`. A chunk with
`chunk_status.extraction_complete == true` is skipped; an incomplete chunk
is re-run. The implementing ticket ships a re-derivation harness that
recomputes every `mention_id` and `assertion_id` from its record fields
and asserts equality — a tripwire against any future ID-rule regression.

Manifest fields (`data/extractions.manifest.json`):

```json
{
  "run_id": "YYYYMMDDTHHMMSSZ-<run_fingerprint>",
  "resumed_from_run_id": "<prior run_id or null>",
  "provider": "anthropic",
  "model": "claude-haiku-4-5",
  "ontology_version": 1,
  "ontology_sha256": "<64 hex>",
  "prompt_template_files": ["prompts/extract_entities_v1.txt", "prompts/extract_relations_v1.txt"],
  "prompt_template_sha256": ["<64 hex>", "<64 hex>"],
  "schema_version": 1,
  "chunks_path": "data/derived/chunks.jsonl",
  "chunks_sha256": "<64 hex>",
  "wall_clock_started_at": "<ISO 8601>",
  "wall_clock_ended_at": "<ISO 8601>",
  "cumulative_tokens": 0,
  "cumulative_cost_usd": 0.0
}
```

The two prompt templates are rendered at run time from `ontology.yaml`'s
per-type allowed-name lists and few-shot examples. An ontology edit is
the only change needed to keep prompts honest; per-type
`extraction_prompt` field references the two grouped template names
(`extract_entities_v1`, `extract_relations_v1`).

## 7. Provider abstractions

Source: `docs/specs/provider-abstractions.md`.

Two Protocol interfaces, not unified. Application code never imports a
provider SDK.

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None   # None when provider does not expose cost

@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str, *,
                 max_tokens: int = 1024,
                 temperature: float = 0.0) -> tuple[str, Usage]: ...

    def generate_json(self, system: str, user: str, schema: dict, *,
                      max_tokens: int = 1024,
                      temperature: float = 0.0) -> tuple[dict, Usage]: ...

@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]: ...
```

`LLMProvider.generate_json` validates parsed output against the supplied
schema; on failure, retries exactly once with a "respond with valid JSON
matching this schema" suffix; on second failure, raises
`LLMValidationError(model_output, schema, validation_error)`. No silent
fallback.

### 7.1 Configuration

Required env vars: `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_PROVIDER`,
`EMBEDDING_MODEL`.

Optional with defaults:

| Var | Type | Default | Description |
|---|---|---|---|
| `LLM_BUDGET_USD` | float | unset | Soft cap on cumulative LLM cost; honored only when provider returns cost |
| `LLM_BUDGET_TOKENS` | int | unset | Soft cap on cumulative input+output tokens |
| `LLM_CACHE_DIR` | path | `data/derived/llm_cache/` | Cache root |
| `LLM_PROVIDER_TIMEOUT_S` | float | 60 | Per-call HTTP timeout |
| `EMBEDDING_BATCH_SIZE` | int | 64 | Max texts per `embed()` call |

Provider-specific secrets (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`OLLAMA_BASE_URL`, etc.) live in `.env` under clearly-commented per-provider
sections. Mismatched keys are not a load-time error; they only fail when
the adapter is invoked.

### 7.2 Adapter registry

```python
LLM_REGISTRY: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicLLM,
    "openai":    OpenAILLM,
    "ollama":    OllamaLLM,
}

EMBEDDING_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "local_sentence_transformers": SentenceTransformersEmbedding,
    "openai":                      OpenAIEmbedding,
    "ollama":                      OllamaEmbedding,
}

def make_llm() -> LLMProvider: ...
def make_embedding() -> EmbeddingProvider: ...
```

Adding a new provider = write adapter class + add one registry line +
add env vars to `.env.example`. Business logic does not learn that a new
provider exists.

### 7.3 `validate_provider_config()`

Runs at the entry point of every script that uses an LLM. Checks:
`LLM_PROVIDER` is a registered adapter; adapter's required secrets are
present; `LLM_MODEL` is non-empty. Does **not** ping the model (the
first call's problem; surfaces a clear connection error).

### 7.4 Budget caps

The wrapper around `LLMProvider` records cumulative `Usage` per call;
after each call checks `cumulative_cost_usd >= LLM_BUDGET_USD` (if cap
set and provider returns cost) or `cumulative_tokens >= LLM_BUDGET_TOKENS`
(if cap set). On cap, raises `BudgetExceeded(cumulative, cap, unit)`.
Calling script catches it, logs the partial run, exits non-zero.
`cost_usd=None` skips the USD check; the token cap still applies.

### 7.5 Cache (canonical JSON + SHA256, not delimiter-concatenated)

Cache identity is built by canonical JSON + SHA256 so that `operation`,
`cache_format_version`, `max_tokens`, `temperature`, and (for
`generate_json`) the schema all participate in the key. Construction:

```python
def _cache_key(*, cache_format_version, operation, provider_name,
               provider_model, system, user, max_tokens, temperature,
               schema_canonical_json=None, texts=None) -> str:
    parts = [cache_format_version, operation, provider_name, provider_model,
             system, user,
             json.dumps(max_tokens, sort_keys=True, separators=(",", ":")),
             json.dumps(temperature, sort_keys=True, separators=(",", ":"))]
    if schema_canonical_json is not None:
        parts.append(schema_canonical_json)
    if texts is not None:
        parts.append(json.dumps(texts, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False))
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

Required identity inputs by operation:

| Operation | Identity inputs (in addition to `cache_format_version`) |
|---|---|
| `generate` | `operation="generate"`, `provider.name`, `provider.model`, exact `system`, exact `user`, `max_tokens`, `temperature` |
| `generate_json` | same as `generate` + canonical schema hash |
| `embed` | `operation="embed"`, `provider.name`, `provider.model`, ordered exact texts (full list preserved) |

Cache files at `${LLM_CACHE_DIR}/{llm,embeddings}/<key>.json`. Provider or
model change → new namespace; prompt change → new key; adapter-code change
→ bump `cache_format_version`, old entries orphan but never silently
returned for the new format.

Callers may add extra identity dimensions (e.g. `#7` adds schema hash,
generation parameters, chunk identity, ontology SHA, prompt-template
SHA) but cannot weaken the canonical identity.

### 7.6 Default provider

The spec pins no default. The architecture-default is whichever provider
`LLM_PROVIDER` names at first-run time. The implementing ticket ships
one working adapter for each of `LLMProvider` and `EmbeddingProvider`,
plus the registry wiring. Anthropic is a reasonable first LLM pick; a
local `sentence-transformers` model is a reasonable first embedding pick
(zero cost, fully reproducible).

Smoke test (§16 acceptance gate): `generate` pong, `generate_json` with a
schema returns the dict, `embed` returns 2 vectors of length `dim` with
non-zero values.

## 8. Entity resolution stage

Source: `docs/specs/entity-resolution.md`.

**Input:** `data/derived/chunks.jsonl`, `data/extractions.jsonl`,
`data/extractions.manifest.json`, `ontology.yaml`.
**Output:** `data/entities.jsonl`, `data/entity_resolutions.jsonl`,
`data/entity_resolution.manifest.json`.

The resolver **never** edits `data/extractions.jsonl`. Neo4j ingestion
(#9) leaves assertion `document` refs as `doc_id` and resolves only
assertion `mention` refs through `data/entity_resolutions.jsonl` to
canonical `e_*` IDs.

### 8.1 Two-stage deterministic policy

**Stage A — exact-normalized groups over `canonical_form`.** Each
extraction record's `canonical_form` field is the Stage A input —
**not** `surface_form`. The pinned normalisation chain, applied in
this exact order to every `canonical_form`:

1. Unicode-normalise to **NFKC**.
2. **Casefold.**
3. Replace Unicode category `Pd` (Dash_Punctuation) characters with
   ASCII `-`. This handles en-dash, em-dash, figure-dash, horizontal
   bar, etc.
4. **Whitespace**: collapse runs of whitespace to a single space;
   trim leading/trailing whitespace.
5. **Wrapper-quote removal (whitelist only).** Strip a leading/trailing
   matching pair only if both ends are one of `"`, `'`, `“`,
   `”`, `‘`, `’`. **No general punctuation stripping**: do not strip
   parentheses, brackets, commas, periods, slashes, colons, or any
   other non-whitelist delimiter.

Then group eligible mentions by `(entity_type, normalize(canonical_form))`.
Each group is a Stage A cluster. Sort `mention_id` values inside each
cluster lexicographically; cluster key for ordering is its smallest
`mention_id`.

**Stage B — complete-link fuzzy clusters within each type partition.**
Distance metric: deterministic **normalised Levenshtein** on the Stage A
normalized strings. **Threshold: `>= 0.92`** (similarity form, frozen;
clusters merge when similarity **at or above** the threshold).
Ordering and tie-breaks are deterministic (smallest `mention_id`).
Hard entity-type separation: `method` and `dataset` cannot land in
the same Stage B cluster even with high similarity.

**No CLI / env threshold override.** The threshold is a frozen constant
in code. Held-out tuning of the threshold is forbidden (eval-schema §6).

### 8.2 Canonical name + ID

For each final cluster, pick the canonical name by deterministic
frequency + tie-break (smallest `mention_id`). `entity_id` is the
semantic-name-addressed hash:

```
entity_id = "e_" + sha256(canonical_json({
    entity_type, canonical_name
}))[:16]
```

`canonical_json` uses `sort_keys=True`, `separators=(",",":")`, UTF-8.

**Note:** `entity_id` is NOT stable across different eligible input or
policy identity. Changed inputs may select a different canonical name
and therefore a different `entity_id`. Byte-identical IDs are guaranteed
only for identical eligible input + identical policy. The manifest's
extraction/input hashes, algorithm version, normalisation version, and
threshold identify the exact generation under which the IDs are
reproducible.

### 8.3 Outputs

`data/entities.jsonl`: one canonical entity per final cluster, sorted by
`(entity_type, entity_id)`. **Required fields per cluster** (full
schema from entity-resolution spec §5):
`entity_id`, `entity_type`, `canonical_name`, `mention_ids` (sorted,
unique), `aliases` (sorted, unique, deduplicated against
`canonical_name`), `resolution_confidence` (float, e.g. `1.0` for
exact-normalized clusters; mean similarity for fuzzy clusters;
deterministic per cluster).

`data/entity_resolutions.jsonl`: full eligible `mention_id → entity_id`
mapping, sorted by `mention_id`. Each record includes `resolution_method`
(`exact_normalized` or `fuzzy`) and `resolution_score` (the Stage A
group size for `exact_normalized`; the complete-link similarity that
caused the merge for `fuzzy`, or `1.0` for singletons).

`data/entity_resolution.manifest.json`: input identity (paths + SHA256
of chunks, extractions, ontology), policy identity (algorithm version,
normalisation version, threshold), counts, output SHA256s.

### 8.4 Hard preflight

Before any output is written, the resolver checks:

1. `data/extractions.manifest.json` exists and describes a successful run.
2. The manifest's `chunks_sha256` equals the current `data/derived/chunks.jsonl`.
3. Every required `chunk_id` in `chunks.jsonl` has exactly one schema-valid
   `chunk_status` record with `extraction_complete: true` in
   `data/extractions.jsonl`.
4. No extraction record refers to an unknown or stale `chunk_id`; all
   extraction input identities are internally consistent.

Any preflight failure aborts the run before any output is replaced.

## 9. Graph ingestion stage

Source: `docs/specs/neo4j-ingestion.md`.

**Input:** `data/manifest.jsonl`, `data/derived/chunks.jsonl`,
`data/entities.jsonl`, `data/entity_resolutions.jsonl`,
`data/entity_resolution.manifest.json`, `ontology.yaml`,
`data/aliases.json` (when present), `data/title_overrides.json` (when
present), `data/fulltext_manifest.jsonl` (when present).
**Output:** populated Neo4j graph, `data/graph_ingestion.manifest.json`.

### 9.1 Schema

| Node label | Required properties | Source |
|---|---|---|
| `Document` | `doc_id` (`^W[0-9]+$`, unique), `arxiv_id`, `title` | `data/manifest.jsonl` |
| `Chunk` | `chunk_id` (`^c_[0-9a-f]{16}$`, unique), `doc_id`, `section_path`, `chunk_index` | `data/derived/chunks.jsonl` |
| `Entity` | `entity_id` (`^e_[0-9a-f]{16}$`, unique), `entity_type`, `canonical_name` | `data/entities.jsonl` |

| Edge | From | To | Source |
|---|---|---|---|
| `HAS_CHUNK` | `Document` | `Chunk` | structural |
| `MENTIONS` | `Chunk` | `Entity` | per-mention from extractions |
| `CITES` | `Document` | `Document` | structural from OpenAlex `referenced_works` |
| `PROPOSES`, `INTRODUCES`, `EVALUATES_ON`, `USES_METHOD`, `REPORTS_METRIC`, `EXTENDS`, `USED_FOR_TASK`, `MEASURES` | typed | typed | assertions from #7 |
| `EVALUATED_ON` | `Entity:method` | `Entity:dataset` | derived closure of `PROPOSES × EVALUATES_ON`, never LLM |

No `Mention` / `Assertion` / run nodes. No chunk-text duplication. No
Neo4j-internal provenance IDs.

### 9.2 Procedure

1. Schema bootstrap (separate, idempotent): create uniqueness
   constraints on `(:Document {doc_id})`, `(:Chunk {chunk_id})`,
   `(:Entity {entity_id})`.
2. Complete preflight: every `chunk_id` and `entity_id` referenced by
   edges exists in the canonical artifact; `chunk_id` membership in
   `chunks.jsonl` matches; `mention_id → entity_id` mapping exists for
   every assertion `kind: "mention"` ref.
3. Transactional graph replacement: delete prior graph content and
   rebuild from canonical artifacts in a single transaction. No partial
   graphs left on failure. On rollback, the graph is restored to its
   pre-run state (whatever was there before the failed transaction),
   not necessarily empty.

`data/graph_ingestion.manifest.json` records input paths + SHA256
(`manifest_sha256`, `aliases_sha256`, `ontology_sha256`, `chunks_sha256`,
`extractions_sha256`, `extractions_manifest_sha256`, `entities_sha256`,
`entity_resolutions_sha256`, `entity_resolution_manifest_sha256`,
`pre-alias citation_source_sha256`), output `graph_content_sha256`,
policy identity, counts (nodes/edges per label/type), and wall-clock.

## 10. Vector ingestion stage

Source: `docs/specs/vector-retrieval.md`.

**Input:** `data/derived/chunks.jsonl`, `EmbeddingProvider` adapter (§7).
**Output:** populated pgvector table `chunks`, plus run metadata.

### 10.1 Embedding model pin

| Property | Value |
|---|---|
| Hugging Face id | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | `384` |
| Max input length | `512` tokens |
| Pooling | mean (model default) |
| Normalization | L2-normalize before returning from adapter |
| Query instruction | `"Represent this sentence for searching relevant passages: "` — **queries only** |
| Passage instruction | none — chunk text embedded verbatim |

Adapter registered under `EmbeddingProvider` Protocol. The query-vs-passage
asymmetry is pinned: any future change updates both fields and re-embeds
the corpus.

### 10.2 No ANN

Exact pgvector cosine search, no ANN index. Corpus is small (182 papers
× tens of chunks ≈ 10⁴ chunks × 384 floats ≈ 14 MB). ANN never appears
unless it demonstrates both a recall@k target vs exact AND a meaningful
latency improvement on a representative chunk sample that does not
touch the 45 held-out benchmark questions.

### 10.3 Store contract — pgvector table `chunks`

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | `text` PRIMARY KEY | `^c_[0-9a-f]{16}$` per chunk-record spec |
| `embedding` | `vector(384)` | L2-normalized |
| `embedding_model` | `text` | `"<provider>:<model>:<revision>"` |
| `source_text_sha256` | `char(64)` | `sha256(chunk.text)` at embedding time |
| `embedded_at` | `timestamptz` | default `now()` |

### 10.4 Stale-embedding detection (four conditions)

A row is stale and re-embedded if any of:

1. `chunk_id` missing from `data/derived/chunks.jsonl` (orphan-row check).
2. Row's `embedding_model` ≠ currently-configured `<provider>:<model>:<revision>`.
3. Row's `source_text_sha256` ≠ `sha256(canonical_chunk.text)`.
4. Canonical artifact has a chunk with no row in pgvector.

The loader reports the count and proceeds to re-embed; never silently
drops or skips.

### 10.5 No silent truncation

The loader refuses to embed and surfaces offending `chunk_id`s + token
counts if any chunk exceeds 512 tokens. The chunking ticket (#6) is
responsible for picking size thresholds that guarantee this property
at chunking time; the loader is the last-line enforcement.

### 10.6 Run metadata (audit only)

Per-run, alongside logs:
- SHA256 of `data/derived/chunks.jsonl` at run time.
- Pinned model revision.
- Timestamp.

The per-row `source_text_sha256` is the authoritative staleness signal;
the run-level SHA is reconstruction aid, not a correctness gate.

## 11. Retrieval stages (the three conditions)

Three retrievers, all composition-only at the orchestrator level
(serving, §14). Each is a standalone implementation following its
per-ticket spec.

### 11.1 Vector-only retriever (#10)

```python
@dataclass(frozen=True)
class VectorRetrievalHit:
    chunk_id: str
    score: float

class VectorRetriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[VectorRetrievalHit]: ...
```

- Score: `1 − pgvector_cosine_distance` (with L2-normalized vectors,
  this is inner-product similarity).
- Order: score desc; tie-break by `chunk_id` asc.
- No similarity threshold.
- Empty store → `[]`. `k > rows` → all rows.
- `k <= 0` → `ValueError(k=...)`. Non-int `k` → `TypeError`.
- Query prep: prepend BGE query instruction before embedding. Passage
  prep: no instruction (caller passes chunk text verbatim).

### 11.2 Graph-only retriever (#11)

```python
@dataclass(frozen=True)
class GraphEvidenceStep:
    step: int                            # 1-based ordinal within the path
    from_id: str                         # typed id (doc_id | entity_id | chunk_id)
    relation: str                        # ontology relation name
    to_id: str                           # typed id (doc_id | entity_id | chunk_id)
    direction: Literal["forward", "reverse"]

@dataclass(frozen=True)
class GraphRetrievalHit:
    chunk_id: str
    score: float
    evidence_path: tuple[GraphEvidenceStep, ...]   # connected, directional
```

`GraphEvidenceStep` is the canonical #11 type. Its JSON serialisation
on the wire is the verbatim object form
`{"step": <int>, "from": <str>, "relation": <str>, "to": <str>,
"direction": <str>}` (i.e. `from_id` is named `from`, `to_id` is
named `to` in the JSON projection; this matches `graph-retrieval.md`
§2 exactly).

**Seed lifting (deterministic):** for each seed type the retriever
loads a frozen seed catalogue at startup:
- **Entity** — every entity in `data/entities.jsonl`, with
  `(entity_id, canonical_name, aliases)`. Matching is exact-normalized
  maximal-span: lower-case the query and the seed string (casefold,
  after NFKC), then find the longest contiguous match with
  Unicode-alphanumeric boundaries (`\w` with `re.UNICODE`).
- **Document** — every `Document.title` in `data/manifest.jsonl`,
  with `(doc_id, title)`. Same matching rule.

Ambiguous matches are retained (no LLM disambiguation). The two
catalogues are searched independently and the union of matched seeds
is the input to traversal.

**Traversal:** forward and reverse over ontology relationships only.
Globally frozen `hop_budget = 3`. Plumbing edges (`HAS_CHUNK`, `MENTIONS`)
do not count as benchmark hops.

**Scoring:** `1.0 / semantic_hop_count` per chunk. Best path per chunk
wins; tie-break canonical-path identity, then `chunk_id`. **No** provider,
fuzzy/vector fallback, learned ranker, or held-out tuning.

**Pure-CITES structural context:** when the only path to a chunk is
through `CITES` (no extracted/derived edges), the terminal-Document
chunk may be returned explicitly as structural context, with
`evidence_path` carrying the path verbatim; #13 lexical status
remains authoritative on support, never labeled textual proof.

### 11.3 Hybrid/routed retriever (#12)

Composes `VectorRetriever` and `GraphRetriever` **unchanged**. The
hybrid invokes **both retrievers for every valid query** with the
globally frozen constants:

- `HYBRID_K_VEC_MULTIPLIER = 2`
- `HYBRID_K_GRAPH_MULTIPLIER = 2`
- `HYBRID_HOP_BUDGET = 3`
- `RRF_K = 60`

No query-side router, no derived-seed back-channel from vector into
graph, no graph-tail call. If #11 seed-lifting finds no graph seed,
`graph_candidates == []` and fusion reduces naturally to the vector
candidate list.

```python
@dataclass(frozen=True)
class HybridRetrievalHit:
    chunk_id: str
    score: float                      # the RRF fused score
    sources: tuple[str, ...]          # subset of {"vector", "graph"}
    hybrid_hit_provenance: HybridHitProvenance

@dataclass(frozen=True)
class HybridHitProvenance:
    in_vector: bool
    in_graph: bool
    vector_rank: int | None
    vector_score: float | None
    graph_rank: int | None
    graph_score: float | None
    graph_evidence_path: tuple[GraphEvidenceStep, ...] | None  # JSON key 'graph_evidence_path' (per serving-surface §6.4); tuple element type is GraphEvidenceStep from §11.2.
```

**Fusion:** Reciprocal Rank Fusion (Cormack 2009) with frozen `RRF_K=60`.
Per chunk: `score = sum(1 / (RRF_K + rank_i))` over retrievers that
returned the chunk (`score` on `HybridRetrievalHit` is the fused score;
there is no separate `rrf_score` field — the fusion value lives in
`score`). `in_vector` / `in_graph` on `HybridHitProvenance` are set
from the per-retriever contribution. Tie-break three-key ordering:
1. `score` desc,
2. `len(sources)` desc (dual-source beats single-source),
3. `chunk_id` asc by UTF-8 byte order.

Graph-only and vector-only receive no modality preference beyond (2).

**No provider/embedding in the hot path.** A disabled v2 fallback
(LLM route-classifier) is recorded in #12 §8 but not enabled in v1.

**Tuning forbidden on held-out.** Every frozen parameter
(`RRF_K`, `HYBRID_K_VEC_MULTIPLIER`, `HYBRID_K_GRAPH_MULTIPLIER`,
`HYBRID_HOP_BUDGET`, the three-key tie-break) is finalised before
benchmark numbers are recorded. Tuning is permitted only on
`eval/dev_questions.jsonl` strictly out-of-band.

## 12. Grounded generation stage

Source: `docs/specs/grounded-generation.md`.

**Input:** a `GenerationRequest` carrying `query` + an ordered
`tuple[EvidenceBlock, ...]`. **Output:** `GenerationResult` (parsed
successfully) or `GenerationParseError` (after one retry). Calls
`LLMProvider.generate` only — never `generate_json`; never reads the
held-out benchmark; never imports a provider SDK.

### 12.1 Type shapes (canonical)

```python
@dataclass(frozen=True)
class EvidenceBlock:
    chunk_id: str        # ^c_[0-9a-f]{16}$
    doc_id: str          # ^W[0-9]+$
    text: str            # verbatim chunk text

@dataclass(frozen=True)
class GenerationRequest:
    query: str
    evidence: tuple[EvidenceBlock, ...]   # caller order preserved exactly
    max_tokens: int = 1024
    temperature: float = 0.0

@dataclass(frozen=True)
class CitationOccurrence:
    chunk_id: str        # ^c_[0-9a-f]{16}$, validated
    claim_id: int
    char_start: int
    char_end: int
    status: Literal[
        "IN_EVIDENCE_SUPPORTING",
        "IN_EVIDENCE_NON_SUPPORTING",
        "OUT_OF_EVIDENCE",
    ]

@dataclass(frozen=True)
class Claim:
    claim_id: int
    text: str
    char_start: int
    char_end: int
    scorable: bool
    attached_chunk_ids: tuple[str, ...]

@dataclass(frozen=True)
class ClaimSupportVerdict:
    claim_id: int
    verdict: Literal["SUPPORTED", "CITED_UNSUPPORTED", "UNCITED"]
    supporting_chunk_ids: tuple[str, ...]
    non_supporting_chunk_ids: tuple[str, ...]
    out_of_evidence_chunk_ids: tuple[str, ...]

@dataclass(frozen=True)
class GenerationResult:
    answer_text: str
    abstained: bool
    citations: tuple[CitationOccurrence, ...]
    claims: tuple[Claim, ...]
    claim_support: tuple[ClaimSupportVerdict, ...]
    usage_calls: tuple[Usage, ...]
    evidence_fingerprint: str
```

No `parsed_ok` / `parse_error` / `cache_hit` / benchmark-metric fields
on `GenerationResult`. The generator never sees retrieval score,
`vector`/`graph`/`hybrid` source label, graph entities, evidence_path,
RRF provenance, or per-retriever ranks; those live in the orchestrator's
`retrieval_trace`.

### 12.2 Prompt artifact

Pinned, versioned: `prompts/grounded_answer_v1.txt`. The generator
reads the file at call time (or memoises). Audit metadata
`prompt_filename`, `prompt_content_sha256` flow into the run/generation
trace and (transitively, via their appearance in the rendered user
prompt) the #16 cache key. The rendered user prompt encodes the
query and the evidence blocks in exact caller order. It does **not**
inject `temperature` or `max_tokens` into the prompt body merely to
vary the cache key — those parameters are passed to
`LLMProvider.generate` and participate in the canonical-JSON cache key
construction in §7.5. Bump the suffix on any non-empty change to the
template text; the SHA changes automatically.

Pinned grounding requirements (also in the prompt file):
- Answer using only the supplied evidence blocks.
- No outside knowledge.
- Every factual claim cites one or more supplied `chunk_id`s.
- Citation tokens BEFORE sentence-ending punctuation.
- Insufficient evidence → output exactly `ABSTAIN` and nothing else.
- No headings. No `References` section. No meta-commentary.

### 12.3 Citation syntax

The only acceptable inline token: `[[c_<16 lowercase hex>]]`. Regex:
`\[\[(c_[0-9a-f]{16})\]\]`. Each token is independent; adjacency is
co-citation. No prefix/suffix/locator/comma-list/whitespace inside the
brackets.

### 12.4 ABSTAIN sentinel

Whole answer is the literal `ABSTAIN` and nothing else. When parsed:
`answer_text == "ABSTAIN"`, `abstained == True`, `citations == ()`,
`claims == ()`, `claim_support == ()`. Any other text containing the
substring `ABSTAIN` is treated as a normal answer.

### 12.5 Zero-evidence behaviour

If `GenerationRequest.evidence` is empty, the generator does **not**
call `LLMProvider`. It returns the deterministic sentinel directly
(`answer_text == "ABSTAIN"`, `abstained == True`, all collections empty,
`usage_calls == ()`, `evidence_fingerprint` from the empty canonical
array). This is a grounding precondition, not a model abstention.

### 12.6 Evidence preflight (input integrity)

Before any provider call (and before §12.5), the implementing ticket
MUST validate every `EvidenceBlock`:
- `query` is `str`.
- `evidence` preserves caller order (no reorder/dedup).
- Every `chunk_id` matches `^c_[0-9a-f]{16}$`.
- Every `doc_id` matches `^W[0-9]+$`.
- `chunk_id` values within one `GenerationRequest` are unique.
- Each `EvidenceBlock(chunk_id, doc_id, text)` matches the current
  canonical `data/derived/chunks.jsonl` record byte-for-byte for
  `(chunk_id, doc_id, text)`.

Any stale/duplicate/unknown/mismatched `EvidenceBlock` raises
`EvidenceIntegrityError`. The generator is not called.

### 12.7 Parsing pipeline

For a non-`ABSTAIN` answer:
1. Segment into claims on `". "`, `"? "`, `"! "`, `"\n\n"`. Empty
   segments dropped. Original order preserved. `char_span` recorded.
2. Extract every `[[c_<16hex>]]` token as a `CitationOccurrence` with
   `char_span`.
3. Attach each citation to exactly one claim by character-span overlap.
   Tie-break by earliest claim containing the citation's leftmost char.
4. Validate each citation's `chunk_id` against the supplied evidence
   set (§12.9).
5. Compute per-claim support (§12.10).

### 12.8 Retry policy

Exactly **one** retry, triggered only by parse/protocol failure:
- malformed `[[…]]` syntax,
- `answer_text` from which no claim can be segmented,
- valid citation token that attaches to no claim.

**Not** triggered by `OUT_OF_EVIDENCE`, lexical-support failures,
uncited claims, `CITED_UNSUPPORTED`, `ABSTAIN`, low
`claim_support_coverage`.

The retry uses one pinned format-repair instruction (single suffix,
implemented by the implementing ticket). Same generation parameters.
The two provider calls produce two independent cache entries (the user
prompt differs by the format-repair suffix) and two `Usage` records,
both on the final `GenerationResult.usage_calls`. Persistent parse
failure after one retry raises `GenerationParseError`.

### 12.9 Per-citation validation statuses

| Status | Condition |
|---|---|
| `IN_EVIDENCE_SUPPORTING` | regex matches AND `chunk_id` is in supplied evidence set AND lexical-support proxy (§12.10) passes for that chunk against the claim |
| `IN_EVIDENCE_NON_SUPPORTING` | regex matches AND `chunk_id` in supplied evidence set AND lexical-support proxy fails |
| `OUT_OF_EVIDENCE` | regex matches AND `chunk_id` not in supplied evidence set |

Malformed bracket syntax is a parse/protocol failure (not a valid
citation); surfaces as parser error, triggers retry. `OUT_OF_EVIDENCE`
is a model outcome, not a parse failure.

### 12.10 Lexical support proxy (frozen constant)

For both claim text and cited chunk text, in this order:
1. Remove direct citation tokens matching `\[\[c_[0-9a-f]{16}\]\]` from
   the claim text first.
2. Unicode-normalize to NFKC.
3. Casefold.
4. Tokenize into maximal contiguous runs of Unicode-alphanumeric chars
   (`\w` with `re.UNICODE`).
5. Retain numeric tokens and tokens containing digits exactly as spelled.

Distinct examples that must remain token-distinguishable: `GPT-4` →
`["GPT","4"]`; `Recall@10` → `["Recall","10"]`; `512 tokens` →
`["512","tokens"]`; `2020` → `["2020"]`.

Containment: `n = min(4, claim_token_count)`. Enumerate every contiguous
n-gram of length `n`; the chunk supports the claim iff at least one
such n-gram occurs contiguously as a sub-list of the chunk's tokens.

If the normalized claim token list is empty: `Claim.scorable = False`;
claim excluded from `claim_support_coverage`'s denominator; no
`SUPPORTED`/`CITED_UNSUPPORTED` verdict; receives `UNCITED` only if no
citation token; if it has citations, those are still parsed and
recorded against `CITED_UNSUPPORTED`.

`n = 4` is a **conservative a-priori v1 constant**, frozen before any
benchmark number is recorded. Any change requires a spec version bump
+ ontology-style prompt-template bump. The 45 held-out questions cannot
be used to tune `n`, normalization, or tokenization.

### 12.11 Per-claim verdicts (with de-duplication)

Dedup `(claim_id, chunk_id)` pairs within one claim for support
evaluation.

| Verdict | Rule |
|---|---|
| `SUPPORTED` | at least one attached citation is `IN_EVIDENCE_SUPPORTING` |
| `CITED_UNSUPPORTED` | has attached citations, none `IN_EVIDENCE_SUPPORTING` |
| `UNCITED` | contains no citation token |

A `Claim` with empty normalized tokens + no citations is `UNCITED`. If
empty + has citations, the citations are still parsed and recorded;
the claim is `CITED_UNSUPPORTED`.

### 12.12 Evidence fingerprinting (audit metadata, NOT mixed into cache key)

```python
def evidence_fingerprint(evidence: tuple[EvidenceBlock, ...]) -> str:
    payload = [
        {"chunk_id": e.chunk_id, "doc_id": e.doc_id,
         "source_text_sha256": hashlib.sha256(e.text.encode("utf-8")).hexdigest()}
        for e in evidence
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

`source_text_sha256` is required because `chunk_id` is content-independent
(derives from `(doc_id, section_path, chunk_index)`, not text). The
fingerprint is **not** mixed into the #16 cache key by this spec — the
underlying `user` prompt already encodes the evidence verbatim, so any
text/order change already changes the #16 key.

### 12.13 Diagnostics on the result (NOT quality targets)

```
claim_support_coverage       = supported_claim_count / scorable_claim_count
cited_unsupported_claim_rate = cited_unsupported_claim_count / total_claim_count
uncited_claim_rate           = uncited_claim_count / total_claim_count
evidence_utilization         = |unique cited in-evidence chunk_ids| /
                               |supplied evidence chunk_ids|
```

Zero-denominator rules: `claim_support_coverage` uses
`scorable_claim_count` in the denominator (`None` if zero);
`evidence_utilization` uses `|supplied evidence chunk_ids|` (`None` if
zero). #14 owns benchmark metrics; this spec does not pick headline
numbers.

### 12.14 Budget accounting

`GenerationResult.usage_calls: tuple[Usage, ...]` records every actual
provider call in call order. On the `GenerationParseError` path, both
calls' `Usage` are written to the run/generation trace before the
exception is raised. `LLM_BUDGET_USD` and `LLM_BUDGET_TOKENS` are
enforced by the #16 wrapper; the generator never bypasses them. A
retry that pushes cumulative counters over cap raises `BudgetExceeded`;
the partial `Usage` records already observed remain on the result and
run trace.

## 13. Serving stage

Source: `docs/specs/serving-surface.md`.

Exactly one application-defined route. Entry point:
`uvicorn src.api:app --host 127.0.0.1 --port 8000`.

### 13.1 Request

```
POST /ask
Content-Type: application/json

{"question": "<string>", "mode": "vector"|"graph"|"hybrid"|null}
```

Validation:
- `question` required and non-empty after `strip()`.
- `mode` optional; absent/null ⇒ `effective_mode = "hybrid"`.
- `extra="forbid"`.
- `Content-Type != application/json` ⇒ `415 unsupported_media_type`.
- Invalid request body ⇒ `422 invalid_request`.

The server owns `k`, hop budget, RRF constants, model, temperature,
prompts, and thresholds. The route does **not** accept any
request-side override of these.

### 13.2 Frozen serving-time constants

| Constant | Value |
|---|---|
| `SERVING_K` | `5` (every mode) |
| Graph `hop_budget` | `3` |
| `RRF_K`, `HYBRID_K_VEC_MULTIPLIER`, `HYBRID_K_GRAPH_MULTIPLIER`, `HYBRID_HOP_BUDGET` | per #12 (§11.3) |

### 13.3 Composition (serves-only, no router)

The route calls each retriever exactly once:
- `vector_retriever.retrieve(question, k=5)`
- `graph_retriever.retrieve(question, k=5, hop_budget=3)`
- `hybrid_retriever.retrieve(question, k=5)`

For `mode == "vector"`, only `vector_retriever` is invoked. Likewise for
`mode == "graph"` and `mode == "hybrid"`. No serving-time router. No
reproduction of #12 hybrid internals (RRF, dedup, ordering, tie-break,
per-source reconstruction).

### 13.4 Generation seam

Ordered hits → `EvidenceBlock` against `data/derived/chunks.jsonl` →
two immutable maps (`chunk_id → EvidenceBlock`, `chunk_id →
retrieval_hit`) → `GroundedGenerator.generate(...)` once per #13.
No serving retry / repair / rewrite / rescoring. Empty evidence ⇒
deterministic `ABSTAIN` with no `LLMProvider` call.

Citations are occurrence-based in answer order. For `IN_EVIDENCE_*`,
`doc_id` is the exact `EvidenceBlock.doc_id` supplied to #13 and
provenance is the matching retrieval hit's mode-tagged object. For
`OUT_OF_EVIDENCE`, `doc_id = null`, `provenance = null`.

**Strict provenance-join invariant:** an `IN_EVIDENCE_*` citation missing
from either map raises `ServingIntegrityError` → `500 internal_error`.
No silent fallback.

Provenance shape (direct JSON projection of retriever-native hit
structure, never re-walked):
- vector: `{mode, rank, score}`
- graph: `{mode, rank, score, graph_evidence_path}` (the
  `graph_evidence_path` field is the verbatim JSON form of #11's
  `evidence_path`; serving-surface §6.4 pins the wire-level key name)
- hybrid: `{mode, rank, score, sources, hybrid_hit_provenance}` where
  `score` is the RRF fused score (verbatim from #12)

No cross-mode score normalization.

### 13.5 ABSTAIN response

`ABSTAIN` ⇒ `HTTP 200`, `answer = "ABSTAIN"`, `abstained = true`,
`citations = []`. Retrieval miss is **not** 404.

### 13.6 Error envelope

Stable non-sensitive shape:
```json
{ "error": { "code": "<string>", "message": "<string>" }, "request_id": "<uuid4>" }
```

Pinned codes:

| Code | HTTP |
|---|---|
| `unsupported_media_type` | 415 |
| `invalid_request` | 422 |
| `evidence_integrity_error` | 503 |
| `retrieval_unavailable` | 503 |
| `generator_unavailable` | 503 |
| `generation_protocol_error` | 502 |
| `internal_error` | 500 |

Never exposes raw question, request body, prompt, model output,
provider error body, chunk text, path, stack trace, or credential.

### 13.7 `request_id`

Opaque UUID4 minted at request ingress **before** validation. Returned
in every response. Logged exactly once. No semantic effect on #16 cache,
retrieval, generation, or #14 benchmark.

### 13.8 Logging

Exactly one structured JSON log per completed request, fields:
`ts`, `event="http_request_completed"`, `request_id`, `mode`,
`status in {ok, abstained, error}`, `http_status`, `latency_ms`,
`retrieval_hit_count`, `citation_count`, `error_code` (when `>= 400`).
Pinned null-vs-zero stage semantics. Never logs raw question,
normalized question, any question hash/HMAC, answer text, evidence
text, prompts, raw provider output, credentials.

### 13.9 Startup preflight

`validate_provider_config()` (§7.3) + required local artifacts +
vector-store connectivity/freshness (#10) + neo4j (#9) + graph content
identity (#11/#9) + adapter wiring. **No synthetic live-LLM call**
unless #16 requires one. All three advertised modes must be healthy at
startup or startup fails non-zero. **No degraded startup, no silent
mode disablement, no `graph→vector` or `hybrid→vector` fallback.**

### 13.10 Frozen v1 exclusions

No UI, websockets, streaming/SSE, auth, rate-limit, accounts, sessions,
persistence, feedback, any request-side model/k/hop/prompt/temperature/
threshold selector, `/benchmark`, `/eval`, `/admin`, `/health`,
`/metrics`, business `/docs`, PII classifier, queues/workers/background
jobs, second service, second production entrypoint, serving-time
router, citation repair, answer rewriting, `README.md`, LangChain.

The serving surface does **not** read or expose benchmark fields
(`qid`, `run_id`, `run_fingerprint`, `expected_answer`, `answer_aliases`,
`hop_bucket`, `gold_evidence_path`, `supporting_chunk_ids`,
`supporting_doc_ids`). The benchmark runner does not depend on serving
either.

## 14. Benchmark stage

Source: `docs/specs/benchmark-runner.md`.

**Input:** `eval/questions.jsonl` (45 records, all `split="held_out"`),
`data/derived/chunks.jsonl`, `ontology.yaml`, the current Neo4j graph,
the current pgvector store, the three retrievers, the grounded
generator, the #16 provider layer.
**Output:** `data/runs/<run_id>/{records.jsonl, report.json}` (gitignored).

### 14.1 Pinned constants

- `GENERATION_K = 5`
- `RETRIEVAL_EVAL_K_VALUES = (5, 20)`
- Graph invocation: `graph_retriever.retrieve(query, k, hop_budget=3)`
  for both `k` values.
- `retrieve(query, 20)[:5] != retrieve(query, 5)` is **not** assumed;
  the runner uses the ordered hits actually returned.

### 14.2 Experimental unit

One record per `(qid, mode)`, containing both `k=5` and `k=20`
retrieval evaluations plus exactly one generation from `k=5`. The
experimental unit is `(qid, mode)`.

### 14.3 Three retrieval conditions

- `vector` — `VectorRetriever` (#10) only.
- `graph` — `GraphRetriever` (#11) only.
- `hybrid` — `HybridRetriever` (#12) only.

Generation is invoked once per `(qid, mode)` against the k=5 retrieval
hits; the generator's input contract (§12.1) is identical across all
three.

### 14.4 Retrieval metrics (required core family)

- `retrieval_recall@k = |top_k ∩ supporting_chunk_ids| / |supporting_chunk_ids|`.
  Undefined when `supporting_chunk_ids == []` (the runner never allows
  that in the shipped file).
- `hit_rate@k = 1` if `top_k ∩ supporting_chunk_ids != ∅`, else `0`.
- `MRR@k = 1 / r` where `r` is the smallest 1-based rank ≤ k whose
  `chunk_id ∈ supporting_chunk_ids`; `0` if no relevant in top k. When
  multiple gold chunks appear, the first relevant rank wins.
- `retrieval_precision@k` is an optional secondary diagnostic; never
  called "citation precision".

### 14.5 Gold-citation metrics (using `C_in`)

- `C_all` = unique syntactically valid `cited_chunk_ids` in the
  generated answer (after #13 parsing; `ABSTAIN` has `C_all = ∅`;
  `GenerationParseError` makes these metrics undefined).
- `C_in` = unique cited chunk_ids whose #13 status is **not**
  `OUT_OF_EVIDENCE`.
- `G` = `supporting_chunk_ids`.

So an `OUT_OF_EVIDENCE` citation cannot receive gold credit, still
penalizes `gold_citation_precision` through `C_all`, and never
increases `gold_citation_recall`.

### 14.6 Deterministic v1 answer-content metrics

- `answer_hit` (from #4 substring/alias rule; always full denominator
  including abstentions and parse errors as zero).
- `normalized_f1` (always-enabled secondary metric, never headline):
  NFKC + casefold + Unicode-alphanumeric tokenization after removing
  `[[c_<16 hex>]]` citation tokens; max F1 over
  `{expected_answer} ∪ answer_aliases`.

No LLM judge, no embedding judge, no learned NLI in v1.

### 14.7 Grounding metrics (conditional mean + `n_defined`)

Grounding metrics (`claim_support_coverage`,
`cited_unsupported_claim_rate`, `uncited_claim_rate`,
`out_of_evidence_citation_rate`, gold-citation metrics) are reported
as **conditional mean + `n_defined`** over the parseable subset so a
7-of-15 cell does not look equivalent to a 15-of-15 cell.

### 14.8 No second-generation nondeterminism probe

A second-generation nondeterminism probe would be a #16 cache hit and is
not a valid probe. The runner never bypasses the cache.

### 14.9 Headline table

`rows = hop_bucket`, `columns = {vector, graph, hybrid}`, cell = mean
`retrieval_recall@5`. Secondary tables in the same shape for
`hit_rate@5`, `MRR@20`, `answer_hit`, `normalized_f1`,
`gold_citation_precision/recall`, `claim_support_coverage`,
`cited_unsupported_claim_rate`, `uncited_claim_rate`,
`out_of_evidence_citation_rate`, `abstain_rate`, `parse_error_rate`.

Macro mean within bucket is primary. No cross-bucket weighting.

### 14.10 Two distinct identifiers

- `run_fingerprint = first 16 hex of sha256(canonical_json(semantic
  inputs))` — excludes timestamps / cache / output / wall-clock; mixes
  `git_sha`, eval file path+SHA, `chunks_sha256`, `ontology_sha256`,
  `graph_ingestion_manifest_sha256`, `graph_content_sha256`,
  `embedding_source_chunks_sha256`, embedding model+revision, retrieval
  k values, graph hop budget, hybrid policy reference, LLM
  provider/model, temperature, max_tokens,
  `prompt_template_sha256`, `cache_format_version`, and the SHAs of all
  five relevant specs (#10/#11/#12/#13/#16), and the scoring/report
  schema version.
- `run_id = YYYYMMDDTHHMMSSZ-<run_fingerprint>` (filesystem-safe UTC
  compact).

Two executions sharing every semantic input have different `run_id` and
the same `run_fingerprint`.

### 14.11 Three statuses, not interchangeable

| Status | Meaning |
|---|---|
| `refused` | Fatal preflight — empty `supporting_chunk_ids`, dirty benchmark state, stale vector store, missing credentials, missing artifacts. Exits non-zero. No headline metrics. |
| `completed` | All three conditions complete cleanly on 45/45. `GenerationParseError` and `ABSTAIN` may still appear. |
| `invalid` | Any retrieval infrastructure error in any condition invalidates that condition. 44/45 is not directly comparable to a complete condition. Valid conditions publish normally. |

No percentage threshold. No degraded completion.

### 14.12 Preflight gates

- `embedding_source_chunks_sha256 == chunks_sha256` (current chunks file).
- #10 stale-row validation passes.
- Required canonical artifacts exist and conform.
- Eval file validates (45 records, all `split="held_out"`, no empty
  `supporting_chunk_ids`).

The runner never invents a hash over live row order.

### 14.13 Reproducibility claims (split)

- **A**: deterministic retrieval reproducibility.
- **B**: deterministic scoring reproducibility (under pinned
  `scoring_version`).
- **C**: provider-conditioned generation may differ at temperature=0.
- **D**: cached generation replay is a #16 property.

The runner does **not** promise byte-identical `records.jsonl` across
fresh executions.

### 14.14 What the runner never modifies

`data/manifest.jsonl`, `data/derived/chunks.jsonl`, `ontology.yaml`, the
Neo4j graph, the pgvector store, the retrievers, the generator, the
#13 validator, the #16 cache, the eval records, held-out parameters,
or `README.md`. The runner only reads.

## 15. Cross-spec consistency audit

This section records the cross-spec audit performed as part of `/to-spec
#1`. Each row is a class of invariant, the spec(s) it cuts across, and
its resolution. Where this document disagrees with a per-ticket spec,
the per-ticket spec wins.

### 15.1 Canonical ID namespaces (resolved)

| ID | Regex | Producer | Appears in |
|---|---|---|---|
| `doc_id` | `^W[0-9]+$` | ingest (openalex_id) | chunks (#2), extractions (#7), entities→document endpoints (#7, #9), graph (#9), benchmark (#14) |
| `chunk_id` | `^c_[0-9a-f]{16}$` | chunker (#6, derivation in #2) | chunks (#2), extractions (#7), pgvector PK (#10), Neo4j uniqueness constraint (#9), retrieval returns (#10/#11/#12), generation (#13), benchmark (#14) |
| `entity_id` | `^e_[0-9a-f]{16}$` | entity-resolution (#8) | entities (#8), graph (#9), graph retrieval seed nodes (#11) |
| `mention_id` | `m_<16 hex>` | extraction (#7) | extractions (#7), entity_resolutions (#8), graph ingest for assertion `mention` refs (#9) |
| `assertion_id` | `a_<16 hex>` | extraction (#7) | extractions (#7), graph ingest (#9) |

Audit: all 15 specs agree on these shapes. ✅ No repair.

### 15.2 Artifact paths

| Path | Lives at | Gitignored |
|---|---|---|
| `data/manifest.jsonl` | repo root (`data/manifest.jsonl`) | **No** (committed, versioned corpus) |
| `data/derived/chunks.jsonl` | `data/derived/` | Yes (under `data/derived/`) |
| `data/extractions.jsonl` + `data/extractions.manifest.json` | `data/` (per extraction-contract §3) | **No** by current `.gitignore` (only `data/derived/` is excluded) — see §15.7 below |
| `data/entities.jsonl` + `data/entity_resolutions.jsonl` + `data/entity_resolution.manifest.json` | `data/` (per entity-resolution §7) | **No** by current `.gitignore` — see §15.7 below |
| `data/graph_ingestion.manifest.json` | `data/` | **No** by current `.gitignore` — see §15.7 below |
| `data/runs/<run_id>/` | `data/runs/` | **No** by current `.gitignore` — see §15.7 below |
| `ontology.yaml` | repo root | **No** (committed, curated) |
| `eval/questions.jsonl` | repo root (`eval/questions.jsonl`) | **No** (committed, frozen) |
| `prompts/grounded_answer_v1.txt` | repo root (`prompts/`) | **No** (committed, pinned) |

### 15.3 Frozen constants

| Constant | Value | Spec |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` | #10 |
| Embedding dim | `384` | #10 |
| Embedding max tokens | `512` | #10 |
| `SERVING_K` | `5` | #15 |
| Graph `hop_budget` | `3` | #11, #12, #15 |
| `RRF_K` | `60` | #12 |
| `HYBRID_K_VEC_MULTIPLIER` | `2` | #12 |
| `HYBRID_K_GRAPH_MULTIPLIER` | `2` | #12 |
| `HYBRID_HOP_BUDGET` | `3` | #12 |
| Entity resolution threshold | `0.92` (complete-link) | #8 |
| Lexical proxy `n` | `min(4, claim_token_count)` | #13 |
| `GENERATION_K` (benchmark) | `5` | #14 |
| `RETRIEVAL_EVAL_K_VALUES` | `(5, 20)` | #14 |
| Eval set size | 45 (~15 per hop bucket) | #4 |
| Hop buckets | 1-hop, 2-hop, 3+ | #4 |

All values consistent across specs. ✅ No repair.

### 15.4 Per-ticket spec audit trails (already on the map)

These consistency repairs were already committed during per-ticket
resolution. They are recorded here so the implementing tickets see
the final state:

- **#7 ID stability** (commit `8ce7fc9`): replaced global counters with
  content-addressed `m_*` / `a_*` IDs.
- **#7 typed assertion endpoints** (commit `d947b93`): replaced
  impossible all-mention endpoints with `{kind, id}` refs.
- **#12 hybrid simplifications** (commit `e3d8aec`): removed
  vector-led graph tail-call; removed `graph` ahead of `vector` tie
  preference. Left three-key tie-break: `rrf_score` desc →
  `len(sources)` desc → `chunk_id` asc.
- **#16 cache repair** (tagged with #13 commit): §5.1 rebuilt with
  canonical-JSON-plus-SHA256 construction; `operation`,
  `cache_format_version`, `max_tokens`, `temperature`, and `generate_json`
  schemas participate in the cache identity.

### 15.5 `LLMProvider` return-type discipline

`LLMProvider.generate` returns `(text, Usage)`. No `cache_hit` field.
Cache observability is the abstraction layer's responsibility. The
generator's `GenerationResult` does not have `cache_hit` or any
benchmark-metric field.

Audit: #13, #16, grounded-generation §2 and §6.3, provider-abstractions §1. ✅ No repair.

### 15.6 Three retrieval-condition equality

The three retrievers (`vector`, `graph`, `hybrid`) each produce their
own hit type (`VectorRetrievalHit`, `GraphRetrievalHit`,
`HybridRetrievalHit`). They share only `chunk_id` and `score` plus the
respective provenance field. Generation conditions are equal: the
generator sees `tuple[EvidenceBlock, ...]` only and never reads the
mode label or per-retriever rank/score.

Audit: #10, #11, #12, #13 §14, #14. ✅ No repair.

### 15.7 `.gitignore` coverage for `data/`-root artifacts (RESOLVED)

**Finding.** `entity-resolution.md` §7 places `data/entities.jsonl`,
`data/entity_resolutions.jsonl`, `data/entity_resolution.manifest.json`
at the repo root under `data/`, NOT under `data/derived/`. The current
`.gitignore` excludes only `data/raw/` and `data/derived/`. By the same
token, `extraction-contract.md` §3 places `data/extractions.jsonl` and
`data/extractions.manifest.json` under `data/`, and `neo4j-ingestion.md`
§10 places `data/graph_ingestion.manifest.json` under `data/`, and
`benchmark-runner.md` §11 places `data/runs/<run_id>/` under `data/`.

The per-ticket specs are mutually consistent on these paths. The
gitignore convention is the loose end.

**Resolution.** Implement a gitignore update in the implementing ticket
for the entity-resolution stage (`/to-tickets` resolves this). The
update adds explicit ignores for:

```
data/extractions.jsonl
data/extractions.manifest.json
data/extractions.rejects.jsonl
data/entities.jsonl
data/entity_resolutions.jsonl
data/entity_resolution.manifest.json
data/graph_ingestion.manifest.json
data/runs/
```

without disturbing the existing `data/derived/` rule. The
reproducibility story for every one of these is "determinism + the
SHA256 record in the benchmark report" (per the existing per-ticket
specs); none needs to be checked in.

This document records the loose end; the implementing ticket implements
the fix. The gitignore change is not in scope for `/to-spec`.

### 15.8 `data/extractions.manifest.json` field set

**Audit.** Extraction-contract §3 and §13 specify the manifest's fields
as a list (provider, model, ontology_version + SHA, prompt template
versions + SHAs, schema_version, chunks-manifest SHA, wall-clock,
resumed_from_run_id, cumulative_tokens, cumulative_cost_usd). The
benchmark-runner spec at §10 references `extractions_manifest_sha256`
as a hash input from #9's ingestion manifest. The neo4j-ingestion spec
at §10 also references `extractions_manifest_sha256` as one of the
input hashes recorded by the graph ingestion manifest.

No conflict: the same path is named consistently across all three
specs. ✅ No repair.

### 15.9 Manifest field naming: `run_id` vs `run_fingerprint`

Both identifiers appear in:
- `data/extractions.manifest.json` (`resumed_from_run_id` references a
  prior run's `run_id`).
- `data/runs/<run_id>/` directory name (uses the full `run_id` string).
- `report.json` and `records.jsonl` (contain both identifiers).
- `#14` acceptance gate definitions.

Both shapes (`YYYYMMDDTHHMMSSZ-<run_fingerprint>` and the 16-hex
`run_fingerprint`) are pinned by #14 and referenced consistently. ✅
No repair.

### 15.10 Generation seam invariants

The generator never reads `eval/questions.jsonl`, `expected_answer`,
`answer_aliases`, `supporting_chunk_ids`, `hop_bucket`. Gold
comparisons are `#14`'s post-hoc job. The retrieval trace
(`retrieval_score`, mode label, graph entities, evidence_path, RRF
provenance, per-retriever ranks) lives in the orchestrator, not on the
generator input.

Audit: #13 §1, §11, §12.6; #14 §13, §14. ✅ No repair.

### 15.11 Standing-preference boundary

`README.md` is user-owned (per AGENTS.md and the map's "Out of scope"
list). No ticket creates, edits, regenerates, appends, or ticketing
README work. `LangChain` is excluded. No author entity type. No
cloud-only infrastructure. No direct provider-SDK imports in business
logic.

Audit: map body + every per-ticket spec's "What this spec leaves
open" / "Acceptance gate" sections. ✅ No repair.

### 15.12 No unresolved contradictions

Every disagreement found during the audit was either resolved by the
already-merged per-ticket consistency repairs (§15.4) or routed to the
implementing ticket (§15.7). None blocks `/to-spec` from closing.

### 15.13 Post-`/to-tickets` consistency repairs

After `/to-tickets` produced #17–#34, an audit pass found the
following drifts between the implementing tickets and the per-ticket
specs. The repairs below were committed to this document; the matching
ticket-body edits live on the respective GitHub issues.

| Repair | Where it surfaces in this doc | Matching per-ticket spec |
|---|---|---|
| §4: do NOT rebuild the curated `data/manifest.jsonl`; the ingest ticket becomes a corpus preflight + parser-regression harness. | §4 | `ontology.md` §4.3 (canonical artifacts are not regenerated); `AGENTS.md` (user-owned committed files preserved) |
| §8.1: Stage A input is `canonical_form` (not `surface_form`); full normalisation chain NFKC + casefold + `Pd→"-"` + whitespace + wrapper-quote whitelist; threshold `>= 0.92`; no CLI/env override. | §8.1 | `entity-resolution.md` §3, §4 |
| §8.3: full output schemas — `entities.jsonl` adds `aliases` and `resolution_confidence`; `entity_resolutions.jsonl` `resolution_score` rule. | §8.3 | `entity-resolution.md` §5 |
| §9.2: rollback semantics — graph is restored to **pre-run** state, not necessarily empty. | §9.2 | `neo4j-ingestion.md` §8 |
| §11.3: `HybridProvenance` → `HybridHitProvenance`; add `in_vector` / `in_graph` booleans; remove redundant `rrf_score` (the fused score lives in `HybridRetrievalHit.score`); three-key tie-break orders by `score` desc. | §11.3 | `hybrid-retrieval.md` §3 |
| §12.2: remove `temperature` / `max_tokens` from the rendered user prompt text — they participate in the #16 cache key structurally, not by prompt injection. | §12.2 | `provider-abstractions.md` §5.1, §5.5; `grounded-generation.md` §6.2 |
| §13.8: log event name is the literal string `"http_request_completed"`; the log record carries `request_id` as a separate field, not as the `event` value. | §13.8 | `serving-surface.md` §7.2 |

`/to-spec` does not reopen the per-ticket specs. These repairs bring
this synthesis doc and the implementing tickets into alignment with
specs already on `main`.

## 16. Global acceptance gate

The implementation is complete when **every** implementing ticket has
its own per-ticket acceptance gate pass, **and** the following
cross-ticket gates pass:

1. `docker compose up` brings up neo4j and pgvector; both healthchecks
   pass.
2. `python -m src.ingest.parse_latex <arxiv_id>` runs against a sample
   tarball and emits expected LaTeX structure (preserving existing user
   work; AGENTS.md).
3. `python -m src.chunker` produces `data/derived/chunks.jsonl`;
   SHA256 is stable across two consecutive runs.
4. `python -m src.extract` produces `data/extractions.jsonl` and
   `data/extractions.manifest.json`; every chunk has
   `chunk_status.extraction_complete == true`; re-derivation harness
   passes for every `mention_id` and `assertion_id`.
5. `python -m src.entity_resolve` produces
   `data/entities.jsonl`, `data/entity_resolutions.jsonl`,
   `data/entity_resolution.manifest.json`; hard preflight passes.
6. `python -m src.build_graph` populates Neo4j and writes
   `data/graph_ingestion.manifest.json`; preflight passes; transactional
   graph replacement runs cleanly.
7. `python -m src.build_vector` populates pgvector; stale-row check
   passes; oversized-chunk check passes; orphan-row check passes.
8. Provider smoke test (§7.6) passes for both `LLMProvider` and
   `EmbeddingProvider`.
9. `python -m src.grounded_gen` runs against a smoke prompt and emits
   `GenerationResult` matching §12.1 shapes (no extra required fields).
10. `python -m src.benchmark` runs the 45 held-out questions across all
    three conditions; produces `data/runs/<run_id>/{records.jsonl,
    report.json}`; headline table is `rows = hop_bucket × cols =
    {vector, graph, hybrid}` with `retrieval_recall@5` per cell.
11. `uvicorn src.api:app --host 127.0.0.1 --port 8000` + `POST /ask`
    returns 200 on a smoke question across all three modes and 415/422
    on bad inputs.
12. `grep -rE "import (anthropic|openai|ollama)" src/` returns matches
    only under `src/providers/` (or wherever adapters live).
13. `README.md` is byte-identical to its pre-implementation state.
14. The §15.7 gitignore update is in place; no new untracked files
    appear under `data/` (excluding `data/raw/` and `data/derived/`)
    after a clean pipeline run.

Each implementing ticket is responsible for its own gate; this list is
the integration-level gate that proves the parts compose.


