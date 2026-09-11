# Serving Surface Specification

Resolution of [#15 — Decide the serving surface](https://github.com/ImmoP/knowledge_graph_rag/issues/15). This document is the **HTTP contract** that `POST /ask` must satisfy. It is the contract, not the implementing code; that ticket is produced by `/to-tickets` once this spec lands.

The serving surface is the thinnest possible layer between an HTTP client and the existing retrieval and generation contracts. Every behavior it exposes is decided by upstream specs (`#10`, `#11`, `#12`, `#13`, `#16`), not invented here. The route does not reimplement retrieval, fusion, ranking, citation parsing, or provider logic; it validates the request, delegates to an application-service module that orchestrates exactly the upstream contracts, and translates a typed result or typed error into an HTTP response.

## 1. Standing constraints

These are the architectural properties the serving surface inherits from upstream specs, the parent map, and the parent ticket. Each is named so the implementing ticket can audit against it.

1. **Single public endpoint.** Exactly one application-defined route, `POST /ask`. No `/v1/` or any other URL version prefix. No `/health`, `/ready`, `/livez`, `/metrics`, `/admin`, `/debug`, `/benchmark`, `/eval`, or business `/docs` route in v1. FastAPI's auto-generated OpenAPI routes may exist as framework behavior but are not part of the supported application contract.
2. **Stateless.** Each `POST /ask` is independent. No session, account, cookie, conversation memory, persistent question/answer storage, or benchmark/eval persistence. Identical requests produce independent results; the server keeps no client state between requests.
3. **Thin route, thin wrapper.** `src/api.py` owns the FastAPI app, the route wiring, the boundary request/response models, request-id minting, latency timing, structured-log emission, and typed-error mapping. A separate application-service module owns all orchestration. The route never contains Cypher, SQL, RRF math, prompt construction, provider-SDK calls, citation parsing, chunk projection, retrieval implementation, or benchmark scoring.
4. **No provider credentials in the request or response.** The endpoint reads the same `.env` as the rest of the system per #16. Provider selection is configuration-driven (`LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`). No credential-bearing headers, no provider-specific request fields, no provider name in responses.
5. **Frozen retrieval constants at the boundary.** `SERVING_K = 5` for every mode; `hop_budget = 3` for graph retrieval. Retrieval k, graph hop budget, `RRF_K`, generation model, temperature, prompts, and scoring thresholds are server-owned. The request cannot override any of them.
6. **Question normalization at the boundary.** `effective_question = question.strip()`. Only leading/trailing Unicode whitespace is removed. Retrieval and generation receive `effective_question`. The raw input is never logged, never persisted, never hashed.
7. **No serving-time router.** There is no LLM, classifier, heuristic, embedding-similarity check, or learned router in the request path. `mode` is either explicit in the request or omitted/null, in which case `effective_mode = "hybrid"` unconditionally.
8. **Hybrid is one call, not two.** `mode = "hybrid"` invokes `HybridRetriever.retrieve(...)` exactly once. The serving layer does not call `VectorRetriever` and `GraphRetriever` separately for a hybrid request, does not invoke RRF, does not multiply k, does not deduplicate, does not inspect candidate lists, and does not reconstruct hybrid provenance.
9. **Generation seam is owned by `#13`.** Ordered retrieval hits → canonical `EvidenceBlock(chunk_id, doc_id, text)` → `GroundedGenerator.generate(...)` exactly once. The serving layer adds no retry, no citation repair, no answer rewriting, no second support check, and no post-hoc scoring. Empty evidence ⇒ deterministic `ABSTAIN` with no `LLMProvider` call. The single `#13` retry on parse/protocol failure belongs to `#13`, not the serving surface.
10. **Citations are occurrence-based and verbatim.** The response emits one `Citation` per `#13` `CitationOccurrence` in answer order. Status vocabulary is copied verbatim from `#13`. Provenance is a direct JSON projection of the retriever's native hit structure; it is never reconstructed from a fresh retriever call and never normalized across modes.
11. **`request_id` is opaque and operational.** One UUID4 per HTTP request, generated at ingress **before** validation, returned in every response, and logged exactly once. It has no semantic effect on cache keys, retrieval, generation, or benchmarking.
12. **Content-free logging.** The single structured request log record contains no raw question, no normalized question, no question-derived hash, no answer text, no evidence text, no prompts, no provider request/response bodies, no credentials, no chunk text, no client IP, and no User-Agent. `request_id` is the only normal v1 request-correlation identifier.
13. **No README change. No implementation code in this spec.**

## 2. Endpoint and entrypoint

### 2.1 The route

```
POST /ask
Content-Type: application/json
```

One application-defined route. One purpose: accept a question and a mode, run the chosen retrieval + grounded generation pipeline once, return the typed result. No other behavior.

### 2.2 Entrypoint

```
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

- Loopback default — local-only v1 serving. Same `.env` as the rest of the system.
- One production entrypoint. No second ASGI target. No middleware chain beyond FastAPI defaults.
- `src.api:app` is the only exported ASGI application object.

### 2.3 Module boundary

```
src/api.py
  - FastAPI app object
  - POST /ask route handler
  - request_id generation (at ingress, before validation)
  - request-body boundary Pydantic model (extra="forbid", Content-Type check)
  - structured-log emission (one record per completed request)
  - latency timing
  - delegates to ask_service.ask(effective_question, effective_mode) -> AskResult | AskError
  - maps AskResult / AskError to the §5 success body or the §6 error envelope
  - knows nothing about Cypher, SQL, RRF, prompts, providers, parsers, scorers, or chunk projection

src/<service_module>.py
  - ask(effective_question, effective_mode) -> AskResult | AskError
  - resolves effective_mode into one of:
      VectorRetriever.retrieve(effective_question, k=SERVING_K)              [mode = vector ]
      GraphRetriever.retrieve( effective_question, k=SERVING_K, hop_budget=3) [mode = graph  ]
      HybridRetriever.retrieve( effective_question, k=SERVING_K)             [mode = hybrid ]
  - resolves retrieval hits into EvidenceBlock objects (#13) using data/derived/chunks.jsonl (#2)
  - holds two immutable maps for the request: chunk_id -> EvidenceBlock, chunk_id -> retrieval hit
  - calls GroundedGenerator.generate(...) once (the allowed #13 retry is owned by #13)
  - constructs AskResult / AskError from GenerationResult and any typed failure
  - no FastAPI, no Request, no Response, no HTTP type imports
```

The route never imports a retrieval, generator, provider, store, prompt, parser, RRF, scorer, or ontology symbol. The service never imports a FastAPI, `Request`, `Response`, or HTTP type.

## 3. Request contract

### 3.1 Schema

```
POST /ask
Content-Type: application/json
```

Body:

```json
{
  "question": "<string>",
  "mode":     "vector" | "graph" | "hybrid" | null
}
```

### 3.2 Field rules

- `question`: required, JSON string. After stripping leading/trailing Unicode whitespace, the result must be non-empty. Retrieval and generation receive `effective_question`, not the raw value.
- `mode`: optional. Allowed values are exactly `"vector"`, `"graph"`, `"hybrid"`, and `null`. The field may be omitted entirely.
- `extra="forbid"` — any additional top-level field is a validation error.

No other fields exist in v1: no `k`, no `hop_budget`, no `model`, no `temperature`, no `prompts`, no `thresholds`, no `top_n`, no `include_text`, no `request_id` echo, no metadata, no auth headers.

### 3.3 Effective mode

```
effective_mode =
    "hybrid"            if mode is absent
    "hybrid"            if mode == null
    mode                if mode in {"vector", "graph", "hybrid"}
```

Any other explicit value (string, number, list, object) is a `422 invalid_request` error.

### 3.4 Normalization

```text
effective_question = question.strip()
if effective_question == "":
    validation fails (422 invalid_request)
```

Only leading/trailing Unicode whitespace is stripped. No case folding, no punctuation rewriting, no inner-whitespace collapsing, no spelling correction. The raw input is never logged or persisted.

### 3.5 Content-Type

A request body whose `Content-Type` is not `application/json` returns `415 unsupported_media_type`. The route handler enforces this explicitly so the contract is not contingent on a framework default.

## 4. Pipeline dispatch — frozen retrieval constants

The route dispatches via a fixed map. `effective_mode` is always one of `"vector"`, `"graph"`, `"hybrid"` by the time dispatch runs — invalid requests are rejected in §3.

```python
# vector
vector_retriever.retrieve(
    effective_question,
    k=SERVING_K,
)

# graph
graph_retriever.retrieve(
    effective_question,
    k=SERVING_K,
    hop_budget=3,
)

# hybrid
hybrid_retriever.retrieve(
    effective_question,
    k=SERVING_K,
)
```

with the constants:

```text
SERVING_K = 5
graph   hop_budget = 3
```

- `SERVING_K = 5` is a serving-side constant for v1. It is numerically aligned with `#14` `GENERATION_K = 5`, but the serving layer does not import benchmark-runner code and is not conceptually dependent on benchmark execution. If an implementation later colocates the two values in a neutral shared constants module, that is acceptable; `#14` is not a runtime dependency of serving.
- `graph hop_budget = 3` is the same value `#11` uses as its internal benchmark default. Passing it explicitly at the boundary keeps the contract auditable against `#11` and is what the implementing ticket asserts against.
- All three retrievers are called exactly once per request, with these signatures, and the serving layer never re-walks their results.

### 4.1 What the serving layer does not do for hybrid

Hybrid runs entirely inside `HybridRetriever`. The serving layer does not:

- multiply k or call either sub-retriever separately for `mode = "hybrid"`;
- invoke RRF;
- inspect, merge, deduplicate, or re-rank candidate lists;
- re-construct hybrid provenance from fresh retriever calls;
- choose between vector-led and graph-led execution paths;
- pass a vector-derived seed into `GraphRetriever`.

Every hybrid internals constant (`HYBRID_K_VEC_MULTIPLIER`, `HYBRID_K_GRAPH_MULTIPLIER`, `HYBRID_HOP_BUDGET`, `RRF_K`, the three-key tie-break) is owned by `#12`. The serving surface sees only the final ordered `HybridRetrievalHit` list.

## 5. Generation seam

For the chosen retriever's ordered top-`SERVING_K` hits, the application service:

1. **Resolves `chunk_id` against the canonical chunk artifact.** Each hit's `chunk_id` is looked up against `data/derived/chunks.jsonl` (#2). Any unknown / stale / duplicate `chunk_id` or `doc_id` raises `EvidenceIntegrityError`.
2. **Constructs `EvidenceBlock` in retrieval order.** Each hit becomes `EvidenceBlock(chunk_id, doc_id, text)` in the order it was returned. No reordering at this step.
3. **Holds two immutable maps** for the duration of the request:
   - `chunk_id -> EvidenceBlock`
   - `chunk_id -> retrieval_hit`
4. **Calls `#13` grounded generation once.** `GroundedGenerator.generate(GenerationRequest(query=effective_question, evidence=tuple(EvidenceBlock, ...)))`. The single allowed `#13` retry on parse/protocol failure belongs to `#13`.
5. **No retry at the serving layer.** No citation repair, no answer rewriting, no support rescoring, no second `LLMProvider` call.
6. **Empty evidence is deterministic.** A zero-length evidence list returns the deterministic `ABSTAIN` sentinel without invoking `LLMProvider`.

No second independent `chunks.jsonl` lookup after generation. Any such lookup added later is an integrity assertion (must equal the supplied `EvidenceBlock.doc_id`), not a primary attribution source.

## 6. Response contracts

### 6.1 Success — `200 OK`

```json
{
  "request_id": "<uuid4>",
  "answer":     "<string or \"ABSTAIN\">",
  "mode":       "vector" | "graph" | "hybrid",
  "abstained":  true | false,
  "citations":  [
    {
      "chunk_id":   "c_<16 hex>",
      "doc_id":     "W..." | null,
      "status":     "IN_EVIDENCE_SUPPORTING" | "IN_EVIDENCE_NON_SUPPORTING" | "OUT_OF_EVIDENCE",
      "provenance": { ...mode-tagged object... } | null
    }
  ]
}
```

- `mode` is `effective_mode`. If the client sent `null` or omitted the field, the response still records the actual mode that ran.
- `citations` preserves the answer-side order of `[[c_<16 hex>]]` occurrences. One response object per occurrence.
- Duplicate `chunk_id` citations remain duplicate occurrences sharing the same `provenance`.
- Status vocabulary is copied verbatim from `#13`.
- Full chunk text is never returned. `EvidenceBlock.text` exists for the generator; the HTTP response does not echo it.
- `OUT_OF_EVIDENCE` entries are kept in the response: `doc_id = null`, `provenance = null`. They are not enriched by any post-generation lookup.

### 6.2 `ABSTAIN`

```json
{
  "request_id": "<uuid4>",
  "answer":     "ABSTAIN",
  "mode":       "vector" | "graph" | "hybrid",
  "abstained":  true,
  "citations":  []
}
```

- A retrieval miss is **not** a 404. It is a successful request that returned `ABSTAIN`.
- `citations: []` is pinned — a partial evidence list is never returned with `ABSTAIN`.
- Two triggers:
  1. Empty resolved evidence ⇒ deterministic `ABSTAIN`, no `LLMProvider` call.
  2. The generator's own `ABSTAIN` sentinel per `#13`.

### 6.3 Citation attribution — provenance-join invariant

`EvidenceBlock` values are constructed exactly from the selected retriever's ordered top-`SERVING_K` hits. Therefore for every `#13` `CitationOccurrence`:

- `status ∈ {IN_EVIDENCE_SUPPORTING, IN_EVIDENCE_NON_SUPPORTING}` must satisfy:
  - the cited `chunk_id` is in the `chunk_id -> EvidenceBlock` map;
  - the cited `chunk_id` is in the `chunk_id -> retrieval_hit` map.
- When both maps contain it:
  - `doc_id` is the exact `EvidenceBlock.doc_id` that was supplied to `#13`;
  - `provenance` is the corresponding retrieval hit's mode-tagged provenance object (§6.4).
  - No canonical-chunk lookup is performed to "enrich" this attribution.
- When either map lacks it: raise an internal `ServingIntegrityError`. That maps to `500 internal_error`. The serving layer never fabricates nullable rank/score/path fields for an allegedly in-evidence citation.
- `status == OUT_OF_EVIDENCE`: `doc_id = null`, `provenance = null`. No canonical-chunk lookup.

This invariant is enforced at the response-build step. It is the rule that prevents answer-side provenance invention.

### 6.4 Provenance shape — verbatim, never reconstructed

Provenance is mode-tagged and is a direct JSON projection of whatever the retriever produced. It is never reconstructed from a fresh retriever call and never normalized across modes.

- **vector** — direct projection of a `#10 VectorRetrievalHit`:

  ```json
  { "mode": "vector", "rank": <int 1..5>, "score": <native vector score> }
  ```

- **graph** — direct projection of a `#11 GraphRetrievalHit`, including the verbatim five-field directional `evidence_path`:

  ```json
  {
    "mode": "graph",
    "rank":  <int>,
    "score": <native graph score>,
    "graph_evidence_path": <verbatim #11 directional path, object form>
  }
  ```

  `graph_evidence_path` is copied verbatim. The serving layer does not validate ontology endpoints, rewrite relation direction, reinterpret direction, or manufacture graph provenance. If the served request was a structural-CITES context, the path reflects that traversal unmodified — but the serving layer does not label the chunk as textual proof of the citation relationship (§6.5).

- **hybrid** — direct projection of a `#12 HybridRetrievalHit`, including the verbatim `#12 HybridHitProvenance`:

  ```json
  {
    "mode": "hybrid",
    "rank":            <int>,                  // fused rank in the RRF output
    "rrf_score":       <float>,                // RRF fused score
    "sources":         <array of "vector" | "graph">,
    "hybrid_hit_provenance": <verbatim #12 HybridHitProvenance>
  }
  ```

  `hybrid_hit_provenance` includes the per-retriever rank, per-retriever raw score, and the verbatim `#11 graph_evidence_path` when the graph contributed. The serving layer does not reconstruct it from fresh calls.

Scores are reported as the producing retriever emitted them. No cross-mode normalization. Cross-mode comparison is the caller's problem; serving only reports retriever-native numbers.

### 6.5 Pure-CITES structural-context carve-out

When a graph hit's evidence path is composed purely of `CITES` edges and the returned chunk is a terminal-`Document` chunk (the structural-context case described by `#11`), the serving layer carries that `CITES` path verbatim into `provenance.graph_evidence_path`. The structural-CITES path is a property of retrieval provenance, not of citation status.

The serving layer does **not** label such a chunk as textual proof of the CITES relationship. The chunk's `#13` lexical citation status (`IN_EVIDENCE_SUPPORTING` / `IN_EVIDENCE_NON_SUPPORTING` / `OUT_OF_EVIDENCE`) remains authoritative in the response. The structural `CITES` provenance does not upgrade or downgrade it and does not add a new status value.

## 7. Request lifecycle and instrumentation

### 7.1 `request_id`

- One UUID4 per HTTP request, generated by the route handler at request ingress **before** validation.
- Returned as a top-level `request_id` field on every success response and inside the `error` envelope on every non-200 response.
- Logged exactly once (see §7.2).
- Not echoed from the client in v1. Opaque. Never reused.
- Has **no semantic effect** on `#16` cache keys, on retrieval, on generation, or on benchmarking. It is operational correlation only.

### 7.2 Structured request log

Exactly one structured JSON record is written per completed HTTP request, after the response body has been built. Fields:

| field | meaning |
|---|---|
| `ts` | framework ISO-8601 UTC timestamp |
| `event` | `"http_request_completed"` (literal) |
| `request_id` | the same UUID4 from §7.1 |
| `mode` | `effective_mode` once known; `null` for failures before validation resolved `mode` |
| `status` | `ok` \| `abstained` \| `error` |
| `http_status` | integer HTTP status |
| `latency_ms` | integer, wall-clock from request ingress to response built |
| `retrieval_hit_count` | integer `>= 0` once retrieval has completed; `null` for failures before retrieval ran |
| `citation_count` | integer `>= 0` once a `#13` `GenerationResult` is parsed; `null` for failures after retrieval but before a parsed result; `0` for `ABSTAIN` and for parse-error responses |
| `error_code` | the envelope's `error.code` iff `http_status >= 400`; absent otherwise |

`status` and `error_code` share a vocabulary:

| condition | `status` | `error_code` |
|---|---|---|
| answer returned | `ok` | — |
| `ABSTAIN` returned | `abstained` | — |
| `GenerationParseError` reached response-build | `error` | `generation_protocol_error` |
| pre-validation failure (Content-Type, malformed JSON, schema, mode) | `error` | `invalid_request` or `unsupported_media_type` |
| `EvidenceIntegrityError` | `error` | `evidence_integrity_error` |
| vector store / graph store unavailable | `error` | `retrieval_unavailable` |
| `LLMProvider` unavailable / `BudgetExceeded` | `error` | `generator_unavailable` |
| `ServingIntegrityError` / any other internal failure | `error` | `internal_error` |

Stage-counter semantics:

| stage reached by the time the log is written | `retrieval_hit_count` | `citation_count` |
|---|---|---|
| pre-validation failure (Content-Type, malformed JSON, schema violation) | `null` | `null` |
| retrieval completed; generation not yet run / errored before `GenerationResult` | actual int | `null` |
| `GenerationResult` parsed (success, `ABSTAIN`, or `GenerationParseError` reached response-build) | actual int | int `>= 0` |

Retrieval miss is `0`; `ABSTAIN` citations is `0`; stage not executed is `null`. The distinction is operational — it lets an operator tell "retrieval returned nothing" from "retrieval didn't run".

The log record never contains: raw question, normalized question, any hash or HMAC of the question, `[[c_*]]` token lists, answer text, `EvidenceBlock.text`, prompts, raw provider output, raw provider request/response bodies, credentials, chunk text, `chunk_id` values, `doc_id` values, retrieval scores, evidence paths, request/response bodies, client IP, User-Agent, or local filesystem paths.

`request_id` is the only normal v1 request-correlation identifier. The framework access log line (e.g. `uvicorn.access`) may exist as framework behavior but is not the canonical record.

## 8. Error envelope and HTTP mappings

Every non-200 response uses one stable non-sensitive envelope shape:

```json
{
  "error": {
    "code":    "<stable_machine_readable_string>",
    "message": "<short, non-sensitive, not localized>"
  },
  "request_id": "<uuid4>"
}
```

`code` is stable per condition. `message` is short and stable per code. No envelope field ever contains: the raw question, the request body, answer text, a prompt, a model raw output, a provider error body, database error strings, chunk text, local paths, stack traces, or credentials.

| condition | HTTP | `error.code` |
|---|---|---|
| `Content-Type != application/json` | 415 | `unsupported_media_type` |
| malformed JSON | 422 | `invalid_request` |
| valid JSON, schema failure (missing `question`, non-string `question`, empty after strip, unknown `mode` value, unknown extra fields) | 422 | `invalid_request` |
| `EvidenceIntegrityError` (unknown / stale / duplicate `chunk_id` or `doc_id` at evidence preflight) | 503 | `evidence_integrity_error` |
| vector store unavailable | 503 | `retrieval_unavailable` |
| graph store unavailable | 503 | `retrieval_unavailable` |
| `LLMProvider` unavailable | 503 | `generator_unavailable` |
| `BudgetExceeded` (`#16`) | 503 | `generator_unavailable` |
| `GenerationParseError` after the single `#13` retry | 502 | `generation_protocol_error` |
| `ServingIntegrityError` (citation provenance invariant violation) | 500 | `internal_error` |
| any other unexpected implementation defect | 500 | `internal_error` |

- `415 unsupported_media_type` is enforced explicitly by the route handler so the contract is not contingent on a framework default.
- `502 generation_protocol_error` is reserved for `#13` protocol-level parse failure, not transport failure.
- `503` covers data-layer freshness and external-service unavailability. `BudgetExceeded` (`#16`) is a service-availability constraint and is mapped here, not as `internal_error`.

## 9. Startup and preflight

Before accepting requests, the service performs a startup preflight. The preflight validates the **local** and **store** preconditions the advertised retrieval modes depend on. It does not invent stronger provider semantics than `#16` already pins: there is no synthetic live-LLM "ping" or generation call at startup unless `#16` itself requires one.

The preflight covers:

- `#16` provider configuration shape, presence, and required secrets;
- required local artifacts (e.g., the canonical `data/derived/chunks.jsonl` readability);
- vector store connectivity and freshness (vector/hybrid prerequisites), per `#10`;
- neo4j connectivity (`#9`);
- graph content/consistency identity (`graph_content_sha256` matches the expected value, per `#9` and `#11`);
- construction of the configured retriever and generator adapters (the wiring evaluates without making a generation call).

The preflight rule is:

> **All three advertised retrieval modes (`vector`, `graph`, `hybrid`) must satisfy their required local / store prerequisites at startup. Otherwise startup fails.**

Failure is non-zero process exit with the failure reason on stderr. There is no:

- degraded startup,
- silent mode disablement,
- `graph → vector` fallback,
- `hybrid → vector` fallback,
- "running anyway" behavior.

A real request-time provider failure remains a request-time failure mapped via §8 (`503 generator_unavailable`); it is not silently absorbed by startup.

## 10. Upstream consistency

- **`#10` (vector retrieval).** `VectorRetriever.retrieve(query, k)` is composed unchanged. The serving layer passes `k = SERVING_K = 5`. No service-level vector re-scoring or re-ranking. No `#10` invariants are altered.
- **`#11` (graph retrieval).** `GraphRetriever.retrieve(query, k, hop_budget)` is composed unchanged. The serving layer passes `k = SERVING_K = 5` and `hop_budget = 3`. The graph `evidence_path` is copied verbatim into response provenance. Pure-CITES structural-context semantics (§6.5) are inherited from `#11` without modification.
- **`#12` (hybrid retrieval).** `HybridRetriever.retrieve(query, k)` is composed unchanged. The serving layer passes `k = SERVING_K = 5`. Hybrid internals (`HYBRID_K_VEC_MULTIPLIER`, `HYBRID_K_GRAPH_MULTIPLIER`, `HYBRID_HOP_BUDGET`, `RRF_K`, the three-key tie-break) are owned by `#12`. The serving layer never reproduces them, never inspects candidate lists, and never passes an externally-derived seed into `GraphRetriever`.
- **`#13` (grounded generation).** `GroundedGenerator.generate(GenerationRequest(query, evidence))` is called exactly once per request. The serving layer does not retry. The single `#13` parse/protocol retry is owned by `#13` and propagates as `502 generation_protocol_error` if it too fails. Citation statuses (`IN_EVIDENCE_SUPPORTING` / `IN_EVIDENCE_NON_SUPPORTING` / `OUT_OF_EVIDENCE`), `ABSTAIN`, and the `evidence_fingerprint` policy are inherited unchanged.
- **`#14` (benchmark runner).** Serving does not import benchmark-runner code and is not conceptually dependent on benchmark execution. `SERVING_K = 5` is a serving-side constant; its numerical alignment with `#14` `GENERATION_K = 5` is incidental. The runner does not depend on serving. The serving route does not read or expose any of the following benchmark-only fields: `qid`, `run_id`, `run_fingerprint`, `expected_answer`, `answer_aliases`, `hop_bucket`, `gold_evidence_path`, `supporting_chunk_ids`, `supporting_doc_ids`. It never reads `eval/questions.jsonl`.
- **`#16` (provider abstractions).** `LLMProvider` and `EmbeddingProvider` are accessed only through their interface signatures. No provider SDK appears in the serving layer. Cache identity and budget behavior are governed exclusively by `#16`; the serving layer does not mint, modify, or read cache keys.
- **`#2` / `#4` / `#5` / `#9` (canonical IDs and content identity).** `chunk_id`, `doc_id`, and graph `evidence_path` shapes are inherited unchanged from `#2`, `#4`, `#5`, and `#9`. The serving layer never invents new ID shapes.

If at implementation time the serving layer appears to need behavior owned by an upstream ticket, that is a signal to revise the upstream ticket — not to add it here.

## 11. Out of scope

The serving surface does **not** introduce, ship, or imply any of the following in v1. They are not deferred-with-stub, not "nice to have"; they are explicitly excluded:

- frontend or UI;
- CLI wrapper around the HTTP endpoint;
- websockets, streaming responses, SSE, long-polling;
- authentication, authorization, rate limiting, accounts, sessions, cookies, request-body API keys, OAuth;
- request/response persistence layer, request log database, audit database;
- background jobs, queues, workers, schedulers, retry queues;
- per-request overrides for model, k, graph depth, prompt, temperature, threshold, top_n, include_text;
- `/benchmark`, `/eval`, `/admin`, `/health`, `/ready`, `/livez`, `/metrics`, `/debug`, or business `/docs` endpoints;
- PII classification, content redaction, embedding-based scrubbing, regex scrubbing, HMAC of question text;
- `README.md` modifications;
- LangChain or any framework of equivalent scope;
- cloud infrastructure services; second service; second production entrypoint;
- direct provider-SDK imports in application code;
- serving-time router, learned router, classifier-based mode selection;
- citation repair, answer rewriting, or post-hoc support rescoring at the serving layer.

## 12. Acceptance gate

This spec closes when committed to `docs/specs/serving-surface.md` and `#15` is marked complete. The implementing ticket closes only when:

1. `POST /ask` exists as the only application-defined route and the v1 entrypoint is `uvicorn src.api:app --host 127.0.0.1 --port 8000`.
2. The request schema (§3) is enforced: `question` required and non-empty after `strip()`; `mode` optional, accepting `"vector"`, `"graph"`, `"hybrid"`, or `null`; `extra="forbid"`; `Content-Type != application/json` returns `415 unsupported_media_type`.
3. The retrieval dispatch (§4) calls each retriever exactly once with `k = SERVING_K = 5` and `hop_budget = 3` for graph; the route never reproduces hybrid internals. Static inspection finds no RRF arithmetic, no `LLMProvider`/`EmbeddingProvider` import, no Cypher, no SQL, no prompt construction, no citation parser, no chunk projection, no benchmark-scoring code in `src/api.py`.
4. The generation seam (§5) resolves retrieval hits into `EvidenceBlock` objects against `data/derived/chunks.jsonl`, holds the two maps, calls `GroundedGenerator.generate(...)` once, returns deterministic `ABSTAIN` without an `LLMProvider` call when evidence is empty, and never retries at the serving layer.
5. The success and `ABSTAIN` responses (§6.1, §6.2) match the pinned shapes; `request_id` is the same UUID4 throughout (§7.1).
6. The provenance-join invariant (§6.3) is enforced: an `IN_EVIDENCE_*` citation whose `chunk_id` is missing from either map raises `ServingIntegrityError` and maps to `500 internal_error`; an `OUT_OF_EVIDENCE` citation has `doc_id = null`, `provenance = null`, and is never enriched by a post-generation `chunks.jsonl` lookup.
7. Pure-CITES structural-context semantics (§6.5) are preserved: the graph `evidence_path` is copied verbatim and `#13` lexical status remains authoritative.
8. The HTTP error envelope and mappings (§8) match the table for every listed condition; no envelope field exposes a raw question, body, prompt, model output, provider error, database error, chunk text, path, stack trace, or credential.
9. The structured request log (§7.2) writes exactly one JSON record per completed request with the pinned field set and the pinned null-vs-zero semantics; the record does not contain any raw question, normalized question, hash/HMAC, answer text, evidence text, prompt, raw provider output, credential, chunk text, `chunk_id`/`doc_id` value, retrieval score, evidence path, or client IP/UA.
10. Startup preflight (§9) rejects startup non-zero when any required local/store precondition for `vector`, `graph`, or `hybrid` fails; no silent mode disablement and no `hybrid → vector` / `graph → vector` fallback is added.
11. A deterministic self-check exercises one fixture per branch: a successful `mode = "vector"` request returning a non-empty answer with `IN_EVIDENCE_SUPPORTING` and `OUT_OF_EVIDENCE` citations; a `mode = "graph"` request whose returned chunk is a pure-CITES structural-context chunk (path retained, `#13` status authoritative); a `mode = "hybrid"` request whose `hybrid_hit_provenance` matches a frozen per-retriever fixture verbatim; a deterministic `ABSTAIN` request driven by an artificial empty `vector_retriever.retrieve`; a `422 invalid_request` fixture on a non-string `question`; a `415 unsupported_media_type` fixture on a non-JSON Content-Type; a `503 retrieval_unavailable` fixture under a forced store-unavailable branch; a `502 generation_protocol_error` fixture under a forced `GenerationParseError`; a `500 internal_error` fixture under a forced `ServingIntegrityError`. Each fixture asserts byte-identical output across repeated runs (modulo `request_id`).

The first six gates validate the request, dispatch, generation, and response contracts. Gates seven and eight validate the failure envelope. Gates nine and ten validate logging and startup. Gate eleven is the runnable self-check that fails if any wiring drifts.
