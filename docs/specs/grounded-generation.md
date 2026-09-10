# Grounded Answer Generation and Citation-Validation Specification

Resolution of [#13 — Decide the grounded answer generation and citation-validation contract](https://github.com/ImmoP/knowledge_graph_rag/issues/13). This document is the **generator and validator contract** that closes #13. The implementing ticket goes to `/to-tickets` once the map is cleared.

The contract covers three things: the generator's input/output shape, the deterministic citation and abstention protocol the generator produces, and the transparent per-citation/per-claim support accounting the validator records on the result. Benchmark metrics, gold comparisons, and headline-metric selection are explicitly [#14](https://github.com/ImmoP/knowledge_graph_rag/issues/14)'s decision and are NOT part of this spec.

## 1. Architectural shape

Three rules, structural:

1. The generator calls `LLMProvider.generate` from #16 only. Application code never imports a provider SDK. Provider/model selection is configuration-driven (`LLM_PROVIDER`, `LLM_MODEL`); this spec pins neither a default provider nor a default model — that decision belongs to #16 (§7) and to runtime configuration.
2. Generation runs only over the **supplied evidence**. The generator is given an ordered list of `EvidenceBlock` and is forbidden by prompt from using knowledge outside that list. There is no fallback to retrieve, no web tool, no follow-up RAG step.
3. Citation validation is a transparent per-claim proxy rooted in lexical containment. No LLM judge, no embedding judge, no learned NLI in v1. The proxy is a pure function of (claim text, chunk text) and is reproducible from inputs alone.

The benchmark runner (#14) consumes the deterministic raw records #13 emits but does not depend on any internal detail of the proxy.

## 2. Type shapes

These are the canonical dataclasses. They are the API surface other tickets program against.

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
    claim_id: int        # 0-based, into claims[]
    char_start: int      # char offset into answer_text
    char_end: int        # exclusive
    status: Literal[
        "IN_EVIDENCE_SUPPORTING",
        "IN_EVIDENCE_NON_SUPPORTING",
        "OUT_OF_EVIDENCE",
    ]

@dataclass(frozen=True)
class Claim:
    claim_id: int                    # 0-based, in segmentation order
    text: str                        # sentence text, citations stripped
    char_start: int                  # char offset into answer_text (inclusive)
    char_end: int                    # exclusive
    scorable: bool                   # False if normalization produced 0 tokens
    attached_chunk_ids: tuple[str, ...]  # unique chunk_ids cited in this claim

@dataclass(frozen=True)
class ClaimSupportVerdict:
    claim_id: int
    verdict: Literal["SUPPORTED", "CITED_UNSUPPORTED", "UNCITED"]
    supporting_chunk_ids: tuple[str, ...]   # subset of attached_chunk_ids
    non_supporting_chunk_ids: tuple[str, ...]
    out_of_evidence_chunk_ids: tuple[str, ...]

@dataclass(frozen=True)
class GenerationResult:
    answer_text: str
    abstained: bool
    citations: tuple[CitationOccurrence, ...]      # every parsed occurrence, in order
    claims: tuple[Claim, ...]                       # segmented, in order
    claim_support: tuple[ClaimSupportVerdict, ...]  # one-to-one with claims
    usage_calls: tuple[Usage, ...]                  # one Usage per provider call
    evidence_fingerprint: str                       # hex SHA256, see §6
```

A successful return is necessarily parsed successfully: `parsed_ok` and `parse_error` are not on the result. A persistent parser failure after the single allowed retry raises `GenerationParseError` and produces no `GenerationResult` (§5).

`GenerationRequest` does **not** carry `system_prompt`, `prompt_template_name`, `prompt_template_version`, or template SHA as caller-supplied fields. The generator owns its prompt artifact (§3); the implementing layer reads it. Audit-only metadata (prompt filename, prompt content SHA) lives in the run/generation trace, not on `GenerationResult`.

## 3. Prompt artifact

The generator owns one pinned, versioned prompt file:

```
prompts/grounded_answer_v1.txt
```

The generator reads the file at call time (or memoises its content). The implementing layer records:

- `prompt_filename` (audit metadata)
- `prompt_content_sha256` = `sha256(file_content.encode("utf-8"))` (audit metadata)

Both flow into the run/generation trace and the #16 cache key (per §6). Callers do not supply prompt text per request.

The pinned grounding requirements the prompt records and the generator contract enforces:

- Answer using only the supplied evidence blocks.
- Do not use outside knowledge.
- Every factual claim must cite one or more supplied chunk_ids.
- Place citation tokens before the sentence-ending punctuation.
- If the supplied evidence is insufficient, output exactly `ABSTAIN` and nothing else.
- No headings.
- No `References` section.
- No meta-commentary.

The rendered prompt includes the query and the evidence blocks in the exact caller-supplied order. Order matters (§6 and §11).

A template-version bump (`v1` → `v2`) is the unit of cache invalidation. The implementing ticket bumps the suffix on any non-empty change to the template text; the SHA changes automatically. The same convention is used by [extraction-contract §7.4](extraction-contract.md) for #7's templates.

## 4. Answer / citation / abstention protocol

### 4.1 Citation syntax

The only acceptable citation form is the inline token:

```
[[c_<16 lowercase hex characters>]]
```

regex: `\[\[(c_[0-9a-f]{16})\]\]`

Examples:

```
DPR was introduced for open-domain question answering
[[c_aabbccddeeff0011]].

It uses BERT [[c_1122334455667788]] and dense retrieval [[c_aabbccddeeff0011]][[c_1122334455667788]].
```

Rules:

- Each token is independent. Adjacency is co-citation, not a list.
- The capture group is exactly `c_[0-9a-f]{16}` matching the canonical `chunk_id` regex from #2.
- A citation token **must** appear before the sentence-ending punctuation that ends the claim it supports.
- No prefix, suffix, locator, comma-list, or whitespace inside the brackets.
- The generator MUST NOT emit a `References:` section, a heading, or a numbered list. The validator treats any such construct as meta-commentary and ignores it; if it prevents claim segmentation the result is a parse/protocol failure.
- Any `[[...]]` that does not match the regex is **malformed bracket syntax** and is a parse/protocol error (§5).

### 4.2 ABSTAIN sentinel

The whole answer is the literal token `ABSTAIN` and nothing else:

```
ABSTAIN
```

When parsed as `ABSTAIN`:

- `answer_text == "ABSTAIN"`.
- `abstained == True`.
- `citations == ()`.
- `claims == ()`.
- `claim_support == ()`.

Any other text containing the substring `ABSTAIN` (in any case) is treated as a normal answer — the prompt's only rule is "ABSTAIN alone". The validator does not interpret partial/sentence-level abstention.

### 4.3 Zero-evidence behaviour

If `GenerationRequest.evidence` is empty, the generator does **not** call `LLMProvider`:

- It returns the deterministic sentinel directly.
- `answer_text == "ABSTAIN"`, `abstained == True`, all collections empty.
- `usage_calls == ()` (no provider call, no `Usage`).
- `evidence_fingerprint` is still the deterministic SHA256 of the empty canonical evidence array (§6).
- This is **not** a model abstention; it is a deterministic grounding precondition. #14 may score it as an incorrect answer when retrieval failed for that question; that judgment is `#14`'s, not this spec's.

## 5. Parser, retry, and parse-failure semantics

### 5.1 Parsing pipeline

The parsing pipeline is a pure function of `answer_text` plus the supplied `evidence`. Steps:

1. If `answer_text == "ABSTAIN"`: emit the sentinel result, `usage_calls = ()`.
2. Otherwise, segment `answer_text` into `Claim` records (§5.2).
3. Extract every `[[c_<16hex>]]` token; record each as a `CitationOccurrence` with `char_span`.
4. Attach each citation to exactly one claim by character-span overlap (§5.3). A valid citation token whose character span does not overlap any claim's span is a protocol failure.
5. Validate each citation's `chunk_id` against the supplied evidence (§7).
6. Compute per-claim support (§8).

### 5.2 Sentence segmentation

A non-`ABSTAIN` answer is segmented into one claim per sentence:

- Split on `". "`, `"? "`, `"! "`, and `"\n\n"`. Each split token is trimmed of leading/trailing whitespace.
- Drop empty segments. Preserve original order.
- Each surviving segment's `char_span = (start, end)` is the inclusive-exclusive offsets in `answer_text`. The split is deterministic; same input → same segments and same spans.
- The generator is prompted to avoid headings, reference sections, and meta-commentary so a non-`ABSTAIN` answer is intended to be entirely claim sentences. The validator does not introduce a procedural/meta sentence class.

`ABSTAIN` produces no claims.

### 5.3 Citation attachment

A citation `[[c_…]]` belongs to the claim whose `char_span` overlaps the citation's character range. If a token sits at a boundary it attaches to the earliest-claimed claim containing its leftmost character.

Attachment invariants:

- Every parsed `CitationOccurrence` attaches to exactly one `Claim`.
- A valid citation token whose character span does not overlap any `Claim` is a **protocol/parse failure**: there is no claim to attach to, which by definition means the model emitted a citation outside any answer sentence. Surface as a parser error.
- Citation tokens are stripped from `Claim.text` before any lexical support comparison.

### 5.4 Retry policy

Exactly **one** retry. The retry is triggered only by parse/protocol failure:

- **Triggers retry**: malformed `[[…]]` syntax (regex mismatch), an `answer_text` from which no claim can be segmented, a valid citation token that attaches to no claim.
- **Does not trigger retry**: `OUT_OF_EVIDENCE` citations, lexical-support failures, uncited claims, `CITED_UNSUPPORTED`, `ABSTAIN`, low `claim_support_coverage`. These are model outcomes, not protocol violations.

The retry uses one pinned format-repair instruction (single suffix, implemented by the implementing ticket). Same generation parameters. The two provider calls produce two independent cache entries (the user prompt differs by the format-repair suffix) and two independent `Usage` records, both recorded on the final `GenerationResult.usage_calls` (or on the `GenerationParseError` path's log).

After one retry, persistent parse failure raises `GenerationParseError`. `#13` does not decide whether the orchestrator or benchmark runner skips the question or aborts the run — `#14` owns that handling.

## 6. Evidence fingerprinting and #16 cache identity

### 6.1 `evidence_fingerprint` (audit metadata, NOT mixed into the cache key)

```python
import hashlib, json

def evidence_fingerprint(evidence: tuple[EvidenceBlock, ...]) -> str:
    payload = [
        {
            "chunk_id":            e.chunk_id,
            "doc_id":              e.doc_id,
            "source_text_sha256":  hashlib.sha256(e.text.encode("utf-8")).hexdigest(),
        }
        for e in evidence                  # exact order preserved; not sorted
    ]
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

Properties:

- The canonical-JSON rules are pinned and reused across the project (see [extraction-contract §6](extraction-contract.md)). Identical inputs and rules produce identical fingerprints. Different inputs produce different fingerprints.
- The evidence array is **not** sorted. The generator sees the caller's exact order, and different order is a different generation input.
- `source_text_sha256` is required because `chunk_id` is **content-independent**: it derives from `(doc_id, section_path, chunk_index)` per #2, not from the chunk text. Stale text on disk and a freshly re-cleaned text can produce the same `chunk_id`. The SHA binds the fingerprint (and, transitively, the cache key below) to the actual text.
- `evidence_fingerprint` is recorded on `GenerationResult` for reproducibility and shows up in benchmark reports alongside any other audit metadata.
- `evidence_fingerprint` is **not** mixed into the #16 cache key by this spec — the underlying `user` prompt already encodes the evidence verbatim, so any text/order change already changes the #16 key.

### 6.2 #16 cache identity — sensitivity requirements

The #16 cache identity MUST change if any of the following change (these are sensitivity requirements; the constructor of the cache key lives in `provider-abstractions.md` §5):

| Dimension | Why it changes the key |
|---|---|
| `provider.name` | #16 outer key |
| `provider.model` | #16 outer key |
| `LLMProvider.generate` vs `LLMProvider.generate_json` vs `EmbeddingProvider.embed` | operation identity (see §7 consistency repair below) |
| `cache_format_version` | adapter-format-change invalidation |
| exact `system` string | #16 key input |
| exact `user` string | #16 key input (the rendered prompt body; includes the prompt filename, content SHA, query, and evidence) |
| exact `embed` ordered text list | #16 key input |
| `prompt_template_filename` and `prompt_template_content_sha256` | mixed into the user-prompt header so any template edit changes the key |
| evidence order (within the rendered user prompt) | changes the user string |
| any evidence `chunk_id`, `doc_id`, or `text` | changes the user string |
| `query` | changes the user string |
| `temperature` | changes the user string header |
| `max_tokens` | changes the user string header |
| canonical schema (for `generate_json`) | schema-hash field in the key |

The generator does **not** inject `temperature`, `max_tokens`, audit hashes, or schema hashes into the rendered user prompt merely to vary the cache key. Doing so would double-count those fields across layers and create spurious cache splits. This is the explicit responsibility split with #16; see `provider-abstractions.md` §5 for the canonical construction.

### 6.3 What `cache_hit` is NOT

A successful return from `GenerationResult` is the answer text. `LLMProvider.generate`'s public return type remains `(text, Usage)` (see `provider-abstractions.md` §1); it does not return a cache-hit field. Cache observability belongs to the abstraction layer / run trace.

## 7. Per-citation validation statuses

Every parsed `CitationOccurrence` receives exactly one of three statuses:

| Status | Condition |
|---|---|
| `IN_EVIDENCE_SUPPORTING` | regex matches `^c_[0-9a-f]{16}$` **and** the `chunk_id` is in the supplied evidence set **and** the lexical-support proxy (§8) passes for that chunk against the claim |
| `IN_EVIDENCE_NON_SUPPORTING` | regex matches **and** the `chunk_id` is in the supplied evidence set **and** the lexical-support proxy fails |
| `OUT_OF_EVIDENCE` | regex matches **and** the `chunk_id` is not in the supplied evidence set |

Malformed bracket syntax is a **parse/protocol failure**, not a valid `CitationOccurrence`. It does not produce a citation record; it surfaces as a parser error and triggers the retry path (§5.4).

`OUT_OF_EVIDENCE` is NOT a parse/protocol failure, because the citation syntax itself is valid. The model produced a token that parses; the token just doesn't resolve to supplied evidence. That is a model outcome and remains observable on the result, not a generator-level error.

The set of supplied evidence used for the membership test is the exact set the caller handed in (i.e. the same list whose SHA is `evidence_fingerprint`).

## 8. Lexical support proxy

This is the transparent, deterministic, pure-function proxy that decides "does this cited chunk contain enough of this claim to count as support?" It is a quality proxy, **not** a semantic-entailment proof.

### 8.1 Normalisation and tokenisation (one pinned rule)

For both claim text and cited chunk text, in this order:

1. Remove direct citation tokens matching `\[\[c_[0-9a-f]{16}\]\]` from the claim text first.
2. Unicode-normalise to **NFKC**.
3. **Casefold**.
4. Tokenise into maximal contiguous runs of Unicode-alphanumeric characters (`\w` with `re.UNICODE`).
5. Retain numeric tokens and tokens containing digits exactly as spelled. Do **not** discard tokens that consist only of digits; do **not** discard tokens that contain a digit.

Distinct examples that must remain token-distinguishable: `GPT-4`, `Recall@10`, `512 tokens`, `2020`. With the rule above, `GPT-4` tokenises as `["GPT", "4"]`, `Recall@10` as `["Recall", "10"]`, `512 tokens` as `["512", "tokens"]`, and `2020` as `["2020"]`. Token-distinguishability across runs is preserved by the same rule in §8.1 every time.

If the normalised claim token list is empty, the sentence is **not a scorable factual claim**: `Claim.scorable = False`, the sentence is excluded from `claim_support_coverage`'s denominator, and it carries no `SUPPORTED` / `CITED_UNSUPPORTED` verdict that requires support. Such claims still receive an `UNCITED` verdict only if they contain no citation token (§9); if they are empty-after-normalisation and contain citations, those citations are still parsed and recorded against `CITED_UNSUPPORTED` because no claim n-gram can be verified.

### 8.2 Containment rule

Let `claim_tokens` be the normalised claim token list (length `k`). Let `n = min(4, k)`. (A claim with fewer than 4 tokens is **not** automatically supported — the proxy is fully symmetric across lengths.)

Enumerate every contiguous `n`-gram of length `n` from `claim_tokens`. For a 3-token claim there is one such n-gram (the 3-token sequence); for a 1-token claim there is one such n-gram (the single token); for a 4+ token claim there are `k - 3` n-grams.

Let `chunk_tokens` be the normalised cited-chunk token list. The chunk supports the claim iff at least one claim n-gram of length `n` occurs contiguously as a sub-list of `chunk_tokens`.

This function is pure: same `(claim_tokens, chunk_tokens, n) →` same verdict. No embedding lookup, no model, no network.

### 8.3 Constants

- `n = 4` is a **conservative a-priori v1 constant**. It is not described as an empirical sweet spot; it is the chosen length boundary for v1, frozen before any benchmark number is recorded.
- Any future change to `n`, the normalization, or the tokenisation requires a versioned contract bump on this spec and an `ontology.yaml`-style bump on the prompt template. The 45 held-out questions in `eval/questions.jsonl` cannot be used to tune any of these parameters (per [eval-schema.md](eval-schema.md) §6).

## 9. Per-claim verdicts and citation de-duplication

For every `Claim` the validator records exactly one of three verdicts:

| Verdict | Rule |
|---|---|
| `SUPPORTED` | the claim has at least one attached citation whose status is `IN_EVIDENCE_SUPPORTING` (§7) |
| `CITED_UNSUPPORTED` | the claim has attached citation(s); **none** of them is `IN_EVIDENCE_SUPPORTING`. `IN_EVIDENCE_NON_SUPPORTING` and `OUT_OF_EVIDENCE` counts on this claim remain separately observable on `ClaimSupportVerdict`. |
| `UNCITED` | the claim contains no citation token |

Citation de-duplication rule (the basis for claim-support accounting):

- For support evaluation, de-duplicate `(claim_id, chunk_id)` pairs within one claim.
- The same chunk cited twice in one claim contributes one `claim_to_chunk` link.
- The same chunk cited once in two different claims contributes two `claim_to_chunk` links.
- Raw `CitationOccurrence` records and their character spans are preserved on `GenerationResult` for debugging. The per-claim verdict uses the deduped link set.

A `Claim` whose normalised token list is empty (`Claim.scorable == False`) and which contains no citation token contributes `UNCITED`. If it has no tokens but contains citation tokens, those tokens are still parsed and the claim is `CITED_UNSUPPORTED` with the relevant counts surfaced on the verdict.

## 10. Diagnostics on the result (not quality targets)

These diagnostics are emitted on the result for the benchmark runner (#14) and the run trace. They are **not** quality targets; this spec does not pick a headline metric, and the model is not required to score any specific value.

```
claim_support_coverage =
    supported_claim_count / scorable_claim_count

cited_unsupported_claim_rate =
    cited_unsupported_claim_count / total_claim_count

uncited_claim_rate =
    uncited_claim_count / total_claim_count

evidence_utilization =
    |unique cited in-evidence chunk_ids| /
    |supplied evidence chunk_ids|
```

Zero-denominator rules:

- `claim_support_coverage` uses `scorable_claim_count` in the denominator. If `scorable_claim_count == 0`, `claim_support_coverage = None`.
- `evidence_utilization` uses `|supplied evidence chunk_ids|` in the denominator. If the supplied set is empty, `evidence_utilization = None`.

`#14` owns benchmark metrics — including any choice of headline numbers, gold comparisons against `supporting_chunk_ids`, exact-match / normalized F1 against `expected_answer`, and the per-bucket × per-condition headline table. `#13` does not select headline metrics and the implementing ticket must not introduce them.

## 11. Boundaries with #4, #10, #11, #12, #14, #16

| Other ticket | Boundary |
|---|---|
| **#4 (eval schema)** | `#13` does NOT read `eval/questions.jsonl`, `expected_answer`, `answer_aliases`, `supporting_chunk_ids`, or `hop_bucket`. Gold comparisons against those fields are `#14`'s job. `#13` emits raw records that `#14` consumes in a post-hoc step. The no-tuning-on-held-out rule applies to every constant in §8. |
| **#10 (vector retrieval)** | Orchestrator loads `data/derived/chunks.jsonl` and projects each chunk to an `EvidenceBlock(chunk_id, doc_id, text)`. Vector scores, ranks, and ANN parameters stay in the orchestrator's `retrieval_trace`, **not** in `evidence`. |
| **#11 (graph retrieval)** | Same projection. Graph `entities`, `evidence_path`, hop-count, and per-entity scores stay in the orchestrator's `retrieval_trace`, **not** in `evidence`. |
| **#12 (hybrid retrieval)** | Same projection. RRF provenance, per-retriever ranks, and merge-policy constants stay in `retrieval_trace`. The hybrid policy's determinism is exactly what makes the §6 cache key sound. |
| **#14 (benchmark runner)** | `#13` is the producer of `GenerationResult` records. `#14` is the consumer that selects headline metrics, joins against gold when running on the held-out benchmark set, and decides the structure of `report.json` (per #14). `#13` does not return benchmark-metric fields. |
| **#16 (provider abstractions)** | Generator calls `LLMProvider.generate` only. The caching and accounting layers in `#16` (with the §5 consistency repair below) own cache construction. The generator contributes the audit metadata (`prompt_filename`, `prompt_content_sha256`) that the wrapped cache key mixes in. |

### 11.1 Consistency repair to `#16`

The `#16` cache key as originally published (`provider-abstractions.md` §5.1) did not include `max_tokens` or `temperature` in its identity, even though changing either parameter changes model output without changing the `user` prompt text. The repair is an abstraction-layer concern, not a `#13`-local concern:

- For `generate`: cache identity now includes `(cache_format_version, operation="generate", provider.name, provider.model, system, user, max_tokens, temperature)`.
- For `generate_json`: cache identity additionally includes `(operation="generate_json", canonical schema hash)`.
- For `embed`: cache identity is `(cache_format_version, operation="embed", provider.name, provider.model, ordered exact texts)`.

Construction uses canonical JSON + SHA256 (see `provider-abstractions.md` §5.1 after the repair; not delimiter-concatenated ad-hoc fields). The public `LLMProvider.generate` return type stays `(text, Usage)` — no `cache_hit` field; cache observability is the abstraction layer's responsibility. `#7`'s stricter extraction-specific cache identity remains valid and is not weakened by this repair.

## 12. Evidence preflight

Before any provider call (and before the §4.3 zero-evidence short circuit), the implementing ticket MUST validate every `EvidenceBlock`:

- `query` is `str`.
- `evidence` preserves the caller's order (no reordering, no deduplication).
- Every `chunk_id` matches `^c_[0-9a-f]{16}$`.
- Every `doc_id` matches `^W[0-9]+$`.
- `chunk_id` values within one `GenerationRequest` are unique.
- Each `EvidenceBlock(chunk_id, doc_id, text)` matches the current canonical `data/derived/chunks.jsonl` record byte-for-byte for `(chunk_id, doc_id, text)`.

Any stale, unknown, mismatched, or duplicate `EvidenceBlock` is an **input-integrity error**: the implementing ticket raises `EvidenceIntegrityError` and the generator is not called. No silent reordering or deduplication.

## 13. Usage and budget accounting

- Every actual provider call returns one `Usage` from #16. `GenerationResult.usage_calls: tuple[Usage, ...]` records them in call order. When the retry path is exercised, the first call's `Usage` is preserved on the result alongside the second call's `Usage`. Total cost and token use for the question is the sum of `usage_calls` and is reconstructable from the result alone.
- On the `GenerationParseError` path, both call `Usage` records (first call, retry) are written to the run/generation trace before the exception is raised. The exception path is not silent on cost.
- `LLM_BUDGET_USD` and `LLM_BUDGET_TOKENS` are enforced by the `#16` wrapper around `LLMProvider`. The generator never bypasses them. A retry that pushes the cumulative counters over cap raises `BudgetExceeded`; the partial `Usage` records already observed remain on the result and the run trace.
- No new accounting fields beyond what `#16` already pins. The cache-accounting audit metadata (`prompt_filename`, `prompt_content_sha256`, `evidence_fingerprint`) is enough to attribute costs to inputs.

## 14. Conditions across vector / graph / hybrid retrieval

The generator's input contract is identical across the three retrieval conditions. `GenerationRequest.evidence` is `tuple[EvidenceBlock, …]` with exactly `(chunk_id, doc_id, text)`. The generator never sees:

- retrieval score
- `vector` / `graph` / `hybrid` source label
- graph entities
- graph `evidence_path`
- RRF provenance
- per-retriever ranks

All of those live in the orchestrator's `retrieval_trace`. Vector / graph / hybrid conditions differ only in **which ordered chunks retrieval hands the orchestrator**. That property is what makes `#14`'s "rows = hop buckets, columns = conditions" comparison isolate retrieval differences from generator behaviour. The same `GenerationResult` contract applies regardless of condition.

## 15. What this spec leaves open

These are explicitly not part of the #13 decision; they belong to implementing tickets or to later specs:

- **Implementing ticket body** (the generator code, prompt-loader, retry suffix wording, run-trace serialiser). The shape is pinned here; the implementing layer writes the body.
- **Benchmark metrics** — `#14`. `#13` does not pick headline numbers and does not name any quality gate.
- **Server surface** — `#15`. The generator API is callable in-process or out-of-process; `#13` does not decide.
- **Citation-style or chunk-text rendering tweaks** — the implementing ticket may add formatting (e.g. one evidence block per bullet, line breaks) inside the prompt body without changing the contract, provided the renderer preserves caller order and includes every supplied `chunk_id` exactly once.

## 16. Acceptance gate

This spec closes when committed to `docs/specs/grounded-generation.md`. The implementing ticket closes when all of these pass:

1. The implementing generator parses and emits `GenerationResult` matching the §2 dataclass shapes (no extra required fields; no required `cache_hit` / `parse_error`).
2. A smoke run with three citations (one `IN_EVIDENCE_SUPPORTING`, one `IN_EVIDENCE_NON_SUPPORTING`, one `OUT_OF_EVIDENCE`) plus one uncited claim produces the matching `CitationOccurrence` / `ClaimSupportVerdict` records — verified by a tiny test using a fixed `answer_text` and a fixed evidence list.
3. Malformed `[[...]]` syntax and a `[[c_…]]` token attached to no claim both trigger the single retry exactly once and then raise `GenerationParseError` on the next failure.
4. `ABSTAIN`-only output produces `abstained=True`, all collections empty, and `usage_calls=()`.
5. Empty `evidence` produces the deterministic zero-evidence sentinel without calling `LLMProvider`.
6. Evidence preflight (§12) raises `EvidenceIntegrityError` on a stale / duplicate / unknown `chunk_id` or `doc_id`, and the provider is not called.
7. `evidence_fingerprint` is stable across runs for identical `(evidence)` and changes when any of `chunk_id`, `doc_id`, or `text` changes, or when array order changes.
8. No benchmark-metric field appears on `GenerationResult`. No reference to `eval/questions.jsonl`, `expected_answer`, `supporting_chunk_ids`, or `hop_bucket` appears in the implementing code.
9. No provider-SDK import appears outside the adapter modules (per `#16` §9.5).

---

This spec resolves ticket #13. It is referenced by #14 (benchmark runner consumes `GenerationResult` records), by #15 (server surface exposes the generator in-process or via the same shape), and by the post-Wayfinder implementing ticket that builds the generator, validator, and prompt loader. No golden data flows from #4 to this spec; the retrieval modes (#10/#11/#12) project into `EvidenceBlock` and stop.
