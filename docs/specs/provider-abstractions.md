# Provider Abstraction Specification

Resolution of [#16 — Define LLM and embedding provider abstractions](https://github.com/ImmoP/knowledge_graph_rag/issues/16). This document is the **contract** the implementing provider layer reads. It is not the implementing ticket — that goes to `/to-tickets` once the map is cleared.

The standing rationale (recorded on the map and on #16): a model provider is a *deployment* choice, not an architecture choice. Application code never imports a provider SDK; switching providers is a config change plus an adapter, not a code change. This spec fixes the seam that makes that property hold.

## 1. Two interfaces, one purpose each

There are exactly two protocols, and they are not unified.

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None  # None when the provider does not expose cost

class LLMProvider(Protocol):
    name: str   # e.g. "anthropic", "openai", "ollama"
    model: str  # provider-specific model identifier

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, Usage]: ...

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[dict, Usage]: ...

class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int  # dimensionality of returned vectors; pinned for the lifetime of the model

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]: ...
```

**Why two interfaces:** they have different return shapes, different batching behavior, different cost dimensions. Unifying them produces an "AIProvider" that lies about its homogeneity.

**Why `Protocol` and not ABC:** structural typing — any adapter class that has the right methods satisfies the protocol without inheriting from a base class. Adapter modules don't need to import this spec to be a valid provider; they only need to define the methods. `Protocol` is the lighter Python idiom and matches the codebase's overall minimalism.

**Why `Usage` is a frozen dataclass:** it carries data, not behavior, and it is the budget-cap signaling channel. Freezing it makes the accounting audit-friendly (no accidental mutation by callers).

**Return-type conventions:**
- `generate` returns `(text, usage)`. The string is the model's natural-language response; it is *not* a JSON object even for prompts that ask for JSON.
- `generate_json` returns `(parsed_dict, usage)`. Parsing and validation happen *inside* the adapter or its wrapping helper. Callers receive a validated dict or get a typed exception.
- `embed` returns `(vectors, usage)`. Vectors are always `list[float]` of length `self.dim`. The adapter is responsible for ensuring this — adapters that batch differently must unbatch before returning.

## 2. `generate_json` semantics

Structured extraction needs validated JSON output. The contract:

1. **Adapter responsibility.** The adapter either uses the provider's native structured-output feature (e.g. Anthropic tool use, OpenAI `response_format=json_schema`) or constructs the prompt so that a clean JSON object is requested.
2. **Validation.** The adapter validates the parsed dict against the provided `schema` (a JSON-Schema-flavoured dict). If the provider returns invalid output, the adapter retries exactly once with a "respond with valid JSON matching this schema" suffix.
3. **Failure mode.** After one retry, if validation still fails, the adapter raises `LLMValidationError(model_output, schema, validation_error)`. The caller catches this and decides whether to fail the run or skip the input. There is no silent fallback.
4. **Schema is opaque to `LLMProvider`.** Schemas are provided by the calling ticket (e.g. extraction ticket passes a JSON Schema for the entity/relation shape). The provider does not know about the ontology.

## 3. Configuration

Provider/model selection is configuration-driven. The implementing ticket reads the following environment variables. Names and shapes are pinned by this spec.

### 3.1 Required

| Variable | Type | Description |
|---|---|---|
| `LLM_PROVIDER` | string | Registry key for the `LLMProvider` adapter to use. E.g. `anthropic`, `openai`, `ollama`. |
| `LLM_MODEL` | string | Model identifier passed to the adapter. E.g. `claude-haiku-4-5`, `gpt-4o-mini`, `llama3.1:8b`. |
| `EMBEDDING_PROVIDER` | string | Registry key for the `EmbeddingProvider` adapter. |
| `EMBEDDING_MODEL` | string | Embedding model identifier. |

### 3.2 Optional with defaults

| Variable | Type | Default | Description |
|---|---|---|---|
| `LLM_BUDGET_USD` | float | unset | Soft cap on cumulative LLM cost in USD. Honored only when the active provider returns `cost_usd`. |
| `LLM_BUDGET_TOKENS` | int | unset | Soft cap on cumulative input+output tokens. Used when `LLM_BUDGET_USD` is unset or the provider has no cost data. |
| `LLM_CACHE_DIR` | path | `data/derived/llm_cache/` | Cache root. Per-provider/model subdirectories are created underneath. |
| `LLM_PROVIDER_TIMEOUT_S` | float | 60 | Per-call HTTP timeout for the active provider. |
| `EMBEDDING_BATCH_SIZE` | int | 64 | Max texts per `embed()` call. Adapters may override. |

### 3.3 Provider-specific secrets

Provider API keys live in environment variables. The implementing ticket (which writes `.env.example`) lists each provider's required keys under a clearly-commented section, not as a flat list that mixes providers:

```dotenv
# .env.example (after this ticket's implementing code lands)
# ---- LLM Provider (set EXACTLY ONE block; leave the rest blank) ----
# Anthropic:
# ANTHROPIC_API_KEY=
# OpenAI:
# OPENAI_API_KEY=
# Ollama (local; no key needed):
# OLLAMA_BASE_URL=http://localhost:11434
# ---- Embedding Provider ----
# (same pattern as LLM)
```

Mismatched keys are not an error at module load — they only fail when the adapter is actually invoked.

### 3.4 Validation at startup

The implementing ticket writes a `validate_provider_config()` function that runs at the entry point of every script that uses an LLM. It checks that:

- `LLM_PROVIDER` is a registered adapter.
- The adapter's required secrets are present in the environment.
- `LLM_MODEL` is non-empty and a string.

It does **not** check that the model is actually reachable; reachability is the first call's problem (which will surface a clear connection error).

## 4. Adapter registry

The implementing ticket ships a single registry module:

```python
# Pseudocode; final structure is the implementing ticket's call.

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

def make_llm() -> LLMProvider:       ...   # reads env, instantiates, returns
def make_embedding() -> EmbeddingProvider: ...
```

**Adding a new provider** is:
1. Write the adapter class implementing the protocol.
2. Add one line to the appropriate registry.
3. Add the provider's required env vars to `.env.example` under a commented section.

No other code changes. Extraction, generation, and retrieval business logic does not learn that the new provider exists.

## 5. Cache

The cache is a property of the abstraction layer, not of the call site.

### 5.1 Key shape

For `generate` / `generate_json`:

```
sha256(f"{provider.name}|{provider.model}|{sha256(system + chr(0x1f) + user)}")
```

For `embed`:

```
sha256(f"{provider.name}|{provider.model}|{sha256(chr(0x1f).join(texts))}")
```

`chr(0x1f)` (unit separator) is used as the inner delimiter because it cannot appear in valid UTF-8 text content. The outer `provider.name` / `provider.model` ensures that switching providers or models cannot reuse a stale cached entry.

### 5.2 Stored value

For `generate`: the raw text response.
For `generate_json`: the parsed dict.
For `embed`: the list-of-lists vector payload.

Cache files are content-addressed JSON under `${LLM_CACHE_DIR}/{llm,embeddings}/<key>.json`. The implementing ticket chooses JSON for portability; binary formats (msgpack, parquet) are valid alternatives if implemented.

### 5.3 Cache invalidation

- **Provider or model change** → new cache key namespace → no contamination.
- **Prompt change** → new key.
- **Adapter code change** (e.g. parsing rule changed) → the implementing ticket bumps a per-adapter `cache_format_version` constant, which is mixed into the key. Old cache entries become orphaned (cheap; just deleted from disk) but never silently returned for the new format.

### 5.4 What is NOT cached

- Budget accounting state (per-process).
- Provider reachability checks.
- Validation retry history.

## 6. Budget caps

A budget cap is a soft limit that aborts the run when crossed. The contract:

1. **Cap reading.** `validate_provider_config()` reads `LLM_BUDGET_USD` and `LLM_BUDGET_TOKENS` and stores them on the active adapter (or on a wrapper that decorates the adapter).
2. **Per-call accounting.** After every `generate` / `generate_json` / `embed` call, the adapter reports `Usage`. The wrapper adds to its cumulative counters.
3. **Cap check.** After each call, the wrapper checks `cumulative_cost_usd >= LLM_BUDGET_USD` (if cap set and provider returns cost) or `cumulative_tokens >= LLM_BUDGET_TOKENS` (if cap set). If exceeded, the wrapper raises `BudgetExceeded(cumulative=..., cap=..., unit="USD"|"tokens")`.
4. **Clean abort.** The calling script catches `BudgetExceeded`, logs the partial run, and exits with a non-zero code. No partial state is left in stores that would mislead later runs.
5. **No cost is not an error.** A provider that returns `cost_usd = None` simply skips the USD-cap check. The token cap still applies.

### 6.1 Units and normalization

Providers report cost in USD (after the implementing ticket's adapter converts from any provider-native currency). Token counts are reported as integers; providers that bill in different units (e.g. characters) must convert to tokens in the adapter (using the chunk-record spec's `len(text) // 4` estimator as a fallback if no precise tokenizer is available).

## 7. Default provider choice

The spec **does not pin a specific default provider**. The architecture-default is: whichever provider `LLM_PROVIDER` env var names at first-run time. This is the right shape because:

- A pinned default would lock the project to a specific provider's account/key status at the moment of writing.
- The user's deployment may differ from the implementer's.
- The whole point of the abstraction is that the default is a *runtime configuration concern*, not an architecture concern.

**What the implementing ticket must ship:** one working adapter for each of `LLMProvider` and `EmbeddingProvider`, plus the registry wiring. The cheapest-to-demonstrate choice at implementation time depends on which provider the user has a key for; the implementing ticket asks the user once and registers that one first. Anthropic is a reasonable first pick (the project already mentions it in `.env.example`), and a local sentence-transformers model is a reasonable first embedding pick (zero cost, fully reproducible). Both choices are interchangeable — that is the architecture property under test.

### 7.1 Quality/cost bar

The one working adapter must clear a smoke test:

1. `generate("reply with exactly: pong", "ping")` returns a string containing `pong`.
2. `generate_json("reply with JSON", "{...}", schema={"type": "object", "required": ["answer"]})` returns a dict with key `answer`.
3. `embed(["hello", "world"])` returns 2 vectors each of length `dim` with non-zero values.

If the smoke test fails, the implementing ticket does not merge. This is a runnable gate.

## 8. What this spec leaves open (implementing ticket owns)

These are explicitly not part of the architecture contract and are the implementing ticket's responsibility:

- **Implementations of specific adapters** (Anthropic, OpenAI, Ollama, sentence-transformers). The spec pins the interface; the ticket writes the bodies.
- **Logging and observability** — what gets logged on each call, whether usage is persisted to a sidecar file for post-hoc accounting, etc.
- **Concurrency model** — sequential is the default; parallel batch embedding requires the implementing ticket to add a flag and reason about ordering in the cache.
- **Retry policy beyond the single validation retry** — if a provider returns 5xx, the adapter may retry with exponential backoff. The spec does not pin the exact policy.
- **Streaming responses** — out of scope for the first implementation. Batch only.
- **`.env.example` layout** — the section structure above is a guide, not a pinned format. The implementing ticket chooses the exact text.

## 9. Acceptance gate

The implementing ticket closes only when all of these pass:

1. **Two adapter registrations** exist in the registries, one for `LLMProvider` and one for `EmbeddingProvider`, each with at least one concrete implementation.
2. **Smoke test (§7.1) passes** for both adapters.
3. **Cache key uniqueness:** switching `LLM_PROVIDER` between two registered adapters produces disjoint cache namespaces (no shared keys). Switching `LLM_MODEL` does likewise.
4. **Budget cap triggers:** a script that sets `LLM_BUDGET_TOKENS=10` and calls `generate` twice with prompts larger than 10 tokens total raises `BudgetExceeded` after the second call.
5. **No provider SDK import** appears outside the adapter modules — verified by `grep -rE "import (anthropic|openai|ollama)" src/` returning matches only under `src/providers/` (or wherever the implementing ticket places adapters).

---

This spec resolves ticket #16. It is referenced by #7 (extraction uses `LLMProvider.generate_json` against an ontology-derived schema), #10 (vector retrieval uses `EmbeddingProvider.embed`), #13 (grounded generation uses `LLMProvider.generate` with a citation-validation step in the calling code), and the implementing ticket for the provider layer itself.