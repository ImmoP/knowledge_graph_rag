"""Provider registry and factory functions (#19).

Resolution of ``docs/specs/provider-abstractions.md`` §4.

Adding a new provider is: write the adapter, add one registry entry,
add the adapter's required env vars to ``.env.example``. No business
logic learns that a new provider exists.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.providers.budget import Budget, BudgetedLLM
from src.providers.cache import CACHE_FORMAT_VERSION as _GLOBAL_CACHE_FORMAT_VERSION
from src.providers.cache import CachePaths, ProviderCache
from src.providers.protocol import EmbeddingProvider, LLMProvider
from src.providers.validation import (
    ENV_EMBEDDING_BATCH_SIZE,
    ENV_EMBEDDING_PROVIDER,
    ENV_LLM_CACHE_DIR,
    ENV_LLM_MODEL,
    ENV_LLM_PROVIDER,
    ENV_LLM_PROVIDER_TIMEOUT_S,
    default_cache_dir,
)


# LLM_REGISTRY: name -> adapter class. Spec §4. Ollama is the one
# production adapter shipping with #19 (the spec does not pin two; the
# acceptance nuance notes cache-namespace isolation must be provable
# generically through test injection).
LLM_REGISTRY: dict[str, type[LLMProvider]] = {}

# EMBEDDING_REGISTRY: name -> adapter class. Spec §4.
EMBEDDING_REGISTRY: dict[str, type[EmbeddingProvider]] = {}


def _register_builtins() -> None:
    """Populate the registries with the adapters shipped in this package.

    Imports happen inside the function so module import does not pay
    the sentence-transformers cost for tests that never exercise the
    embedding adapter.
    """
    if not LLM_REGISTRY:
        from src.providers.adapters.ollama import OllamaLLM, PROVIDER_KEY as OLLAMA_KEY

        LLM_REGISTRY[OLLAMA_KEY] = OllamaLLM
    if not EMBEDDING_REGISTRY:
        from src.providers.adapters.sentence_transformers import (
            PROVIDER_KEY as ST_KEY,
            SentenceTransformersEmbedding,
        )

        EMBEDDING_REGISTRY[ST_KEY] = SentenceTransformersEmbedding


_register_builtins()


# --- cache helper ----------------------------------------------------


def _build_cache(env: dict | None, *, cache_format_version: str | None = None) -> ProviderCache:
    src = env if env is not None else dict(os.environ)
    raw = src.get(ENV_LLM_CACHE_DIR)
    root = Path(raw) if raw else default_cache_dir()
    return ProviderCache(
        CachePaths.from_root(root),
        cache_format_version=cache_format_version or _GLOBAL_CACHE_FORMAT_VERSION,
    )


# --- factories -------------------------------------------------------


def _budget_from_env(env: dict) -> Budget:
    """Read the optional budget caps into a fresh :class:`Budget`."""
    usd_raw = env.get("LLM_BUDGET_USD")
    tokens_raw = env.get("LLM_BUDGET_TOKENS")
    usd_cap = float(usd_raw) if usd_raw not in (None, "") else None
    tokens_cap = int(float(tokens_raw)) if tokens_raw not in (None, "") else None
    return Budget(usd_cap=usd_cap, tokens_cap=tokens_cap)


def make_llm(env: dict | None = None, *, with_budget: bool = True) -> LLMProvider:
    """Build the configured ``LLMProvider``.

    Reads ``LLM_PROVIDER`` and ``LLM_MODEL`` (and the optional cache /
    timeout / budget env vars) and returns the registered adapter,
    wrapped in :class:`BudgetedLLM` if any budget cap is set. When
    ``with_budget`` is ``False``, the wrapper is skipped — tests use
    this to drive the bare adapter.
    """
    _register_builtins()
    src = env if env is not None else dict(os.environ)
    provider_name = src.get(ENV_LLM_PROVIDER, "")
    model = src.get(ENV_LLM_MODEL, "")
    if not provider_name or provider_name not in LLM_REGISTRY:
        raise KeyError(f"LLM provider {provider_name!r} not registered")
    adapter_cls = LLM_REGISTRY[provider_name]

    kwargs: dict = {"model": model}
    adapter_cache_version = getattr(adapter_cls, "CACHE_FORMAT_VERSION", None)
    if provider_name == "ollama":
        timeout_raw = src.get(ENV_LLM_PROVIDER_TIMEOUT_S)
        if timeout_raw not in (None, ""):
            kwargs["timeout_s"] = float(timeout_raw)
        kwargs["cache"] = _build_cache(src, cache_format_version=adapter_cache_version)
    adapter = adapter_cls(**kwargs)

    if not with_budget:
        return adapter
    budget = _budget_from_env(src)
    if budget.usd_cap is None and budget.tokens_cap is None:
        return adapter
    return BudgetedLLM(adapter, budget)


def make_embedding(env: dict | None = None) -> EmbeddingProvider:
    """Build the configured ``EmbeddingProvider``."""
    _register_builtins()
    src = env if env is not None else dict(os.environ)
    provider_name = src.get(ENV_EMBEDDING_PROVIDER, "")
    model = src.get("EMBEDDING_MODEL", "")
    if not provider_name or provider_name not in EMBEDDING_REGISTRY:
        raise KeyError(f"Embedding provider {provider_name!r} not registered")
    adapter_cls = EMBEDDING_REGISTRY[provider_name]

    kwargs: dict = {"model_id": model}
    batch_raw = src.get(ENV_EMBEDDING_BATCH_SIZE)
    if batch_raw not in (None, ""):
        kwargs["batch_size"] = int(float(batch_raw))
    kwargs["cache"] = _build_cache(
        src, cache_format_version=getattr(adapter_cls, "CACHE_FORMAT_VERSION", None)
    )
    return adapter_cls(**kwargs)


def reset_for_tests() -> None:
    """Clear registries. Tests that inject providers call this between cases."""
    LLM_REGISTRY.clear()
    EMBEDDING_REGISTRY.clear()