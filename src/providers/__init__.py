"""Provider layer for #19 — the seam between application logic and
external LLM / embedding providers.

Two protocols (:class:`LLMProvider`, :class:`EmbeddingProvider`) plus a
content-addressed cache, a budget wrapper, and a registry of adapters.
Application code consumes only the protocols and the
:func:`make_llm` / :func:`make_embedding` factories; it never imports a
provider SDK directly.
"""

from __future__ import annotations

from src.providers import cache as _cache
from src.providers import validation as _validation
from src.providers.budget import Budget, BudgetExceeded, BudgetedLLM
from src.providers.cache import (
    CACHE_FORMAT_VERSION,
    ProviderCache,
    schema_hash,
)
from src.providers.protocol import (
    EmbeddingProvider,
    LLMProvider,
    LLMValidationError,
    Usage,
)
from src.providers.registry import (
    EMBEDDING_REGISTRY,
    LLM_REGISTRY,
    make_embedding,
    make_llm,
)
from src.providers.validation import validate_provider_config


__all__ = [
    # protocol
    "LLMProvider",
    "EmbeddingProvider",
    "LLMValidationError",
    "Usage",
    # budget
    "Budget",
    "BudgetedLLM",
    "BudgetExceeded",
    # cache
    "ProviderCache",
    "CACHE_FORMAT_VERSION",
    "schema_hash",
    # registry
    "LLM_REGISTRY",
    "EMBEDDING_REGISTRY",
    "make_llm",
    "make_embedding",
    # validation
    "validate_provider_config",
]