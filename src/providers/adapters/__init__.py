"""Adapter implementations for the provider layer (#19).

Only one production LLM adapter (Ollama) and one production embedding
adapter (local sentence-transformers) ship with #19, per the spec's
"one working adapter for each" requirement. Switching providers is a
registry-line edit plus env config — application code never imports
this module directly.
"""

from __future__ import annotations

from src.providers.adapters.ollama import OllamaLLM, PROVIDER_KEY as OLLAMA_KEY
from src.providers.adapters.sentence_transformers import (
    MODEL_IDENTITY as SENTENCE_TRANSFORMERS_IDENTITY,
    PROVIDER_KEY as SENTENCE_TRANSFORMERS_KEY,
    REVISION as SENTENCE_TRANSFORMERS_REVISION,
    SentenceTransformersEmbedding,
)


__all__ = [
    "OllamaLLM",
    "OLLAMA_KEY",
    "SentenceTransformersEmbedding",
    "SENTENCE_TRANSFORMERS_KEY",
    "SENTENCE_TRANSFORMERS_IDENTITY",
    "SENTENCE_TRANSFORMERS_REVISION",
]