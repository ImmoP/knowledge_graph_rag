"""Extraction pipeline (#23).

Resolution of `docs/specs/extraction-contract.md`. Reads
``data/derived/chunks.jsonl`` and ``ontology.yaml``, calls the
``LLMProvider`` exactly twice per chunk (entity phase + relation
phase), and writes ``data/extractions.jsonl`` plus
``data/extractions.manifest.json`` (plus ``data/extractions.rejects.jsonl``
for any rejected candidates).

The CLI entry is ``python -m src.extract``. Library callers construct
an ``LLMProvider`` and call :func:`src.extract.pipeline.run_extraction`
directly.
"""

from __future__ import annotations

from src.extract import ids, prompts, schema, validation
from src.extract.pipeline import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_ONTOLOGY_PATH,
    DEFAULT_OUTPUT_JSONL,
    DEFAULT_OUTPUT_MANIFEST,
    DEFAULT_OUTPUT_REJECTS,
    RunResult,
    run_extraction,
)


__all__ = [
    "DEFAULT_CHUNKS_PATH",
    "DEFAULT_ONTOLOGY_PATH",
    "DEFAULT_OUTPUT_JSONL",
    "DEFAULT_OUTPUT_MANIFEST",
    "DEFAULT_OUTPUT_REJECTS",
    "RunResult",
    "run_extraction",
    "ids",
    "prompts",
    "schema",
    "validation",
]