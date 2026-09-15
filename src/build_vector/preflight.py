"""Chunk preflight for pgvector embedding ingestion (#26, impl-spec §10).

Loads ``data/derived/chunks.jsonl``, validates every record against the
chunk-record §3 schema, and fails closed on any oversize chunk.

The oversize check uses a **real-tokenizer count** of the chunk text
(via the adapter's ``count_tokens`` seam) — vector-retrieval.md §1.3
forbids approximations, so the chunk-record field ``token_estimate``
is NOT used for this gate. ``token_estimate`` is still loaded and
shape-validated; the loader uses it for nothing else.

The preflight never imports the ``EmbeddingProvider`` adapter or any
provider SDK. It only inspects the canonical artifact and calls a
caller-supplied ``token_counter`` callable. The loader wires the
adapter's :meth:`SentenceTransformersEmbedding.count_tokens`; tests
inject a deterministic stand-in.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from src.vector_store_validator import MAX_TOKENS


CHUNK_ID_RE = re.compile(r"^c_[0-9a-f]{16}$")
DOC_ID_RE = re.compile(r"^W[0-9]+$")

_REQUIRED_FIELDS = frozenset(
    {"chunk_id", "doc_id", "section_path", "chunk_index", "text",
     "token_estimate", "char_start", "char_end"}
)


class PreflightError(RuntimeError):
    """Raised when the chunks preflight fails. The pgvector ``chunks``
    table is not touched; no partial state survives."""


class OversizeChunkError(PreflightError):
    """A chunk's REAL tokenizer count exceeds the 512-token budget.

    The count comes from the pinned BGE tokenizer via the loader's
    ``token_counter`` callable (vector-retrieval.md §1.3). Surfaces
    ``chunk_id`` + actual measured token count + ``token_estimate``
    for diagnostic clarity. The loader passes
    :meth:`SentenceTransformersEmbedding.count_tokens`; tests inject a
    deterministic stand-in.
    """

    def __init__(
        self,
        *,
        chunk_id: str,
        token_count: int,
        token_estimate: int,
    ) -> None:
        super().__init__(
            f"oversize chunk: chunk_id={chunk_id} "
            f"token_count={token_count} token_estimate={token_estimate} "
            f"budget={MAX_TOKENS}"
        )
        self.chunk_id = chunk_id
        self.token_count = token_count
        self.token_estimate = token_estimate


# ---------------------------------------------------------------------------
# Record dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkRecord:
    """A validated chunk-record §3 row from the canonical artifact.

    Carries everything the loader needs to embed + write a pgvector
    row, plus the SHA256 of the chunk text at preflight time so the
    per-row ``source_text_sha256`` matches the canonical chunk-record
    byte stream (impl-spec §10.3 column ``source_text_sha256``).
    """

    chunk_id: str
    doc_id: str
    section_path: list
    chunk_index: int
    text: str
    token_estimate: int
    char_start: int
    char_end: int

    @property
    def source_text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Path defaults — mirrors ``src.build_graph.preflight``.
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "derived" / "chunks.jsonl"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise PreflightError(f"chunks file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreflightError(
                    f"chunks.jsonl line {line_no}: invalid JSON: {exc}"
                ) from exc


def _check_exact(record: dict, required: frozenset, *, where: str) -> None:
    """chunk-record §3 producer-shape compliance. The persisted field set
    must match the required set exactly — no missing, no extra."""
    keys = set(record)
    missing = required - keys
    extra = keys - required
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing={sorted(missing)}")
        if extra:
            problems.append(f"extra={sorted(extra)}")
        raise PreflightError(f"{where}: {'; '.join(problems)}")


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------


def run_preflight(
    chunks_path: Path,
    token_counter: Callable[[str], int],
) -> list[ChunkRecord]:
    """Load + validate every chunk in the canonical artifact.

    Token-counter contract: ``token_counter(text)`` returns the number
    of model input tokens the pinned BGE tokenizer produces for
    ``text`` (including its normal special-token treatment). The
    counter MUST NOT silently truncate; it returns the real count of
    the full input. The loader passes
    :meth:`SentenceTransformersEmbedding.count_tokens`; tests inject
    a deterministic stand-in. No provider SDK import is performed
    here.

    Returns the parsed :class:`ChunkRecord` list (frozen, in file
    order). Fails closed on any schema or oversize violation; the
    pgvector store is not touched.
    """
    if not callable(token_counter):
        raise TypeError("run_preflight requires a callable token_counter")

    records: list[ChunkRecord] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(_read_jsonl(chunks_path), start=1):
        _check_exact(raw, _REQUIRED_FIELDS, where=f"chunks.jsonl line {line_no}")

        chunk_id = raw["chunk_id"]
        if not isinstance(chunk_id, str) or not CHUNK_ID_RE.match(chunk_id):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: invalid chunk_id {chunk_id!r}"
            )
        if chunk_id in seen_ids:
            raise PreflightError(
                f"chunks.jsonl line {line_no}: duplicate chunk_id {chunk_id!r}"
            )
        seen_ids.add(chunk_id)

        doc_id = raw["doc_id"]
        if not isinstance(doc_id, str) or not DOC_ID_RE.match(doc_id):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: invalid doc_id {doc_id!r}"
            )

        text = raw["text"]
        if not isinstance(text, str):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: text must be str"
            )

        token_estimate = raw["token_estimate"]
        if not isinstance(token_estimate, int):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: token_estimate must be int"
            )

        # vector-retrieval.md §1.3 last-line enforcement: measure with
        # the pinned BGE tokenizer, NOT the chunk-record approximation.
        # ``token_estimate`` is loaded for shape parity but NOT used
        # to gate the embedding.
        token_count = token_counter(text)
        if token_count > MAX_TOKENS:
            raise OversizeChunkError(
                chunk_id=chunk_id,
                token_count=token_count,
                token_estimate=token_estimate,
            )

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section_path=list(raw["section_path"]),
                chunk_index=int(raw["chunk_index"]),
                text=text,
                token_estimate=token_estimate,
                char_start=int(raw["char_start"]),
                char_end=int(raw["char_end"]),
            )
        )
    return records