"""Chunk preflight for pgvector embedding ingestion (#26, impl-spec §10).

Loads ``data/derived/chunks.jsonl`` and validates every record against
``docs/specs/chunk-record.md`` §2 + §3 + §5. Every required field is
type-checked AND value-checked; malformed records fail closed with
:class:`PreflightError` (the loader never silently coerces a value).

Validation order (one pass over the file):

1. exact required-field set, no extras / omissions;
2. ``chunk_id`` is a string matching ``^c_[0-9a-f]{16}$``;
3. duplicate ``chunk_id`` across the whole file fails;
4. ``doc_id`` is a string matching ``^W[0-9]+$`` (manifest
   membership is an upstream canonical-artifact guarantee per
   chunk-record §1; this check is the local contract);
5. ``section_path`` is a ``list[str]`` with non-empty components;
6. ``chunk_index`` is an ``int`` (``bool`` rejected), ``>= 0``;
7. duplicate ``chunk_index`` within one ``doc_id`` fails;
8. ``text`` is a non-empty ``str``;
9. ``token_estimate`` is an ``int`` (``bool`` rejected), equal to
   ``len(text) // 4``;
10. ``char_start`` and ``char_end`` are ``int`` (``bool`` rejected);
    the canonical escape-hatch semantics from ``src/build_graph``'s
    preflight are enforced — either both are ``-1`` or
    ``char_end > char_start >= 0``;
11. ``chunk_id`` is re-derived under chunk-record §3 from
    ``(doc_id, section_path, chunk_index)`` and required to match the
    record's value byte-for-byte;
12. ``token_estimate`` is loaded for shape parity but is NOT used to
    gate embedding — the **real** BGE tokenizer count via the
    adapter's ``count_tokens`` seam is the independent ``<=512`
    gate (vector-retrieval.md §1.3 last-line enforcement);
13. file records are lexicographically sorted by ``chunk_id``
    (chunk-record §1 contract).

The preflight never imports the ``EmbeddingProvider`` adapter or any
provider SDK. It only inspects the canonical artifact and calls a
caller-supplied ``token_counter`` callable. The loader wires the
adapter's :m `` ``SentenceTransformersEmbedding.count_tokens`` ``;
tests inject a deterministic stand-in.

``data/manifest.jsonl`` membership is intentionally NOT re-checked
here — the canonical chunks file is the source of truth per the
issue spec, and the build_graph preflight already asserts
``chunk.doc_id`` in ``documents`` (#25). Re-reading the manifest here
would add a second canonical-artifact input that the spec forbids
for this ticket.
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
    section_path: list[str]
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
    """chunk-record §2 producer-shape compliance. The persisted field set
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


def _derive_chunk_id(*, doc_id: str, section_path: list[str], chunk_index: int) -> str:
    """Re-derive ``chunk_id`` per chunk-record §3:

    ``"c_" + sha1(f"{doc_id}|{json.dumps(section_path)}|{chunk_index}")[:16]``.

    Re-used here (vs. importing ``src.build_graph.schema.derive_chunk_id``)
    to keep the build_vector preflight standalone — the chunk-record
    spec is a canonical contract, not a build_graph implementation
    detail, and the algorithm is five lines.
    """
    # Match the canonical producer ``src/chunk/chunker.py`` byte-for-byte:
    # ``json.dumps(section_path)`` with its normal defaults — no
    # ``ensure_ascii=False``, no other overrides. ``json.dumps``'s
    # default is ``ensure_ascii=True``, so non-ASCII section
    # names are escaped to ``\uXXXX`` form. Using
    # ``ensure_ascii=False`` here would produce a different SHA1 for
    # any non-ASCII section path and silently corrupt re-derivation.
    payload = f"{doc_id}|{json.dumps(section_path)}|{chunk_index}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return "c_" + digest[:16]


def _is_int_not_bool(value: object) -> bool:
    """``True`` only for ``int`` values that are NOT ``bool``
    (Python's ``bool`` is a subclass of ``int``)."""
    return isinstance(value, int) and not isinstance(value, bool)

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

    # Track chunk_id order for the sort-by-chunk_id contract (chunk-
    # record §1). The validator compares successive chunk_ids and
    # raises if any pair is out of order.
    prev_chunk_id: str | None = None
    seen_ids: set[str] = set()
    # Per-doc chunk_index uniqueness (chunk-record §2: "Two chunks in
    # the same document always have distinct chunk_index values").
    per_doc_indices: dict[str, set[int]] = {}
    records: list[ChunkRecord] = []

    for line_no, raw in enumerate(_read_jsonl(chunks_path), start=1):
        if not isinstance(raw, dict):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: chunk record must be a JSON object"
            )
        _check_exact(raw, _REQUIRED_FIELDS, where=f"chunks.jsonl line {line_no}")

        # 2. chunk_id
        chunk_id = raw["chunk_id"]
        if not isinstance(chunk_id, str) or not CHUNK_ID_RE.match(chunk_id):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: invalid chunk_id {chunk_id!r}"
            )
        # 13. sort-by-chunk_id — check ordering BEFORE duplicate
        # detection so the failure message names the offender pair.
        if prev_chunk_id is not None and chunk_id < prev_chunk_id:
            raise PreflightError(
                f"chunks.jsonl line {line_no}: records not sorted by "
                f"chunk_id ({prev_chunk_id!r} > {chunk_id!r})"
            )
        # 3. duplicate chunk_id
        if chunk_id in seen_ids:
            raise PreflightError(
                f"chunks.jsonl line {line_no}: duplicate chunk_id {chunk_id!r}"
            )
        seen_ids.add(chunk_id)
        prev_chunk_id = chunk_id

        # 4. doc_id (manifest membership is an upstream guarantee
        # — chunk-record §1; the local contract is the regex shape).
        doc_id = raw["doc_id"]
        if not isinstance(doc_id, str) or not DOC_ID_RE.match(doc_id):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: invalid doc_id {doc_id!r}"
            )

        # 5. section_path
        section_path = raw["section_path"]
        if not isinstance(section_path, list) or not all(
            isinstance(component, str) and bool(component)
            for component in section_path
        ):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: section_path must be "
                "list[str] with non-empty components"
            )
        # Snapshot as list[str]; we don't trust the input object beyond
        # this point.
        section_path = [str(c) for c in section_path]

        # 6. chunk_index
        chunk_index = raw["chunk_index"]
        if not _is_int_not_bool(chunk_index) or chunk_index < 0:
            raise PreflightError(
                f"chunks.jsonl line {line_no}: chunk_index must be a "
                "non-negative int"
            )

        # 7. duplicate chunk_index within one doc_id. ``per_doc_indices``
        # is the constant-time structure; membership is the check.
        indices = per_doc_indices.setdefault(doc_id, set())
        if chunk_index in indices:
            raise PreflightError(
                f"chunks.jsonl line {line_no}: duplicate chunk_index "
                f"{chunk_index} within doc_id {doc_id!r}"
            )
        indices.add(chunk_index)

        # 8. text (non-empty str)
        text = raw["text"]
        if not isinstance(text, str) or not text:
            raise PreflightError(
                f"chunks.jsonl line {line_no}: text must be a non-empty str"
            )

        # 9. token_estimate == len(text) // 4 (int, not bool)
        token_estimate = raw["token_estimate"]
        expected_token_estimate = len(text) // 4
        if (
            not _is_int_not_bool(token_estimate)
            or token_estimate != expected_token_estimate
        ):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: token_estimate must equal "
                f"len(text) // 4 ({expected_token_estimate}), "
                f"got {token_estimate!r}"
            )

        # 10. char_start / char_end: int (not bool) + escape-hatch
        char_start = raw["char_start"]
        char_end = raw["char_end"]
        if (
            not _is_int_not_bool(char_start)
            or not _is_int_not_bool(char_end)
        ):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: char_start and char_end "
                "must be integers"
            )
        # Escape-hatch semantics: either BOTH are -1 (the chunker
        # could not compute them) OR char_end > char_start >= 0. This
        # is the exact rule the build_graph preflight enforces.
        if char_start == -1:
            if char_end != -1:
                raise PreflightError(
                    f"chunks.jsonl line {line_no}: char_end must be -1 "
                    "when char_start uses the -1 escape hatch"
                )
        elif not (char_end > char_start >= 0):
            raise PreflightError(
                f"chunks.jsonl line {line_no}: offsets must satisfy "
                "char_end > char_start >= 0 or both be -1"
            )

        # 11. chunk_id re-derivation (chunk-record §3)
        derived = _derive_chunk_id(
            doc_id=doc_id,
            section_path=section_path,
            chunk_index=chunk_index,
        )
        if derived != chunk_id:
            raise PreflightError(
                f"chunks.jsonl line {line_no}: chunk_id {chunk_id!r} "
                f"does not re-derive under chunk-record §3 "
                f"(expected {derived!r})"
            )

        # 12. REAL-tokenizer oversize gate (vector-retrieval §1.3).
        # token_estimate is loaded for shape parity above but is NOT
        # the gate — the pinned BGE tokenizer is.
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
                section_path=section_path,
                chunk_index=chunk_index,
                text=text,
                token_estimate=token_estimate,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return records