"""Tests for ``src.vector_store_validator`` (#26).

Covers impl-spec §10.4 + §10.6:

* Happy-path: ``validate_vector_store`` returns a
  :class:`VectorStoreIdentity` whose
  ``embedding_source_chunks_sha256`` matches the SHA256 of the exact
  ``data/derived/chunks.jsonl`` bytes on disk (not a re-serialization)
  and whose ``embedding_model_identity`` equals the configured
  identity.
* Stale-row scenario: a row whose ``embedding_model`` differs from
  the configured identity makes the validator fail closed with the
  offending ``chunk_id`` named.
* Orphan-row scenario: a row whose ``chunk_id`` is absent from the
  canonical artifact makes the validator fail closed with the
  orphan ``chunk_id`` named.
* Source-text SHA mismatch scenario: a row whose
  ``source_text_sha256`` differs from ``sha256(canonical_chunk.text)``
  is named in the error.
* Public API regression: the production signature is
  ``validate_vector_store(chunks_path, configured_embedding_identity)``
  — the two required positional / keyword inputs only. Tests inject
  deterministic rows via the private ``_validate_vector_store``
  worker, which ``#27`` / ``#33`` do not need to import.
* Static single-source-of-truth guard: the freshness predicates
  appear exactly once in ``src/vector_store_validator.py`` and
  nowhere else in the repo (defends against #27 / #33 re-implementing
  them).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vector_store_validator import (  # noqa: E402
    MAX_TOKENS,
    VectorStoreIdentity,
    VectorStoreValidationError,
    _validate_vector_store,
    validate_vector_store,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Synthetic canonical artifact builders.
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "W100000001",
    section_path: list | None = None,
    chunk_index: int = 0,
    token_estimate: int | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "section_path": section_path or ["Intro"],
        "chunk_index": chunk_index,
        "text": text,
        "token_estimate": token_estimate if token_estimate is not None else len(text) // 4,
        "char_start": 0,
        "char_end": len(text),
    }


def _write_chunks(path: Path, chunks: list[dict]) -> None:
    """Write the canonical chunks artifact with deterministic bytes.

    Using ``json.dumps(..., separators=(",", ":"))`` keeps the bytes
    stable across Python sessions so the file SHA matches across
    processes. The validator hashes the on-disk bytes — never the
    parsed records — so the same artifact produces the same identity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, separators=(",", ":")) + "\n")


def _row(
    chunk_id: str,
    text: str,
    embedding_model: str = "local_sentence_transformers:BAAI/bge-small-en-v1.5:5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "embedding_model": embedding_model,
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


PINED_IDENTITY = (
    "local_sentence_transformers:BAAI/bge-small-en-v1.5:"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)


def _run(rows):
    """Test helper: drive the private ``_validate_vector_store`` with a
    deterministic ``rows_provider``. Production code uses the public
    :func:`validate_vector_store` instead."""
    return _validate_vector_store(
        chunks_path=path,  # set per test
        configured_embedding_identity=PINED_IDENTITY,
        rows_provider=lambda: list(rows),
    )


# Per-test path binding helper so the helper above is not a closure
# of state shared across tests.
class _PerTest:
    def __init__(self):
        self.path: Path | None = None


_pt = _PerTest()


def _validate(rows):
    return _validate_vector_store(
        chunks_path=_pt.path,
        configured_embedding_identity=PINED_IDENTITY,
        rows_provider=lambda: list(rows),
    )


# ---------------------------------------------------------------------------
# Happy-path.
# ---------------------------------------------------------------------------


class HappyPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _pt.path = self.chunks_path
        self.chunks = [
            _chunk("c_0000000000000001", "alpha body"),
            _chunk("c_0000000000000002", "beta body", chunk_index=1),
            _chunk("c_0000000000000003", "gamma body", chunk_index=2),
        ]
        _write_chunks(self.chunks_path, self.chunks)
        self.rows = [_row(c["chunk_id"], c["text"]) for c in self.chunks]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_identity_matches_file_bytes_and_configured_model(self) -> None:
        expected_sha = hashlib.sha256(self.chunks_path.read_bytes()).hexdigest()
        identity = _validate(self.rows)
        self.assertIsInstance(identity, VectorStoreIdentity)
        self.assertEqual(identity.embedding_source_chunks_sha256, expected_sha)
        self.assertEqual(identity.embedding_model_identity, PINED_IDENTITY)

    def test_identity_is_frozen_dataclass(self) -> None:
        identity = _validate(self.rows)
        with self.assertRaises(Exception):
            identity.embedding_model_identity = "tampered"  # type: ignore[misc]

    def test_public_api_call_shape(self) -> None:
        """Production call: ``validate_vector_store`` is invoked with
        exactly the two required inputs (``#27`` and ``#33`` seam)."""
        # The validator imports rows via the production path; we cannot
        # exercise a live pgvector here, but the function must be
        # callable with the public positional / keyword inputs only.
        try:
            validate_vector_store(
                chunks_path=self.chunks_path,
                configured_embedding_identity=PINED_IDENTITY,
            )
        except RuntimeError as exc:
            # Missing PGVECTOR_* env vars -> runtime error from the
            # production row reader, NOT a TypeError about
            # ``rows_provider``. The public surface is what matters.
            self.assertNotIn("rows_provider", str(exc))
            self.assertNotIn("unexpected keyword", str(exc))
        else:
            # If pgvector happened to be reachable + correct shape,
            # this would return a real identity.
            pass  # pragma: no cover - environment-dependent


# ---------------------------------------------------------------------------
# Stale-row scenario (wrong embedding_model).
# ---------------------------------------------------------------------------


class StaleRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _pt.path = self.chunks_path
        self.chunks = [
            _chunk("c_0000000000000001", "alpha"),
            _chunk("c_0000000000000002", "beta"),
        ]
        _write_chunks(self.chunks_path, self.chunks)
        self.rows = [_row(c["chunk_id"], c["text"]) for c in self.chunks]
        # Drift row 2 to a different (wrong) model identity.
        self.rows[1]["embedding_model"] = (
            "local_sentence_transformers:BAAI/bge-small-en-v1.5:wrongrevision"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_wrong_embedding_model_is_named(self) -> None:
        with self.assertRaises(VectorStoreValidationError) as ctx:
            _validate(self.rows)
        self.assertEqual(ctx.exception.predicate, "embedding_model_mismatch")
        self.assertEqual(ctx.exception.offender_ids, ["c_0000000000000002"])
        self.assertIn("configured=", str(ctx.exception))


# ---------------------------------------------------------------------------
# Orphan-row scenario.
# ---------------------------------------------------------------------------


class OrphanRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _pt.path = self.chunks_path
        self.chunks = [_chunk("c_0000000000000001", "only-canonical")]
        _write_chunks(self.chunks_path, self.chunks)
        self.rows = [
            _row("c_0000000000000001", "only-canonical"),
            _row("c_deadbeefdeadbeef", "orphan body"),
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_orphan_chunk_id_is_named(self) -> None:
        with self.assertRaises(VectorStoreValidationError) as ctx:
            _validate(self.rows)
        self.assertEqual(ctx.exception.predicate, "orphan-row")
        self.assertEqual(ctx.exception.offender_ids, ["c_deadbeefdeadbeef"])


# ---------------------------------------------------------------------------
# source_text_sha256 mismatch.
# ---------------------------------------------------------------------------


class SourceShaMismatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _pt.path = self.chunks_path
        self.chunks = [_chunk("c_0000000000000001", "canonical text")]
        _write_chunks(self.chunks_path, self.chunks)
        # Row claims an SHA that does NOT match the canonical text.
        self.rows = [{
            "chunk_id": "c_0000000000000001",
            "embedding_model": PINED_IDENTITY,
            "source_text_sha256": "0" * 64,
        }]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sha_mismatch_is_named(self) -> None:
        with self.assertRaises(VectorStoreValidationError) as ctx:
            _validate(self.rows)
        self.assertEqual(ctx.exception.predicate, "source_text_sha256_mismatch")
        self.assertEqual(ctx.exception.offender_ids, ["c_0000000000000001"])


# ---------------------------------------------------------------------------
# Missing-row (canonical chunk with no pgvector row).
# ---------------------------------------------------------------------------


class MissingRowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.chunks_path = Path(self.tmp.name) / "chunks.jsonl"
        _pt.path = self.chunks_path
        self.chunks = [
            _chunk("c_0000000000000001", "alpha"),
            _chunk("c_0000000000000002", "beta"),
        ]
        _write_chunks(self.chunks_path, self.chunks)
        self.rows = [_row("c_0000000000000001", "alpha")]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_canonical_chunk_is_named(self) -> None:
        with self.assertRaises(VectorStoreValidationError) as ctx:
            _validate(self.rows)
        self.assertEqual(ctx.exception.predicate, "missing-row")
        self.assertEqual(ctx.exception.offender_ids, ["c_0000000000000002"])


# ---------------------------------------------------------------------------
# File-bytes identity (no re-serialization).
# ---------------------------------------------------------------------------


class FileBytesIdentityTests(unittest.TestCase):
    def test_sha_matches_file_bytes_not_parsed_records(self) -> None:
        """The validator's SHA must be ``hashlib.sha256(file_bytes)``,
        not a re-serialization of parsed records. Renormalizing a
        parsed record would yield a different SHA."""
        tmp = tempfile.TemporaryDirectory()
        try:
            chunks_path = Path(tmp.name) / "chunks.jsonl"
            chunks = [_chunk("c_0000000000000001", "alpha")]
            _write_chunks(chunks_path, chunks)
            _pt.path = chunks_path
            expected = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
            identity = _validate([_row("c_0000000000000001", "alpha")])
            self.assertEqual(identity.embedding_source_chunks_sha256, expected)
        finally:
            tmp.cleanup()


# ---------------------------------------------------------------------------
# Static single-source-of-truth guard.
# ---------------------------------------------------------------------------


class SingleSourceOfTruthGuardTests(unittest.TestCase):
    """``vector-retrieval.md`` + impl-spec §10 require the freshness
    predicates (one-row-per-chunk, no orphans, model equality, sha
    equality) to be defined in exactly one place. ``#27`` and ``#33``
    import from this module; they MUST NOT re-implement the
    predicates. The guard greps the repo for the predicate strings
    and asserts they appear only in the validator source (and the
    validator tests, which are allowed to assert on the strings).
    """

    PREDICATES = {
        "orphan-row",
        "embedding_model_mismatch",
        "source_text_sha256_mismatch",
        "missing-row",
    }

    def _iter_repo_sources(self) -> list[Path]:
        skip_dirs = {
            ".venv",
            ".venv312",
            ".git",
            "__pycache__",
            "node_modules",
        }
        out: list[Path] = []
        for p in REPO_ROOT.rglob("*.py"):
            rel = p.relative_to(REPO_ROOT)
            if any(part in skip_dirs for part in rel.parts):
                continue
            if rel.parts and rel.parts[0].startswith((".",)):
                continue
            out.append(p)
        return out

    def test_predicates_appear_only_in_validator_module_and_tests(self) -> None:
        allowed_files = {
            REPO_ROOT / "src" / "vector_store_validator.py",
            REPO_ROOT / "tests" / "test_vector_store_validator.py",
        }
        offenders: list[tuple[Path, str, str]] = []
        for path in self._iter_repo_sources():
            if path in allowed_files:
                continue
            text = path.read_text(encoding="utf-8")
            for predicate in self.PREDICATES:
                if re.search(rf'["\']{re.escape(predicate)}["\']', text):
                    offenders.append((path, predicate, "literal string"))

        if offenders:
            msg = "\n".join(
                f"  {p.relative_to(REPO_ROOT)}: {predicate} ({reason})"
                for p, predicate, reason in offenders
            )
            self.fail(
                "Freshness predicates must be defined exactly once, in\n"
                "src/vector_store_validator.py. Reuse the validator\n"
                "instead of duplicating the predicates:\n" + msg
            )

    def test_predicate_strings_appear_in_validator(self) -> None:
        validator_path = REPO_ROOT / "src" / "vector_store_validator.py"
        text = validator_path.read_text(encoding="utf-8")
        for predicate in self.PREDICATES:
            self.assertIn(
                predicate, text,
                f"predicate {predicate!r} must be defined in vector_store_validator.py"
            )

    def test_signature_is_unchanged(self) -> None:
        """``#27`` and ``#33`` import ``validate_vector_store`` and
        ``VectorStoreIdentity``; the production signature has exactly
        the two required inputs."""
        sig = inspect.signature(validate_vector_store)
        self.assertEqual(
            list(sig.parameters),
            ["chunks_path", "configured_embedding_identity"],
        )


# ---------------------------------------------------------------------------
# MAX_TOKENS guard.
# ---------------------------------------------------------------------------


class MaxTokensGuardTests(unittest.TestCase):
    def test_max_tokens_constant_matches_pinned_spec(self) -> None:
        self.assertEqual(MAX_TOKENS, 512)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()