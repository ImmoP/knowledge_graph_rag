"""ID re-derivation harness (#23).

Implements ``docs/specs/extraction-contract.md`` §15.9: a tripwire
against any future regression that re-introduces a position-dependent
ID rule. For every ``mention`` and ``assertion`` record in
``data/extractions.jsonl``, recompute the ID from the record's own
fields and assert equality. Clean-run and resumed-run common records
must have byte-identical IDs.

The harness is a pure helper (no I/O of its own); callers feed it a
sequence of records and a kind label. The implementing ticket ships
both a programmatic entry point and a small CLI that reads
``data/extractions.jsonl`` and exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from src.extract import ids as ids_module


@dataclass(frozen=True)
class IdDerivationError(RuntimeError):
    """Raised when a record's stored ID disagrees with the re-derived one."""

    record_kind: str
    chunk_id: str
    stored_id: str
    derived_id: str

    def __post_init__(self):
        super().__init__(
            f"{self.record_kind} for {self.chunk_id!r}: stored "
            f"{self.stored_id!r} != derived {self.derived_id!r}"
        )


def verify_records(records: list[dict]) -> int:
    """Re-derive every ID in ``records`` and report any mismatch.

    Records are JSONL-shaped dicts with a ``kind`` field. The function
    is generic over both ``mention`` and ``assertion`` records and
    ignores non-record lines (e.g. ``chunk_status``).

    Returns the count of records verified. Raises
    :class:`IdDerivationError` on the first mismatch found (the spec
    calls for assertion equality, not collection; any single failure
    invalidates the file).
    """
    verified = 0
    for rec in records:
        kind = rec.get("kind")
        if kind == "mention":
            stored = rec.get("mention_id")
            derived = ids_module.rederive_mention_id(rec)
            if stored != derived:
                raise IdDerivationError(
                    record_kind="mention",
                    chunk_id=rec.get("chunk_id", "<unknown>"),
                    stored_id=stored or "<missing>",
                    derived_id=derived,
                )
            verified += 1
        elif kind == "assertion":
            stored = rec.get("assertion_id")
            derived = ids_module.rederive_assertion_id(rec)
            if stored != derived:
                raise IdDerivationError(
                    record_kind="assertion",
                    chunk_id=rec.get("chunk_id", "<unknown>"),
                    stored_id=stored or "<missing>",
                    derived_id=derived,
                )
            verified += 1
    return verified


def main(argv: list[str] | None = None) -> int:
    """CLI entry: re-derive every ID in the JSONL at ``argv[1]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m src.extract.verify <path/to/extractions.jsonl>", file=sys.stderr)
        return 2
    path = Path(args[0])
    if not path.is_file():
        print(f"verify: input file not found: {path}", file=sys.stderr)
        return 2
    import json

    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    try:
        n = verify_records(records)
    except IdDerivationError as exc:
        print(f"verify: {exc}", file=sys.stderr)
        return 1
    print(f"verify: {n} id-bearing record(s) re-derived OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())