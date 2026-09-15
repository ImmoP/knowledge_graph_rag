"""In-memory Neo4j driver mock for ``tests/test_build_graph.py``.

The mock records every Cypher statement and supports:

- ``session.run(stmt, **params)`` returning a :class:`FakeResult`.
- ``session.execute_write(fn)`` invoking ``fn`` with a :class:`FakeTx`
  in write mode; CREATE statements populate an in-memory store and
  the in-transaction integrity check queries observe it.
- ``session.execute_read(fn)`` invoking ``fn`` with a :class:`FakeTx`
  in read mode; MATCH...RETURN queries iterate the in-memory store.
- ``session(database=...)`` context-managed.
- A configurable uniqueness-constraint policy that raises on duplicate
  primary keys (acceptance gate #3).
- A configurable mid-run failure injection (acceptance gate #6).

The mock's write store is shared across sessions on the same driver,
mirroring a real database. ``FakeDriver.shared_nodes`` and
``FakeDriver.shared_relationships`` are the global stores a test may
inspect or pre-populate to simulate stale or extra graph content.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class FakeResult:
    def consume(self) -> None:
        return None

    def single(self, *, strict: bool = False):
        return None


class _FakeRecord:
    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._fields.values())[key]
        return self._fields[key]

    def keys(self):
        return list(self._fields.keys())


# Statement patterns.
_LABEL_RE = re.compile(r"\(n:([A-Za-z:]+)\)")
_ANCHOR_FROM_RE = re.compile(r"\{\s*([a-z_]+)\s*:\s*\$from_id\s*\}")
_ANCHOR_TO_RE = re.compile(r"\{\s*([a-z_]+)\s*:\s*\$to_id\s*\}")
_REL_TYPE_RE = re.compile(r"\[r:([A-Z_]+)\]")


def _node_logical_id(props: dict) -> str:
    """Mirror §8.1's label-independent property preference once the
    caller has selected the node class. Chunk IDs precede doc_id
    because Chunk nodes intentionally carry both."""
    for key in ("chunk_id", "entity_id", "doc_id"):
        value = props.get(key)
        if value:
            return str(value)
    return ""
def _node_labels_for_anchor(anchor_key: str) -> list[str]:
    if anchor_key == "doc_id":
        return ["Document"]
    if anchor_key == "chunk_id":
        return ["Chunk"]
    return ["Entity"]


class FakeTx:
    """A single transactional unit. ``write`` toggles capture mode."""

    def __init__(self, parent: "FakeSession", *, write: bool) -> None:
        self.parent = parent
        self.write = write
        self.fail_after = parent.fail_after
        self.statement_count = 0

    def run(self, statement: str, **params: Any) -> "FakeResult":
        self.statement_count += 1
        self.parent.statements.append((statement, params))
        if (
            self.fail_after is not None
            and self.statement_count >= self.fail_after
        ):
            raise RuntimeError(
                "injected mid-transaction failure (acceptance gate #6)"
            )
        upper = statement.strip().upper()
        mutating = (
            upper.startswith("CREATE")
            or "DETACH DELETE" in upper
            or (" CREATE " in upper and "]->(" in statement)
        )
        if self.write and mutating:
            return self.parent._capture_write(statement, params)
        return self.parent._capture_read(statement, params)

    def commit(self) -> None:
        self.parent.committed = True

    def rollback(self) -> None:
        self.parent.rolled_back = True


class FakeSession:
    def __init__(
        self,
        *,
        duplicate_anchor: tuple | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.database = None
        self.statements: list[tuple[str, dict]] = []
        self.committed = False
        self.rolled_back = False
        self.duplicate_anchor = duplicate_anchor
        self.fail_after = fail_after
        self.driver: FakeDriver | None = None  # wired by FakeDriver.session
        self._unique_seen: set[tuple] = set()

    # ---------------------------- WRITE PATH ----------------------------
    def _capture_write(self, statement: str, params: dict) -> "FakeResult":
        driver = self.driver
        if "DETACH DELETE" in statement:
            if driver is not None:
                driver.shared_nodes.clear()
                driver.shared_relationships.clear()
            return FakeResult()
        # CREATE (n:Label1:Label2) SET n = $props
        m = _LABEL_RE.search(statement)
        if m and statement.strip().upper().startswith("CREATE"):
            labels = m.group(1).split(":")
            props = params.get("props", {})
            self._check_unique(labels, props)
            if driver is not None:
                driver.shared_nodes.append(
                    {"labels": labels, "props": dict(props)}
                )
            return FakeResult()
        # MATCH ... CREATE (a)-[r:TYPE]->(b) SET r = $props
        if "MATCH" in statement and "CREATE" in statement and "]->(" in statement:
            type_match = _REL_TYPE_RE.search(statement)
            if not type_match:
                return FakeResult()
            rel_type = type_match.group(1)
            from_id = params.get("from_id", "")
            to_id = params.get("to_id", "")
            props = params.get("props", {})
            from_m = _ANCHOR_FROM_RE.search(statement)
            to_m = _ANCHOR_TO_RE.search(statement)
            from_key = from_m.group(1) if from_m else "doc_id"
            to_key = to_m.group(1) if to_m else "chunk_id"
            from_node = self._find_node(driver, from_key, from_id)
            to_node = self._find_node(driver, to_key, to_id)
            from_labels = list(from_node["labels"]) if from_node else []
            to_labels = list(to_node["labels"]) if to_node else []
            if driver is not None:
                driver.shared_relationships.append(
                    {
                        "type": rel_type,
                        "from_id": str(from_id),
                        "from_labels": from_labels,
                        "to_id": str(to_id),
                        "to_labels": to_labels,
                        "props": dict(props),
                    }
                )
            return FakeResult()
        return FakeResult()

    def _check_unique(self, labels: list[str], props: dict) -> None:
        if self.duplicate_anchor is None:
            return
        label, prop = self.duplicate_anchor
        if label not in labels:
            return
        value = props.get(prop)
        if value is None:
            return
        key = (label, prop, value)
        if key in self._unique_seen:
            raise RuntimeError(
                f"uniqueness constraint violation on {label}.{prop}={value!r}"
            )
        self._unique_seen.add(key)

    def _find_node(
        self, driver: "FakeDriver | None", anchor_key: str, anchor_id: Any
    ) -> dict | None:
        """Resolve a MATCH endpoint exactly as Neo4j would: by the
        anchor property actually named in the statement."""
        if driver is None:
            return None
        for n in driver.shared_nodes:
            if n["props"].get(anchor_key) != anchor_id:
                continue
            if anchor_key == "doc_id" and "Document" not in n["labels"]:
                continue
            return n
        return None

    # ---------------------------- READ PATH -----------------------------
    def _capture_read(self, statement: str, params: dict) -> "FakeResult":
        driver = self.driver
        nodes = list(driver.shared_nodes) if driver is not None else []
        rels = list(driver.shared_relationships) if driver is not None else []

        # MATCH (n) WHERE ... RETURN count(n)
        if "RETURN count(n)" in statement:
            if "Document" in statement:
                count = sum(1 for n in nodes if "Document" in n["labels"])
            elif "Chunk" in statement:
                count = sum(1 for n in nodes if "Chunk" in n["labels"])
            elif "Entity" in statement:
                count = sum(1 for n in nodes if "Entity" in n["labels"])
            else:
                count = len(nodes)
            return FakeScalarResult(count)

        # MATCH (n)-[r:HAS_CHUNK]->(c) ...
        if "HAS_CHUNK" in statement and "OPTIONAL MATCH" in statement:
            bad = 0
            for n in nodes:
                if "Chunk" not in n["labels"]:
                    continue
                target = _node_logical_id(n["props"])
                incoming = sum(
                    1
                    for r in rels
                    if r["type"] == "HAS_CHUNK" and r["to_id"] == target
                )
                if incoming != 1:
                    bad += 1
            return FakeScalarResult(bad)

        # MENTIONS multiplicity by mention_id.
        if "MENTIONS" in statement and "WITH r.mention_id" in statement:
            seen: dict[str, int] = {}
            for r in rels:
                if r["type"] != "MENTIONS":
                    continue
                mid = r["props"].get("mention_id", "")
                seen[mid] = seen.get(mid, 0) + 1
            bad = sum(1 for c in seen.values() if c != 1)
            return FakeScalarResult(bad)

        # Extracted ontology assertion multiplicity.
        if "type(r) IN" in statement and "WITH r.assertion_id" in statement:
            seen = {}
            for r in rels:
                if r["type"] not in {
                    "PROPOSES",
                    "INTRODUCES",
                    "EVALUATES_ON",
                    "USES_METHOD",
                    "REPORTS_METRIC",
                    "EXTENDS",
                    "USED_FOR_TASK",
                    "MEASURES",
                }:
                    continue
                aid = r["props"].get("assertion_id", "")
                seen[aid] = seen.get(aid, 0) + 1
            bad = sum(1 for c in seen.values() if c != 1)
            return FakeScalarResult(bad)

        # MATCH ()-[r:TYPE]->() RETURN count(r)
        m = re.search(r"\[r:([A-Z_]+)\].*RETURN\s+count\(r\)", statement)
        if m:
            rtype = m.group(1)
            count = sum(1 for r in rels if r["type"] == rtype)
            return FakeScalarResult(count)

        # Readback: MATCH (n) ... RETURN labels, logical_id, props
        if "RETURN labels(n)" in statement and "logical_id" in statement:
            return FakeRowsResult(
                [
                    _FakeRecord(
                        labels=n["labels"],
                        logical_id=_node_logical_id(n["props"]),
                        props=n["props"],
                    )
                    for n in nodes
                ]
            )

        # Readback: MATCH (a)-[r]->(b) ... RETURN type(r), ...
        if "RETURN type(r)" in statement:
            return FakeRowsResult(
                [
                    _FakeRecord(
                        rel_type=r["type"],
                        from_id=r["from_id"],
                        from_labels=r["from_labels"],
                        to_id=r["to_id"],
                        to_labels=r["to_labels"],
                        props=r["props"],
                    )
                    for r in rels
                ]
            )
        return FakeScalarResult(0)

    def run(self, statement: str, **params: Any) -> "FakeResult":
        # Top-level session.run (no transaction).
        self.statements.append((statement, params))
        return self._capture_read(statement, params)

    @contextmanager
    def begin_transaction(self):
        tx = FakeTx(parent=self, write=True)
        try:
            yield tx
        except Exception:
            if not tx.parent.committed:
                tx.parent.rolled_back = True
            raise

    def execute_write(self, fn: Callable[[FakeTx], None]) -> None:
        tx = FakeTx(parent=self, write=True)
        try:
            fn(tx)
            if not tx.parent.rolled_back and not tx.parent.committed:
                tx.commit()
        except Exception:
            tx.rollback()
            raise

    def execute_read(self, fn: Callable[[FakeTx], None]) -> None:
        tx = FakeTx(parent=self, write=False)
        fn(tx)


class FakeScalarResult(FakeResult):
    def __init__(self, value: int) -> None:
        self._value = value

    def single(self, *, strict: bool = False):
        if strict:
            return _FakeRecord(c=self._value)
        return self._value

    def __iter__(self):
        yield self


class FakeRowsResult(FakeResult):
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self, *, strict: bool = False):
        if strict and len(self._rows) != 1:
            raise RuntimeError(
                f"single(strict=True) expected exactly one row, got {len(self._rows)}"
            )
        return self._rows[0] if self._rows else None


class FakeDriver:
    """Top-level driver.

    Shares the same in-memory node and relationship stores across all
    sessions, mirroring a real database. The stores are publicly
    accessible so a test may pre-populate or post-mutate to simulate
    stale / extra / missing graph content.
    """

    def __init__(
        self,
        *,
        duplicate_anchor: tuple | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.duplicate_anchor = duplicate_anchor
        self.fail_after = fail_after
        self.sessions: list[FakeSession] = []
        self.shared_nodes: list[dict] = []
        self.shared_relationships: list[dict] = []

    @contextmanager
    def session(self, database: str | None = None):
        session = FakeSession(
            duplicate_anchor=self.duplicate_anchor,
            fail_after=self.fail_after,
        )
        session.database = database
        session.driver = self
        self.sessions.append(session)
        yield session