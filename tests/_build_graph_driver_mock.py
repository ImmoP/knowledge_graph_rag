"""In-memory Neo4j driver mock for ``tests/test_build_graph.py``.

The mock records every Cypher statement and supports:

- ``session.run(stmt, **params)`` returning a :class:`FakeResult` whose
  ``.consume()`` is a no-op.
- ``session.execute_write(fn)`` invoking ``fn`` with a :class:`FakeTx`.
- ``session(database=...)`` context-managed.
- A configurable uniqueness-constraint policy that raises on duplicate
  primary keys (verifies acceptance gate #3). The driver accepts an
  optional ``pre_seed`` tuple ``(label, prop, value)``; the FIRST session
  it produces will treat that value as already-present, so the first
  CREATE statement in the transaction collides with it.
- A configurable post-write rollback simulation: pass ``fail_after=N``
  to inject an exception after ``N`` statements have run inside the
  transaction, exercising the §7 rollback path.
- ``last_graph_content_sha256()`` returning the SHA the orchestrator
  verifies against (acceptance gate §8.2 post-commit read).

The mock is intentionally narrow; it tracks only what the tests need
to assert and never claims to be a Neo4j emulation.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class FakeResult:
    def consume(self) -> None:
        return None

    def single(self, *, strict: bool = False):
        return None


class FakeTx:
    def __init__(self, parent: "FakeSession", fail_after: int | None) -> None:
        self.parent = parent
        self.fail_after = fail_after
        self.statement_count = 0

    def run(self, statement: str, **params: Any) -> FakeResult:
        self.statement_count += 1
        self.parent.statements.append((statement, params))
        if (
            self.fail_after is not None
            and self.statement_count >= self.fail_after
        ):
            raise RuntimeError(
                "injected mid-transaction failure (acceptance gate #6)"
            )
        if statement.strip().upper().startswith("CREATE"):
            self.parent._enforce_unique_create(statement, params)
        return FakeResult()

    def commit(self) -> None:
        self.parent.committed = True

    def rollback(self) -> None:
        self.parent.rolled_back = True


class FakeSession:
    def __init__(
        self,
        *,
        duplicate_anchor: tuple[str, str] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.database = None
        self.statements: list[tuple[str, dict]] = []
        self.committed = False
        self.rolled_back = False
        self.duplicate_anchor = duplicate_anchor
        self.fail_after = fail_after
        self.nodes_by_anchor: dict[tuple[str, str], dict] = {}

    def _enforce_unique_create(self, statement: str, params: dict) -> None:
        if self.duplicate_anchor is None:
            return
        label, prop = self.duplicate_anchor
        if (
            f":{label}" not in statement
            and not statement.startswith(f"CREATE (n:{label}")
        ):
            return
        # The mock sees ``params`` as ``{"props": {...}}`` because the
        # production code passes its binding dict as a single kwarg.
        value = params.get(prop)
        if value is None:
            props = params.get("props")
            if isinstance(props, dict):
                value = props.get(prop)
        if value is None:
            return
        key = (label, prop)
        if key in self.nodes_by_anchor:
            raise RuntimeError(
                f"uniqueness constraint violation on {label}.{prop}={value!r}"
            )
        self.nodes_by_anchor[key] = params

    def run(self, statement: str, **params: Any) -> FakeResult:
        self.statements.append((statement, params))
        return FakeResult()

    @contextmanager
    def begin_transaction(self):
        tx = FakeTx(parent=self, fail_after=self.fail_after)
        try:
            yield tx
        except Exception:
            if not tx.parent.committed:
                tx.parent.rolled_back = True
            raise

    def execute_write(self, fn: Callable[[FakeTx], None]) -> None:
        tx = FakeTx(parent=self, fail_after=self.fail_after)
        try:
            fn(tx)
            if not tx.parent.rolled_back and not tx.parent.committed:
                tx.commit()
        except Exception:
            tx.rollback()
            raise


class FakeDriver:
    """Top-level driver. ``duplicate_anchor`` and ``fail_after`` are
    forwarded to every new session. ``pre_seed`` (label, prop, value)
    pre-populates the FIRST session's ``nodes_by_anchor`` so the first
    CREATE in the transaction collides with it."""

    def __init__(
        self,
        *,
        duplicate_anchor: tuple[str, str] | None = None,
        fail_after: int | None = None,
        projected_sha256: str | None = None,
        pre_seed: tuple[str, str, str] | None = None,
    ) -> None:
        self.duplicate_anchor = duplicate_anchor
        self.fail_after = fail_after
        self.projected_sha256 = projected_sha256
        self.pre_seed = pre_seed
        self.sessions: list[FakeSession] = []
        self._seeded = False

    @contextmanager
    def session(self, database: str | None = None):
        session = FakeSession(
            duplicate_anchor=self.duplicate_anchor,
            fail_after=self.fail_after,
        )
        session.database = database
        if self.pre_seed is not None and not self._seeded:
            label, prop, value = self.pre_seed
            session.nodes_by_anchor[(label, prop)] = {prop: value}
            self._seeded = True
        self.sessions.append(session)
        yield session

    def last_graph_content_sha256(self) -> str:
        if self.projected_sha256 is not None:
            return self.projected_sha256
        return ""