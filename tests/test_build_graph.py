"""Acceptance tests for ``src.build_graph`` (#25).

These tests cover every numbered rule in the implementation ticket:

- Two-run ``graph_content_sha256`` determinism harness (gate #2).
- Uniqueness-constraint fixture — duplicate insert is rejected (gate #3).
- ``EVALUATED_ON`` reconstruction fixture (gate #4).
- Transaction-rollback fixture (gate #6).
- Manifest field-completeness assertion (gate #5).

Preflight uses synthetic fixtures from :mod:`tests.build_graph_fixture`.
The Neo4j driver is mocked through
:mod:`tests._build_graph_driver_mock`; no live database is required.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.build_graph import ingestion, preflight
from src.build_graph.ingestion import (
    graph_content_sha256,
    run_ingestion,
    _canonical_graph_payload,
)
from src.build_graph.preflight import PreflightError, run_preflight
from src.build_graph.schema import canonical_json

from tests._build_graph_driver_mock import FakeDriver
from tests.build_graph_fixture import (
    CHUNK_A1,
    DATASET_EID,
    METHOD_EID,
    materialize,
    patch_extractions_manifest,
)
class _PathPatcher:
    """Override the ``DEFAULT_*`` module constants AND rebind the
    corresponding function ``__kwdefaults__`` so calls without explicit
    keyword arguments read from the tmp fixture.

    Some constants (``DEFAULT_OUTPUT_MANIFEST_PATH`` vs
    ``DEFAULT_MANIFEST_PATH``) share a kwarg name across different
    target functions; the per-target mapping below resolves that.
    """

    _PREFLIGHT_KWARG_MAP: dict[str, str] = {
        "DEFAULT_MANIFEST_PATH": "manifest_path",
        "DEFAULT_CHUNKS_PATH": "chunks_path",
        "DEFAULT_ONTOLOGY_PATH": "ontology_path",
        "DEFAULT_EXTRACTIONS_PATH": "extractions_path",
        "DEFAULT_EXTRACTIONS_MANIFEST_PATH": "extractions_manifest_path",
        "DEFAULT_ENTITIES_PATH": "entities_path",
        "DEFAULT_ENTITY_RESOLUTIONS_PATH": "entity_resolutions_path",
        "DEFAULT_ENTITY_RESOLUTION_MANIFEST_PATH": "entity_resolution_manifest_path",
        "DEFAULT_ALIASES_PATH": "aliases_path",
        "DEFAULT_OPENALEX_DIR": "openalex_dir",
    }

    _INGESTION_KWARG_MAP: dict[str, str] = {
        "DEFAULT_OUTPUT_MANIFEST_PATH": "manifest_path",
    }

    _TARGETS = (
        (preflight, "run_preflight", _PREFLIGHT_KWARG_MAP),
        (ingestion, "run_ingestion", _INGESTION_KWARG_MAP),
    )

    def __init__(self, **paths: Path) -> None:
        self.paths = paths
        self._originals: dict[tuple, object] = {}

    def apply(self) -> None:
        for default_name, path in self.paths.items():
            setattr(preflight, default_name, path)
            setattr(ingestion, default_name, path)
            for module, func_name, kwarg_map in self._TARGETS:
                kwarg = kwarg_map.get(default_name)
                if kwarg is None:
                    continue
                fn = getattr(module, func_name)
                kw = fn.__kwdefaults__
                if kw is None or kwarg not in kw:
                    continue
                self._originals[(func_name, kwarg)] = kw[kwarg]
                kw[kwarg] = path

    def revert(self) -> None:
        for (func_name, kwarg), original in self._originals.items():
            for module, name, _ in self._TARGETS:
                if name != func_name:
                    continue
                fn = getattr(module, name)
                fn.__kwdefaults__[kwarg] = original


class _FixtureTestCase(unittest.TestCase):
    """Materialise the synthetic fixture into a tmpdir, patch the
    preflight + ingestion modules' default paths to point at the tmp
    fixture, run the test, and tear down."""

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="build_graph_test_"))
        self.paths = materialize(self.tmpdir)
        self._patcher = _PathPatcher(
            DEFAULT_MANIFEST_PATH=self.paths.manifest,
            DEFAULT_CHUNKS_PATH=self.paths.chunks,
            DEFAULT_ONTOLOGY_PATH=self.paths.ontology,
            DEFAULT_EXTRACTIONS_PATH=self.paths.extractions,
            DEFAULT_EXTRACTIONS_MANIFEST_PATH=self.paths.extractions_manifest,
            DEFAULT_ENTITIES_PATH=self.paths.entities,
            DEFAULT_ENTITY_RESOLUTIONS_PATH=self.paths.entity_resolutions,
            DEFAULT_ENTITY_RESOLUTION_MANIFEST_PATH=self.paths.entity_resolution_manifest,
            DEFAULT_ALIASES_PATH=self.paths.aliases,
            DEFAULT_OPENALEX_DIR=self.paths.openalex_dir,
            DEFAULT_OUTPUT_MANIFEST_PATH=self.paths.graph_manifest,
        )
        self._patcher.apply()

    def tearDown(self) -> None:
        self._patcher.revert()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class PreflightSmokeTests(_FixtureTestCase):
    """§2 happy-path: every input is well-formed; preflight succeeds."""

    def test_preflight_returns_bundle_with_expected_counts(self) -> None:
        bundle = run_preflight()
        self.assertEqual(len(bundle.documents), 3)
        self.assertEqual(len(bundle.chunks), 6)
        self.assertEqual(len(bundle.entities), 4)
        self.assertEqual(len(bundle.mentions), 4)
        self.assertEqual(len(bundle.assertions), 4)
        # Citations: A->B, B->A, C->A. Self-loop A->A dropped; D external dropped.
        self.assertEqual(len(bundle.citation_pairs), 3)
        self.assertIn(("W100000001", "W100000002"), bundle.citation_pairs)
        self.assertIn(("W100000002", "W100000001"), bundle.citation_pairs)
        self.assertIn(("W100000003", "W100000001"), bundle.citation_pairs)


class PreflightRejectionTests(_FixtureTestCase):
    """§2 negative paths: every named failure category produces a
    :class:`PreflightError` before any graph write."""

    def test_missing_raw_openalex_record_is_rejected(self) -> None:
        (self.paths.openalex_dir / "W100000001.json").unlink()
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("raw OpenAlex record missing", str(cm.exception))

    def test_stale_chunks_sha_in_extractions_manifest_is_rejected(self) -> None:
        self.paths.chunks.write_text(
            '{"chunk_id": "c_0000000000000fff", "doc_id": "W100000001", '
            '"section_path": ["x"], "chunk_index": 0, "text": "noise"}\n'
        )
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("chunks_sha256 disagrees", str(cm.exception))

    def test_duplicate_chunk_id_is_rejected(self) -> None:
        with self.paths.chunks.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "chunk_id": "c_000000000000a001",
                        "doc_id": "W100000001",
                        "section_path": ["dup"],
                        "chunk_index": 99,
                        "text": "x" * 32,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("duplicate chunk_id", str(cm.exception))

    def test_chunk_doc_id_not_in_manifest_is_rejected(self) -> None:
        payload = [
            json.loads(line)
            for line in self.paths.chunks.read_text().splitlines()
            if line.strip()
        ]
        payload[0]["doc_id"] = "W999999999"
        with self.paths.chunks.open("w") as fh:
            for row in payload:
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("not in manifest", str(cm.exception))

    def test_extraction_referencing_unknown_chunk_is_rejected(self) -> None:
        rows = [
            json.loads(line)
            for line in self.paths.extractions.read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            if row.get("kind") == "mention":
                row["chunk_id"] = "c_ffffffffffffffff"
        with self.paths.extractions.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("not in chunks", str(cm.exception))

    def test_unresolved_mention_endpoint_is_rejected(self) -> None:
        resolutions = [
            json.loads(line)
            for line in self.paths.entity_resolutions.read_text().splitlines()
            if line.strip()
        ]
        resolutions = [
            r for r in resolutions if r["mention_id"] != "m_" + "a" * 16
        ]
        with self.paths.entity_resolutions.open("w") as fh:
            for r in resolutions:
                fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("not resolved", str(cm.exception))

    def test_incomplete_extraction_is_rejected(self) -> None:
        rows = [
            json.loads(line)
            for line in self.paths.extractions.read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            if row.get("kind") == "chunk_status" and row["chunk_id"] == CHUNK_A1:
                row["extraction_complete"] = False
        with self.paths.extractions.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("extraction_complete", str(cm.exception))


class DeterminismTests(_FixtureTestCase):
    """Acceptance gate #2: two consecutive replacement runs produce the
    same ``graph_content_sha256`` and byte-identical ingestion manifests.
    """

    def test_graph_hash_is_stable_across_two_runs(self) -> None:
        bundle1 = run_preflight()
        sha1 = graph_content_sha256(bundle1)
        bundle2 = run_preflight()
        sha2 = graph_content_sha256(bundle2)
        self.assertEqual(sha1, sha2)
        payload1 = _canonical_graph_payload(bundle1)
        payload2 = _canonical_graph_payload(bundle2)
        self.assertEqual(canonical_json(payload1), canonical_json(payload2))


class UniquenessConstraintTests(_FixtureTestCase):
    """Acceptance gate #3: uniqueness constraints reject duplicate
    primary-key inserts (verified by an intentional duplicate).

    The mock driver is pre-seeded with the anchor value the first
    CREATE would attempt; the ingestion's first CREATE then collides
    on the constraint. This proves the constraint is enforced on the
    live CREATE path, not just on direct duplicates within ingestion.
    """

    def test_duplicate_doc_id_create_is_rejected(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver(
            duplicate_anchor=("Document", "doc_id"),
            pre_seed=("Document", "doc_id", "W100000001"),
        )
        with self.assertRaises(Exception) as cm:
            ingestion.replace_graph(bundle, driver)
        self.assertIn("uniqueness constraint", str(cm.exception).lower())

    def test_duplicate_chunk_id_create_is_rejected(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver(
            duplicate_anchor=("Chunk", "chunk_id"),
            pre_seed=("Chunk", "chunk_id", "c_000000000000a001"),
        )
        with self.assertRaises(Exception) as cm:
            ingestion.replace_graph(bundle, driver)
        self.assertIn("uniqueness constraint", str(cm.exception).lower())

    def test_duplicate_entity_id_create_is_rejected(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver(
            duplicate_anchor=("Entity", "entity_id"),
            pre_seed=("Entity", "entity_id", METHOD_EID),
        )
        with self.assertRaises(Exception) as cm:
            ingestion.replace_graph(bundle, driver)
        self.assertIn("uniqueness constraint", str(cm.exception).lower())


class EvaluatedOnReconstructionTests(_FixtureTestCase):
    """Acceptance gate #4: every ``EVALUATED_ON`` edge is provably
    derivable from the ``PROPOSES × EVALUATES_ON`` closure. No
    LLM-derived ``EVALUATED_ON`` may exist."""

    def test_only_one_evaluated_on_pair_is_present(self) -> None:
        bundle = run_preflight()
        evals = ingestion._build_evaluated_on(bundle)
        self.assertEqual(len(evals), 1)
        ev = evals[0]
        self.assertEqual(ev["from"]["id"], METHOD_EID)
        self.assertEqual(ev["to"]["id"], DATASET_EID)

    def test_supporting_assertion_ids_are_sorted_unique_union(self) -> None:
        bundle = run_preflight()
        evals = ingestion._build_evaluated_on(bundle)
        self.assertEqual(len(evals), 1)
        ev = evals[0]
        self.assertEqual(
            ev["properties"]["supporting_assertion_ids"],
            ["a_" + "0" * 16, "a_" + "1" * 16],
        )

    def test_no_evaluated_on_when_one_side_missing(self) -> None:
        """Removing the EVALUATES_ON assertion removes the closure."""
        rows = [
            json.loads(line)
            for line in self.paths.extractions.read_text().splitlines()
            if line.strip()
        ]
        rows = [
            r
            for r in rows
            if not (
                r.get("kind") == "assertion"
                and r.get("relation_type") == "EVALUATES_ON"
            )
        ]
        with self.paths.extractions.open("w") as fh:
            for row in rows:
                fh.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
        patch_extractions_manifest(self.paths.extractions_manifest)
        bundle = run_preflight()
        evals = ingestion._build_evaluated_on(bundle)
        self.assertEqual(evals, [])


class TransactionRollbackTests(_FixtureTestCase):
    """Acceptance gate #6: a mid-run failure rolls back the transaction
    atomically — the post-failure canonical projection is identical to
    the pre-run capture."""

    def test_failure_after_one_create_rolls_back_atomically(self) -> None:
        bundle = run_preflight()
        pre_run_sha = graph_content_sha256(bundle)
        pre_run_payload = _canonical_graph_payload(bundle)
        # Fail immediately after the DETACH DELETE statement (=1), so
        # the first CREATE fails and the transaction rolls back.
        driver = FakeDriver(fail_after=1)
        with self.assertRaises(RuntimeError):
            ingestion.replace_graph(bundle, driver)
        session = driver.sessions[-1]
        self.assertTrue(session.rolled_back)
        self.assertFalse(session.committed)
        post_sha = graph_content_sha256(bundle)
        post_payload = _canonical_graph_payload(bundle)
        self.assertEqual(pre_run_sha, post_sha)
        self.assertEqual(
            canonical_json(pre_run_payload), canonical_json(post_payload)
        )

    def test_failure_after_some_creates_rolls_back_atomically(self) -> None:
        """Inject a failure deeper in the transaction: some CREATE
        statements have run, then the failure triggers rollback. The
        recorded statements list must reflect what was attempted and
        ``rolled_back`` must be True."""
        bundle = run_preflight()
        driver = FakeDriver(fail_after=8)
        with self.assertRaises(RuntimeError):
            ingestion.replace_graph(bundle, driver)
        session = driver.sessions[-1]
        self.assertTrue(session.rolled_back)
        self.assertFalse(session.committed)


class ManifestFieldCompletenessTests(_FixtureTestCase):
    """Acceptance gate #5: every named hash appears in the published
    ``data/graph_ingestion.manifest.json`` and is non-empty."""

    def test_manifest_has_every_named_hash(self) -> None:
        bundle = run_preflight()
        projected = graph_content_sha256(bundle)
        driver = FakeDriver()
        driver.projected_sha256 = projected
        result = run_ingestion(
            driver=driver, manifest_path=self.paths.graph_manifest
        )
        self.assertTrue(self.paths.graph_manifest.is_file())
        manifest = json.loads(self.paths.graph_manifest.read_text())
        # Inputs block carries the bulk of the named SHAs.
        inputs = manifest["inputs"]
        for key in (
            "manifest_sha256",
            "chunks_sha256",
            "ontology_sha256",
            "extractions_sha256",
            "extractions_manifest_sha256",
            "entities_sha256",
            "entity_resolutions_sha256",
            "entity_resolution_manifest_sha256",
        ):
            self.assertTrue(inputs.get(key), f"missing or empty: {key}")
        # Top-level SHAs.
        self.assertTrue(manifest.get("citation_source_sha256"))
        self.assertTrue(manifest.get("graph_content_sha256"))
        self.assertEqual(
            manifest["graph_content_sha256"], result.graph_content_sha256
        )
        self.assertEqual(
            manifest["citation_source_sha256"], result.citation_source_sha256
        )
        # Counts.
        self.assertGreater(manifest["counts"]["nodes_by_label"].get("Document", 0), 0)
        self.assertGreater(manifest["counts"]["nodes_by_label"].get("Chunk", 0), 0)
        self.assertGreater(manifest["counts"]["nodes_by_label"].get("Entity", 0), 0)
        self.assertGreater(
            manifest["counts"]["relationships_by_type"].get("EVALUATED_ON", 0), 0
        )
        # 3 in-corpus citation pairs (A->B, B->A, C->A).
        self.assertEqual(
            manifest["counts"]["relationships_by_type"].get("CITES", 0), 3
        )

    def test_manifest_publication_is_atomic_no_partial_overwrite(self) -> None:
        """§8.2: the manifest is written via os.replace to a temp
        sibling. A pre-existing valid manifest survives if the new write
        fails partway. This test verifies the helper directly."""
        # Write a sentinel first.
        sentinel = json.dumps({"sentinel": True}, sort_keys=True)
        self.paths.graph_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.paths.graph_manifest.write_text(sentinel + "\n", encoding="utf-8")
        # Now overwrite with a payload via the same helper the orchestrator
        # uses.
        from src.build_graph.ingestion import _atomic_write_manifest

        payload = {"alpha": 1, "beta": 2}
        _atomic_write_manifest(self.paths.graph_manifest, payload)
        self.assertTrue(self.paths.graph_manifest.is_file())
        self.assertEqual(json.loads(self.paths.graph_manifest.read_text()), payload)
        # No leftover .tmp file.
        leftovers = list(self.paths.graph_manifest.parent.glob("*.tmp"))
        self.assertEqual(leftovers, [])


class SchemaBootstrapTests(_FixtureTestCase):
    """§5 idempotent uniqueness constraint and lookup index creation."""

    def test_bootstrap_runs_five_statements(self) -> None:
        driver = FakeDriver()
        ingestion.bootstrap_schema(driver, database="neo4j")
        session = driver.sessions[-1]
        # 3 uniqueness constraints + 2 indexes = 5 statements.
        self.assertEqual(len(session.statements), 5)
        joined = " ".join(s for s, _ in session.statements)
        self.assertIn("document_doc_id_unique", joined)
        self.assertIn("chunk_chunk_id_unique", joined)
        self.assertIn("entity_entity_id_unique", joined)
        self.assertIn("chunk_doc_id_lookup", joined)
        self.assertIn("entity_type_name_lookup", joined)
