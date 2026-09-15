"""Acceptance + regression tests for ``src.build_graph`` (#25).

Coverage:

- §2 preflight happy path and every named rejection category.
- Resolution provenance ownership (#8 §7.2) reaching MENTIONS.
- Entity-resolution manifest hash cross-check (§2.1 rule 4).
- chunk_id / assertion_id re-derivation (§2.1 rules 2 and 6).
- Raw-OpenAlex SHA cross-check (§2.1 rule 9).
- Exact §4.6 relationship property sets.
- §7 in-transaction integrity checks and rollback.
- §8.1 canonical hash determinism.
- §8.2 production-style post-commit readback.
- §8.2 manifest determinism and field completeness.

Preflight consumes synthetic fixtures from
:mod:`tests.build_graph_fixture`, whose IDs are content-derived under
the real producer contracts. The Neo4j driver is mocked through
:mod:`tests._build_graph_driver_mock`, which routes every query through
the same ``session`` / ``execute_write`` / ``execute_read`` surface the
production code uses; no test passes by echoing a pre-stamped hash.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.build_graph import ingestion, preflight
from src.build_graph.ingestion import (
    IngestionError,
    _canonical_graph_payload,
    graph_content_sha256,
    readback_graph_content_sha256,
    replace_graph,
    run_ingestion,
)
from src.build_graph.preflight import PreflightError, run_preflight
from src.build_graph.schema import canonical_json

from tests._build_graph_driver_mock import FakeDriver
from tests.build_graph_fixture import (
    CHUNK_A1_ID,
    DATASET_EID,
    METHOD_EID,
    materialize,
    patch_extractions_manifest,
)


#: Repository-relative scratch root. Keeping fixtures inside the repo
#: preserves the production "repository-relative POSIX, fail closed"
#: manifest path contract instead of weakening it for tests. Mirrors
#: the existing ``tests/_entity_resolution_tmp/`` precedent; gitignored.
_SCRATCH_ROOT = Path(__file__).resolve().parent / "_build_graph_tmp"


class _PathPatcher:
    """Override the ``DEFAULT_*`` module constants AND rebind the
    matching function ``__kwdefaults__`` so calls without explicit
    keyword arguments read from the scratch fixture.

    ``DEFAULT_OUTPUT_MANIFEST_PATH`` and ``DEFAULT_MANIFEST_PATH`` share
    the kwarg name ``manifest_path`` across different target functions;
    the per-target mapping below disambiguates them.
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
    """Materialise the synthetic fixture into a repo-relative scratch
    directory, patch the default paths, run the test, tear down."""

    def setUp(self) -> None:
        _SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmpdir = Path(tempfile.mkdtemp(prefix="case_", dir=str(_SCRATCH_ROOT)))
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

    # -- small helpers shared by several tests --------------------------
    def _read_jsonl(self, path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    def _ingest(self, driver: FakeDriver):
        return run_ingestion(driver=driver, manifest_path=self.paths.graph_manifest)

    def _refresh_entity_resolution_input_hashes(self) -> None:
        """Keep the #8 producer manifest coherent after a test mutates
        upstream extraction artifacts on purpose."""
        import hashlib

        payload = json.loads(self.paths.entity_resolution_manifest.read_text())
        payload["inputs"]["extractions_sha256"] = hashlib.sha256(
            self.paths.extractions.read_bytes()
        ).hexdigest()
        payload["inputs"]["extractions_manifest_sha256"] = hashlib.sha256(
            self.paths.extractions_manifest.read_bytes()
        ).hexdigest()
        self.paths.entity_resolution_manifest.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n"
        )


class PreflightSmokeTests(_FixtureTestCase):
    """§2 happy path: every input is well-formed; preflight succeeds."""

    def test_preflight_returns_bundle_with_expected_counts(self) -> None:
        bundle = run_preflight()
        self.assertEqual(len(bundle.documents), 3)
        self.assertEqual(len(bundle.chunks), 6)
        self.assertEqual(len(bundle.entities), 6)
        self.assertEqual(len(bundle.mentions), 6)
        self.assertEqual(len(bundle.assertions), 4)
        self.assertEqual(
            sorted(bundle.citation_pairs),
            [
                ("W100000001", "W100000002"),
                ("W100000002", "W100000001"),
                ("W100000003", "W100000001"),
            ],
        )

    def test_entity_resolution_algorithm_identity_is_captured(self) -> None:
        """§8.2 requires the resolution algorithm identity in the manifest;
        the bundle must carry it out of the #8 manifest verbatim."""
        bundle = run_preflight()
        self.assertEqual(bundle.er_algorithm_version, "entity_resolution_v1")
        self.assertEqual(bundle.er_normalization_version, "resolution_name_v1")
        self.assertEqual(bundle.er_similarity_metric, "normalized_levenshtein")
        self.assertEqual(bundle.er_threshold, 0.92)


class ResolutionProvenanceTests(_FixtureTestCase):
    """§2.1 rule 5 + §4.2: resolution method/score come from
    ``entity_resolutions.jsonl``, never from extraction records."""

    def test_fuzzy_resolution_metadata_reaches_mentions_unchanged(self) -> None:
        bundle = run_preflight()
        fuzzy = {
            mid: (m.resolution_method, m.resolution_score)
            for mid, m in bundle.mentions.items()
            if m.resolution_method == "fuzzy"
        }
        self.assertEqual(len(fuzzy), 1, "fixture ships exactly one fuzzy merge")
        (method, score), = fuzzy.values()
        self.assertEqual(method, "fuzzy")
        self.assertEqual(score, 0.95)
        # And it must survive into the MENTIONS projection unchanged.
        rels = ingestion._build_mentions_relationships(bundle)
        fuzzy_rels = [
            r for r in rels if r["properties"]["resolution_method"] == "fuzzy"
        ]
        self.assertEqual(len(fuzzy_rels), 1)
        self.assertEqual(fuzzy_rels[0]["properties"]["resolution_score"], 0.95)

    def test_extraction_record_cannot_supply_resolution_metadata(self) -> None:
        """A forged ``_resolution_method`` on an extraction record must be
        rejected by the exact-shape gate, never silently honoured."""
        rows = self._read_jsonl(self.paths.extractions)
        for row in rows:
            if row.get("kind") == "mention":
                row["_resolution_method"] = "fuzzy"
                row["_resolution_score"] = 0.5
                break
        self._write_jsonl(self.paths.extractions, rows)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("mention record", str(cm.exception))

    def test_invalid_resolution_method_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.entity_resolutions)
        rows[0]["resolution_method"] = "embedding_knn"
        self._write_jsonl(self.paths.entity_resolutions, rows)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("resolution_method", str(cm.exception))

    def test_exact_normalized_with_non_unit_score_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.entity_resolutions)
        for row in rows:
            if row["resolution_method"] == "exact_normalized":
                row["resolution_score"] = 0.99
                break
        self._write_jsonl(self.paths.entity_resolutions, rows)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("must have score 1.0", str(cm.exception))

    def test_fuzzy_score_below_pinned_threshold_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.entity_resolutions)
        for row in rows:
            if row["resolution_method"] == "fuzzy":
                row["resolution_score"] = 0.5
                break
        self._write_jsonl(self.paths.entity_resolutions, rows)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("outside", str(cm.exception))

    def test_entity_type_mismatch_between_mention_and_entity_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.entity_resolutions)
        # Point the method mention at the dataset entity.
        for row in rows:
            if row["entity_id"] == METHOD_EID:
                row["entity_id"] = DATASET_EID
                break
        self._write_jsonl(self.paths.entity_resolutions, rows)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("entity_type", str(cm.exception))

    def test_mention_without_resolution_mapping_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.entity_resolutions)
        self._write_jsonl(self.paths.entity_resolutions, rows[:-1])
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("no resolution mapping", str(cm.exception))


class PreflightRejectionTests(_FixtureTestCase):
    """§2 negative paths: every named failure category produces a
    :class:`PreflightError` before any graph write."""

    def test_missing_raw_openalex_record_is_rejected(self) -> None:
        (self.paths.openalex_dir / "W100000001.json").unlink()
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("raw OpenAlex record missing", str(cm.exception))

    def test_raw_openalex_sha_drift_is_rejected(self) -> None:
        """§2.1 rule 9: the raw record's bytes must hash to the
        manifest's recorded ``sha256``."""
        raw = self.paths.openalex_dir / "W100000001.json"
        raw.write_text(raw.read_text() + " ")  # byte-level drift only
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("raw OpenAlex SHA drift", str(cm.exception))

    def test_stale_chunks_sha_in_extractions_manifest_is_rejected(self) -> None:
        manifest = json.loads(self.paths.extractions_manifest.read_text())
        manifest["chunks_sha256"] = "0" * 64
        self.paths.extractions_manifest.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("chunks_sha256 disagrees", str(cm.exception))

    def test_duplicate_chunk_id_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.chunks)
        rows.append(dict(rows[0]))  # exact duplicate primary key
        self._write_jsonl(self.paths.chunks, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("duplicate chunk_id", str(cm.exception))

    def test_non_rederiving_chunk_id_is_rejected(self) -> None:
        """§2.1 rule 2: chunk_id must re-derive from
        ``(doc_id, section_path, chunk_index)``."""
        rows = self._read_jsonl(self.paths.chunks)
        rows[0]["chunk_id"] = "c_" + "0" * 16
        self._write_jsonl(self.paths.chunks, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("does not re-derive under the chunk contract", str(cm.exception))

    def test_chunk_doc_id_not_in_manifest_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.chunks)
        rows[0]["doc_id"] = "W999999999"
        self._write_jsonl(self.paths.chunks, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("not in manifest", str(cm.exception))

    def test_non_rederiving_assertion_id_is_rejected(self) -> None:
        """§2.1 rule 6: assertion_id must re-derive under #23."""
        rows = self._read_jsonl(self.paths.extractions)
        for row in rows:
            if row.get("kind") == "assertion":
                row["assertion_id"] = "a_" + "0" * 16
                break
        self._write_jsonl(self.paths.extractions, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("does not re-derive from #23 contract", str(cm.exception))

    def test_ontology_endpoint_type_mismatch_is_rejected(self) -> None:
        """§2.1 rule 6: a mention endpoint's entity_type must match the
        relation's ontology endpoint type."""
        rows = self._read_jsonl(self.paths.extractions)
        # A dataset mention lives in the same chunk as the PROPOSES
        # assertion, so the same-chunk gate passes and the endpoint-type
        # gate is the one that must fire.
        same_chunk_dataset_mid = next(
            r["mention_id"]
            for r in rows
            if r.get("kind") == "mention"
            and r["entity_type"] == "dataset"
            and r["chunk_id"] == CHUNK_A1_ID
        )
        # Repoint PROPOSES (document -> method) at a dataset mention.
        for row in rows:
            if row.get("kind") == "assertion" and row["relation_type"] == "PROPOSES":
                row["object_ref"] = {"kind": "mention", "id": same_chunk_dataset_mid}
                from src.extract.ids import rederive_assertion_id

                row["assertion_id"] = rederive_assertion_id(row)
                break
        self._write_jsonl(self.paths.extractions, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("expects 'method'", str(cm.exception))

    def test_dangling_mention_endpoint_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.extractions)
        for row in rows:
            if row.get("kind") == "assertion" and row["relation_type"] == "PROPOSES":
                row["object_ref"] = {"kind": "mention", "id": "m_" + "f" * 16}
                from src.extract.ids import rederive_assertion_id

                row["assertion_id"] = rederive_assertion_id(row)
                break
        self._write_jsonl(self.paths.extractions, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("dangling mention ref", str(cm.exception))

    def test_cross_chunk_mention_endpoint_is_rejected(self) -> None:
        """A PROPOSES in chunk A may not reference a mention living in
        chunk B; the same-chunk surface gate must reject it."""
        rows = self._read_jsonl(self.paths.extractions)
        other_chunk_mid = next(
            r["mention_id"]
            for r in rows
            if r.get("kind") == "mention"
            and r["entity_type"] == "method"
            and r["chunk_id"] != CHUNK_A1_ID
        )
        for row in rows:
            if (
                row.get("kind") == "assertion"
                and row["relation_type"] == "PROPOSES"
                and row["chunk_id"] == CHUNK_A1_ID
            ):
                row["object_ref"] = {"kind": "mention", "id": other_chunk_mid}
                from src.extract.ids import rederive_assertion_id

                row["assertion_id"] = rederive_assertion_id(row)
                break
        self._write_jsonl(self.paths.extractions, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("dangling mention ref", str(cm.exception))

    def test_extraction_referencing_unknown_chunk_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.extractions)
        for row in rows:
            if row.get("kind") == "mention":
                row["chunk_id"] = "c_" + "f" * 16
                break
        self._write_jsonl(self.paths.extractions, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("not in chunks", str(cm.exception))

    def test_incomplete_extraction_is_rejected(self) -> None:
        rows = self._read_jsonl(self.paths.extractions)
        for row in rows:
            if row.get("kind") == "chunk_status" and row["chunk_id"] == CHUNK_A1_ID:
                row["extraction_complete"] = False
        self._write_jsonl(self.paths.extractions, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("extraction_complete", str(cm.exception))


class EntityResolutionManifestTests(_FixtureTestCase):
    """§2.1 rule 4: the #8 manifest's input hashes must match the current
    canonical artifacts and its output hashes must match the current
    entity / resolution files."""

    def _mutate_er_manifest(self, path_in_manifest: list[str], value) -> None:
        payload = json.loads(self.paths.entity_resolution_manifest.read_text())
        cursor = payload
        for key in path_in_manifest[:-1]:
            cursor = cursor[key]
        cursor[path_in_manifest[-1]] = value
        self.paths.entity_resolution_manifest.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n"
        )

    def test_stale_input_chunks_sha_is_rejected(self) -> None:
        self._mutate_er_manifest(["inputs", "chunks_sha256"], "0" * 64)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("inputs.chunks_sha256 disagrees", str(cm.exception))

    def test_stale_input_extractions_sha_is_rejected(self) -> None:
        self._mutate_er_manifest(["inputs", "extractions_sha256"], "0" * 64)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("inputs.extractions_sha256 disagrees", str(cm.exception))

    def test_stale_output_entities_sha_is_rejected(self) -> None:
        self._mutate_er_manifest(["outputs", "entities_sha256"], "0" * 64)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn("outputs.entities_sha256 disagrees", str(cm.exception))

    def test_stale_output_entity_resolutions_sha_is_rejected(self) -> None:
        self._mutate_er_manifest(["outputs", "entity_resolutions_sha256"], "0" * 64)
        with self.assertRaises(PreflightError) as cm:
            run_preflight()
        self.assertIn(
            "outputs.entity_resolutions_sha256 disagrees", str(cm.exception)
        )


class RelationshipPropertySetTests(_FixtureTestCase):
    """§4.6: exact persisted property sets on every relationship class."""

    def test_mentions_carries_exactly_the_eight_required_properties(self) -> None:
        bundle = run_preflight()
        rels = ingestion._build_mentions_relationships(bundle)
        self.assertTrue(rels)
        expected = {
            "mention_id",
            "chunk_id",
            "char_start",
            "char_end",
            "source_text_sha256",
            "confidence",
            "resolution_method",
            "resolution_score",
        }
        for rel in rels:
            self.assertEqual(set(rel["properties"]), expected)

    def test_extracted_assertions_carry_exactly_six_required_properties(self) -> None:
        bundle = run_preflight()
        rels = ingestion._build_extracted_assertions(bundle)
        self.assertTrue(rels)
        expected = {
            "assertion_id",
            "chunk_id",
            "char_start",
            "char_end",
            "source_text_sha256",
            "confidence",
        }
        for rel in rels:
            self.assertEqual(set(rel["properties"]), expected)

    def test_structural_relationships_carry_no_properties(self) -> None:
        bundle = run_preflight()
        for rel in ingestion._build_has_chunk_relationships(bundle):
            self.assertEqual(rel["properties"], {})
        for rel in ingestion._build_citations(bundle):
            self.assertEqual(rel["properties"], {})

    def test_evidence_values_match_the_producer_records(self) -> None:
        """The projection must copy #23 evidence verbatim, not recompute."""
        bundle = run_preflight()
        rels = {
            r["properties"]["mention_id"]: r["properties"]
            for r in ingestion._build_mentions_relationships(bundle)
        }
        for mid, mention in bundle.mentions.items():
            props = rels[mid]
            self.assertEqual(props["char_start"], mention.evidence["char_start"])
            self.assertEqual(props["char_end"], mention.evidence["char_end"])
            self.assertEqual(
                props["source_text_sha256"], mention.evidence["source_text_sha256"]
            )
            self.assertEqual(props["confidence"], mention.confidence)


class EvaluatedOnReconstructionTests(_FixtureTestCase):
    """§4.5: EVALUATED_ON is the exact PROPOSES x EVALUATES_ON closure."""

    def test_only_one_evaluated_on_pair_is_present(self) -> None:
        bundle = run_preflight()
        evals = ingestion._build_evaluated_on(bundle)
        self.assertEqual(len(evals), 1)
        ev = evals[0]
        self.assertEqual(ev["from"]["id"], METHOD_EID)
        self.assertEqual(ev["to"]["id"], DATASET_EID)
        self.assertEqual(ev["from"]["kind"], "Method")
        self.assertEqual(ev["to"]["kind"], "Dataset")

    def test_supporting_assertion_ids_are_sorted_unique_union(self) -> None:
        bundle = run_preflight()
        ev = ingestion._build_evaluated_on(bundle)[0]
        supporting = ev["properties"]["supporting_assertion_ids"]
        expected = sorted(
            a.assertion_id
            for a in bundle.assertions.values()
            if a.relation_type in {"PROPOSES", "EVALUATES_ON"}
        )
        self.assertEqual(supporting, expected)
        self.assertEqual(supporting, sorted(set(supporting)))

    def test_no_evaluated_on_when_one_side_missing(self) -> None:
        rows = self._read_jsonl(self.paths.extractions)
        rows = [
            r
            for r in rows
            if not (
                r.get("kind") == "assertion"
                and r.get("relation_type") == "EVALUATES_ON"
            )
        ]
        for row in rows:
            if row.get("kind") == "chunk_status" and row["chunk_id"] == CHUNK_A1_ID:
                row["phases"]["relation"]["assertion_count"] = 1
        self._write_jsonl(self.paths.extractions, rows)
        patch_extractions_manifest(self.paths.extractions_manifest)
        self._refresh_entity_resolution_input_hashes()
        bundle = run_preflight()
        self.assertEqual(ingestion._build_evaluated_on(bundle), [])


class DeterminismTests(_FixtureTestCase):
    """Acceptance gate #2: identical inputs produce identical hashes."""

    def test_graph_hash_is_stable_across_two_runs(self) -> None:
        bundle1 = run_preflight()
        bundle2 = run_preflight()
        self.assertEqual(graph_content_sha256(bundle1), graph_content_sha256(bundle2))
        self.assertEqual(
            canonical_json(_canonical_graph_payload(bundle1)),
            canonical_json(_canonical_graph_payload(bundle2)),
        )

    def test_two_identical_ingestions_produce_identical_hashes(self) -> None:
        first = self._ingest(FakeDriver())
        second = self._ingest(FakeDriver())
        self.assertEqual(
            first.graph_content_sha256, second.graph_content_sha256
        )
        self.assertEqual(
            first.citation_source_sha256, second.citation_source_sha256
        )

    def test_two_identical_ingestions_produce_byte_identical_manifests(self) -> None:
        self._ingest(FakeDriver())
        first_bytes = self.paths.graph_manifest.read_bytes()
        self._ingest(FakeDriver())
        second_bytes = self.paths.graph_manifest.read_bytes()
        self.assertEqual(first_bytes, second_bytes)


class SchemaBootstrapTests(_FixtureTestCase):
    """§5 idempotent uniqueness constraints and lookup indexes."""

    def test_bootstrap_runs_five_statements(self) -> None:
        driver = FakeDriver()
        ingestion.bootstrap_schema(driver, database="neo4j")
        joined = " ".join(s for s, _ in driver.sessions[-1].statements)
        for name in (
            "document_doc_id_unique",
            "chunk_chunk_id_unique",
            "entity_entity_id_unique",
            "chunk_doc_id_lookup",
            "entity_type_name_lookup",
        ):
            self.assertIn(name, joined)
        self.assertIn("IF NOT EXISTS", joined)

    def test_bootstrap_is_idempotent_across_reruns(self) -> None:
        driver = FakeDriver()
        ingestion.bootstrap_schema(driver, database="neo4j")
        first = list(driver.sessions[-1].statements)
        ingestion.bootstrap_schema(driver, database="neo4j")
        second = list(driver.sessions[-1].statements)
        self.assertEqual(first, second)
        # Re-running must not create graph content.
        self.assertEqual(driver.shared_nodes, [])
        self.assertEqual(driver.shared_relationships, [])


class UniquenessConstraintTests(_FixtureTestCase):
    """Acceptance gate #3: duplicate primary keys are rejected."""

    def _assert_duplicate_rejected(self, anchor: tuple) -> None:
        bundle = run_preflight()
        driver = FakeDriver(duplicate_anchor=anchor)
        label, prop = anchor
        # Pre-seed the anchor so the first matching CREATE collides.
        first = next(
            n
            for n in _canonical_graph_payload(bundle)["nodes"]
            if label in n["labels"]
        )
        with driver.session() as session:
            session._unique_seen.add((label, prop, first["properties"][prop]))
            seeded = session._unique_seen
        with self.assertRaises(RuntimeError) as cm:
            # Reuse the seeded set for the replacement session.
            original_session = FakeDriver.session

            def patched(self_driver, database=None):
                ctx = original_session(self_driver, database)
                gen = ctx.__enter__()
                gen._unique_seen = seeded
                return _CtxWrap(ctx, gen)

            FakeDriver.session = patched
            try:
                replace_graph(bundle, driver)
            finally:
                FakeDriver.session = original_session
        self.assertIn("uniqueness constraint", str(cm.exception).lower())

    def test_duplicate_doc_id_create_is_rejected(self) -> None:
        self._assert_duplicate_rejected(("Document", "doc_id"))

    def test_duplicate_chunk_id_create_is_rejected(self) -> None:
        self._assert_duplicate_rejected(("Chunk", "chunk_id"))

    def test_duplicate_entity_id_create_is_rejected(self) -> None:
        self._assert_duplicate_rejected(("Entity", "entity_id"))


class _CtxWrap:
    """Minimal context-manager shim used by the uniqueness tests."""

    def __init__(self, ctx, value):
        self._ctx = ctx
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


class InTransactionIntegrityTests(_FixtureTestCase):
    """§7 step 4: referential-integrity checks run INSIDE the write
    transaction and roll the whole replacement back on mismatch."""

    def test_zero_match_relationship_create_is_caught_in_transaction(self) -> None:
        """A relationship CREATE that matches no endpoints leaves the
        relationship uncreated; the in-transaction count check fires."""
        bundle = run_preflight()
        driver = FakeDriver()
        original = FakeDriver.session

        def dropping_session(self_driver, database=None):
            ctx = original(self_driver, database)
            gen = ctx.__enter__()
            real_capture = gen._capture_write

            def capture(statement, params):
                # Simulate a MATCH that finds no endpoints: the CREATE
                # silently produces nothing.
                if "MATCH" in statement and "]->(" in statement:
                    if "HAS_CHUNK" in statement:
                        from tests._build_graph_driver_mock import FakeResult

                        return FakeResult()
                return real_capture(statement, params)

            gen._capture_write = capture
            return _CtxWrap(ctx, gen)

        FakeDriver.session = dropping_session
        try:
            with self.assertRaises(IngestionError) as cm:
                replace_graph(bundle, driver)
        finally:
            FakeDriver.session = original
        self.assertIn("HAS_CHUNK", str(cm.exception))

    def test_node_count_mismatch_causes_rollback(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver()
        original = FakeDriver.session

        def dropping_session(self_driver, database=None):
            ctx = original(self_driver, database)
            gen = ctx.__enter__()
            real_capture = gen._capture_write

            def capture(statement, params):
                # Drop one Document CREATE so the node count is short.
                props = params.get("props") or {}
                if statement.strip().upper().startswith("CREATE (N:DOCUMENT"):
                    if props.get("doc_id") == "W100000001":
                        from tests._build_graph_driver_mock import FakeResult

                        return FakeResult()
                return real_capture(statement, params)

            gen._capture_write = capture
            return _CtxWrap(ctx, gen)

        FakeDriver.session = dropping_session
        try:
            with self.assertRaises(IngestionError) as cm:
                replace_graph(bundle, driver)
        finally:
            FakeDriver.session = original
        self.assertIn("count mismatch", str(cm.exception))

    def test_relationship_count_mismatch_causes_rollback(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver()
        original = FakeDriver.session

        def dropping_session(self_driver, database=None):
            ctx = original(self_driver, database)
            gen = ctx.__enter__()
            real_capture = gen._capture_write

            def capture(statement, params):
                if "[r:CITES]" in statement:
                    from tests._build_graph_driver_mock import FakeResult

                    return FakeResult()
                return real_capture(statement, params)

            gen._capture_write = capture
            return _CtxWrap(ctx, gen)

        FakeDriver.session = dropping_session
        try:
            with self.assertRaises(IngestionError) as cm:
                replace_graph(bundle, driver)
        finally:
            FakeDriver.session = original
        self.assertIn("CITES", str(cm.exception))


class TransactionRollbackTests(_FixtureTestCase):
    """Acceptance gate #6: a mid-run failure rolls the transaction back
    and leaves the previous graph snapshot unchanged."""

    def test_failure_after_one_create_rolls_back_atomically(self) -> None:
        bundle = run_preflight()
        # Establish a committed "previous" graph first.
        driver = FakeDriver()
        replace_graph(bundle, driver)
        pre_run_nodes = [dict(n) for n in driver.shared_nodes]
        pre_run_rels = [dict(r) for r in driver.shared_relationships]
        pre_run_sha = readback_graph_content_sha256(driver)

        # Now inject a failure immediately after the DETACH DELETE.
        driver.fail_after = 2
        with self.assertRaises(RuntimeError):
            replace_graph(bundle, driver)
        session = driver.sessions[-1]
        self.assertTrue(session.rolled_back)
        self.assertFalse(session.committed)
        # Restore the snapshot the way a real rollback would, then prove
        # the visible projection is byte-identical to the pre-run state.
        driver.shared_nodes[:] = pre_run_nodes
        driver.shared_relationships[:] = pre_run_rels
        driver.fail_after = None
        self.assertEqual(readback_graph_content_sha256(driver), pre_run_sha)

    def test_failure_mid_way_through_creates_rolls_back(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver(fail_after=8)
        with self.assertRaises(RuntimeError):
            replace_graph(bundle, driver)
        session = driver.sessions[-1]
        self.assertTrue(session.rolled_back)
        self.assertFalse(session.committed)

    def test_failed_preflight_leaves_existing_manifest_byte_identical(self) -> None:
        # Publish a good manifest first.
        self._ingest(FakeDriver())
        good_bytes = self.paths.graph_manifest.read_bytes()
        # Now break an input and re-run; the manifest must not change.
        raw = self.paths.openalex_dir / "W100000001.json"
        raw.write_text(raw.read_text() + " ")
        with self.assertRaises(PreflightError):
            run_ingestion(
                driver=FakeDriver(), manifest_path=self.paths.graph_manifest
            )
        self.assertEqual(self.paths.graph_manifest.read_bytes(), good_bytes)


class PostCommitReadbackTests(_FixtureTestCase):
    """§8.2: the post-commit verifier reconstructs the canonical
    projection from the ACTUAL committed graph via the standard driver
    read API and compares hashes. No pre-stamped shortcut exists."""

    def test_readback_matches_the_pre_write_hash(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver()
        expected = replace_graph(bundle, driver)
        self.assertEqual(readback_graph_content_sha256(driver), expected)

    def test_readback_detects_missing_content(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver()
        expected = replace_graph(bundle, driver)
        driver.shared_nodes.pop()
        self.assertNotEqual(readback_graph_content_sha256(driver), expected)

    def test_readback_detects_extra_stale_content(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver()
        expected = replace_graph(bundle, driver)
        driver.shared_nodes.append(
            {"labels": ["Document"], "props": {"doc_id": "W999999999", "title": "Stale"}}
        )
        self.assertNotEqual(readback_graph_content_sha256(driver), expected)

    def test_readback_detects_wrong_properties(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver()
        expected = replace_graph(bundle, driver)
        for node in driver.shared_nodes:
            if "Document" in node["labels"]:
                node["props"]["title"] = "TAMPERED"
                break
        self.assertNotEqual(readback_graph_content_sha256(driver), expected)

    def test_readback_detects_wrong_relationship_type(self) -> None:
        bundle = run_preflight()
        driver = FakeDriver()
        expected = replace_graph(bundle, driver)
        for rel in driver.shared_relationships:
            if rel["type"] == "CITES":
                rel["type"] = "PROPOSES"
                break
        self.assertNotEqual(readback_graph_content_sha256(driver), expected)

    def test_orchestrator_raises_when_readback_disagrees(self) -> None:
        """A committed graph that drifts from the expected projection
        must abort the run before the manifest is published."""
        bundle = run_preflight()
        original = ingestion.readback_graph_content_sha256
        ingestion.readback_graph_content_sha256 = lambda driver, database="neo4j": "0" * 64
        try:
            with self.assertRaises(IngestionError) as cm:
                run_ingestion(
                    driver=FakeDriver(), manifest_path=self.paths.graph_manifest
                )
        finally:
            ingestion.readback_graph_content_sha256 = original
        self.assertIn("post-commit readback hash mismatch", str(cm.exception))
        self.assertFalse(self.paths.graph_manifest.exists())


class ManifestTests(_FixtureTestCase):
    """§8.2: field completeness, determinism, and forbidden content."""

    def test_manifest_has_every_named_hash(self) -> None:
        result = self._ingest(FakeDriver())
        manifest = json.loads(self.paths.graph_manifest.read_text())
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
            "aliases_sha256",
        ):
            self.assertTrue(inputs.get(key), f"missing or empty: {key}")
        self.assertTrue(manifest["citation_source_sha256"])
        self.assertTrue(manifest["graph_content_sha256"])
        self.assertEqual(
            manifest["graph_content_sha256"], result.graph_content_sha256
        )
        self.assertEqual(
            manifest["citation_source_sha256"], result.citation_source_sha256
        )

    def test_manifest_records_entity_resolution_algorithm_identity(self) -> None:
        self._ingest(FakeDriver())
        manifest = json.loads(self.paths.graph_manifest.read_text())
        algo = manifest["entity_resolution_algorithm"]
        self.assertEqual(algo["algorithm_version"], "entity_resolution_v1")
        self.assertEqual(algo["normalization_version"], "resolution_name_v1")
        self.assertEqual(algo["similarity_metric"], "normalized_levenshtein")
        self.assertEqual(algo["threshold"], 0.92)

    def test_manifest_contains_no_wall_clock_or_host_data(self) -> None:
        self._ingest(FakeDriver())
        blob = self.paths.graph_manifest.read_text()
        payload = json.loads(blob)

        def _keys(node, acc):
            if isinstance(node, dict):
                for k, v in node.items():
                    acc.add(k)
                    _keys(v, acc)
            elif isinstance(node, list):
                for v in node:
                    _keys(v, acc)
            return acc

        keys = _keys(payload, set())
        for forbidden in (
            "wall_clock_seconds",
            "wall_clock_started_at",
            "wall_clock_ended_at",
            "timestamp",
            "hostname",
            "host",
        ):
            self.assertNotIn(forbidden, keys)

    def test_manifest_paths_are_repository_relative(self) -> None:
        self._ingest(FakeDriver())
        manifest = json.loads(self.paths.graph_manifest.read_text())
        for key, value in manifest["inputs"].items():
            if not key.endswith("_path") or value is None:
                continue
            self.assertFalse(
                value.startswith("/"), f"{key} must be repository-relative: {value}"
            )

    def test_manifest_counts_match_the_canonical_projection(self) -> None:
        result = self._ingest(FakeDriver())
        manifest = json.loads(self.paths.graph_manifest.read_text())
        counts = manifest["counts"]
        self.assertEqual(counts["nodes_by_label"], result.node_counts)
        self.assertEqual(
            counts["relationships_by_type"], result.relationship_counts
        )
        self.assertEqual(counts["nodes_by_label"]["Document"], 3)
        self.assertEqual(counts["nodes_by_label"]["Chunk"], 6)
        self.assertEqual(counts["relationships_by_type"]["HAS_CHUNK"], 6)
        self.assertEqual(counts["relationships_by_type"]["CITES"], 3)
        self.assertEqual(counts["relationships_by_type"]["EVALUATED_ON"], 1)

    def test_manifest_publication_is_atomic(self) -> None:
        self.paths.graph_manifest.parent.mkdir(parents=True, exist_ok=True)
        self.paths.graph_manifest.write_text('{"sentinel": true}\n')
        self._ingest(FakeDriver())
        payload = json.loads(self.paths.graph_manifest.read_text())
        self.assertNotIn("sentinel", payload)
        self.assertEqual(
            list(self.paths.graph_manifest.parent.glob("*.tmp")), []
        )


class ProviderBoundaryTests(unittest.TestCase):
    """§9: ingestion must not depend on any provider SDK, LLM,
    embedding, or pgvector code."""

    def test_no_provider_or_vector_imports_in_build_graph(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "build_graph"
        forbidden = ("anthropic", "openai", "ollama", "psycopg", "pgvector",
                     "sentence_transformers", "torch")
        for path in sorted(root.glob("*.py")):
            source = path.read_text()
            for token in forbidden:
                self.assertNotIn(
                    f"import {token}", source, f"{path.name} imports {token}"
                )
                self.assertNotIn(
                    f"from {token}", source, f"{path.name} imports from {token}"
                )
