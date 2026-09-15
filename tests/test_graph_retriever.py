"""Deterministic graph-retriever contract tests for Issue #28."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.entity_resolution import normalized_resolution_name  # noqa: E402
from src.retrieval.graph import (  # noqa: E402
    GRAPH_HOP_BUDGET,
    GraphEvidenceError,
    GraphEvidenceStep,
    GraphFreshnessError,
    GraphRetrievalHit,
    GraphRetriever,
    GraphSeed,
    Neo4jGraphRetriever,
    StaleDerivedProvenanceError,
    _OfficialGraphReadDriver,
    _Path,
    _SemanticEdge,
    _as_assertion_resolver,
    _as_chunk_children_provider,
    _canonical_relationship_identity,
    _directed_identity,
    canonical_evidence_path_json,
    collect_path_evidence,
    document_candidates,
    entity_candidates,
    matched_seeds,
    normalize_query,
    semantic_relation_types,
    traverse_semantic_paths,
    validate_graph_freshness,
)


class ForbiddenDriver:
    def __init__(self) -> None:
        self.accessed = False

    def __getattr__(self, name):
        self.accessed = True
        raise AssertionError(f"driver accessed through {name}")


def retriever(*, entities=None, documents=None):
    driver = ForbiddenDriver()
    return Neo4jGraphRetriever(
        driver, entities=entities or [], documents=documents or []
    ), driver


_FRESH_INPUT_PATHS = {
    "manifest": "data/manifest.jsonl",
    "chunks": "data/derived/chunks.jsonl",
    "ontology": "ontology.yaml",
    "extractions": "data/extractions.jsonl",
    "extractions_manifest": "data/extractions.manifest.json",
    "entities": "data/entities.jsonl",
    "entity_resolutions": "data/entity_resolutions.jsonl",
    "entity_resolution_manifest": "data/entity_resolutions.manifest.json",
    "aliases": "data/aliases.json",
}


def _freshness_fixture(graph_sha="a" * 64, citation_sha="c" * 64):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    inputs = {"ontology_version": 1}
    for name, relative_path in _FRESH_INPUT_PATHS.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        inputs[f"{name}_path"] = relative_path
        inputs[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "inputs": inputs,
        "citation_source_sha256": citation_sha,
        "graph_content_sha256": graph_sha,
        "ontology": {
            key: inputs[key]
            for key in ("ontology_path", "ontology_sha256", "ontology_version")
        },
    }
    manifest_path = root / "graph_ingestion.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return temporary, root, manifest_path


class TestPublicContracts(unittest.TestCase):
    def test_public_dataclasses_are_frozen_and_exact(self):
        step = GraphEvidenceStep(1, "e_1", "PROPOSES", "W1", "reverse")
        hit = GraphRetrievalHit("c_1", 1.0, (step,))
        self.assertEqual(
            [field.name for field in dataclasses.fields(GraphEvidenceStep)],
            ["step", "from_id", "relation", "to_id", "direction"],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(GraphRetrievalHit)],
            ["chunk_id", "score", "evidence_path"],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            step.relation = "CITES"  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            hit.score = 0.5  # type: ignore[misc]

    def test_protocol_and_default_hop_budget_contract(self):
        self.assertEqual(GRAPH_HOP_BUDGET, 3)
        self.assertIsInstance(retriever()[0], GraphRetriever)
        self.assertEqual(
            inspect.signature(GraphRetriever.retrieve).parameters["hop_budget"].default,
            GRAPH_HOP_BUDGET,
        )

    def test_canonical_evidence_serialization_has_exact_wire_keys(self):
        path = (
            GraphEvidenceStep(1, "e_ä", "PROPOSES", "W2", "reverse"),
            GraphEvidenceStep(2, "W2", "CITES", "W3", "forward"),
        )
        self.assertEqual(
            canonical_evidence_path_json(path),
            '[{"direction":"reverse","from":"e_ä","relation":"PROPOSES","step":1,"to":"W2"},{"direction":"forward","from":"W2","relation":"CITES","step":2,"to":"W3"}]',
        )


class TestInputValidation(unittest.TestCase):
    def test_invalid_inputs_fail_before_driver_access(self):
        retr, driver = retriever()
        for query in (None, 1, True):
            with self.subTest(query=query), self.assertRaises(TypeError):
                retr.retrieve(query, 1)  # type: ignore[arg-type]
        for k in (True, 1.5, "1", None):
            with self.subTest(k=k), self.assertRaises(TypeError):
                retr.retrieve("DPR", k)  # type: ignore[arg-type]
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValueError):
                retr.retrieve("DPR", k)
        for budget in (True, 1.5, "1", None):
            with self.subTest(budget=budget), self.assertRaises(TypeError):
                retr.retrieve("DPR", 1, budget)  # type: ignore[arg-type]
        for budget in (0, -1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                retr.retrieve("DPR", 1, budget)
        self.assertFalse(driver.accessed)

    def test_whitespace_query_short_circuits_before_driver_access(self):
        retr, driver = retriever(entities=[{"entity_id": "e_dpr", "canonical_name": "DPR", "aliases": []}])
        self.assertEqual(retr.retrieve("\u2003\t\n", 1), [])
        self.assertFalse(driver.accessed)


class TestNormalizationAndMatching(unittest.TestCase):
    def test_query_normalization_is_unicode_nfkc_casefold_whitespace_trim_only(self):
        self.assertEqual(normalize_query("  ＤＰＲ\u2003Straße\n"), "dpr strasse")
        self.assertEqual(normalize_query(' "DPR" '), '"dpr"')

    def test_candidate_normalization_reuses_entity_resolution_function(self):
        self.assertIn("normalized_resolution_name", inspect.getsource(sys.modules[entity_candidates.__module__]))
        self.assertEqual(
            entity_candidates([{"entity_id": "e_1", "canonical_name": "“DPR”", "aliases": []}])[1].normalized_text,
            normalized_resolution_name("“DPR”"),
        )

    def test_boundary_and_punctuation_behavior(self):
        candidates = entity_candidates([{"entity_id": "e_dpr", "canonical_name": "DPR", "aliases": []}])
        self.assertEqual(matched_seeds("modelDPRv2", candidates), [])
        self.assertEqual(matched_seeds("my_DPR", candidates)[0].canonical_id, "e_dpr")
        self.assertEqual(matched_seeds("(DPR!)", candidates)[0].canonical_id, "e_dpr")
        self.assertEqual(matched_seeds("DPR", candidates)[0].canonical_id, "e_dpr")

    def test_strict_maximal_span_keeps_equal_and_partial_overlaps(self):
        containing = entity_candidates(
            [
                {"entity_id": "e_short", "canonical_name": "RAG", "aliases": []},
                {"entity_id": "e_long", "canonical_name": "RAG system", "aliases": []},
            ]
        )
        self.assertEqual(
            [seed.canonical_id for seed in matched_seeds("RAG system", containing)],
            ["e_long"],
        )
        equal = entity_candidates(
            [
                {"entity_id": "e_first", "canonical_name": "DPR", "aliases": []},
                {"entity_id": "e_second", "canonical_name": "DPR", "aliases": []},
            ]
        )
        self.assertEqual(
            [seed.canonical_id for seed in matched_seeds("DPR", equal)],
            ["e_first", "e_second"],
        )
        partial = entity_candidates(
            [
                {"entity_id": "e_one", "canonical_name": "ab-?", "aliases": []},
                {"entity_id": "e_two", "canonical_name": "?-cd", "aliases": []},
            ]
        )
        self.assertEqual(
            [seed.canonical_id for seed in matched_seeds(" ab-?-cd ", partial)],
            ["e_one", "e_two"],
        )


class TestSeedLifting(unittest.TestCase):
    def test_document_entity_deduplication_and_ordering(self):
        candidates = entity_candidates(
            [
                {"entity_id": "e_z", "canonical_name": "DPR", "aliases": ["Dense Passage Retrieval", "DPR"]},
                {"entity_id": "e_a", "canonical_name": "DPR", "aliases": []},
            ]
        ) + document_candidates(
            [
                {"doc_id": "W9", "title": "DPR"},
                {"doc_id": "W1", "title": "DPR"},
            ]
        )
        self.assertEqual(
            [(seed.node_kind, seed.canonical_id) for seed in matched_seeds("DPR", candidates)],
            [("Document", "W1"), ("Document", "W9"), ("Entity", "e_a"), ("Entity", "e_z")],
        )

    def test_alias_and_explicit_ids_are_candidates(self):
        candidates = entity_candidates(
            [{"entity_id": "e_7", "canonical_name": "Canonical", "aliases": ["Alias"]}]
        ) + document_candidates([{"doc_id": "W123", "title": "Document Title"}])
        self.assertEqual(
            [(seed.node_kind, seed.canonical_id) for seed in matched_seeds("Alias W123 e_7", candidates)],
            [("Document", "W123"), ("Entity", "e_7")],
        )

    def test_cross_catalogue_containment_preserves_both_seeds(self):
        retr, _driver = retriever(
            entities=[
                {
                    "entity_id": "e_dpr_system",
                    "canonical_name": "DPR system",
                    "aliases": [],
                }
            ],
            documents=[{"doc_id": "W1", "title": "DPR"}],
        )
        self.assertEqual(
            [(seed.node_kind, seed.canonical_id) for seed in retr.lift_seeds("DPR system")],
            [("Document", "W1"), ("Entity", "e_dpr_system")],
        )

    def test_retriever_exposes_lifted_seeds_without_traversal(self):
        retr, driver = retriever(
            entities=[{"entity_id": "e_1", "canonical_name": "DPR", "aliases": []}],
            documents=[{"doc_id": "W1", "title": "DPR paper"}],
        )
        self.assertEqual(
            [(seed.node_kind, seed.canonical_id) for seed in retr.lift_seeds("DPR paper")],
            [("Document", "W1"), ("Entity", "e_1")],
        )
        with self.assertRaises(GraphFreshnessError):
            retr.retrieve("DPR paper", 5)
        self.assertFalse(driver.accessed)

# ---------------------------------------------------------------------------
# Phase 2 — semantic traversal fixture helpers.
# ---------------------------------------------------------------------------

# Frozen v1 ontology relationship set derived from `ontology.yaml`; the
# graph retriever derives it the same way via ``semantic_relation_types``.
_SEMANTIC_RELATIONS = frozenset(
    {
        "CITES",
        "PROPOSES",
        "INTRODUCES",
        "EVALUATES_ON",
        "USES_METHOD",
        "REPORTS_METRIC",
        "EXTENDS",
        "EVALUATED_ON",
        "USED_FOR_TASK",
        "MEASURES",
    }
)


def _edge(
    relation: str,
    source_id: str,
    target_id: str,
    *,
    assertion_id: str = "",
    chunk_id: str = "",
    supporting_assertion_ids=(),
) -> _SemanticEdge:
    return _SemanticEdge(
        relation=relation,
        source_id=source_id,
        target_id=target_id,
        assertion_id=assertion_id,
        chunk_id=chunk_id,
        supporting_assertion_ids=tuple(supporting_assertion_ids),
    )


def _path_steps(path):
    return path.to_evidence_path()


def _seeds(*ids):
    from src.retrieval.graph import GraphSeed

    return [GraphSeed("Document", cid) for cid in ids]


class TestOntologyDerivation(unittest.TestCase):
    def test_semantic_relation_types_excludes_plumbing(self):
        ontology = {
            "relation_types": [
                {"name": "CITES", "population_mode": "structural"},
                {"name": "PROPOSES", "population_mode": "extracted"},
                {"name": "EVALUATED_ON", "population_mode": "derived"},
                {"name": "HAS_CHUNK", "population_mode": "structural"},
                {"name": "MENTIONS", "population_mode": "extracted"},
            ]
        }
        self.assertEqual(
            semantic_relation_types(ontology),
            frozenset({"CITES", "PROPOSES", "EVALUATED_ON"}),
        )

    def test_semantic_relation_types_matches_ontology_yaml(self):
        ontology = {
            "relation_types": [
                {"name": "CITES", "population_mode": "structural"},
                {"name": "PROPOSES", "population_mode": "extracted"},
                {"name": "INTRODUCES", "population_mode": "extracted"},
                {"name": "EVALUATES_ON", "population_mode": "extracted"},
                {"name": "USES_METHOD", "population_mode": "extracted"},
                {"name": "REPORTS_METRIC", "population_mode": "extracted"},
                {"name": "EXTENDS", "population_mode": "extracted"},
                {"name": "EVALUATED_ON", "population_mode": "derived"},
                {"name": "USED_FOR_TASK", "population_mode": "extracted"},
                {"name": "MEASURES", "population_mode": "extracted"},
            ]
        }
        self.assertEqual(semantic_relation_types(ontology), _SEMANTIC_RELATIONS)

    def test_semantic_relation_types_rejects_bad_inputs(self):
        self.assertEqual(semantic_relation_types(None), frozenset())
        self.assertEqual(semantic_relation_types({}), frozenset())
        self.assertEqual(semantic_relation_types({"relation_types": "nope"}), frozenset())


class TestCanonicalIdentity(unittest.TestCase):
    def test_extracted_relationship_identity_uses_assertion_id(self):
        edge = _edge("PROPOSES", "W1", "e_dpr", assertion_id="a_1", chunk_id="c_1")
        self.assertEqual(
            _canonical_relationship_identity(edge),
            ("ASSERTION", "a_1"),
        )

    def test_cites_identity_uses_canonical_endpoint_pair(self):
        cites = _edge("CITES", "W1", "W2")
        self.assertEqual(
            _canonical_relationship_identity(cites),
            ("CITES", "W1", "W2"),
        )

    def test_evaluated_on_identity_uses_method_dataset_pair(self):
        evald = _edge(
            "EVALUATED_ON",
            "e_dpr",
            "e_nq",
            supporting_assertion_ids=("a_1", "a_2"),
        )
        self.assertEqual(
            _canonical_relationship_identity(evald),
            ("EVALUATED_ON", "e_dpr", "e_nq"),
        )

    def test_directed_identity_distinguishes_forward_and_reverse(self):
        cites = _edge("CITES", "W1", "W2")
        forward = _directed_identity(cites, "forward")
        reverse = _directed_identity(cites, "reverse")
        self.assertEqual(forward, ("CITES", "W1", "W2", "forward"))
        self.assertEqual(reverse, ("CITES", "W1", "W2", "reverse"))
        self.assertNotEqual(forward, reverse)

    def test_extracted_identity_independent_of_relation_metadata(self):
        # Two extracted edges that share assertion_id but differ in
        # relation metadata must collapse to the same canonical base
        # identity. ``relation`` is intentionally not part of the
        # identity so a re-labelled driver row does not create a
        # phantom distinct relationship.
        proposes = _edge("PROPOSES", "W1", "e_alpha", assertion_id="a_x")
        introduces = _edge("INTRODUCES", "W1", "e_alpha", assertion_id="a_x")
        self.assertEqual(
            _canonical_relationship_identity(proposes),
            _canonical_relationship_identity(introduces),
        )
        self.assertEqual(
            _canonical_relationship_identity(proposes),
            ("ASSERTION", "a_x"),
        )

    def test_extracted_relationship_missing_assertion_id_fails_closed(self):
        # An extracted relation without assertion_id cannot satisfy the
        # canonical identity contract and must raise rather than invent
        # an identity from the endpoint pair.
        from src.retrieval.graph import GraphTraversalError

        edge = _edge("PROPOSES", "W1", "e_alpha")
        with self.assertRaises(GraphTraversalError):
            _canonical_relationship_identity(edge)
        with self.assertRaises(GraphTraversalError):
            traverse_semantic_paths(
                _seeds("W1"),
                adjacency={"W1": [edge], "e_alpha": []},
                relation_set=frozenset({"PROPOSES"}),
            )

    def test_reverse_cites_identity_preserves_canonical_pair(self):
        # CITES stored W1 -> W2: the canonical (citing, cited) pair
        # never flips regardless of traversal direction.
        cites = _edge("CITES", "W1", "W2")
        self.assertEqual(
            _directed_identity(cites, "reverse"),
            ("CITES", "W1", "W2", "reverse"),
        )

    def test_reverse_evaluated_on_identity_preserves_canonical_pair(self):
        # EVALUATED_ON stored method -> dataset: the canonical
        # (method, dataset) pair never flips regardless of direction.
        evald = _edge(
            "EVALUATED_ON",
            "e_method",
            "e_dataset",
            supporting_assertion_ids=("a_1", "a_2"),
        )
        self.assertEqual(
            _directed_identity(evald, "reverse"),
            ("EVALUATED_ON", "e_method", "e_dataset", "reverse"),
        )


class TestTraversalBasics(unittest.TestCase):
    def test_one_hop_forward_traversal(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(
            _path_steps(paths[0]),
            (GraphEvidenceStep(1, "W1", "CITES", "W2", "forward"),),
        )

    def test_one_hop_reverse_traversal(self):
        # Seed W2 traverses the reverse orientation of the CITES edge
        # stored W1 -> W2. The driver surfaces the edge at both
        # endpoints so traversal sees it from the target side too.
        shared_edge = _edge("CITES", "W1", "W2")
        adjacency = {
            "W1": [],
            "W2": [shared_edge],
        }
        paths = traverse_semantic_paths(
            _seeds("W2"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(
            _path_steps(paths[0]),
            (GraphEvidenceStep(1, "W2", "CITES", "W1", "reverse"),),
        )
    def test_two_hop_connected_path(self):

        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("PROPOSES", "W2", "e_dpr", assertion_id="a_1")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        two_hop = [path for path in paths if len(path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(
            _path_steps(two_hop[0]),
            (
                GraphEvidenceStep(1, "W1", "CITES", "W2", "forward"),
                GraphEvidenceStep(2, "W2", "PROPOSES", "e_dpr", "forward"),
            ),
        )


    def test_source_has_no_forbidden_retrieval_dependencies(self):
        source = (Path(__file__).resolve().parents[1] / "src/retrieval/graph.py").read_text()
        for forbidden in ("langchain", "LLMProvider", "EmbeddingProvider", "pgvector", "rapidfuzz", "sklearn"):
            self.assertNotIn(forbidden, source)
    def test_three_hop_connected_path(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("PROPOSES", "W2", "e_dpr", assertion_id="a_1")],
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq", supporting_assertion_ids=("a_1",))],
            "e_nq": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
            hop_budget=3,
        )
        three_hop = [path for path in paths if len(path.steps) == 3]
        self.assertEqual(len(three_hop), 1)
        self.assertEqual(
            _path_steps(three_hop[0]),
            (
                GraphEvidenceStep(1, "W1", "CITES", "W2", "forward"),
                GraphEvidenceStep(2, "W2", "PROPOSES", "e_dpr", "forward"),
                GraphEvidenceStep(3, "e_dpr", "EVALUATED_ON", "e_nq", "forward"),
            ),
        )

    def test_hop_budget_truncates_at_three(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("CITES", "W2", "W3")],
            "W3": [_edge("CITES", "W3", "W4")],
            "W4": [_edge("CITES", "W4", "W5")],
            "W5": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
            hop_budget=GRAPH_HOP_BUDGET,
        )
        self.assertTrue(all(len(path.steps) <= 3 for path in paths))
        lengths = sorted(len(path.steps) for path in paths)
        self.assertIn(3, lengths)
        self.assertNotIn(4, lengths)

    def test_no_zero_hop_paths(self):
        adjacency = {"W1": [_edge("CITES", "W1", "W2")], "W2": []}
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0].steps), 1)

    def test_empty_seeds_yield_no_paths(self):
        paths = traverse_semantic_paths(
            [],
            adjacency={"W1": [_edge("CITES", "W1", "W2")]},
            relation_set=_SEMANTIC_RELATIONS,
        )
        self.assertEqual(paths, [])

    def test_only_ontology_relations_appear_in_evidence_paths(self):
        adjacency = {
            "W1": [
                _edge("CITES", "W1", "W2"),
                _edge("PROPOSES", "W1", "e_dpr", assertion_id="a_1"),
            ],
            "W2": [_edge("INTRODUCES", "W2", "e_nq", assertion_id="a_2")],
            "e_dpr": [],
            "e_nq": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        seen_relations = {
            step.relation for path in paths for step in path.to_evidence_path()
        }
        self.assertTrue(seen_relations.issubset(_SEMANTIC_RELATIONS))
        self.assertNotIn("HAS_CHUNK", seen_relations)
        self.assertNotIn("MENTIONS", seen_relations)

class TestPlumbingExclusion(unittest.TestCase):
    def test_has_chunk_excluded_from_semantic_path(self):
        adjacency = {
            "W1": [_edge("HAS_CHUNK", "W1", "c_1")],
            "c_1": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        self.assertEqual(paths, [])

    def test_mentions_excluded_from_semantic_path(self):
        from src.retrieval.graph import GraphSeed

        adjacency = {
            "c_1": [_edge("MENTIONS", "c_1", "e_dpr")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            [GraphSeed("Chunk", "c_1")],
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        self.assertEqual(paths, [])

    def test_plumbing_edges_do_not_consume_semantic_hops(self):
        # Mix plumbing + semantic edges; semantic edges still expand
        # even when plumbing neighbours exist.
        adjacency = {
            "W1": [
                _edge("HAS_CHUNK", "W1", "c_1"),
                _edge("CITES", "W1", "W2"),
            ],
            "W2": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0].steps), 1)
        self.assertEqual(paths[0].steps[0].relation, "CITES")


class TestDeterministicOrdering(unittest.TestCase):
    def test_shuffled_adjacency_yields_byte_equal_results(self):
        edges_a = [
            _edge("CITES", "W1", "W2"),
            _edge("PROPOSES", "W1", "e_alpha", assertion_id="a_alpha"),
        ]
        edges_b = list(reversed(edges_a))
        relation_set = frozenset({"CITES", "PROPOSES"})
        paths_a = traverse_semantic_paths(
            _seeds("W1"),
            adjacency={"W1": edges_a, "W2": [], "e_alpha": []},
            relation_set=relation_set,
        )
        paths_b = traverse_semantic_paths(
            _seeds("W1"),
            adjacency={"W1": edges_b, "W2": [], "e_alpha": []},
            relation_set=relation_set,
        )
        self.assertEqual(
            [path.to_evidence_path() for path in paths_a],
            [path.to_evidence_path() for path in paths_b],
        )
        # CITES sorts before PROPOSES (canonical sort order).
        self.assertEqual(paths_a[0].steps[0].relation, "CITES")
        self.assertEqual(paths_a[1].steps[0].relation, "PROPOSES")


class TestRepeatedRelationshipRejection(unittest.TestCase):
    def test_same_directed_identity_cannot_repeat(self):
        # Cycle: W1 CITES W2 (forward), W2 CITES W3 (forward), W3 CITES W1 (forward)
        # reaches W1 again; from W1 the only forward expansion is
        # W1->W2 forward which is the same directed identity
        # (CITES, W1, W2, forward) already used by step 1 and is rejected.
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("CITES", "W2", "W3")],
            "W3": [_edge("CITES", "W3", "W1")],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=frozenset({"CITES"}),
        )
        # No 4-hop (repeating) paths; only 1, 2, 3 hop paths.
        self.assertTrue(all(len(path.steps) <= 3 for path in paths))
        self.assertFalse(any(len(path.steps) == 4 for path in paths))

    def test_same_underlying_relationship_forward_and_reverse_are_distinct(self):
        # Two-step forward+reverse traversal: from W1 the forward
        # orientation goes W1->W2; from W2 the reverse orientation of
        # the same canonical edge goes back W2->W1. Directed identities
        # differ (forward vs reverse) so both steps coexist.
        shared_edge = _edge("CITES", "W1", "W2")
        adjacency = {
            "W1": [shared_edge],
            "W2": [shared_edge],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=frozenset({"CITES"}),
        )
        two_hop = [path for path in paths if len(path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(two_hop[0].steps[0].traversal_direction, "forward")
        self.assertEqual(two_hop[0].steps[1].traversal_direction, "reverse")

    def test_node_revisit_allowed_with_different_relationship_identity(self):
        adjacency = {
            "W1": [
                _edge("CITES", "W1", "W2"),
                _edge("PROPOSES", "W1", "e_alpha", assertion_id="a_alpha"),
            ],
            "W2": [_edge("PROPOSES", "W2", "e_alpha", assertion_id="a_beta")],
            "e_alpha": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=frozenset({"CITES", "PROPOSES"}),
        )
        # Two paths reach e_alpha: direct W1->e_alpha (1-hop) and
        # W1->W2->e_alpha (2-hop). Both PROPOSES assertions have distinct
        # assertion_id bases, so neither directed identity repeats.
        targets = [path.steps[-1].to_id for path in paths]
        self.assertEqual(targets.count("e_alpha"), 2)

    def test_extracted_relationship_repeats_rejected_by_assertion_id(self):
        # W1 -> e_alpha (PROPOSES, a_same) and back e_alpha -> W1
        # (PROPOSES, a_same): the second forward step shares the same
        # directed identity (ASSERTION, PROPOSES, a_same, forward) and is
        # rejected.
        adjacency = {
            "W1": [_edge("PROPOSES", "W1", "e_alpha", assertion_id="a_same")],
            "e_alpha": [_edge("PROPOSES", "e_alpha", "W1", assertion_id="a_same")],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=frozenset({"PROPOSES"}),
        )
        # Only the 1-hop forward path survives; 2-hop is rejected.
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0].steps), 1)

class TestPathInvariants(unittest.TestCase):
    def test_step_numbering_is_one_based(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("PROPOSES", "W2", "e_dpr", assertion_id="a_1")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"), adjacency=adjacency, relation_set=_SEMANTIC_RELATIONS
        )
        two_hop = [path for path in paths if len(path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual([step.step for step in two_hop[0].to_evidence_path()], [1, 2])

    def test_evidence_paths_are_connected(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("PROPOSES", "W2", "e_dpr", assertion_id="a_1")],
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq")],
            "e_nq": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"), adjacency=adjacency, relation_set=_SEMANTIC_RELATIONS, hop_budget=3
        )
        for path in paths:
            steps = path.to_evidence_path()
            for prev, current in zip(steps, steps[1:]):
                self.assertEqual(prev.to_id, current.from_id)


class TestNoNeo4jInternalIds(unittest.TestCase):
    def test_source_text_has_no_neo4j_internal_id_references(self):
        from pathlib import Path

        source = Path("src/retrieval/graph.py").read_text()
        forbidden = ("id(", "elementId(", "_Neo4j", "neo4j_id")
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_module_has_no_neo4j_import(self):
        from pathlib import Path

        source = Path("src/retrieval/graph.py").read_text()
        self.assertNotIn("from neo4j", source)
        self.assertNotIn("import neo4j", source)

class TestPhase2PublicSeam(unittest.TestCase):
    def test_retriever_seam_uses_injected_adjacency(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        shared_edge = _edge("CITES", "W1", "W2")
        driver = ForbiddenDriver()
        retr = Neo4jGraphRetriever(
            driver,
            documents=[{"doc_id": "W1", "title": "DPR paper"}],
            relation_set=frozenset({"CITES"}),
            manifest_path=manifest_path,
            repository_root=root,
            graph_content_sha256_reader=lambda _driver, _database: "a" * 64,
            citation_source_sha256_reader=lambda *_: "c" * 64,
        )
        paths = retr.traverse_semantic_paths("DPR paper", adjacency={"W1": [shared_edge], "W2": []})
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].to_evidence_path()[0].relation, "CITES")
        self.assertFalse(driver.accessed)


class TestSelfLoopTraversal(unittest.TestCase):
    def test_self_loop_is_traversable_in_both_directions(self):
        # A canonical self-loop (``source_id == target_id``) is not
        # suppressed. Forward and reverse are both valid; they yield
        # two distinct 1-hop paths with distinct directed identities.
        edge = _edge("PROPOSES", "W1", "W1", assertion_id="a_self")
        adjacency = {"W1": [edge]}
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=frozenset({"PROPOSES"}),
            hop_budget=1,
        )
        self.assertEqual(len(paths), 2)
        seen = sorted(
            (step.from_id, step.relation, step.to_id, step.direction)
            for path in paths for step in path.to_evidence_path()
        )
        self.assertEqual(
            seen,
            [
                ("W1", "PROPOSES", "W1", "forward"),
                ("W1", "PROPOSES", "W1", "reverse"),
            ],
        )
    def test_self_loop_same_directed_identity_cannot_repeat(self):
        # With hop_budget=1 the self-loop yields exactly two 1-hop paths
        # (one forward, one reverse). The repeated-identity rule
        # prevents a same-direction repeat, which is exercised below
        # when we extend a forward 1-hop path and confirm that the
        # forward branch is rejected at hop 2.
        edge = _edge("PROPOSES", "W1", "W1", assertion_id="a_self")
        adjacency = {"W1": [edge]}
        one_hop = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=frozenset({"PROPOSES"}),
            hop_budget=1,
        )
        self.assertEqual(len(one_hop), 2)
        identities = sorted(path.directed_identities for path in one_hop)
        self.assertEqual(
            identities,
            sorted([
                (("ASSERTION", "a_self", "forward"),),
                (("ASSERTION", "a_self", "reverse"),),
            ]),
        )

    def test_reverse_cites_traversal_uses_stored_endpoints(self):
        # CITES stored W1 -> W2 traversed in reverse still reports the
        # canonical (citing, cited) pair in the directed identity; the
        # step's from_id/to_id reflect traversal order, but the
        # canonical base never flips.
        edge = _edge("CITES", "W1", "W2")
        # The same canonical edge must be discoverable from both
        # endpoints so forward and reverse expansions both fire.
        adjacency = {"W1": [edge], "W2": [edge]}
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=frozenset({"CITES"}),
            hop_budget=2,
        )
        two_hop = [path for path in paths if len(path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        steps = two_hop[0].to_evidence_path()
        self.assertEqual((steps[0].from_id, steps[0].to_id, steps[0].direction),
                         ("W1", "W2", "forward"))
        self.assertEqual((steps[1].from_id, steps[1].to_id, steps[1].direction),
                         ("W2", "W1", "reverse"))
        self.assertEqual(
            two_hop[0].directed_identities,
            (
                ("CITES", "W1", "W2", "forward"),
                ("CITES", "W1", "W2", "reverse"),
            ),
        )

    def test_reverse_evaluated_on_traversal_uses_stored_endpoints(self):
        # EVALUATED_ON stored method -> dataset traversed in reverse
        # preserves the canonical (method, dataset) pair.
        edge = _edge(
            "EVALUATED_ON",
            "e_method",
            "e_dataset",
            supporting_assertion_ids=("a_1", "a_2"),
        )
        # The same canonical edge must be discoverable from both
        # endpoints so forward and reverse expansions both fire.
        adjacency = {"e_method": [edge], "e_dataset": [edge]}
        paths = traverse_semantic_paths(
            _seeds("e_method"),
            adjacency=adjacency,
            relation_set=frozenset({"EVALUATED_ON"}),
            hop_budget=2,
        )
        two_hop = [path for path in paths if len(path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        steps = two_hop[0].to_evidence_path()
        self.assertEqual((steps[1].from_id, steps[1].to_id, steps[1].direction),
                         ("e_dataset", "e_method", "reverse"))
        self.assertEqual(
            two_hop[0].directed_identities,
            (
                ("EVALUATED_ON", "e_method", "e_dataset", "forward"),
                ("EVALUATED_ON", "e_method", "e_dataset", "reverse"),
            ),
        )

# ---------------------------------------------------------------------------
# Phase 3 — path-specific evidence collection fixtures.
# ---------------------------------------------------------------------------

# Pure-Python read seams so these tests do not require a live Neo4j driver.
# The mapping is consumed by ``collect_path_evidence`` exactly the way
# the real driver seam will resolve supporting assertions later.
def _static_assertion_resolver(edges_by_assertion_id):
    def _resolve(assertion_id):
        return edges_by_assertion_id.get(assertion_id)
    return _resolve


def _static_chunk_children(children_by_doc_id):
    prepared = {
        doc_id: tuple(sorted({chunk_id for chunk_id in ids}))
        for doc_id, ids in children_by_doc_id.items()
    }
    def _children(doc_id):
        return prepared.get(doc_id, ())
    return _children


def _empty_chunk_children(_doc_id):
    return ()



class TestExtractedAssertionEvidence(unittest.TestCase):
    def test_one_hop_proposes_returns_exact_chunk_id(self):
        # A single PROPOSES edge carrying chunk_id="c_assertion" yields
        # exactly that chunk_id as evidence — no terminal-Document
        # HAS_CHUNK expansion.
        adjacency = {
            "W1": [_edge("PROPOSES", "W1", "e_dpr",
                         assertion_id="a_p", chunk_id="c_assertion")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_empty_chunk_children,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ("c_assertion",))
        self.assertEqual(evidence[0].origin, "assertion")

    def test_multiple_extracted_edges_return_sorted_unique_union(self):
        adjacency = {
            "W1": [_edge("PROPOSES", "W1", "e_dpr",
                         assertion_id="a_p1", chunk_id="c_b")],
            "e_dpr": [_edge("EVALUATES_ON", "e_dpr", "e_nq",
                            assertion_id="a_e1", chunk_id="c_a")],
            "e_nq": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_empty_chunk_children,
        )
        two_hop = [ev for ev in evidence if len(ev.path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(two_hop[0].chunk_ids, ("c_a", "c_b"))

    def test_duplicate_chunk_id_is_deduplicated(self):
        adjacency = {
            "W1": [
                _edge("PROPOSES", "W1", "e_dpr",
                      assertion_id="a_p", chunk_id="c_shared"),
                _edge("INTRODUCES", "W1", "e_nq",
                      assertion_id="a_i", chunk_id="c_shared"),
            ],
            "e_dpr": [],
            "e_nq": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_empty_chunk_children,
        )
        one_hop = [ev for ev in evidence if len(ev.path.steps) == 1]
        self.assertEqual(len(one_hop), 2)
        for ev in one_hop:
            self.assertEqual(ev.chunk_ids, ("c_shared",))

    def test_extracted_assertion_missing_chunk_id_fails_closed(self):
        adjacency = {
            "W1": [_edge("PROPOSES", "W1", "e_dpr",
                         assertion_id="a_p", chunk_id="")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        with self.assertRaises(GraphEvidenceError):
            collect_path_evidence(
                paths,
                adjacency=adjacency,
                assertion_resolver=_static_assertion_resolver({}),
                chunk_children=_empty_chunk_children,
            )

    def test_evidence_read_from_exact_traversed_assertion(self):
        # Two parallel PROPOSES edges share endpoints but have distinct
        # assertion_id + chunk_id; the retriever must read the exact
        # traversed object's chunk_id, not an arbitrary same-relation edge.
        adjacency = {
            "W1": [
                _edge("PROPOSES", "W1", "e_dpr",
                      assertion_id="a_one", chunk_id="c_one"),
                _edge("PROPOSES", "W1", "e_dpr",
                      assertion_id="a_two", chunk_id="c_two"),
            ],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_empty_chunk_children,
        )
        chunks = {ev.chunk_ids[0] for ev in evidence}
        self.assertEqual(chunks, {"c_one", "c_two"})


class TestDerivedEvaluatedOnEvidence(unittest.TestCase):
    def test_single_same_document_witness_returns_both_chunks(self):
        adjacency = {
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq",
                            supporting_assertion_ids=("a_e", "a_p"))],
            "e_nq": [],
        }
        resolver = _static_assertion_resolver({
            "a_p": _edge("PROPOSES", "W1", "e_dpr",
                         assertion_id="a_p", chunk_id="c_p"),
            "a_e": _edge("EVALUATES_ON", "W1", "e_nq",
                         assertion_id="a_e", chunk_id="c_e"),
        })
        paths = traverse_semantic_paths(
            [GraphSeed("Entity", "e_dpr")],
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=resolver,
            chunk_children=_empty_chunk_children,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ("c_e", "c_p"))
        self.assertEqual(evidence[0].origin, "assertion")

    def test_multiple_supporting_assertion_ids_sorted_unique(self):
        adjacency = {
            "W1": [
                _edge("PROPOSES", "W1", "e_dpr",
                      assertion_id="a_p", chunk_id="c_p"),
                _edge("EVALUATES_ON", "W1", "e_nq",
                      assertion_id="a_e", chunk_id="c_e"),
            ],
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq",
                            supporting_assertion_ids=("a_p", "a_e"))],
            "e_nq": [],
        }
        resolver = _static_assertion_resolver({
            "a_p": _edge("PROPOSES", "W1", "e_dpr",
                         assertion_id="a_p", chunk_id="c_p"),
            "a_e": _edge("EVALUATES_ON", "W1", "e_nq",
                         assertion_id="a_e", chunk_id="c_e"),
        })
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
            hop_budget=3,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=resolver,
            chunk_children=_empty_chunk_children,
        )
        two_hop = [ev for ev in evidence if len(ev.path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(two_hop[0].chunk_ids, ("c_e", "c_p"))

    def test_duplicate_support_chunk_ids_deduplicated(self):
        adjacency = {
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq",
                            supporting_assertion_ids=("a_e", "a_p1", "a_p2"))],
            "e_nq": [],
        }
        resolver = _static_assertion_resolver({
            "a_p1": _edge("PROPOSES", "W1", "e_dpr",
                          assertion_id="a_p1", chunk_id="c_shared"),
            "a_p2": _edge("PROPOSES", "W1", "e_dpr",
                          assertion_id="a_p2", chunk_id="c_shared"),
            "a_e": _edge("EVALUATES_ON", "W1", "e_nq",
                         assertion_id="a_e", chunk_id="c_shared"),
        })
        paths = traverse_semantic_paths(
            [GraphSeed("Entity", "e_dpr")],
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=resolver,
            chunk_children=_empty_chunk_children,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ("c_shared",))

    def test_missing_supporting_assertion_fails_closed(self):
        adjacency = {
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq",
                            supporting_assertion_ids=("a_missing",))],
            "e_nq": [],
        }
        paths = traverse_semantic_paths(
            [GraphSeed("Entity", "e_dpr")],
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        with self.assertRaises(StaleDerivedProvenanceError):
            collect_path_evidence(
                paths,
                adjacency=adjacency,
                assertion_resolver=_static_assertion_resolver({}),
                chunk_children=_empty_chunk_children,
            )

    def test_support_resolves_to_non_extracted_relation_fails_closed(self):
        adjacency = {
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq",
                            supporting_assertion_ids=("a_p",))],
            "e_nq": [],
        }
        resolver = _static_assertion_resolver({
            "a_p": _edge("CITES", "W1", "W2"),
        })
        paths = traverse_semantic_paths(
            [GraphSeed("Entity", "e_dpr")],
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        with self.assertRaises(StaleDerivedProvenanceError):
            collect_path_evidence(
                paths,
                adjacency=adjacency,
                assertion_resolver=resolver,
                chunk_children=_empty_chunk_children,
            )

    def test_supporting_assertion_without_chunk_id_fails_closed(self):
        adjacency = {
            "e_dpr": [_edge("EVALUATED_ON", "e_dpr", "e_nq",
                            supporting_assertion_ids=("a_p",))],
            "e_nq": [],
        }
        resolver = _static_assertion_resolver({
            "a_p": _edge("PROPOSES", "W1", "e_dpr",
                         assertion_id="a_p", chunk_id=""),
        })
        paths = traverse_semantic_paths(
            [GraphSeed("Entity", "e_dpr")],
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        with self.assertRaises(StaleDerivedProvenanceError):
            collect_path_evidence(
                paths,
                adjacency=adjacency,
                assertion_resolver=resolver,
                chunk_children=_empty_chunk_children,
            )


class TestDerivedProvenanceConsistency(unittest.TestCase):
    def _derived_paths(self, support_ids):
        adjacency = {
            "e_method": [_edge("EVALUATED_ON", "e_method", "e_dataset",
                               supporting_assertion_ids=support_ids)],
            "e_dataset": [],
        }
        paths = traverse_semantic_paths(
            [GraphSeed("Entity", "e_method")],
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        return adjacency, paths

    def _assert_stale(self, support_ids, resolved):
        adjacency, paths = self._derived_paths(support_ids)
        with self.assertRaises(StaleDerivedProvenanceError):
            collect_path_evidence(
                paths,
                adjacency=adjacency,
                assertion_resolver=_static_assertion_resolver(resolved),
                chunk_children=_empty_chunk_children,
            )

    def test_multiple_same_document_witnesses_return_all_chunks(self):
        adjacency, paths = self._derived_paths(("a_e1", "a_e2", "a_p1", "a_p2"))
        resolver = _static_assertion_resolver({
            "a_p1": _edge("PROPOSES", "W1", "e_method",
                          assertion_id="a_p1", chunk_id="c_p1"),
            "a_e1": _edge("EVALUATES_ON", "W1", "e_dataset",
                          assertion_id="a_e1", chunk_id="c_e1"),
            "a_p2": _edge("PROPOSES", "W2", "e_method",
                          assertion_id="a_p2", chunk_id="c_p2"),
            "a_e2": _edge("EVALUATES_ON", "W2", "e_dataset",
                          assertion_id="a_e2", chunk_id="c_e2"),
        })
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=resolver,
            chunk_children=_empty_chunk_children,
        )
        self.assertEqual(evidence[0].chunk_ids, ("c_e1", "c_e2", "c_p1", "c_p2"))

    def test_reports_metric_support_fails_closed(self):
        self._assert_stale(
            ("a_metric",),
            {"a_metric": _edge("REPORTS_METRIC", "W1", "e_metric",
                                assertion_id="a_metric", chunk_id="c_metric")},
        )

    def test_other_extracted_relation_support_fails_closed(self):
        self._assert_stale(
            ("a_intro",),
            {"a_intro": _edge("INTRODUCES", "W1", "e_method",
                               assertion_id="a_intro", chunk_id="c_intro")},
        )

    def test_proposes_target_not_derived_method_fails_closed(self):
        self._assert_stale(
            ("a_e", "a_p"),
            {
                "a_p": _edge("PROPOSES", "W1", "e_other_method",
                             assertion_id="a_p", chunk_id="c_p"),
                "a_e": _edge("EVALUATES_ON", "W1", "e_dataset",
                             assertion_id="a_e", chunk_id="c_e"),
            },
        )

    def test_evaluates_on_target_not_derived_dataset_fails_closed(self):
        self._assert_stale(
            ("a_e", "a_p"),
            {
                "a_p": _edge("PROPOSES", "W1", "e_method",
                             assertion_id="a_p", chunk_id="c_p"),
                "a_e": _edge("EVALUATES_ON", "W1", "e_other_dataset",
                             assertion_id="a_e", chunk_id="c_e"),
            },
        )

    def test_cross_document_unmatched_pair_fails_closed(self):
        self._assert_stale(
            ("a_e", "a_p"),
            {
                "a_p": _edge("PROPOSES", "W1", "e_method",
                             assertion_id="a_p", chunk_id="c_p"),
                "a_e": _edge("EVALUATES_ON", "W2", "e_dataset",
                             assertion_id="a_e", chunk_id="c_e"),
            },
        )

    def test_resolver_assertion_id_mismatch_fails_closed(self):
        self._assert_stale(
            ("a_expected",),
            {"a_expected": _edge("PROPOSES", "W1", "e_method",
                                  assertion_id="a_different", chunk_id="c_p")},
        )

    def test_empty_supporting_assertion_ids_fail_closed(self):
        self._assert_stale((), {})


class TestMixedPathEvidence(unittest.TestCase):
    def test_cites_then_proposes_uses_only_assertion_chunks(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("PROPOSES", "W2", "e_dpr",
                         assertion_id="a_p", chunk_id="c_p")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_static_chunk_children({
                "W2": ["c_doc_1", "c_doc_2"],
                "W1": ["c_doc_0"],
            }),
        )
        two_hop = [ev for ev in evidence if len(ev.path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(two_hop[0].chunk_ids, ("c_p",))

    def test_proposes_only_path_ignores_terminal_chunks(self):
        adjacency = {
            "W1": [_edge("PROPOSES", "W1", "e_dpr",
                         assertion_id="a_p", chunk_id="c_p")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_static_chunk_children({"W1": ["c_doc_1"]}),
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ("c_p",))

    def test_cites_then_evaluated_on_uses_resolved_derived_chunks(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("EVALUATED_ON", "W2", "e_nq",
                         supporting_assertion_ids=("a_e", "a_p"))],
            "e_nq": [],
        }
        resolver = _static_assertion_resolver({
            "a_p": _edge("PROPOSES", "W3", "W2",
                         assertion_id="a_p", chunk_id="c_p"),
            "a_e": _edge("EVALUATES_ON", "W3", "e_nq",
                         assertion_id="a_e", chunk_id="c_e"),
        })
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=resolver,
            chunk_children=_static_chunk_children({"W2": ["c_doc_1"]}),
        )
        two_hop = [ev for ev in evidence if len(ev.path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(two_hop[0].chunk_ids, ("c_e", "c_p"))
    def test_multiple_cites_and_assertion_uses_only_assertion_chunks(self):
        # Multiple CITES steps then a PROPOSES — the structural
        # HAS_CHUNK fallback must not fire for any intermediate or
        # terminal Document.
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("CITES", "W2", "W3")],
            "W3": [_edge("PROPOSES", "W3", "e_dpr",
                         assertion_id="a_p", chunk_id="c_p")],
            "e_dpr": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
            hop_budget=4,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_static_chunk_children({
                "W1": ["c_doc_1"],
                "W2": ["c_doc_2"],
                "W3": ["c_doc_3a", "c_doc_3b"],
            }),
        )
        three_hop = [ev for ev in evidence if len(ev.path.steps) == 3]
        self.assertEqual(len(three_hop), 1)


class TestPureCitesPathEvidence(unittest.TestCase):
    def test_one_hop_pure_cites_returns_terminal_document_chunks(self):
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_static_chunk_children({
                "W1": ["c_doc_1"],
                "W2": ["c_terminal_a", "c_terminal_b"],
            }),
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ("c_terminal_a", "c_terminal_b"))
        self.assertEqual(evidence[0].origin, "structural_cites")

    def test_two_hop_pure_cites_returns_only_final_terminal(self):
        # Intermediate Document W2's chunks must not appear.
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("CITES", "W2", "W3")],
            "W3": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
            hop_budget=4,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_static_chunk_children({
                "W1": ["c_doc_1"],
                "W2": ["c_intermediate_1", "c_intermediate_2"],
                "W3": ["c_terminal_a", "c_terminal_b"],
            }),
        )
        two_hop = [ev for ev in evidence if len(ev.path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(
            two_hop[0].chunk_ids, ("c_terminal_a", "c_terminal_b")
        )
        self.assertEqual(two_hop[0].origin, "structural_cites")

    def test_two_hop_pure_cites_chunks_sorted_and_deduplicated(self):
        # The provider returns unsorted duplicates; evidence must be
        # sorted and unique.
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_static_chunk_children({
                "W2": ["c_z", "c_a", "c_z", "c_b"],
            }),
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ("c_a", "c_b", "c_z"))

    def test_pure_cites_path_with_empty_terminal_chunks_yields_empty(self):
        adjacency = {"W1": [_edge("CITES", "W1", "W2")], "W2": []}
        paths = traverse_semantic_paths(
            _seeds("W1"), adjacency=adjacency, relation_set=_SEMANTIC_RELATIONS
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_empty_chunk_children,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ())
        self.assertEqual(evidence[0].origin, "structural_cites")

        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_empty_chunk_children,
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chunk_ids, ())
        self.assertEqual(evidence[0].origin, "structural_cites")

    def test_pure_cites_path_intermediate_chunks_excluded(self):
        # Assertion: only the TERMINAL Document contributes
        # ``HAS_CHUNK`` chunks; intermediate Documents' chunks are
        # NOT included.
        adjacency = {
            "W1": [_edge("CITES", "W1", "W2")],
            "W2": [_edge("CITES", "W2", "W3")],
            "W3": [],
        }
        paths = traverse_semantic_paths(
            _seeds("W1"),
            adjacency=adjacency,
            relation_set=_SEMANTIC_RELATIONS,
            hop_budget=4,
        )
        evidence = collect_path_evidence(
            paths,
            adjacency=adjacency,
            assertion_resolver=_static_assertion_resolver({}),
            chunk_children=_static_chunk_children({
                "W1": ["c_W1_se"],
                "W2": ["c_W2_int"],
                "W3": ["c_W3_terminal"],
            }),
        )
        # Filter to the two-hop path (the inner one-hop W1->W2 path
        # would only return c_W2_int on its own).
        two_hop = [ev for ev in evidence if len(ev.path.steps) == 2]
        self.assertEqual(len(two_hop), 1)
        self.assertEqual(two_hop[0].chunk_ids, ("c_W3_terminal",))



class _MemoryGraphDriver:
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def semantic_adjacency(self, node_id):
        return self.adjacency.get(node_id, ())


class TestRetrievalPipeline(unittest.TestCase):
    def _retriever(self, adjacency, *, chunks=None, entities=None, documents=None):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        return Neo4jGraphRetriever(
            _MemoryGraphDriver(adjacency),
            entities=entities or [],
            documents=documents or [{"doc_id": "W1", "title": "Source"}],
            relation_set=_SEMANTIC_RELATIONS,
            manifest_path=manifest_path,
            repository_root=root,
            graph_content_sha256_reader=lambda *_: "a" * 64,
            citation_source_sha256_reader=lambda *_: "c" * 64,
            assertion_resolver=_static_assertion_resolver({
                edge.assertion_id: edge
                for edges in adjacency.values() for edge in edges if edge.assertion_id
            }),
            chunk_children=_static_chunk_children(chunks or {}),
        )

    def test_scores_paths_reduces_chunks_and_orders_after_global_union(self):
        short = _edge("PROPOSES", "W1", "e_short", assertion_id="a_short", chunk_id="c_shared")
        long_a = _edge("PROPOSES", "W1", "e_mid", assertion_id="a_long_a", chunk_id="c_z")
        long_b = _edge("EXTENDS", "e_mid", "e_end", assertion_id="a_long_b", chunk_id="c_shared")
        retr = self._retriever({"W1": [long_a, short], "e_short": [], "e_mid": [long_b], "e_end": []})
        hits = retr.retrieve("Source", 5)
        self.assertEqual([(hit.chunk_id, hit.score) for hit in hits], [("c_shared", 1.0), ("c_z", 1.0)])
        self.assertEqual(hits[0].evidence_path[0].relation, "PROPOSES")

    def test_pure_cites_returns_terminal_chunks_and_respects_k_after_reduction(self):
        cite = _edge("CITES", "W1", "W2")
        retr = self._retriever({"W1": [cite], "W2": []}, chunks={"W2": ["c_b", "c_a"]})
        self.assertEqual(
            [(hit.chunk_id, hit.score) for hit in retr.retrieve("Source", 1)],
            [("c_a", 1.0)],
        )

    def test_equal_score_paths_use_canonical_utf8_tie_break_and_ignore_input_order(self):
        left = _edge("PROPOSES", "W1", "e_left", assertion_id="a_left", chunk_id="c_shared")
        right = _edge("INTRODUCES", "W1", "e_right", assertion_id="a_right", chunk_id="c_shared")
        paths = [
            traverse_semantic_paths(_seeds("W1"), adjacency={"W1": [edge], edge.target_id: []}, relation_set=_SEMANTIC_RELATIONS, hop_budget=1)[0]
            for edge in (left, right)
        ]
        expected = min(paths, key=lambda path: canonical_evidence_path_json(path.to_evidence_path()).encode("utf-8"))
        first = self._retriever({"W1": [left, right], "e_left": [], "e_right": []}).retrieve("Source", 5)
        second = self._retriever({"W1": [right, left], "e_left": [], "e_right": []}).retrieve("Source", 5)
        self.assertEqual(first, second)
        self.assertEqual(first[0].evidence_path, expected.to_evidence_path())

    def test_scores_one_two_three_hop_paths_and_path_chunks_equally(self):
        one = _edge("PROPOSES", "W1", "e1", assertion_id="a1", chunk_id="c_one")
        two_a = _edge("PROPOSES", "W1", "e2", assertion_id="a2", chunk_id="c_two")
        two_b = _edge("EXTENDS", "e2", "e3", assertion_id="a3", chunk_id="c_two_b")
        three_a = _edge("PROPOSES", "W1", "e4", assertion_id="a4", chunk_id="c_three")
        three_b = _edge("EXTENDS", "e4", "e5", assertion_id="a5", chunk_id="c_three_b")
        three_c = _edge("USES_METHOD", "e5", "e6", assertion_id="a6", chunk_id="c_three_c")
        hits = self._retriever({"W1": [one, two_a, three_a], "e1": [], "e2": [two_b], "e3": [], "e4": [three_b], "e5": [three_c], "e6": []}).retrieve("Source", 10)
        scores = {hit.chunk_id: hit.score for hit in hits}
        self.assertEqual(scores["c_one"], 1.0)
        self.assertEqual(scores["c_two_b"], 0.5)
        self.assertEqual(scores["c_three_c"], 1.0 / 3.0)

    def test_multi_seed_evidence_is_globally_reduced_and_ordered(self):
        doc_edge = _edge("PROPOSES", "W1", "e_doc", assertion_id="a_doc", chunk_id="c_shared")
        entity_edge = _edge("INTRODUCES", "e_seed", "e_entity", assertion_id="a_entity", chunk_id="c_shared")
        entity_other = _edge("USES_METHOD", "e_seed", "e_other", assertion_id="a_other", chunk_id="c_b")
        retr = self._retriever(
            {"W1": [doc_edge], "e_doc": [], "e_seed": [entity_edge, entity_other], "e_entity": [], "e_other": []},
            entities=[{"entity_id": "e_seed", "canonical_name": "Entity Seed", "aliases": []}],
        )
        hits = retr.retrieve("Source Entity Seed", 10)
        self.assertEqual([hit.chunk_id for hit in hits], ["c_b", "c_shared"])
        self.assertEqual(len(hits), 2)
    def test_valid_seeded_graph_without_paths_returns_empty(self):
        self.assertEqual(self._retriever({"W1": []}).retrieve("Source", 5), [])

    def test_valid_pure_cites_terminal_without_chunks_returns_empty(self):
        cite = _edge("CITES", "W1", "W2")
        self.assertEqual(self._retriever({"W1": [cite], "W2": []}).retrieve("Source", 5), [])

    def test_chunks_from_same_pure_cites_path_share_its_score(self):
        cite = _edge("CITES", "W1", "W2")
        hits = self._retriever({"W1": [cite], "W2": []}, chunks={"W2": ["c_b", "c_a"]}).retrieve("Source", 5)
        self.assertEqual([(hit.chunk_id, hit.score) for hit in hits], [("c_a", 1.0), ("c_b", 1.0)])

class _OfficialSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self, query, **parameters):
        self.driver.calls.append((query, parameters))
        return self.driver.rows(parameters)


class _OfficialDriver:
    def __init__(self, rows):
        self.rows, self.calls, self.sessions = rows, [], []

    def session(self, *, database, default_access_mode):
        self.database = database
        self.sessions.append(default_access_mode)
        return _OfficialSession(self)


class TestOfficialDriverAdapters(unittest.TestCase):
    def test_adjacency_uses_logical_ids_and_read_only_query(self):
        driver = _OfficialDriver(lambda _: [{
            "relation": "PROPOSES", "source_id": "W1", "target_id": "e1",
            "assertion_id": "a1", "chunk_id": "c1", "supporting_assertion_ids": None,
        }])
        adapter = _OfficialGraphReadDriver(driver, "neo4j", frozenset({"PROPOSES"}))
        self.assertEqual(tuple(adapter.semantic_adjacency("W1")), (_edge("PROPOSES", "W1", "e1", assertion_id="a1", chunk_id="c1"),))
        query, parameters = driver.calls[0]
        self.assertEqual(driver.database, "neo4j")
        self.assertIn("labels(a)", query)
        self.assertNotIn("CREATE", query.upper())
        self.assertNotIn("MERGE", query.upper())
        self.assertEqual(parameters, {"relations": ["PROPOSES"], "node_id": "W1"})
        self.assertEqual(driver.sessions, ["READ"])

    def test_assertion_resolver_fails_closed_for_duplicate_records(self):
        row = {"relation": "PROPOSES", "source_id": "W1", "target_id": "e1", "assertion_id": "a1", "chunk_id": "c1", "supporting_assertion_ids": None}
        driver = _OfficialDriver(lambda _: [row, row])
        resolver = _as_assertion_resolver(driver, "neo4j")
        with self.assertRaises(StaleDerivedProvenanceError):
            resolver("a1")
        self.assertEqual(driver.sessions, ["READ"])

    def test_chunk_children_are_unique_and_sorted(self):
        driver = _OfficialDriver(lambda _: [
            {"chunk_id": "c2"}, {"chunk_id": "c1"}, {"chunk_id": "c2"},
        ])
        provider = _as_chunk_children_provider(driver, "neo4j")
        self.assertEqual(tuple(provider("W1")), ("c1", "c2"))
        self.assertEqual(driver.sessions, ["READ"])

class TestGraphFreshness(unittest.TestCase):
    def _validate(self, manifest_path, root, graph_sha="a" * 64, citation_sha="c" * 64):
        validate_graph_freshness(
            object(),
            manifest_path=manifest_path,
            repository_root=root,
            graph_content_sha256_reader=lambda _driver, _database: graph_sha,
            citation_source_sha256_reader=lambda _manifest, _aliases, _openalex: citation_sha,
        )

    def test_matching_manifest_inputs_and_graph_pass(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        self._validate(manifest_path, root)

    def test_missing_malformed_and_non_object_manifest_fail_closed(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        manifest_path.unlink()
        with self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root)
        manifest_path.write_text("{", encoding="utf-8")
        with self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root)
        manifest_path.write_text("[]", encoding="utf-8")
        with self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root)

    def test_missing_or_malformed_graph_hash_fails_closed(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        manifest = json.loads(manifest_path.read_text())
        for value in (None, "upper" * 16):
            manifest["graph_content_sha256"] = value
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root)

    def test_changed_or_missing_canonical_input_fails_closed(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        input_path = root / _FRESH_INPUT_PATHS["chunks"]
        input_path.write_bytes(b"changed")
        with self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root)
        input_path.unlink()
        with self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root)

    def test_citation_digest_missing_or_malformed_fails_closed(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        manifest = json.loads(manifest_path.read_text())
        for value in (None, "", "C" * 64, "g" * 64, "c" * 63):
            if value is None:
                manifest.pop("citation_source_sha256", None)
            else:
                manifest["citation_source_sha256"] = value
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(GraphFreshnessError):
                self._validate(manifest_path, root)

    def test_citation_drift_and_reader_failure_fail_before_graph_readback(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        calls = []
        with self.assertRaises(GraphFreshnessError):
            validate_graph_freshness(
                object(), manifest_path=manifest_path, repository_root=root,
                citation_source_sha256_reader=lambda *_: "d" * 64,
                graph_content_sha256_reader=lambda *_: calls.append("graph") or "a" * 64,
            )
        self.assertEqual(calls, [])
        with self.assertRaises(GraphFreshnessError):
            validate_graph_freshness(
                object(), manifest_path=manifest_path, repository_root=root,
                citation_source_sha256_reader=lambda *_: (_ for _ in ()).throw(RuntimeError()),
                graph_content_sha256_reader=lambda *_: "a" * 64,
            )

    def test_symlinked_canonical_input_outside_root_fails_closed(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        outside = root.parent / "outside-input"
        outside.write_bytes(b"chunks")
        self.addCleanup(outside.unlink)
        (root / _FRESH_INPUT_PATHS["chunks"]).unlink()
        (root / _FRESH_INPUT_PATHS["chunks"]).symlink_to(outside)
        with self.assertRaises(GraphFreshnessError):
            self._validate(manifest_path, root)

    def test_production_citation_recomputation_detects_raw_openalex_drift(self):
        temporary, root, ingestion_manifest = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        openalex_dir = root / "data/raw/openalex"
        openalex_dir.mkdir(parents=True)
        raw_path = openalex_dir / "W1.json"
        raw_path.write_text(json.dumps({"referenced_works": ["https://openalex.org/W2"]}))
        canonical_manifest = root / _FRESH_INPUT_PATHS["manifest"]
        canonical_manifest.write_text(json.dumps({
            "openalex_id": "W1", "title": "Paper",
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        }) + "\n")
        aliases_path = root / _FRESH_INPUT_PATHS["aliases"]
        aliases_path.write_text("{}")
        manifest = json.loads(ingestion_manifest.read_text())
        for name in ("manifest", "aliases"):
            path = root / _FRESH_INPUT_PATHS[name]
            manifest["inputs"][f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        from src.build_graph.preflight import _build_citation_source, _load_aliases, _validate_documents
        documents = _validate_documents(canonical_manifest, openalex_dir)
        _, source_bytes = _build_citation_source(documents, openalex_dir, _load_aliases(aliases_path))
        manifest["citation_source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        ingestion_manifest.write_text(json.dumps(manifest))
        validate_graph_freshness(
            object(), manifest_path=ingestion_manifest, repository_root=root,
            openalex_dir=openalex_dir, graph_content_sha256_reader=lambda *_: "a" * 64,
        )
        raw_path.write_text(json.dumps({"referenced_works": ["W3"]}))
        with self.assertRaises(GraphFreshnessError):
            validate_graph_freshness(
                object(), manifest_path=ingestion_manifest, repository_root=root,
                openalex_dir=openalex_dir, graph_content_sha256_reader=lambda *_: "a" * 64,
            )

    def test_ontology_disagreement_graph_readback_failure_and_mismatch_fail_closed(self):
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        manifest = json.loads(manifest_path.read_text())
        manifest["ontology"]["ontology_sha256"] = "b" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root)
        manifest["ontology"]["ontology_sha256"] = manifest["inputs"]["ontology_sha256"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(GraphFreshnessError):
            validate_graph_freshness(
                object(), manifest_path=manifest_path, repository_root=root,
                graph_content_sha256_reader=lambda _driver, _database: (_ for _ in ()).throw(RuntimeError()),
                citation_source_sha256_reader=lambda *_: "c" * 64,
            )
        with self.assertRaises(GraphFreshnessError): self._validate(manifest_path, root, "b" * 64)

    def test_retriever_short_circuits_before_freshness_and_validates_before_driver(self):
        calls = []
        retr = Neo4jGraphRetriever(
            ForbiddenDriver(), documents=[{"doc_id": "W1", "title": "paper"}],
            relation_set=frozenset({"CITES"}),
            graph_content_sha256_reader=lambda _driver, _database: calls.append("read") or "a" * 64,
            citation_source_sha256_reader=lambda *_: "c" * 64,
        )
        with self.assertRaises(ValueError): retr.retrieve("paper", 1, 0)
        self.assertEqual(retr.retrieve(" ", 1), [])
        self.assertEqual(retr.retrieve("unmatched", 1), [])
        self.assertEqual(calls, [])
        temporary, root, manifest_path = _freshness_fixture()
        self.addCleanup(temporary.cleanup)
        retr._manifest_path, retr._repository_root = manifest_path, root
        with self.assertRaises(AssertionError): retr.traverse_semantic_paths("paper")
        self.assertEqual(calls, ["read"])
