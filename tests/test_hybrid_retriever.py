"""Focused deterministic contract tests for Issue #29 hybrid fusion."""

from __future__ import annotations

import dataclasses
import inspect
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.graph import GraphEvidenceStep, GraphRetrievalHit
from src.retrieval.hybrid import (
    HYBRID_HOP_BUDGET,
    HYBRID_K_GRAPH_MULTIPLIER,
    HYBRID_K_VEC_MULTIPLIER,
    RRF_K,
    HybridHitProvenance,
    HybridRetrievalHit,
    HybridRetriever,
    RRFHybridRetriever,
)
from src.retrieval.vector import VectorRetrievalHit


class FakeVectorRetriever:
    def __init__(self, hits=(), error: Exception | None = None):
        self.hits = list(hits)
        self.error = error
        self.calls = []

    def retrieve(self, query: str, k: int):
        self.calls.append((query, k))
        if self.error:
            raise self.error
        return list(self.hits)


class FakeGraphRetriever:
    def __init__(self, hits=(), error: Exception | None = None):
        self.hits = list(hits)
        self.error = error
        self.calls = []

    def retrieve(self, query: str, k: int, hop_budget: int = 3):
        self.calls.append((query, k, hop_budget))
        if self.error:
            raise self.error
        return list(self.hits)


def vector(chunk_id: str, score: float) -> VectorRetrievalHit:
    return VectorRetrievalHit(chunk_id, score)


def graph(chunk_id: str, score: float, path=()) -> GraphRetrievalHit:
    return GraphRetrievalHit(chunk_id, score, path)


def retriever(vector_hits=(), graph_hits=()):
    vectors = FakeVectorRetriever(vector_hits)
    graphs = FakeGraphRetriever(graph_hits)
    return RRFHybridRetriever(vectors, graphs), vectors, graphs


def ranked_vectors(target: str, rank: int, score: float) -> list[VectorRetrievalHit]:
    return [vector(f"v-{index:03}", -index) for index in range(1, rank)] + [vector(target, score)]


def ranked_graphs(target: str, rank: int, score: float, path=()) -> list[GraphRetrievalHit]:
    return [graph(f"g-{index:03}", -index) for index in range(1, rank)] + [graph(target, score, path)]


class TestPublicContract(unittest.TestCase):
    def test_constants_dataclass_shapes_identity_and_protocol_are_exact(self):
        self.assertEqual(
            (HYBRID_K_VEC_MULTIPLIER, HYBRID_K_GRAPH_MULTIPLIER, HYBRID_HOP_BUDGET, RRF_K),
            (2, 2, 3, 60),
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(HybridHitProvenance)],
            ["in_vector", "in_graph", "vector_rank", "vector_score", "graph_rank", "graph_score", "graph_evidence_path"],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(HybridRetrievalHit)],
            ["chunk_id", "score", "sources", "provenance"],
        )
        provenance = HybridHitProvenance(True, False, 1, 0.5, None, None, None)
        hit = HybridRetrievalHit("c", 1.0, ("vector",), provenance)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            provenance.in_vector = False  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            hit.score = 0.0  # type: ignore[misc]
        from src.retrieval.hybrid import GraphEvidenceStep as ReExportedStep
        import src.retrieval.hybrid as hybrid

        self.assertIs(ReExportedStep, GraphEvidenceStep)
        self.assertFalse(hasattr(hybrid, "HybridProvenance"))
        self.assertNotIn("rrf_score", [field.name for field in dataclasses.fields(HybridHitProvenance)])
        with self.assertRaises(TypeError):
            HybridRetrievalHit("c", 1.0, ("vector",), hybrid_hit_provenance=provenance)  # type: ignore[call-arg]
        self.assertEqual(list(inspect.signature(HybridRetriever.retrieve).parameters), ["self", "query", "k"])
        self.assertIsInstance(retriever()[0], HybridRetriever)


class TestValidationAndCalls(unittest.TestCase):
    def test_invalid_and_whitespace_inputs_do_not_call_upstream(self):
        hybrid, vectors, graphs = retriever()
        for query, k, error in ((None, 1, TypeError), (1, 1, TypeError), ("q", True, TypeError), ("q", 1.5, TypeError), ("q", "1", TypeError), ("q", 0, ValueError), ("q", -1, ValueError)):
            with self.subTest(query=query, k=k), self.assertRaises(error):
                hybrid.retrieve(query, k)  # type: ignore[arg-type]
        self.assertEqual(hybrid.retrieve(" \t\n", 1), [])
        self.assertEqual(vectors.calls, [])
        self.assertEqual(graphs.calls, [])

    def test_valid_query_calls_both_once_with_frozen_arguments(self):
        hybrid, vectors, graphs = retriever()
        self.assertEqual(hybrid.retrieve("  literal query  ", 4), [])
        self.assertEqual(vectors.calls, [("  literal query  ", 8)])
        self.assertEqual(graphs.calls, [("  literal query  ", 8, 3)])

    def test_both_retrievers_are_called_when_either_side_is_empty(self):
        for vector_hits, graph_hits, expected in (
            ((), (graph("g", 0.5),), "g"),
            ((vector("v", 0.5),), (), "v"),
        ):
            with self.subTest(expected=expected):
                hybrid, vectors, graphs = retriever(vector_hits, graph_hits)
                self.assertEqual([hit.chunk_id for hit in hybrid.retrieve("q", 1)], [expected])
                self.assertEqual(vectors.calls, [("q", 2)])
                self.assertEqual(graphs.calls, [("q", 2, 3)])

    def test_failures_propagate_without_extra_call(self):
        vector_error = RuntimeError("vector failed")
        graphs = FakeGraphRetriever()
        hybrid = RRFHybridRetriever(FakeVectorRetriever(error=vector_error), graphs)
        with self.assertRaises(RuntimeError) as caught:
            hybrid.retrieve("q", 1)
        self.assertIs(caught.exception, vector_error)
        self.assertEqual(graphs.calls, [])
        graph_error = RuntimeError("graph failed")
        vectors = FakeVectorRetriever()
        hybrid = RRFHybridRetriever(vectors, FakeGraphRetriever(error=graph_error))
        with self.assertRaises(RuntimeError) as caught:
            hybrid.retrieve("q", 1)
        self.assertIs(caught.exception, graph_error)
        self.assertEqual(vectors.calls, [("q", 2)])


class TestRRF(unittest.TestCase):
    def test_rank_table_and_raw_score_independence(self):
        cases = (
            (1, None, 1 / 61),
            (2, None, 1 / 62),
            (None, 1, 1 / 61),
            (None, 3, 1 / 63),
            (1, 2, 1 / 61 + 1 / 62),
            (2, 3, 1 / 62 + 1 / 63),
        )
        for vector_rank, graph_rank, expected in cases:
            with self.subTest(vector_rank=vector_rank, graph_rank=graph_rank):
                vector_hits = ranked_vectors("target", vector_rank, 0.001) if vector_rank else ()
                graph_hits = ranked_graphs("target", graph_rank, 999999.0) if graph_rank else ()
                hybrid, _vectors, _graphs = retriever(vector_hits, graph_hits)
                hit = next(hit for hit in hybrid.retrieve("q", 4) if hit.chunk_id == "target")
                self.assertEqual(hit.score, expected)

        first, _vectors, _graphs = retriever(ranked_vectors("target", 2, -1e300), ranked_graphs("target", 3, 1e300))
        second, _vectors, _graphs = retriever(ranked_vectors("target", 2, 1e300), ranked_graphs("target", 3, -1e300))
        first_hit = next(hit for hit in first.retrieve("q", 4) if hit.chunk_id == "target")
        second_hit = next(hit for hit in second.retrieve("q", 4) if hit.chunk_id == "target")
        self.assertEqual(first_hit.score, second_hit.score)
        self.assertEqual(first_hit.score, 1 / 62 + 1 / 63)


class TestProvenance(unittest.TestCase):
    def test_absence_and_graph_evidence_identity_are_exact(self):
        path = (GraphEvidenceStep(1, "e", "CITES", "d", "forward"),)
        hybrid, _vectors, _graphs = retriever([vector("vector-only", 0.42)], [graph("graph-only", 0.75, path)])
        hits = {hit.chunk_id: hit for hit in hybrid.retrieve("q", 2)}
        vector_provenance = hits["vector-only"].provenance
        self.assertTrue(vector_provenance.in_vector)
        self.assertFalse(vector_provenance.in_graph)
        self.assertIsNotNone(vector_provenance.vector_rank)
        self.assertIsNotNone(vector_provenance.vector_score)
        self.assertIsNone(vector_provenance.graph_rank)
        self.assertIsNone(vector_provenance.graph_score)
        self.assertIsNone(vector_provenance.graph_evidence_path)
        graph_provenance = hits["graph-only"].provenance
        self.assertFalse(graph_provenance.in_vector)
        self.assertTrue(graph_provenance.in_graph)
        self.assertIsNone(graph_provenance.vector_rank)
        self.assertIsNone(graph_provenance.vector_score)
        self.assertIsNotNone(graph_provenance.graph_rank)
        self.assertIsNotNone(graph_provenance.graph_score)
        self.assertIs(graph_provenance.graph_evidence_path, path)


class TestOrdering(unittest.TestCase):
    def test_equal_score_dual_source_precedes_single_source(self):
        vector_hits = [vector("single", 0.1)] + [vector(f"v-{index:03}", 0.1) for index in range(2, 62)] + [vector("dual", 0.1)]
        graph_hits = [graph(f"g-{index:03}", 0.1) for index in range(1, 62)] + [graph("dual", 0.1)]
        hybrid, _vectors, _graphs = retriever(vector_hits, graph_hits)
        hits = hybrid.retrieve("q", 31)
        by_id = {hit.chunk_id: hit for hit in hits}
        self.assertEqual(by_id["dual"].score, by_id["single"].score)
        self.assertEqual(by_id["dual"].score, 1 / 61)
        self.assertLess([hit.chunk_id for hit in hits].index("dual"), [hit.chunk_id for hit in hits].index("single"))
        self.assertEqual(len(by_id["dual"].sources), 2)
        self.assertEqual(len(by_id["single"].sources), 1)

    def test_equal_single_source_scores_use_chunk_id_without_modality_preference(self):
        hybrid, _vectors, _graphs = retriever([vector("z", 0.1)], [graph("a", 0.1)])
        hits = hybrid.retrieve("q", 2)
        self.assertEqual([hit.chunk_id for hit in hits], ["a", "z"])
        self.assertEqual(hits[0].sources, ("graph",))
        self.assertEqual(hits[1].sources, ("vector",))

    def test_final_key_uses_utf8_byte_order(self):
        ids = ["ä", "a"]
        hybrid, _vectors, _graphs = retriever(
            [vector("ä", 0.1), vector("a", 0.1)],
            [graph("a", 0.1), graph("ä", 0.1)],
        )
        self.assertEqual(
            [hit.chunk_id for hit in hybrid.retrieve("q", 2)],
            sorted(ids, key=lambda chunk_id: chunk_id.encode("utf-8")),
        )


class TestResultSizeAndDeterminism(unittest.TestCase):
    def test_truncation_happens_after_global_fusion(self):
        hybrid, vectors, graphs = retriever(
            [vector("vector-first", 1.0), vector("promoted", 0.5)],
            [graph("promoted", 0.5)],
        )
        hits = hybrid.retrieve("q", 1)
        self.assertEqual([hit.chunk_id for hit in hits], ["promoted"])
        self.assertEqual(vectors.calls, [("q", 2)])
        self.assertEqual(graphs.calls, [("q", 2, 3)])

    def test_equivalent_recreated_inputs_produce_equal_values(self):
        def fixture():
            return retriever(
                [vector("v", 0.91), vector("both", 0.2)],
                [graph("both", 0.8, (GraphEvidenceStep(1, "e", "CITES", "d", "forward"),))],
            )[0]

        self.assertEqual(fixture().retrieve("q", 2), fixture().retrieve("q", 2))


class TestStaticGuards(unittest.TestCase):
    def test_hybrid_source_has_no_forbidden_dependency_or_tuning_reference(self):
        source = (Path(__file__).resolve().parents[1] / "src/retrieval/hybrid.py").read_text()
        for forbidden in (
            "eval/questions.jsonl", "eval/dev_questions.jsonl", "LLMProvider", "EmbeddingProvider",
            "pgvector", "psycopg", "neo4j", "LangChain", "sentence_transformers", "fuzzy",
            "rerank", "cross-encoder", "cross_encoder", "GraphRetrievalHit", "VectorRetrievalHit",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
