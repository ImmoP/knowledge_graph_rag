"""Deterministic reciprocal-rank fusion over vector and graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from src.retrieval.graph import GraphEvidenceStep, GraphRetriever
from src.retrieval.vector import VectorRetriever

HYBRID_K_VEC_MULTIPLIER = 2
HYBRID_K_GRAPH_MULTIPLIER = 2
HYBRID_HOP_BUDGET = 3
RRF_K = 60


@dataclass(frozen=True)
class HybridHitProvenance:
    in_vector: bool
    in_graph: bool
    vector_rank: int | None
    vector_score: float | None
    graph_rank: int | None
    graph_score: float | None
    graph_evidence_path: tuple[GraphEvidenceStep, ...] | None


@dataclass(frozen=True)
class HybridRetrievalHit:
    chunk_id: str
    score: float
    sources: tuple[Literal["vector", "graph"], ...]
    provenance: HybridHitProvenance


@runtime_checkable
class HybridRetriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[HybridRetrievalHit]: ...


class RRFHybridRetriever(HybridRetriever):
    """Fuse the fixed-depth ranked outputs of the two retriever ports."""

    def __init__(self, vector_retriever: VectorRetriever, graph_retriever: GraphRetriever) -> None:
        self._vector_retriever = vector_retriever
        self._graph_retriever = graph_retriever

    def retrieve(self, query: str, k: int) -> list[HybridRetrievalHit]:
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        _validate_k(k)
        if not query.strip():
            return []

        vector_hits = self._vector_retriever.retrieve(
            query, k * HYBRID_K_VEC_MULTIPLIER
        )
        graph_hits = self._graph_retriever.retrieve(
            query,
            k * HYBRID_K_GRAPH_MULTIPLIER,
            hop_budget=HYBRID_HOP_BUDGET,
        )

        vector_by_chunk = {}
        for rank, hit in enumerate(vector_hits, start=1):
            vector_by_chunk.setdefault(hit.chunk_id, (rank, hit.score))

        graph_by_chunk = {}
        for rank, hit in enumerate(graph_hits, start=1):
            graph_by_chunk.setdefault(hit.chunk_id, (rank, hit.score, hit.evidence_path))

        fused: list[HybridRetrievalHit] = []
        for chunk_id in vector_by_chunk.keys() | graph_by_chunk.keys():
            vector = vector_by_chunk.get(chunk_id)
            graph = graph_by_chunk.get(chunk_id)
            sources = tuple(
                source
                for source, present in (("graph", graph is not None), ("vector", vector is not None))
                if present
            )
            score = (1.0 / (RRF_K + vector[0]) if vector else 0.0) + (
                1.0 / (RRF_K + graph[0]) if graph else 0.0
            )
            fused.append(
                HybridRetrievalHit(
                    chunk_id=chunk_id,
                    score=score,
                    sources=sources,
                    provenance=HybridHitProvenance(
                        in_vector=vector is not None,
                        in_graph=graph is not None,
                        vector_rank=vector[0] if vector else None,
                        vector_score=vector[1] if vector else None,
                        graph_rank=graph[0] if graph else None,
                        graph_score=graph[1] if graph else None,
                        graph_evidence_path=graph[2] if graph else None,
                    ),
                )
            )
        return sorted(
            fused, key=lambda hit: (-hit.score, -len(hit.sources), hit.chunk_id.encode("utf-8"))
        )[:k]


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got k={k}")


__all__ = [
    "GraphEvidenceStep",
    "HYBRID_HOP_BUDGET",
    "HYBRID_K_GRAPH_MULTIPLIER",
    "HYBRID_K_VEC_MULTIPLIER",
    "HybridHitProvenance",
    "HybridRetrievalHit",
    "HybridRetriever",
    "RRFHybridRetriever",
    "RRF_K",
]
