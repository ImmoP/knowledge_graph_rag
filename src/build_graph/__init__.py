"""Neo4j graph ingestion (#25, #9).

Implements ``docs/specs/neo4j-ingestion.md`` and impl-spec §9 verbatim.
The pipeline:

  1. ``preflight.run_preflight`` — the §2 hard preflight; raises
     :class:`src.build_graph.preflight.PreflightError` on any failure.
  2. ``ingestion.bootstrap_schema`` — idempotent uniqueness constraints
     and lookup indexes from §5 (runs separately from the data
     transaction per §7).
  3. ``ingestion.replace_graph`` — single-write-transaction replacement
     per §7; deletes prior graph data and rebuilds from canonical
     artifacts in one transaction with post-write integrity checks.
  4. ``ingestion.run_ingestion`` — top-level orchestrator: preflight,
     bootstrap, replace, compute canonical hash, atomically publish
     ``data/graph_ingestion.manifest.json`` per §8.

This module never reads LLMProvider, EmbeddingProvider, pgvector, or any
provider SDK. It loads ``NEO4J_URI``, ``NEO4J_USER``, ``NEO4J_PASSWORD``
from the repository ``.env`` (gitignored) and connects only through the
official ``neo4j`` Python driver. No credential is written to logs,
graph properties, or the ingestion manifest (§9).
"""