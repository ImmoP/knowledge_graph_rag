"""Benchmark dataset package.

The benchmark compares the locally-running Qwen3 8B with the
Manifest-gateway-hosted GPT-5.6 Terra on the **same** chunk set,
**same** prompts, **same** ontology, **same** schema, **same**
parameters. Selection stability is the contract: every benchmark run
operates on the byte-identical frozen chunk set committed in this
package. The two smoke chunks ``c_0001921855a8156b`` and
``c_001452fd142ce075`` are explicitly pinned for the four-call
initial smoke comparison so the operator can confirm the gateway
end-to-end before opening the larger dataset.

The package layout is intentionally minimal:

* ``select.py`` — deterministic chunk-set selection.
* ``smoke_chunks.py`` — pinned smoke chunks and ordering.
* ``manifest.py`` — frozen manifest (selection algorithm version,
  seed, SHA-256 of the resulting chunk set, counts).

The frozen dataset files (``benchmark_chunks.jsonl``,
``benchmark_chunks.sha256``) live in this package and are produced
by ``scripts/benchmark/build_dataset.py``. Re-running the build
must produce a byte-identical chunk set and manifest; any drift
fails closed.
"""