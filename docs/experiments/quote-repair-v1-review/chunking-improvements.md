# Chunking Improvement Options (Scientific Limits, Reviewer-Oriented)

**Status:** analysis only — no code or chunk artefact is changed.
**Scope:** the three review entries (``a_44999b3871c3f58f``,
``a_0a775699fcc8a5ce``, ``a_2814c002a3ebd912``) and the chunk
``c_001452fd142ce075`` that supplies their evidence.
**Contract:** the V2 extraction contract
(``docs/specs/extraction-contract.md`` §3, §5, §8) is **not** modified.
No human review labels are assigned.

## 1. Per-Assertion Self-Sufficiency (own-chunk evidence only)

This section answers: *which assertions can be supported purely from
the local chunk text, and which would require text outside the chunk?*

| Assertion | Object | What the chunk evidence alone supports | What still lies outside the chunk |
|---|---|---|---|
| ``a_44999b3871c3f58f`` | ``Answer Coverage at Top-20/100`` (metric) | The metric name appears as a verbatim substring. The words *Retrieval results (Answer Coverage at Top-20/100) on …* are present. | The numeric value of the metric (no Top-20/100 numbers in the chunk). The system name (DPR/ANCE/BM25 are absent from the span). |
| ``a_0a775699fcc8a5ce`` | ``Natural Questions (NQ)`` (dataset) | The dataset mention appears as a verbatim substring. The connector phrase *… on Natural Questions (NQ) and …* is present. | The numeric NQ value (in ``Tables/openqa.tex``, ends at cleaned-text offset ``37059`` — **before** the chunk start ``37074``). The system name. The binding sentence *Evaluation.tex:19*. |
| ``a_2814c002a3ebd912`` | ``Trivial QA (TQA)`` (dataset) | Identical pattern to the NQ assertion: dataset mention and connector phrase present. | Identical pattern: TQA numeric value, system name, ``Evaluation.tex:19``. |

**Conclusion:** all three assertions can be **structurally** validated
against the local chunk (the V2 gates pass), but **scientifically** each
assertion needs at least one of:

1. The numeric table body in ``Tables/openqa.tex`` (outside the chunk);
2. The prose binding in ``Evaluation.tex:19`` (outside the chunk);
3. A system-name occurrence that the chunk does not contain.

The existing chunk boundary therefore sets the structural ceiling of
the contract's evidence_span gate, not the scientific ceiling of the
reviewer's ``supported / unsupported / unclear`` decision.

## 2. Why the Current Chunk Boundary Stops Where It Does

The chunker is sized for ``TOKEN_BUDGET = 512`` in
``src/chunk/chunker.py`` (cap on ``len(text) // 4``). The chunk
``c_001452fd142ce075`` starts at cleaned-text offset ``37074`` and runs
to ``37909`` (835 chars / ~209 estimated tokens). The chunk
boundary was chosen by the upstream chunker when the source tree was
extracted; the manifest does not change it.

Two structural facts are observable from the local material:

* The ``Tables/openqa.tex`` body, which holds the NQ/TQA Top-20/100
  numbers, ends at cleaned-text offset ``37059`` — i.e. **15 characters
  before** the chunk start. The chunker therefore cut the body off by
  exactly one floating-environment boundary (the ``\input`` of
  ``Tables/openqa.tex`` is completed just before the caption that the
  chunk keeps).
* The sentence in ``Evaluation.tex:19`` that names DPR lives in a
  different ``.tex`` file and is included from ``Evaluation.tex`` by a
  separate ``\input``; it sits in a different ``\section`` block. It is
  therefore unreachable by any boundary tweak that keeps the caption
  and its label inside the chunk.

## 3. Chunking Improvement Options (none implemented here)

Each option below is described as a **future change**, not as a
delivered fix. The intent is to make the options comparable to the
reviewer, not to commit to one.

### Option 1 — Enlarge the chunk by ~80 chars on the left

* **What changes:** shift the chunk start from cleaned-text ``37074``
  to ``~36994`` to include the ``Tables/openqa.tex`` tabular body up to
  the row that contains the NQ Top-20/100 value (and the matching TQA
  row).
* **What it would gain:** the missing numeric values for both datasets
  and the metric, *within the same chunk*, become evidence_span-
  admissible. Both ``EVALUATES_ON`` assertions gain a numeric anchor
  that the structural gate can re-use (currently only the *connector*
  phrase is in the chunk).
* **What it would NOT gain:** the system name. ``DPR`` does not appear
  in the body of ``Tables/openqa.tex`` as a system — the table is a
  method-vs-dataset layout. The system name still lives in
  ``Evaluation.tex:19`` only.
* **Contract impact:** zero. ``evidence_quote`` remains a verbatim
  substring of ``chunk.text``; the relation count and entity
  extraction gates are unchanged. The chunk shape (size, count) is
  allowed to vary per ``source chunking strategy`` (§3.4).
* **Risk:** the chunk text length grows by ~80 chars but stays well
  below ``TOKEN_BUDGET = 512``. No new mention spans are introduced.

### Option 2 — Enlarge the chunk by one paragraph upward to capture
``Evaluation.tex:19``

* **What changes:** shift the chunk start back through ``Evaluation.tex``
  to include the binding sentence *It also outperforms DPR in passage
  retrieval in OpenQA (Table~\ref{tab:openqa}).*
* **What it would gain:** the missing system name (``DPR``) inside the
  chunk, plus the dataset family (``OpenQA``) which is implicit in the
  caption but explicit in the prose.
* **What it would NOT gain:** the numeric values (still in
  ``Tables/openqa.tex``).
* **Contract impact:** zero — same as Option 1.
* **Risk:** the chunk then crosses a ``\section`` boundary. The current
  chunker is happy to cross sections if the budget allows, but the
  chunk's ``section_path`` metadata would need to reflect the wider
  span, and downstream consumers that filter on ``section_path`` would
  see one fewer chunk boundary. This is a downstream-visibility risk,
  not a contract risk.

### Option 3 — Cross-chunk `evidence_quote` with a windowed pointer

* **What changes:** allow ``evidence_quote`` to be a literal that is
  byte-equivalent to a substring of the chunk's ``text`` plus a
  reference-only offset into an adjacent chunk's text.
* **Why we describe it but do NOT propose it here:** the V2 contract
  fixes ``evidence_quote`` as a verbatim substring of ``chunk.text``
  (extraction contract §3 deterministic span semantics, §8.3 evidence
  integrity gate). Any cross-chunk ``evidence_quote`` would break the
  evidence integrity gate by definition. Listing it here as an option
  serves only to mark the architectural limit; it cannot be adopted
  without a contract change, which is out of scope.

### Option 4 — Re-chunk the source so the caption and its **own** body
are co-located

* **What changes:** split or merge source files so that the
  ``\caption{Retrieval results (Answer Coverage at Top-20/100) …}`` and
  the ``Tables/openqa.tex`` rows it labels land in the same chunk.
* **What it would gain:** the natural scientific grouping
  (caption+body) and Option 1's gain at the same time.
* **Why we describe it but do NOT propose it here:** the source tree is
  the arXiv tarball at ``data/raw/latex/2007.00808.tar.gz``. The
  chunker accepts the upstream ``ms.tex`` as-is. Re-chunking would
  require either (a) editing the source tree before extraction (which
  is not a chunking change but a source-tree change) or (b) introducing
  a re-chunking layer that violates the chunker's current
  "structure-aware" assumption. Neither is in scope.

## 4. Recommended Next Step (no implementation now)

The reviewer-relevant gap is the **system-name gap**, not the
**numeric-value gap**. The numeric values alone do not establish a
``supported`` verdict under ``gold_set/GUIDELINES.md`` §4.1.3 and §4.1.5
either; the binding sentence is the discriminating piece of evidence.
Therefore:

* Option 2 (enlarge by one paragraph upward to capture the binding
  prose) is the smallest change that closes the most consequential
  scientific gap. Option 1 is the smallest change that closes the
  structural numeric gap. Both are cheap to implement offline once a
  re-chunked source is regenerated.
* Option 3 is rejected on contract grounds.
* Option 4 is rejected on out-of-scope grounds (it modifies the source
  tree, not the chunker).
* None of these changes is delivered in this branch. The branch
  carries the manifest update, the offline diagnostic test for
  ``c_012d2ff7efe77f40``, and the bounded plan for a single-chunk
  targeted model trial; the chunker, the prompts, the ontology, and
  the V2 contract are not touched.

## 5. What This Document Does NOT Claim

* No new ``supported / unsupported / unclear`` verdict is assigned.
* No assertion is moved from one confidence bucket to another.
* No claim is made about Precision/Recall on any gold subset.
* No claim is made about retrieval or downstream answer quality.
* No change to ``src/chunk/chunker.py``, ``src/extract/pipeline_v2.py``,
  ``prompts/extract_entities_v2.txt``, ``prompts/extract_relations_v2.txt``,
  ``ontology.yaml``, or any test fixture is in this branch.