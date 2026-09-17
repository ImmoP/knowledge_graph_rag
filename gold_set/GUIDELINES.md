# Gold-Set v1 — Authoring Guidelines and Adjudication Rubric

```text
annotation_guideline_version = 1
```

This document is the normative human-annotation protocol for the
extraction-contract v2 gold-set pilot defined in #48. It governs how
the 120-chunk pilot is annotated, how disagreements are resolved, and
how the per-chunk JSON records under
`gold_set/chunks/<chunk_id>.json` are authored. The document is
documentation only; it does not implement the sampler, the annotation
tooling, the leakage audit, or the pilot runner.

Every `gold-record.v1` produced under this version MUST include
`annotation_guideline_sha256 = sha256(this file)` as one of its
reproducibility pins (#48 §10).

## 0. Authority and precedence

- The authoritative vocabulary is `ontology.yaml` at the time of
  authoring. The `annotation_guideline_version` and
  `ontology_version` recorded on each chunk record jointly define the
  annotation contract.
- The authoritative extraction-contract semantics for v2 are pinned
  by #46 (fail-closed exact evidence alignment) and #49
  (deterministic exact-span v2 aligner). This document restates the
  relevant rules as they apply to a human annotator; on any
  conflict, #46 and #49 win.
- The authoritative pilot methodology and adjudication protocol are
  pinned by #48. This document restates #48 §3 (annotation
  protocol), §4 (adjudication protocol), and §10 (versioning) as a
  single per-chunk operating procedure. On any conflict, #48 wins.

## 1. Scope and annotation unit

The annotation unit is one canonical chunk from
`data/derived/chunks.jsonl`. Every per-chunk record is keyed by the
chunk's `chunk_id` (the stable, content-independent chunk
  identifier defined by the chunk-record contract; never edited).

- `chunk.text` is the single source of evidence. Annotators MUST
  work against `chunk.text` with exact Python-string equality.
  `char_start` / `char_end` are Python string character offsets into
  `chunk.text`. No Unicode, case, or whitespace normalisation; no
  paraphrasing. UTF-8 byte integrity is separately pinned by
  `source_text_sha256` and is not the same as character-offset
  matching.
- Gold records are pre-entity-resolution and chunk-local. Canonical
  forms are recorded per entity but are not used as matching keys in
  this protocol.
- The 45 held-out retrieval/QA questions
  (`eval/questions.jsonl`) are NOT an annotation source. The
  annotator MUST NOT consult the held-out questions, their expected
  answers, or their `supporting_chunk_ids` / `gold_evidence_path` to
  decide any label. The annotation tooling MUST refuse to render any
  chunk whose `chunk_id` is in the held-out exclusion list
  (#48 §9.2).

## 2. Entity inclusion and exclusion

Entity types covered by this guideline are the four extracted
ontology entity types:

| entity_type | population_mode |
| --- | --- |
| `method`    | extracted |
| `dataset`   | extracted |
| `metric`    | extracted |
| `task`      | extracted |

`document` and `chunk` are structural entity types and are NOT
annotated as gold mentions; they are referenced by `doc_id` /
`chunk_id` in relation endpoints.

An annotator records a mention iff ALL of the following hold:

1. The mention's `surface_form` is a verbatim substring of
   `chunk.text` (Python-string equality on
   `chunk.text[char_start:char_end]`).
2. The literal is used in the local chunk as one of the four
   ontology entity types above.
3. A canonical form can be assigned without inventing an entity
   absent from the text.
4. The mention is not resolvable only via a cross-chunk reference.

A mention is NOT annotated when any of the following holds:

- The surface form is not a verbatim substring of `chunk.text`.
- The intended type is `document` or `chunk`, or any other type not
  in the four extracted ontology entity types.
- The surface form is a common noun whose local usage does not
  refer to the ontology entity (e.g. `attention` used as a generic
  English noun rather than the named mechanism).
- The reference can only be resolved from a different chunk.

### 2.1 Entity rubric — `method`

**Inclusion rule.** A literal in `chunk.text` is annotated as
`method` when the chunk uses it as the name of a specific
retrieval, generation, or representation method discussed in the
corpus (e.g. `DPR`, `ColBERT`, `RAG`, `REALM`,
`Fusion-in-Decoder`).

**Exclusion / ambiguity rule.** Common nouns or verbs that happen
to appear in method-name context (e.g. `attention`, `model`,
`index`) are excluded unless the local usage clearly refers to the
named mechanism. Surface forms that are canonical-only (no
verbatim substring in `chunk.text`) are excluded.

**Worked example (positive, synthetic).** Chunk text contains:

> "We compare Dense Passage Retrieval (DPR) against BM25 on the
> same retrieval task."

Three `method` mentions in this sentence: `Dense Passage
Retrieval`, `DPR`, `BM25`. They are three distinct entity records
even though the canonical form of the first and second is
identical.

**Worked example (exclusion / ambiguity, synthetic).** Chunk text
contains:

> "We use a single attention head per query and report the standard
> scoring metric."

`attention` is excluded as `method`: the local usage is the
generic English noun, not the named mechanism. `scoring metric` is
excluded as `metric`: no named metric literal appears.

### 2.2 Entity rubric — `dataset`

**Inclusion rule.** A literal in `chunk.text` is annotated as
`dataset` when the chunk uses it as the name of a specific
evaluation or training dataset (e.g. `Natural Questions`,
`HotpotQA`, `SQuAD`, `ELI5`).

**Exclusion / ambiguity rule.** Generic references to `the
dataset`, `this corpus`, `our training set` are excluded. A
literal that names a dataset only by paraphrase (e.g. `the
open-domain QA benchmark released by Google`) is excluded; the
canonical name MUST appear verbatim in `chunk.text`.

**Worked example (positive, synthetic).** Chunk text contains:

> "We evaluate on Natural Questions, HotpotQA, and TriviaQA."

Three `dataset` mentions: `Natural Questions`, `HotpotQA`,
`TriviaQA`.

**Worked example (exclusion / ambiguity, synthetic).** Chunk text
contains:

> "We evaluate on the benchmark introduced by the Google team."

No mention is annotated: no verbatim dataset name appears.

### 2.3 Entity rubric — `metric`

**Inclusion rule.** A literal in `chunk.text` is annotated as
`metric` when the chunk uses it as the name of a specific
retrieval or QA evaluation metric (e.g. `Recall@k`, `Exact Match`,
`top-k accuracy`).

**Exclusion / ambiguity rule.** The English word `accuracy` used
as a generic descriptor (not as the canonical name of a metric
literal) is excluded. The phrase `we report the metric` with no
literal name is excluded.

**Worked example (positive, synthetic).** Chunk text contains:

> "We report Exact Match and F1 scores on the development split."

`Exact Match` is a `metric` mention. `F1` is also a `metric`
mention when used as the canonical metric name. `scores` is not
annotated.

**Worked example (exclusion / ambiguity, synthetic).** Chunk text
contains:

> "We report high accuracy on the development split."

`accuracy` is excluded as `metric`: it is used as a generic English
noun, not as the name of a specific metric literal in the chunk.

### 2.4 Entity rubric — `task`

**Inclusion rule.** A literal in `chunk.text` is annotated as
`task` when the chunk uses it as the name of a specific retrieval,
QA, or generation task (e.g. `Open-Domain QA`, `Multi-Hop QA`,
`Passage Retrieval`, `Re-ranking`).

**Exclusion / ambiguity rule.** Generic phrases like `question
answering` used as a description of what the paper does, without
being the canonical task name in the local context, are excluded.
Generic verb phrases (`answering questions`, `retrieving passages`)
are excluded.

**Worked example (positive, synthetic).** Chunk text contains:

> "We focus on Open-Domain QA and Multi-Hop QA in our experiments."

Two `task` mentions: `Open-Domain QA`, `Multi-Hop QA`.

**Worked example (exclusion / ambiguity, synthetic).** Chunk text
contains:

> "Our method is designed for question answering."

`question answering` is excluded as `task`: the chunk uses it as a
generic descriptor, not as the canonical task name. If the same
chunk elsewhere contained `Open-Domain QA` verbatim, that surface
form would be annotated and this descriptor would still be
excluded.

## 3. Aliases, acronyms, and repeated literals

### 3.1 Distinct surface forms are distinct mentions

Distinct surface forms produce distinct entity records even when
they share a canonical form. `Dense Passage Retrieval` and `DPR`
are two distinct mentions, each with its own `entity_id` and its
own `occurrences_in_chunk` list. Canonical-form clustering is
downstream (`entity_resolve.py`).

### 3.2 Acronym and expansion

When both an acronym and its expansion occur in the chunk, both
are annotated as separate mentions. Both records MAY carry the
same `canonical_form`. The matching rule (#48 §6.1) matches
verbatim substrings, not canonical forms.

**Worked example (synthetic).** Chunk text contains:

> "We compare Dense Passage Retrieval (DPR) against BM25."

Entity records:

- `surface_form="Dense Passage Retrieval"`,
  `canonical_form="DPR"`,
  `occurrences_in_chunk=[{occurrence_index=0, char_start=…,
  char_end=…}]`.
- `surface_form="DPR"`, `canonical_form="DPR"`,
  `occurrences_in_chunk=[{occurrence_index=0, char_start=…,
  char_end=…}]`.
- `surface_form="BM25"`, `canonical_form="BM25"`,
  `occurrences_in_chunk=[{occurrence_index=0, char_start=…,
  char_end=…}]`.

### 3.3 Repeated literal handling

Every exact left-to-right, non-overlapping occurrence of the
surface form in `chunk.text` MUST be enumerated in
`occurrences_in_chunk`. The enumeration rule mirrors the v2
aligner (#49): after a match at `start`, the next search begins at
`start + len(surface_form)`. This guarantees a single
deterministic ordering and prevents overlapping or adjacent
matches from being treated as separate occurrences.

Occurrence indices are zero-based. `occurrence_index` MUST be a
non-negative in-range integer (`type(index) is int`; `bool` is NOT
accepted as `int`).

For each gold entity record, the annotator commits to one
occurrence as the record's `span` (`char_start`, `char_end`).

- The default committed occurrence is `occurrence_index == 0`
  (first occurrence in chunk order).
- Deviation from the first-occurrence default requires an
  annotation note. The note MUST explain why the first occurrence
  is grammatically ambiguous in a way the chosen occurrence is
  not.

**Worked example (repeated literal, synthetic).** Chunk text
contains:

> "DPR outperforms BM25 on NQ. The training procedure for DPR
> uses in-batch negatives."

`surface_form="DPR"` occurs twice. `occurrences_in_chunk`:

- `occurrence_index=0` at "DPR outperforms BM25".
- `occurrence_index=1` at "training procedure for DPR".

The committed span is `occurrence_index=0` (default). No note is
required.

**Worked example (deviation from default, synthetic).** Chunk
text contains:

> "In the related-work section, we cite the preprint titled
> 'DPR for low-resource domains'. In this paper, DPR is the
> retriever we evaluate."

`surface_form="DPR"` occurs twice. `occurrence_index=0` is the
`DPR` inside the quoted preprint title `'DPR for low-resource
domains'`: that occurrence is grammatically the head noun of a
quoted bibliographic title, not a use of the named method.
`occurrence_index=1` is the `DPR` in the sentence `DPR is the
retriever we evaluate`, which is a direct use of the named method
as the retriever. The annotator commits to
`occurrence_index=1` and records in `note`: "occurrence 0 is the
method name inside a quoted paper title, not a use of the method;
occurrence 1 is the intended method use."

### 3.4 `occurrence_index` semantics — alignment with #46 / #49

`occurrence_index` semantics in gold MUST match the v2 aligner
exactly (#46 resolution, #49 acceptance items 1–8):

- Zero occurrences of the surface form → reject. No gold record.
- One occurrence + `occurrence_index` absent → accept (defaults to
  0).
- One occurrence + `occurrence_index == 0` → accept.
- One occurrence + `occurrence_index != 0` → reject.
- ≥ 2 occurrences + `occurrence_index` absent → reject.
- ≥ 2 occurrences + valid `occurrence_index` → accept the indexed
  occurrence.
- Negative, out-of-range, non-int, or `bool` `occurrence_index` →
  reject.

The v2 runtime aligner (#49) independently scans `chunk.text` to
produce its own exact left-to-right, non-overlapping occurrence
enumeration via `str.find`. Gold `occurrences_in_chunk` records
the expected enumeration for annotation and scoring; it MUST be
the same deterministic enumeration the runtime aligner produces.
Gold records do NOT drive runtime extraction alignment. The
committed span MUST equal the occurrence at the annotator's
selected index.

## 4. Relation inclusion and exclusion

This guideline covers the eight extracted relation types from the
current `ontology.yaml`:

| relation_type     | from (endpoint type) | to (endpoint type) |
| ----------------- | -------------------- | ------------------ |
| `PROPOSES`        | document             | method             |
| `INTRODUCES`      | document             | dataset            |
| `EVALUATES_ON`    | document             | dataset            |
| `USES_METHOD`     | document             | method             |
| `REPORTS_METRIC`  | document             | metric             |
| `EXTENDS`         | method               | method             |
| `USED_FOR_TASK`   | dataset              | task               |
| `MEASURES`        | metric               | task               |

`CITES` is a structural relation and is NOT a gold annotation
under this guideline. `EVALUATED_ON` is a derived relation and is
NOT a gold annotation under this guideline; it is produced
downstream by entity resolution from `PROPOSES` × `EVALUATES_ON`.

A relation is annotated iff ALL of the following hold:

1. `relation_type` is one of the eight extracted relation types in
   the table above.
2. Every endpoint is admissible:
   - `subject_ref.kind == "document"` ⇒
     `subject_ref.id == chunk.doc_id`.
   - `subject_ref.kind == "mention"` ⇒ `subject_ref.id` is an
     `entity_id` of a same-chunk gold mention whose `entity_type`
     matches the relation's `from` endpoint type.
   - The same rule applies symmetrically to `object_ref` against
     the relation's `to` endpoint type.
3. A single contiguous verbatim evidence span can be selected
   under §5 below.
4. The relation is not resolvable only via a cross-chunk
   reference.

A relation is NOT annotated when any of the following holds:

- The relation type is structural (`CITES`) or derived
  (`EVALUATED_ON`).
- Either endpoint is a mention from a different chunk.
- The intended evidence cannot be expressed as a single
  contiguous verbatim substring of `chunk.text`.

### 4.1 Per-relation rubric

For each of the eight extracted relation types, this section pins
the relation meaning, the legal endpoint types, and a worked
example.

#### 4.1.1 `PROPOSES` (document → method)

**Meaning.** The current chunk's document introduces or
originates a method. `subject_ref.kind == "document"`
(id = chunk.doc_id); `object_ref.kind == "mention"` (id of a
same-chunk gold `method` mention).

**Worked example (synthetic).** Chunk text contains:

> "This paper introduces Dense Passage Retrieval (DPR), a method
> for open-domain question answering."

Relation record:

- `relation_type="PROPOSES"`.
- `subject_ref={kind:"document", id:<chunk.doc_id>}`.
- `object_ref={kind:"mention", id:<entity_id for DPR mention>}`.
- `evidence_quote="introduces Dense Passage Retrieval (DPR)"`.

#### 4.1.2 `INTRODUCES` (document → dataset)

**Meaning.** The current chunk's document releases or proposes a
dataset. `subject_ref.kind == "document"`;
`object_ref.kind == "mention"` (id of a same-chunk gold `dataset`
mention).

**Worked example (synthetic).** Chunk text contains:

> "We release Natural Questions, a new dataset for open-domain
> QA."

Relation record:

- `relation_type="INTRODUCES"`.
- `subject_ref={kind:"document", id:<chunk.doc_id>}`.
- `object_ref={kind:"mention", id:<entity_id for Natural
  Questions mention>}`.
- `evidence_quote="We release Natural Questions, a new dataset"`.

#### 4.1.3 `EVALUATES_ON` (document → dataset)

**Meaning.** The current chunk's document reports experimental
results on a dataset. `subject_ref.kind == "document"`;
`object_ref.kind == "mention"` (id of a same-chunk gold `dataset`
mention).

**Worked example (synthetic).** Chunk text contains:

> "We evaluate DPR on Natural Questions and report top-k
> accuracy."

Relation record (one per dataset):

- `relation_type="EVALUATES_ON"`.
- `subject_ref={kind:"document", id:<chunk.doc_id>}`.
- `object_ref={kind:"mention", id:<entity_id for Natural
  Questions mention>}`.
- `evidence_quote="evaluate DPR on Natural Questions"`.

#### 4.1.4 `USES_METHOD` (document → method)

**Meaning.** The current chunk's document applies or builds on a
method without originating it. `subject_ref.kind == "document"`;
`object_ref.kind == "mention"` (id of a same-chunk gold `method`
mention).

**Worked example (synthetic).** Chunk text contains:

> "Our system uses Fusion-in-Decoder to combine passage evidence."

Relation record:

- `relation_type="USES_METHOD"`.
- `subject_ref={kind:"document", id:<chunk.doc_id>}`.
- `object_ref={kind:"mention", id:<entity_id for Fusion-in-Decoder
  mention>}`.
- `evidence_quote="uses Fusion-in-Decoder to combine passage
  evidence"`.

#### 4.1.5 `REPORTS_METRIC` (document → metric)

**Meaning.** The current chunk's document reports a metric in its
results. `subject_ref.kind == "document"`;
`object_ref.kind == "mention"` (id of a same-chunk gold `metric`
mention).

**Worked example (synthetic).** Chunk text contains:

> "We report Exact Match on the development split."

Relation record:

- `relation_type="REPORTS_METRIC"`.
- `subject_ref={kind:"document", id:<chunk.doc_id>}`.
- `object_ref={kind:"mention", id:<entity_id for Exact Match
  mention>}`.
- `evidence_quote="report Exact Match on the development split"`.

#### 4.1.6 `EXTENDS` (method → method)

**Meaning.** Method A builds on or generalises method B. Both
endpoints are mentions of `method` in the same chunk.
`subject_ref.kind == "mention"` (id of a same-chunk gold `method`
mention); `object_ref.kind == "mention"` (id of a same-chunk gold
`method` mention).

When direction is ambiguous in the local text (e.g. "ColBERT is
similar to ColBERTv2"), the annotator commits to the
ontology-defined direction (`EXTENDS` subject = the newer method,
object = the older method) and captures the ambiguity in `note`
(#48 §3.8).

**Worked example (synthetic).** Chunk text contains:

> "ColBERTv2 extends ColBERT by introducing more expressive
> per-token embeddings."

Relation record:

- `relation_type="EXTENDS"`.
- `subject_ref={kind:"mention", id:<entity_id for ColBERTv2
  mention>}`.
- `object_ref={kind:"mention", id:<entity_id for ColBERT
  mention>}`.
- `evidence_quote="ColBERTv2 extends ColBERT by introducing"`.

#### 4.1.7 `USED_FOR_TASK` (dataset → task)

**Meaning.** A dataset supports or is standard for a task. Both
endpoints are mentions: `subject_ref.kind == "mention"` (id of a
same-chunk gold `dataset` mention); `object_ref.kind == "mention"`
(id of a same-chunk gold `task` mention).

**Worked example (synthetic).** Chunk text contains:

> "HotpotQA is a standard benchmark for Multi-Hop QA."

Relation record:

- `relation_type="USED_FOR_TASK"`.
- `subject_ref={kind:"mention", id:<entity_id for HotpotQA
  mention>}`.
- `object_ref={kind:"mention", id:<entity_id for Multi-Hop QA
  mention>}`.
- `evidence_quote="HotpotQA is a standard benchmark for
  Multi-Hop QA"`.

#### 4.1.8 `MEASURES` (metric → task)

**Meaning.** A metric evaluates performance on a task. Both
endpoints are mentions: `subject_ref.kind == "mention"` (id of a
same-chunk gold `metric` mention); `object_ref.kind == "mention"`
(id of a same-chunk gold `task` mention).

**Worked example (synthetic).** Chunk text contains:

> "Recall@k is the standard metric for Passage Retrieval."

Relation record:

- `relation_type="MEASURES"`.
- `subject_ref={kind:"mention", id:<entity_id for Recall@k
  mention>}`.
- `object_ref={kind:"mention", id:<entity_id for Passage Retrieval
  mention>}`.
- `evidence_quote="Recall@k is the standard metric for Passage
  Retrieval"`.

## 5. Relation evidence

Every gold relation record carries exactly one contiguous literal
`evidence_quote`. The `evidence_quote` MUST be a verbatim
substring of `chunk.text` (Python-string equality on
`chunk.text[char_start:char_end]`). There is NO numeric maximum
quote-length rule in v1; #45 and #48 explicitly defer a bounded
quote-length policy until pilot distributions are known.

### 5.1 Selection rule

The annotator selects the **shortest contiguous verbatim
substring** of `chunk.text` that:

1. Contains or directly expresses the asserted relation.
2. Fully contains the aligned span of every mention-endpoint
   (`subject_ref` and `object_ref` when their `kind == "mention"`).
   Document endpoints impose no mention-span containment
   requirement.
3. Is non-empty after exact-string matching.

The selected span MUST be a contiguous substring of `chunk.text`.
No multi-span evidence. No fuzzy or paraphrased evidence. No
whitespace, case, or Unicode normalisation. No semantic repair.

### 5.2 Admissible one-span evidence

**Worked example (synthetic).** Chunk text contains:

> "This paper introduces Dense Passage Retrieval (DPR), a method
> for open-domain question answering."

For `PROPOSES(document → DPR)`, the admissible evidence quote is
`introduces Dense Passage Retrieval (DPR)`. The span fully
contains the `DPR` mention span (and `Dense Passage Retrieval`
mention span) and directly expresses the relation.

### 5.3 Endpoint-not-contained — invalid relation

When the only candidate evidence_quote does not fully contain
every mention-endpoint span, the relation is NOT admissible as a
v2 relation and MUST NOT be recorded.

**Worked example (invalid, synthetic).** Chunk text contains:

> "We use Fusion-in-Decoder in our pipeline. The training
> procedure is described in the appendix."

Mentions of `Fusion-in-Decoder` (method) and `pipeline` (no
ontology entity) exist. If the annotator proposes
`USES_METHOD(document → Fusion-in-Decoder)` and attempts to use
`The training procedure is described in the appendix` as
evidence_quote, the relation is rejected: the evidence span does
NOT contain the `Fusion-in-Decoder` mention span, so endpoint
containment fails.

### 5.4 Cross-chunk invalid relation

Cross-chunk mention endpoints are forbidden in v2. A relation
whose mention endpoint resolves to a mention in a different
chunk is NOT annotated. If a relation is asserted across two
chunks, neither chunk records it; the chunk-local note flags the
cross-chunk candidate for downstream resolution.

**Worked example (invalid, synthetic).** Chunk A asserts
"ColBERTv2 is described in the next section." Chunk B contains
`ColBERTv2` and `ColBERT`. The annotator MUST NOT record an
`EXTENDS(ColBERTv2 → ColBERT)` relation in chunk B that relies on
chunk A's assertion: the relation's endpoints (both `method`
mentions) are present in chunk B, but the asserted relation is
not locally supported by chunk B's text. If chunk B's text does
not itself express the `EXTENDS` claim, no relation is recorded
in chunk B; the cross-chunk candidate is flagged in note.

## 6. Confidence rubric

Every gold mention and every gold relation carries exactly one
`confidence` label from the closed set
`{stated, inferred, hedged}`. The label is assigned by the
annotator from the local chunk text under the operational rules
below. Model-emitted confidence scores MUST NOT be used to set
the gold label.

### 6.1 Operational rules

**`stated`.** The chunk text contains a literal assertion that
makes the entity fact or the relation hold. A direct declarative
sentence, a parenthetical definition, or a verbatim quotation
qualifies.

**`inferred`.** The fact is supported by information present in
the local `chunk.text` but is not expressed as a direct
declarative assertion. The supporting evidence may be a local
table row, a compact result line, a numbered equation, a list
item, or other chunk-internal structural evidence (e.g. a
results table cell "DPR    NQ    41.5" implies
`EVALUATES_ON(document → Natural Questions)` without a literal
sentence asserting the relation). External scientific knowledge
MUST NOT create an annotation that the chunk itself does not
support. When the chunk is silent on certainty, default to
`inferred`.

**`hedged`.** The chunk text expresses the entity or the relation
with explicit uncertainty, modal qualification, or speculative
language (e.g. "may improve", "could be extended to",
"we speculate that", "potentially supports"). Verbs or adverbs
that signal uncertainty MUST be present in the chunk text.

### 6.2 Worked examples (synthetic)

**`stated` — relation.** Chunk text contains:

> "We release Natural Questions, a new dataset for open-domain
> QA."

`INTRODUCES(document → Natural Questions)` is `stated`: the
sentence directly asserts the dataset release.

**`inferred` — relation.** Chunk text contains:

> "DPR    NQ    41.5"

(A table row, synthetic.) `EVALUATES_ON(document → Natural
Questions)` is `inferred`: the row presents an empirical
evaluation, but the sentence-level assertion is not present.

**`hedged` — relation.** Chunk text contains:

> "ColBERTv2 may extend ColBERT by introducing more expressive
> per-token embeddings."

`EXTENDS(ColBERTv2 → ColBERT)` is `hedged`: the modal verb `may`
is present in the chunk text.

**`stated` — mention.** Chunk text contains:

> "We use Dense Passage Retrieval (DPR) as the retriever."

The `DPR` mention is `stated`: the chunk directly names the
retriever method.

**`inferred` — mention.** Chunk text contains:

> "Our retrieval pipeline is described below."

No method mention is annotated: the chunk does not name a
verbatim method literal, so no record exists. This example is
illustrative of when the mention itself is absent rather than
`inferred`.

### 6.3 Confidence is diagnostic metadata

Confidence is recorded as diagnostic metadata on the gold
record. It is NOT part of the primary entity/relation F1
selection metric (#48 refinement 2). The diagnostic metric
`confidence_agreement_rate` (#48 §5.4) measures agreement
between gold confidence labels and model confidence labels on
validated extractions. Confidence labels do not enter the
match predicate (#48 §6.4).

If Cohen's κ < 0.7 on any confidence cell (`stated`, `inferred`,
or `hedged`) during the 24-chunk random review sample (§9),
confidence-label stability is escalated and reconsidered. The
default mitigation is full-double annotation on that confidence
cell type (§9 escalation trigger); the field MAY be deprecated
in a future `gold-record.vN` if reproducibility remains
unsatisfactory.

## 7. Ambiguity policy

The annotator resolves ambiguity under the following operational
rules:

- **Literal / local reading wins.** The annotator prefers the
  literal reading of `chunk.text` over external scientific
  knowledge. When a surface form has both a generic meaning and
  a named-entity meaning, the named-entity meaning is selected
  only when the local usage clearly refers to the named entity.
- **No held-out QA consultation.** The annotator MUST NOT resolve
  ambiguity by consulting the 45 held-out retrieval/QA questions,
  their expected answers, or any `supporting_chunk_ids` /
  `gold_evidence_path` (§1). Held-out questions are not an
  annotation source.
- **Unresolved ambiguity → note / review.** When two plausible
  interpretations remain, the annotator commits to the preferred
  interpretation under the literal / local rule and records the
  rejected alternative in the chunk-local `note` for adjudication
  (§9).

### 7.1 Common-noun-vs-method example (synthetic)

Chunk text contains:

> "We use a single attention head per query and report the
> standard scoring metric."

`attention` is excluded as `method`: the local usage is the
generic English noun, not the named attention mechanism. The
chunk does not contain a verbatim surface form that anchors the
named-entity reading. The annotator records no method mention at
the position of "attention". If a separate chunk elsewhere
contained `Multi-Head Attention` verbatim, that surface form
would be annotated as `method` regardless of the local reading
here.

### 7.2 Relation-direction example (synthetic)

Chunk text contains:

> "ColBERTv2 is similar to ColBERT."

`ColBERT` and `ColBERTv2` are both `method` mentions. The
`EXTENDS` direction under #48 §3.8 is subject = the newer
method (`ColBERTv2`), object = the older method (`ColBERT`).
The annotator records the `EXTENDS` relation with that
direction and captures the local ambiguity ("similar to")
in `note`. The relation is NOT inverted to follow the literal
surface order.

## 8. Zero-record chunks

Both of the following records are valid gold annotations under
this guideline:

```json
{"entities": [], "relations": []}
```

### 8.1 True-negative vs incomplete annotation

A zero-record chunk is a **true negative** when the annotator
has explicitly scanned `chunk.text` for the four extracted
entity types and for the eight extracted relation types and
found no admissible mention or relation. The chunk record MUST
be authored with empty arrays and SHOULD carry a one-line note
describing why the chunk is genuinely empty (e.g. "references
section only", "no verbatim ontology term in this chunk",
"table-only chunk with no assertion sentences").

A zero-record chunk is **incomplete annotation** when the
annotator has not actually scanned the chunk text or has
recorded empty arrays by default without verifying. Incomplete
annotation is a quality defect: it is indistinguishable from a
true negative to the matching pipeline but degrades the gold
set's reliability. The annotation tooling SHOULD require an
explicit empty-chunk note to permit saving a zero-record chunk.

### 8.2 Relations must be empty when entities are empty

When `entities` is empty, `relations` MUST also be empty. A
relation whose endpoints are both mentions requires the
mentions to exist as same-chunk gold entities; with no entities,
no relations are admissible. The annotation tooling SHOULD
enforce this structural invariant at write time.

### 8.3 Worked example (synthetic)

Chunk text contains:

> "Acknowledgements. We thank the anonymous reviewers for their
> feedback."

Entities: `[]`. Relations: `[]`. Note:
`"acknowledgements-only chunk; no ontology entity or relation
literal present"`.

## 9. Adjudication protocol

This section pins the v1 pilot adjudication protocol defined by
#48 §4. The protocol is a single-annotator pilot with
adjudication on disagreement signals. It does NOT silently
upgrade the entire pilot to full-double annotation absent the
κ < 0.7 escalation trigger.

The protocol is deterministic and reproducible: a second
annotator with only this document, the canonical chunks, the
ontology, and the annotation tooling SHOULD be able to
reproduce the reference set.

### 9.1 Steps

1. **Initial annotation by `ann-A`** for all 120 chunks
   (`single_pass`). The per-chunk record carries
   `adjudication_status: "single_pass"`,
   `annotator_id: "ann-A"`, `adjudicated_by: null`.
2. **Self-consistency check.** After a 7-day wash, `ann-A`
   re-reads her own annotation for every chunk and flags any
   cell where her second-pass answer disagrees with her
   first-pass. The 7-day wash is the minimal interval; shorter
   intervals are NOT acceptable. The self-disagreement rate is
   measured and reported (#48 §5.6) as a primary QA signal on
   the pilot. Chunks with self-disagreement proceed to step 3.
3. **Independent review by `ann-B`** for every
   self-disagreement chunk PLUS a uniform / deterministic 20 %
   random sample of the pilot (24 chunks). `ann-B` writes
   `adjudication_status: "double_pass"` and her review verdict
   to an `ann_b_review` sidecar field per chunk. The 20 %
   random sample lets the protocol compute Cohen's κ on
     cell-level agreement without reviewing every chunk.
4. **Adjudication by `ann-C`** on every chunk where `ann-A`
   self-disagrees OR `ann-A` vs `ann-B` Cohen's-κ cell flag.
   `ann-C` writes the canonical record with
   `adjudication_status: "adjudicated"`,
   `adjudicated_by: "ann-C"`. The default rule is that
   `ann-C`'s record is canonical; `ann-A`'s and `ann-B`'s
   records are preserved as `ann_a_record` / `ann_b_record`
   sidecars under `gold_set/reviews/<chunk_id>.json`.
5. **Canonical record location.** The final canonical gold
   record lives at `gold_set/chunks/<chunk_id>.json`. Review
   sidecars live under `gold_set/reviews/<chunk_id>.json` and
   are gitignored.

### 9.2 Escalation trigger

Cohen's κ is computed per cell on the 24-chunk random sample
(#48 §4). Cells are:

- per entity_type × confidence (`stated` / `inferred` /
  `hedged`),
- per relation_type × confidence,
- per endpoint-resolution kind (`document` vs `mention`).

**Escalation trigger.** If Cohen's κ < 0.7 on any entity,
relation, or confidence cell across the 24-chunk random
sample, the protocol escalates to full-double annotation on
that cell type across the whole pilot. The escalation MUST be
scoped to the failing cell type. The protocol MUST NOT
silently convert the entire pilot to full-double annotation
absent the trigger.

The annotation guideline itself MAY be revised (with
`annotation_guideline_version` bumped and `annotation_guideline_sha256`
updated) if guideline ambiguity is identified as the root
cause; affected chunks are then re-annotated under the new
guideline version (#48 §5.6).

## 10. Worked examples — summary index

Every example in this document is synthetic unless explicitly
labelled otherwise. Synthetic examples are constructed to obey
the exact v2 contract semantics (#46, #49) and the gold-record
schema (#48 §2).

- **Entity positive + exclusion / ambiguity**:
  - method — §2.1
  - dataset — §2.2
  - metric — §2.3
  - task — §2.4
- **Repeated literal / `occurrence_index`** — §3.3
- **Acronym + expansion** — §3.2
- **Relation worked examples** (one per extracted relation):
  - `PROPOSES` — §4.1.1
  - `INTRODUCES` — §4.1.2
  - `EVALUATES_ON` — §4.1.3
  - `USES_METHOD` — §4.1.4
  - `REPORTS_METRIC` — §4.1.5
  - `EXTENDS` — §4.1.6
  - `USED_FOR_TASK` — §4.1.7
  - `MEASURES` — §4.1.8
- **Document → mention endpoint** — §4.1.1, §4.1.2, §4.1.3,
  §4.1.4, §4.1.5
- **Mention → mention endpoint** — §4.1.6, §4.1.7, §4.1.8
- **Endpoint-not-contained invalid relation** — §5.3
- **Cross-chunk invalid relation** — §5.4
- **`stated` confidence** — §6.2
- **`inferred` confidence** — §6.2
- **`hedged` confidence** — §6.2
- **Zero-entity / zero-relation chunk** — §8.3

## 11. Per-chunk annotator checklist

For each canonical chunk, the annotator (or the annotation
tooling) verifies the following before saving the gold record.
This checklist is compact and mechanically mirrorable by later
annotation tooling.

1. **Canonical chunk identity checked.** `chunk_id` matches the
   row in `data/derived/chunks.jsonl`; `doc_id` matches the
   chunk's `doc_id`; `chunk.text` is the verbatim text of that
   row.
2. **Literal evidence exact.** Every `surface_form` and every
   `evidence_quote` satisfies
   `chunk.text[char_start:char_end] == literal` under exact
   Python-string equality (no Unicode / case / whitespace
   normalisation). `char_start` / `char_end` are Python string
   character offsets into `chunk.text`.
3. **Ontology type valid.** Every `entity_type` is one of
   `method | dataset | metric | task`; every `relation_type` is
   one of the eight extracted relation types in §4.
4. **Occurrences enumerated.** Every entity record lists every
   exact left-to-right non-overlapping occurrence of its
   `surface_form` in `chunk.text` in `occurrences_in_chunk`,
   indexed from 0.
5. **Committed occurrence chosen.** The record's `span`
   (`char_start`, `char_end`) equals the
   `occurrences_in_chunk[occurrence_index]` entry the
   annotator committed to. The default committed occurrence is
   `occurrence_index == 0`; deviation is captured in `note`.
6. **Endpoint types valid.** Every relation endpoint's
   `kind`/`id` matches the endpoint table in §4:
   `kind: "document"` ⇒ `id == chunk.doc_id`;
   `kind: "mention"` ⇒ `id` is a same-chunk gold
   `entity_id` whose `entity_type` matches the relation's
   endpoint type.
7. **Evidence contiguous.** Every relation record carries
   exactly one contiguous `evidence_quote`. No multi-span
   evidence. No fuzzy or paraphrased evidence. No numeric
   quote-length cap.
8. **Endpoint containment valid.** Every relation's
   `evidence_span` fully contains the aligned span of every
   mention-kind endpoint. Document-kind endpoints impose no
   mention-span containment requirement.
9. **Confidence assigned.** Every entity and every relation
   record carries one of `stated | inferred | hedged` per §6.
10. **Ambiguity note added if needed.** When the annotator's
    interpretation was non-trivial under §7, the rejected
    alternative is captured in `note` for adjudication.
11. **Zero-record decision considered.** When the chunk has
    no admissible entity or relation, the record carries
    `entities: []` and `relations: []` and a one-line note
    explaining why the chunk is genuinely empty (§8.1).
12. **Held-out QA not consulted.** The annotator did NOT use
    any of the 45 held-out questions, their expected answers, or
    their `supporting_chunk_ids` / `gold_evidence_path` to
    decide any label on this chunk. The annotation tooling
    MUST refuse to render any chunk in the held-out exclusion
    list, which is the operational enforcement of this rule.
