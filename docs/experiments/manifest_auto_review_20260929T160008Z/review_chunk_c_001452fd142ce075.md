# Manual review — Chunk `c_001452fd142ce075`

| Field | Value |
|---|---|
| Review label (provider blinded) | `rev_manifest_auto_54dfd813` |
| Unblinded provider label | `manifest-auto` |
| `chunk_id` | `c_001452fd142ce075` |
| `doc_id` | `W3038572442` |
| Document title | "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval" (ANCE; arXiv `2007.00808`, OpenAlex `W3038572442`) |
| Chunk source length | 835 characters |
| `source_text_sha256` | `4a2900139582bcea20c1200f509398ddb91b1d791b098505ca459724f081d335` (verified against chunk text) |
| Phase | entity + relation; **fully processed** (`processed: true`) |
| Accepted mentions | 4 (per `metrics_manifest-auto.json`) |
| Accepted assertions | 4 (per `metrics_manifest-auto.json`) |

This is the floating-row caption of two side-by-side tables in the ANCE
paper (a LaTeX fragment containing `tab:openqa` and `tab.prod`). It is
the only chunk in the run whose relation phase produced non-empty
output.

## 1. Verbatim chunk text (full, 835 bytes)

```
}}
         {
         \caption{Retrieval results (Answer Coverage at Top-20/100) on
         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]]. \vspace{-0.35cm}

          \label{tab:openqa}}}

        \ttabbox{\resizebox{0.35\textwidth}{!}
     {
\begin{tabular}{rrr|r} \hline \hline
\textbf{Corpus Size} & \textbf{Dim} & \textbf{Search} &
\textbf{Gain}  
\\ \hline
250 Million & 768 & KNN & +18.4\% \\
8 Billion & 64 & KNN & +14.2\% \\
8 Billion & 64 & ANN & +15.5\% \\ \hline

\end{tabular}
}}
      {\caption{Relative gains in the first stage retrieval of a commercial search engine. The gains are from changing the training of a production DR model to ANCE. \vspace{-0.123cm}


      \label{tab.prod}
          }
      }

   \end{floatrow}
\end{figure}
```

## 2. Accepted mentions

| `mention_id` | `entity_type` | `surface_form` (verbatim) | `canonical_form` | `confidence` | offsets `[cs,ce)` | Annotator label |
|---|---|---|---|---|---|---|
| `m_9e85ca5fde092683` | `dataset` | `Natural Questions (NQ)` | `Natural Questions` | `stated` | `[97,119)` | **supported** |
| `m_0045e51965d8059f` | `dataset` | `Trivial QA (TQA)` | `Trivia QA` | `inferred` | `[124,140)` | **supported (with note)** |
| `m_da34cbe93057b973` | `method` | `DR` | `Dense Retrieval` | `inferred` | `[706,708)` | **supported (with note)** |
| `m_b0dfffc9fd61a4d2` | `method` | `ANCE` | `ANCE` | `stated` | `[718,722)` | **supported** |

All four `char_start`/`char_end` offsets reproduce the `surface_form`
verbatim from the chunk text; the pipeline's evidence-alignment gate is
correctly satisfied. SHA-256 of the chunk text matches the per-mention
`source_text_sha256`.

### Evidence with surrounding context

`m_9e85ca5fde092683` — `Natural Questions (NQ)` at `[97,119)`:

```
\caption{Retrieval results (Answer Coverage at Top-20/100) on
         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]]. \vspace{-0.35cm}
```

`m_0045e51965d8059f` — `Trivial QA (TQA)` at `[124,140)`:

```
\caption{Retrieval results (Answer Coverage at Top-20/100) on
         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]]. \vspace{-0.35cm}
```

`m_da34cbe93057b973` — `DR` at `[706,708)` and
`m_b0dfffc9fd61a4d2` — `ANCE` at `[718,722)`:

```
... of a commercial search engine. The gains are from changing the training of a production DR model to ANCE. \vspace{-0.123cm}
```

### Annotator notes per mention

- **Natural Questions**: clean. The caption names the dataset verbatim
  with its abbreviation NQ; the canonical form matches the ontology
  example. Confidence `stated` is correct — the chunk is plainly naming
  it.
- **Trivia QA** (`Trivial QA (TQA)`): the chunk text says **"Trivial QA
  (TQA)"**, which is a typo in the original paper — see also
  `Tables/joint.tex` in the ANCE LaTeX source (`/tmp/latex_ance/`),
  where the same caption is written verbatim. Elsewhere in the same
  paper the dataset is rendered as **"Trivia QA (TQA)"**
  (`Tables/efficiency_qa.tex`, `Experiment.tex`:
  *"TriviaQA (TQA)~\citep{joshi2017triviaqa}"*). The pipeline accepted
  the verbatim `surface_form` and stored the canonical form
  `Trivia QA`, which is the correct dataset identity (Joshi et al.
  2017). The mention is **supported**. The `confidence="inferred"`
  label is mildly under-stated (the chunk plainly names the dataset,
  same as NQ), but is not a defect — the assessor conservatively marked
  the caption as naming-with-abbreviation only, not full natural-language
  statement. Outside-the-chunk context (other captions in the same
  paper) confirms the spelling.
- **Dense Retrieval** (`DR`): the chunk token `DR` is an abbreviation
  used inline in the production table caption, not a self-contained
  method name. The canonical form `Dense Retrieval` is the textbook
  expansion; the paper's own `Experiment.tex` calls it *"a
  production-quality DR model"* in the same paragraph. The mention is
  **supported**, but it is worth flagging that the abbreviation is
  ambiguous in isolation (could also stand for "Document Retriever"
  etc.). The caption's companion sentence ("changing the training ...
  to ANCE") removes the ambiguity within this chunk.
- **ANCE**: clean. The token appears verbatim and unambiguously names
  the proposed method.

## 3. Accepted assertions

| `assertion_id` | `relation_type` | subject | object | `evidence_quote` | offsets `[cs,ce)` | Annotator label |
|---|---|---|---|---|---|---|
| `a_186ebfe9b1cd3758` | `USES_METHOD` | `document W3038572442` | mention `m_da34cbe93057b973` (Dense Retrieval) | `The gains are from changing the training of a production DR model to ANCE` | `[649,722)` | **unsupported (false-positive schema / wrong relation type)** |
| `a_48f1cfb9f80bca7b` | `USES_METHOD` | `document W3038572442` | mention `m_b0dfffc9fd61a4d2` (ANCE) | same | `[649,722)` | **unsupported (false-positive relation type)** |
| `a_4fd1879b3d8f18b7` | `EVALUATES_ON` | `document W3038572442` | mention `m_0045e51965d8059f` (Trivia QA) | `\caption{Retrieval results (Answer Coverage at Top-20/100) on \n         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~` | `[25,162)` | **supported** |
| `a_c690e0b1f2dfec3a` | `EVALUATES_ON` | `document W3038572442` | mention `m_9e85ca5fde092683` (Natural Questions) | same | `[25,162)` | **supported** |

All four `evidence_quote` strings reproduce verbatim from the chunk
text; SHA-256 and the per-record `source_text_sha256` agree. The
pipeline's evidence-alignment and containment gates pass.

### Evidence with surrounding context

`a_186ebfe9b1cd3758` (USES_METHOD → DR) and `a_48f1cfb9f80bca7b`
(USES_METHOD → ANCE), spanning `[649,722)`:

```
... of a commercial search engine. The gains are from changing the training of a production DR model to ANCE. \vspace{-0.123cm}
```

The full chunk context for these offsets is:

```
the offline gains in various corpus sizes, encoding dimensions, and exact/approximate search.

We also study the effectiveness of ANCE in the first stage retrieval of a commercial search engine's production system. We change the training of a production-quality DR model to ANCE, and evaluate
```

(This `Experiment.tex` block is **outside the chunk** — included here
only to disambiguate the abbreviation `DR` in context. The in-chunk
text alone is the caption shown above.)

`a_4fd1879b3d8f18b7` (EVALUATES_ON → Trivia QA) and
`a_c690e0b1f2dfec3a` (EVALUATES_ON → Natural Questions), spanning
`[25,162)`:

```
}}
         {
         \caption{Retrieval results (Answer Coverage at Top-20/100) on
         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]]. \vspace{-0.35cm}
```

### Annotator notes per assertion

- **`a_c690e0b1f2dfec3a` EVALUATES_ON(W3038572442 → Natural Questions)**:
  **supported**. The caption is *"Retrieval results ... on Natural
  Questions (NQ)"* — the chunk is plainly reporting retrieval results
  on NQ. Endpoint types match the ontology (`document` → `dataset`).
- **`a_4fd1879b3d8f18b7` EVALUATES_ON(W3038572442 → Trivia QA)**:
  **supported**. Same caption, same schema. The `surface_form` is
  `Trivial QA (TQA)` (typo in source paper), but `canonical_form` is
  `Trivia QA` — the dataset identity. The chunk reports retrieval
  results on this dataset; the assertion is faithful.
- **`a_186ebfe9b1cd3758` USES_METHOD(W3038572442 → Dense Retrieval)**:
  **unsupported (relation type is wrong / misleading)**. The
  ontology defines `USES_METHOD` as *"Document applies or builds on
  a method without originating it"*. The chunk says *"a production
  DR model"* — `DR` here denotes the *kind of model the production
  system was running* before the change. The paper does not "use" or
  "build on" generic Dense Retrieval as a method; rather, it *replaces*
  it with ANCE. Two defensible readings exist:
    1. **Strict**: the chunk describes an external (production)
       system that uses DR; the ANCE paper studies that swap. Under
       this reading the paper does *apply* ANCE to a DR-based
       production system, but `USES_METHOD(DR)` from the ANCE paper
       remains weak — the paper is *about* the swap, not about *using
       DR*. **Annotator verdict: unsupported**.
    2. Lenient: the paper does use DR as the baseline it improves.
       But this still maps better to a different relation (a
       PROPOSES-of-ANCE with EVALUATES_ON framing), and `USES_METHOD`
       is not the cleanest fit. The mention of DR is incidental, not
       a "uses" claim.

    The annotator's verdict is **unsupported as written**; if a human
    wanted to keep a relation here, `USES_METHOD(ANCE)` (a_48f1…) is
    the closer fit, and even that should likely be `PROPOSES(ANCE)`.
- **`a_48f1cfb9f80bca7b` USES_METHOD(W3038572442 → ANCE)**:
  **unsupported (false-positive relation type)**. The ANCE paper
  *originates* ANCE — see ontology definition of `USES_METHOD`
  (*"Document applies or builds on a method without originating
  it"*) and `PROPOSES` (*"Document introduces or originates a
  method"*). The caption does not literally say *"we propose ANCE"*,
  but the paper is *the* ANCE paper; the correct relation is
  `PROPOSES(W3038572442 → ANCE)`, not `USES_METHOD`. Same caveat:
  the chunk text alone describes the swap to ANCE in a production
  system (i.e., an application), not the proposal; even so,
  `USES_METHOD` is the wrong ontology edge for a paper that is the
  canonical citation for ANCE. **Annotator verdict: unsupported as
  written; correct relation would be PROPOSES.**

### Potential missing relation (within-chunk)

- **`PROPOSES(W3038572442 → ANCE)`** — supported by corpus-level
  knowledge (ANCE paper), but the chunk text only describes *using*
  ANCE in a production A/B. Within the chunk's evidence span the
  relation is not directly asserted; corpus-context only. **Out of
  scope for chunk-local review.** Note in the missing-relations
  section below.

## 4. Possible false negatives (plausible missing mentions / relations)

The annotator flagged the following as scientifically defensible
within-chunk items that the run did **not** produce:

| Type | Item | Why it is in-chunk | Grounded in-chunk text |
|---|---|---|---|
| Missing mention (entity) | `metric` `Answer Coverage` | "Retrieval results (Answer Coverage at Top-20/100) on ..." | `Answer Coverage` is verbatim in the caption at offset `[42,57)` (after `\caption{Retrieval results (`). The ontology allows a `metric` entity type. The pipeline missed it. |
| Missing assertion | `REPORTS_METRIC(W3038572442 → Answer Coverage)` | Same caption, same offsets as the two `EVALUATES_ON` assertions above | The caption is literally *"Retrieval results (Answer Coverage at Top-20/100) on NQ and TQA"* — both the metric and the datasets are in the same span. The pipeline extracted both datasets but did not extract the metric or the corresponding `REPORTS_METRIC` relation. |

### Why `Answer Coverage` matters

The user brief explicitly asks whether this entity may have been
overlooked. **Yes — it is a clear false negative.** Three independent
confirmations:

1. The token `Answer Coverage` is literally present in the chunk text
   inside the `evidence_quote` of both `EVALUATES_ON` assertions
   (`a_4fd1879b3d8f18b7`, `a_c690e0b1f2dfec3a`).
2. The ontology defines a `metric` entity type (line 85 of
   `ontology.yaml`); `Answer Coverage` is a textbook retrieval/QA
   metric (semantically `Recall@k` restricted to answer-string
   presence — see `Experiment.tex` line 18: *"The metrics are
   Coverage@20/100, which evaluate whether the Top-20/100 retrieved
   passages include the answer."*).
3. The ontology's `REPORTS_METRIC(document → metric)` relation is
   exactly the edge needed for *"Retrieval results (Answer Coverage
   at Top-20/100)"*. With `Answer Coverage` as a missing mention,
   `REPORTS_METRIC` could not have been emitted anyway.

The first stage of the pipeline accepted the citation marker
`[[CITE:karpukhin2020dense]]` literally and the abbreviation
`Top-20/100` literally (which is not in the ontology's entity types),
but did not extract `Answer Coverage`. This is a real precision
*and* recall miss: a missed mention that also unlocks a missed
relation.

### Why `Trivial QA (TQA)` → `Trivia QA` is fine (the "normalization" the user flagged)

The user asked whether the surface form `Trivial QA` was normalised to
`Trivia QA`. **It was, and the normalisation is correct.** The chunk
text says `Trivial QA (TQA)` (a typo in the source paper) and the
accepted mention's `canonical_form` is `Trivia QA`. Reasoning:

- The pipeline contract allows `canonical_form` to be a normalised
  name (the prompt itself gives `"DPR"` for *"Dense Passage
  Retrieval"* as the worked example).
- The dataset whose arXiv ID and BibTeX key is `joshi2017triviaqa` is
  called **TriviaQA** (one word) in the Joshi et al. 2017 paper.
  Outside-the-chunk context in the same ANCE paper confirms this:
    - `Experiment.tex` line 16: *"TriviaQA (TQA)~\citep{joshi2017triviaqa}"*
    - `Tables/efficiency_qa.tex`: *"Trivia QA (TQA)"*
    - `Tables/openqa.tex`, `Tables/openqa_retrieval_full.tex`,
      `Tables/openqa_e2e.tex`: header reads `\textbf{TQA}` with the
      caption spelling out *"Trivia QA (TQA)"*.
  So **the typo is local to `Tables/joint.tex`** (the source of this
  chunk). The pipeline's normalised `Trivia QA` matches the
  identifier used everywhere else in the same paper and matches the
  dataset's own canonical name.

- `confidence="inferred"` is a fair hedge given the in-chunk typo.
  A more aggressive reader could set this to `stated` (the caption
  is plainly naming the dataset), but `inferred` is not wrong.

There is no scientific error here; the only concern is whether a future
gold set will treat the `surface_form` `Trivial QA (TQA)` as a
distinct entity from `Trivia QA (TQA)`. The run's answer is "no", and
the annotator agrees with that answer.

## 5. Scientific assessment of the two USES_METHOD relations

The user asked specifically whether the two `USES_METHOD` relations
(directed at `DR` and `ANCE`) are scientifically defensible. The
annotator's verdict for both:

- **`USES_METHOD(W3038572442 → ANCE)`** is **scientifically
  incorrect**. The ANCE paper *originates* ANCE. The ontology's
  `USES_METHOD` definition explicitly excludes the originating paper:
  *"applies or builds on a method without originating it"*. The
  correct relation is `PROPOSES(W3038572442 → ANCE)`, but that
  relation is not strictly grounded in the chunk text (the chunk
  only says *"training ... to ANCE"*; the word "propose" does not
  appear). Corpus-context makes `PROPOSES` clearly true, but
  chunk-context alone makes the *strongest* grounded relation
  something like "the chunk describes the paper applying ANCE in a
  production system" — which is closer to `USES_METHOD(ANCE)` than to
  `PROPOSES`, but the run emits the wrong *direction of fit*: this is
  the originating paper and the relation type is "applies without
  originating".

- **`USES_METHOD(W3038572442 → DR)`** is **scientifically weak /
  borderline**. The chunk says *"a production DR model"* — `DR` is a
  description of the *pre-change* production system's model class.
  The ANCE paper does not "use" generic Dense Retrieval as a method;
  it studies what happens when a DR-based production system is
  retrained with ANCE. The relation type is at best a stretch, and
  the production-system context (the third-party commercial search
  engine) makes "uses" even less apt.

In short: **both `USES_METHOD` edges look like precision errors** —
the extracted spans are real text spans, but the chosen ontology
relation type does not reflect the science.

## 6. Scientific assessment of the two EVALUATES_ON relations

The user asked specifically about both `EVALUATES_ON` edges.

- **`EVALUATES_ON(W3038572442 → Natural Questions)`**: **supported**.
  Caption says *"Retrieval results ... on Natural Questions (NQ)"*.
  Endpoint types match; the surface form is verbatim in the chunk;
  the dataset identity is correct (Kwiatkowski et al. 2019).
- **`EVALUATES_ON(W3038572442 → Trivia QA)`**: **supported**, with
  the same caveat about the source-typo `Trivial QA` (the canonical
  name `Trivia QA` is correct).

No false positive or false negative concerns for the
`EVALUATES_ON` direction. Both are grounded in the same caption span
that already passes the pipeline gates; the annotator would not flag
either for change.

## 7. Annotation summary table

| Item | Label |
|---|---|
| `m_9e85ca5fde092683` (Natural Questions) | supported |
| `m_0045e51965d8059f` (Trivia QA via "Trivial QA") | supported (with note on source typo) |
| `m_da34cbe93057b973` (Dense Retrieval via "DR") | supported (with note on abbreviation) |
| `m_b0dfffc9fd61a4d2` (ANCE) | supported |
| `a_c690e0b1f2dfec3a` (`EVALUATES_ON` → NQ) | supported |
| `a_4fd1879b3d8f18b7` (`EVALUATES_ON` → Trivia QA) | supported |
| `a_186ebfe9b1cd3758` (`USES_METHOD` → DR) | unsupported (false-positive relation type / weak fit) |
| `a_48f1cfb9f80bca7b` (`USES_METHOD` → ANCE) | unsupported (false-positive relation type — should be PROPOSES or omitted) |
| *(missing)* `metric` `Answer Coverage` | **false negative** — clear in-chunk miss |
| *(missing)* `REPORTS_METRIC(W3038572442 → Answer Coverage)` | **false negative** — implied by same caption |

## 8. Notes on what was *not* examined

- The 38 other `processed: false` records in `review_records.jsonl`
  are out of scope (the run only processed two chunks; the rest are
  review-package scaffolding entries generated by the seed).
- The provider identity (`manifest-auto`) is blinded behind
  `rev_manifest_auto_54dfd813` per the reviewer prompt; this
  unblinding happens after labels are recorded (post-hoc) and is
  noted here only for traceability, not as a quality judgement.
- The chunk's citation marker `[[CITE:karpukhin2020dense]]` would, at
  the structural layer, generate `CITES(W3038572442 → W3015883388)` —
  not within the scope of an LLM-extraction review and not part of
  this chunk's accepted outputs.