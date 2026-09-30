# Review-Paket: fünf ursprünglich leere Chunks

**Unabhängige, zunächst nicht annotierte Vorlage.** Alle Kandidaten sind Vorschläge; es werden keine Gold-Labels erzeugt. Zeichenpositionen sind `[start,end)` im unveränderten Chunktext.

## Provenienz

Quelle der eingefrorenen Originaltexte: `/home/immo/manifest-two-chunk.WB08yOG3/recording/inputs/chunks.jsonl`; Datensatz-SHA256 `38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985`; Recording-Manifest-SHA256 `9d6a91f7f879ac550abf45c07f41f436df4b46bdc38b7bae301c274b02c9bfb8`. Im historischen Lauf waren alle fünf technisch `extraction_complete=true`, jeweils 0 Mentions und 0 Assertions. Historische Rohantworten fehlen.

## Gemeinsames Review-Formular (je Chunk)

- Reviewer 1 / Datum: 
- Reviewer 2 / Datum: 
- Gesamturteil: ☐ tatsächlich leer ☐ mögliche Entity-Auslassung ☐ mögliche Relations-Auslassung ☐ beides ☐ unentscheidbar
- Kontextbedarf / fehlende Quellen: 
- Reviewer-1-Begründung: 
- Reviewer-2-Begründung: 
- Konfliktauflösung: 

### Mögliche Entitäten

| Surface Form | Intervall | Typvorschlag | Status/Begründung Reviewer 1 | Reviewer 2 |
|---|---:|---|---|---|
| | | | | |

### Mögliche Relationen

| Subjekt | Relation | Objekt | Evidenz + Intervall | Reviewer 1 | Reviewer 2 |
|---|---|---|---|---|---|
| | | | | | |

## Chunk 1: `c_7fdad3ec8f3e783b`

- Dokument: `W3120014137`
- Abschnitt: `Method` → `Passage Reranking`
- Text-SHA256: `d7d7d03a974501481fd8fbb06ffed8ca8131e11b3d702446a536be938a9b7052`
- Technischer Status: historisch leer/complete

```text
Given an initially retrieved passage list $\mathbf{R}$ and top-N predictions of the reader $\mathbf{A}^{[:N]}$, \ours forms a reranked passage list $\mathbf{R'}$ as follows.
\ours scans $\mathbf{R}$ from the beginning of the list and appends to $\mathbf{R'}$ every passage $p \in \mathbf{R}$ if $p$ contains any reader prediction $a \in \mathbf{A}^{[:N]}$ after string normalization (removing articles and punctuation) and tokenization. Then, the remaining passages are appended to $\mathbf{R'}$ according to their original order.

Intuitively, if the reader prediction is perfect, the retrieval accuracy after reranking is guaranteed to be optimal.
Specifically, if the reader prediction is correct, it is guaranteed that the retrieval accuracy after reranking is better, since \ours moves all passages containing the correct answer to the top (or at least the same if those passages are \textit{all} at the top before reranking). If the reader prediction is wrong, \ours could still be better if the predicted answer co-occurs with the correct answer, the same, or worse if the predicted answer is misleading.
In practice, if the reader performs reasonably well, \ours is also likely to rerank passages well.
Overall, we observe quantitatively that \ours leads to consistent gains in terms of both retrieval accuracy and QA performance without refining the retriever (reader) or even any training itself despite the noise in reader predictions.
```

**Automatische Vorschläge, nicht annotiert:** `\ours` an mehreren Stellen; `reader`, `retriever` und `QA` sind generische Begriffe. Makrobedeutung und konkreter Systemname fehlen. Relationen zwischen Dokument und `\ours` nicht automatisch ableiten; Ko-Erwähnung ist kein Beleg.

## Chunk 2: `c_2a9f84f5fe0ce2ce`

- Dokument: `W2955868579`
- Abschnitt: `Deep-Learning-Based Methods` → `Typical Deep-Learning Methods`
- Text-SHA256: `52dad8902fe6070da3407674c20d183351e1faf5acd6d62f67cd99318882bc6f`
- Technischer Status: historisch leer/complete

```text
\t\t\t\makecell[l]{Hierarchical Attention Flow\\ (Zhu et al. 2018) [[CITE:zhu2018hierarchical]]}&Conventional&RNN&\makecell[c]{Bidirectional\\One-Hop}&\makecell[c]{Option\\ Selector}\\
\t\t\t\midrule
\t\t\t\makecell[l]{ELMo + improved Bi-DAF\\ (Peters et al. 2018) [[CITE:peters2018deep]]}& Contextual & RNN &\makecell[c]{Bidirectional\\One-Hop}&\makecell[c]{ Span\\ Extractor}\\
\t\t\t\midrule
\t\t\tGPT (Radford et al. 2018)[[CITE:radford2018improving]]& Contextual&/ &/ &\makecell[c]{Option\\ Selector}\\
\t\t\t\midrule
\t\t\tBERT (Devlin et al. 2018) [[CITE:devlin2019bert]]& Contextual&/ &/ &\makecell[c]{Span\\ Extractor}\\
\t\t\t\bottomrule
\t\t\t\end{tabular}}
\t\t\end{table}
```

**Automatische Kandidaten, nicht annotiert:** `Hierarchical Attention Flow`, `ELMo`, `Bi-DAF`, `GPT`, `BERT` als mögliche `method`; `RNN` möglicherweise generisches Architekturkonzept. Tabellenzellen und Makros müssen als Originaltext geprüft werden. Autoren/Zitate sind kein Methodenname. Keine Relation aus derselben Tabellenzeile ohne explizite Ontologieaussage.

## Chunk 3: `c_c0de5074475c9c0f`

- Dokument: `W3119866685`
- Abschnitt: `Designing Models with Data, Model, and Expert-Parallelism` → `Data Parallelism`
- Text-SHA256: `22bd98e62ff620aa3933df773477ff8385ca8364aab9a67576d1a497ee98e63b`
- Technischer Status: historisch leer/complete

```text
When training data parallel models, which is the standard for distributed training, then all cores are allocated to the data-parallel dimension or $n = N, m = 1$.
This has the advantage that no communication is needed until the entire forward and backward pass is finished and the gradients need to be then aggregated across all cores. 
This corresponds to the left-most column of Figure \ref{fig:core_layouts}.
```

**Automatische Vorschläge, nicht annotiert:** `data parallel` / `data-parallel` / `distributed training` sind generische Konzepte; kein klar benannter Ontologie-Kandidat. `Figure` ist strukturell, nicht Methode. Review muss entscheiden, ob ein generisches Konzept ausgeschlossen wird.

## Chunk 4: `c_61090bb37bc1e7a9`

- Dokument: `W3119866685`
- Abschnitt: `Pseudo Code for Switch Transformers`
- Text-SHA256: `250de138d33b731503594a3826623a7b7b51ab19a5298539498fdb0fa05a8ba9`
- Technischer Status: historisch leer/complete

```text
\end{lstlisting}
\caption{Pseudo code of the Switch Transformer layer in Mesh Tensorflow.}
\end{figure}


\clearpage
\bibliography{references}


\end{document}
```

**Automatische Kandidaten, nicht annotiert:** `Switch Transformer`, `Mesh Tensorflow` als mögliche `method`. `\clearpage`/`\bibliography`/`\end` sind LaTeX-Struktur; `\CLEAR` ist hier nicht vorhanden. Caption belegt Benennung im Dokument, aber keine Relation wie `USES_METHOD` ohne weitere Aussage.

## Chunk 5: `c_d285f0a62ff9479b`

- Dokument: `W3022373106`
- Abschnitt: `Proposed Method` → `Residual-based Learning`
- Text-SHA256: `e85b3bb542c43d6d76058b0a6e077aac2a9794391431db75f06dc2e9f2cf35d5`
- Technischer Status: historisch leer/complete

```text
Through the course of training, the neural model learns to encode the semantic patterns that are not captured by text surface forms. When training finishes, the two models will  work together, as \CLEAR.
```

**Automatischer Vorschlag, nicht annotiert:** `\CLEAR` könnte ein benanntes System/Verfahren sein, aber die tatsächliche Makrodefinition fehlt. `neural model`, `two models` und `text surface forms` sind generisch. Keine Relation aus “work together” ableiten, solange die Endpoint-Identitäten unbekannt sind.

## Wissenschaftliche Sicherheitsregeln

Originaltext zuerst prüfen; danach erst technische Outputs. Vollständiger Kontext außerhalb des Chunks ist separat zu hashen und zu referenzieren. Menschliche Entscheidungen bleiben in diesem Paket leer. Kein Vorschlag wird in ein Gold-Set importiert. Recall ist erst nach vollständiger Referenzannotation zulässig.
