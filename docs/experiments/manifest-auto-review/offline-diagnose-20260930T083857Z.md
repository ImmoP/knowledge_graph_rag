# Offline-Diagnose Manifest-Auto-Lauf `20260930T083857Z-c69654`

**Auftrag:** Offline-Analyse des abgeschlossenen 10-Chunk-Laufs.
**Quellen:** `~/migration-incoming/manifest-auto-autopilot-live/20260930T083857Z-c69654/`,
`src/benchmark/dataset/benchmark_chunks.jsonl`,
bestehende V2-Aligner (`src/extract/align.py`),
experimentelle Whitespace-Reparatur (`src/extract/experimental_quote_repair.py`),
bestehender Benchmark-Evaluator (`src/benchmark/evaluate.py`).
**Keine** neuen Modellanfragen. **Keine** Änderung der kanonischen V2-Validierung.
**Keine** Modifikation der bisherigen Live-Ergebnisse, der Gold-Sets oder der
extractions.jsonl / rejects.jsonl / run_summary.json / metrics.jsonl dieses Laufs.

Strukturierte Rohdiagnose: `offline_diagnosis_20260930T091300Z.json`.
Diagnoseskript: `scripts/experiments/manifest_auto_offline_review/diagnose_20260930T083857Z.py`.

## 1. Aggregatzähler

Aus den `chunk_status`-Einträgen, akzeptierten Records und Rejects:

| Kennzahl | Wert |
|---|---:|
| `chunks_total` | 10 |
| `chunks_complete` (`extraction_complete=true`, unabhängig von Inhalt) | 5 |
| `chunks_complete_with_accepted_records` (complete und mind. 1 Mention **oder** 1 Assertion) | 0 |
| `chunks_complete_empty` (complete, aber 0 Mentions **und** 0 Assertions) | 5 |
| `chunks_incomplete_partial` (incomplete, aber ≥ 1 akzeptierter Record) | 5 |
| `mentions_accepted` | 13 |
| `assertions_accepted` | 9 |
| `candidates_rejected` | 7 |

Rejects nach Phase:

| Phase | Anzahl |
|---|---:|
| entity | 4 |
| relation | 3 |

Rejects nach Phase und Grund:

| Phase / Grund | Anzahl |
|---|---:|
| `entity/INVALID_OCCURRENCE_INDEX` | 4 |
| `relation/ENDPOINT_NOT_CONTAINED` | 2 |
| `relation/ZERO_OCCURRENCES` | 1 |

Die oben separat ausgewiesenen Kennzahlen werden ab diesem Commit vom
Benchmark-Evaluator ohne Änderung der bestehenden Felder unter
`metrics_<label>.json` mitgeliefert. Siehe Abschnitt 6.

## 2. Klassifikation der sieben Rejects (belegte Ursachen)

### 2.1 `c_ca129edd34052284` — Relation / `ZERO_OCCURRENCES` (eindeutig rekonstruierbar)

**Modell-Output:** `evidence_quote = "with the Attentive and Impatient Readers performing best across both datasets."`

**Belegt:** Der String kommt im Chunk nicht vor. Die experimentelle
Whitespace-Reparatur liefert eindeutig (`status=repaired`,
`span=[1201,1279)`) den Originalspan
`"with the\nAttentive and Impatient Readers performing best across both datasets."`.
Die einzige Abweichung zwischen Modell-Zitat und Originalspan ist ein
einzelner ` ` → `\n` Tausch an Offset 8 (nach `"with the"`); davor und danach
sind beide Strings byteweise identisch (gemeinsamer Präfix 8, gemeinsamer
Suffix 69, Gesamtlänge 78). Eine Wiederholung des rekonstruierten Spans
existiert im Chunk nicht (`re.finditer` liefert genau einen Treffer bei
Position 1201). Es liegt **keine** Absatzgrenze im rekonstruierten Span.

**Folgerung:** Die Reparatur ist eineindeutig und semantikneutral. Der
rekonstruierte Span würde durch den unveränderten V2-Aligner exakt gefunden
und enthält die akzeptierte Mention `Attentive and Impatient Readers`
(`[1210,1241)`, `m_1a6115d8b84689e0`). Ob diese Mention tatsächlich Endpoint
des verworfenen Relationskandidaten war und ob Endpoint-, Ontologie- und
Evidenzgate passieren, ist ohne die nicht persistierte Rohantwort nicht prüfbar.

**Hinweis zur wissenschaftlichen Bewertung:** Die technische
Rekonstruierbarkeit besagt **nicht**, dass die aus dem rekonstruierten
Span ableitbare Relation fachlich korrekt wäre. Die fachliche
Bewertung bleibt offen und ist ohne externes Wissen oder Gold Set nicht
möglich.

### 2.2 `c_6914ed3f4c5362fb` — Relation / `ENDPOINT_NOT_CONTAINED`

**Modell-Output:** `evidence_quote = "We can go further by\nequipping the model with the ability to reread from the document as each query\ntoken is read."`
mit Endpunkten `subject_ref = m_00fae178a6bd4bd7 ("Attentive Reader")` und
`object_ref = m_f262be7dc5fdf7f1 ("Impatient Reader")`.

**Belegt:** Der Belegspan `[165,279)` existiert wörtlich im Chunk
(`exact_quote_present=true`). Beide Mention-Endpunkte (`[37,53)` bzw.
`[15,31)`) liegen außerhalb dieses Spans. **Die referenzierten Entitäten
kommen im Beleg gar nicht vor.** Es handelt sich um eine **Modell-seitig
falsche Wahl des Belegspans**, nicht um eine Abweichung im Literal.

Der gleiche Chunk enthält **eine** akzeptierte Relation
`W2949615363 --PROPOSES--> m_f262be7dc5fdf7f1` mit Evidence
`"The Impatient Reader"` `[11,31)`. Diese ist endpoint-konform; sie
enthält aber nur eines der beiden Endpunkte.

### 2.3 `c_8a84177f479aea1f` — Relation / `ENDPOINT_NOT_CONTAINED`

**Modell-Output:** `evidence_quote = "it takes 12 to 16 hours with 8 GPUs for the DPR reader"`
mit `subject_ref = W3120014137 (document)` und
`object_ref = m_28e70c94eb305208 (mention "DPR")`.

**Belegt:** Belegspan `[258,312)` existiert wörtlich im Chunk
(`exact_quote_present=true`). `subject_ref` ist ein Dokument und ohne
Beschränkung (`contained=true`). `object_ref` zeigt auf Mention-Id
`m_28e70c94eb305208`, die im Chunk-Output eine eigene, **separate**
DPR-Mention an Span `[540,543)` ist. Dieser Mention-Span liegt **außerhalb**
des Belegspans `[258,312)`.

**Subtiler Befund:** Im Belegspan `[258,312)` selbst kommt der String
`"DPR"` an Position `[302,305)` vor — dort ist er aber **kein** vom Modell
emittierter Mention-Endpunkt (das Modell hat den DPR-Mention-Endpunkt
auf die spätere Stelle `[540,543)` gelegt). Die Ursache ist eine
Inkonsistenz zwischen Mention-Span und gewählter Relation.

Der gleiche Chunk enthält zwei akzeptierte Relationen mit Evidence
`"it takes around 3.5/11 min to generate answers on the NQ/Trivia test set with 1 GPU"`
an Span `[436,519)` und Endpunkten `NQ` bzw. `Trivia` — diese sind
endpoint-konform.

### 2.4 `c_9c4a6e0f2cab05e3` — Entity / `INVALID_OCCURRENCE_INDEX` (×2)

**Modell-Output:** Zwei Entity-Kandidaten mit `occurrence_index = 1` und
`occurrence_index = 2` (aus `extractions.rejects.jsonl`). Die
Rohantworten sind im Run-Output **nicht** persistiert
(`raw_responses/` ist leer).

**Belegbar:** Die alignierten, akzeptierten Mentions dieses Chunks sind
`question answering` an `[64,82)` und `biomedical question answering` an
`[53,82)` — beide ohne sichtbaren `occurrence_index` im gespeicherten
Record. Da der V2-Aligner akzeptierte Records nur dann aufnimmt, wenn
`occurrence_index` entweder weggelassen wird (bei unique literal) **oder**
einen gültigen Index liefert, hat das Modell für diese zwei Mentions
einen gültigen Index emittiert (sonst wären sie verworfen worden).

**Hypothesen zur Ursache der beiden Rejects** (ohne Rohantwort nicht
eindeutig entscheidbar):

| Hypothese | Plausibilität |
|---|---|
| Modell emittierte Literal `"biomedical question answering"` (kommt 2× vor an `[53,200)`) mit `occurrence_index=2` → INVALID (≥2 bei count=2) | mittel |
| Modell emittierte Literal `"question"` (kommt 3× vor an `[64,123,211)`) mit `occurrence_index=2` → wäre **akzeptiert** (gültig) | entfällt |
| Modell emittierte Literal `"answering"` (kommt 3× vor an `[73,132,220)`) mit `occurrence_index=2` → wäre **akzeptiert** (gültig) | entfällt |
| Modell emittierte Literal `"biomedical"` (kommt 3× vor an `[53,200,276)`) mit `occurrence_index=1` oder `=2` → wäre akzeptiert | entfällt |
| Modell emittierte ein **seltenes** Literal mit count=0 oder count=1 und einem Index jenseits dieses Counts | offen (Rohantwort nötig) |

**Folgerung:** Ohne Rohantwort keine eindeutige Zuordnung. **Falsche
Nummerierung (1- statt 0-basiert)** ist eine mögliche Ursache, aber nicht
belegbar. **Mehrfach vorkommender Text mit überschießendem Index** ist
eine alternative mögliche Ursache. Beide Varianten benötigen den
gespeicherten Modell-Output zur Klärung.

### 2.5 `c_abdfc58c76ee1139` — Entity / `INVALID_OCCURRENCE_INDEX` (×2)

**Modell-Output:** Zwei Entity-Kandidaten mit `occurrence_index = None`
(aus `extractions.rejects.jsonl`). Die akzeptierten Mentions dieses
Chunks sind `DrQA`, `SQuAD`, `exact match accuracy`, `open-domain setting`
— alle **unique** im Chunk.

**Belegbar:** Die zwei Rejects müssen Literale betreffen, die im Chunk
**mehrfach** vorkommen. Heuristik auf Wortebene (≥ 5 Zeichen):

| Literal | Vorkommen |
|---|---:|
| `multi-passage` | 2 (`[28,339)`) |

Das ist der einzige Wort-Treffer mit Mehrfachvorkommen in dem Chunk.
Eine plausible Modell-Emission wäre eine Mention
`surface_form = "multi-passage"` ohne `occurrence_index`. Diese
Kombination wird vom Aligner mit `INVALID_OCCURRENCE_INDEX` abgelehnt.

**Hypothese (plausibel, nicht belegt):** Das Modell hat eine Chunk-
übergreifende Phrase (`"multi-passage"` oder ein sinnverwandter Ausdruck
aus dem Sektionskopf `"Multi-Passage Machine Reading Comprehension"`) als
Mention emittiert, ohne einen disambiguierenden Index zu liefern. Ein
akzeptierbarer Workaround wäre `occurrence_index = 0`.

**Hinweis:** Die akzeptierten Relationen dieses Chunks
(`DrQA USES_METHOD`, `DrQA EVALUATES_ON SQuAD`, `SQuAD USED_FOR_TASK
open-domain setting`, `SQuAD MEASURES exact match accuracy`,
`exact match accuracy REPORTS_METRIC`) sind sämtlich endpoint-konform
und nutzen die akzeptierten Mention-Spans — die zwei Entity-Rejects
betreffen nur zwei zusätzliche, abgelehnte Mention-Kandidaten.

## 3. Fünf vollständig leere Chunks (heuristische Sichtung)

Diese Einschätzung ist **explizit nicht wissenschaftlich bestätigt**.
Sie liefert ausschließlich einen begrenzten Textauszug und eine
heuristische Indikation möglicherweise extrahierbarer Oberflächenformen.

### 3.1 `c_7fdad3ec8f3e783b` (`W3120014137`, Section `Passage Reranking`)

Auszug (verbatim, head 600 Zeichen):
> Given an initially retrieved passage list `$\mathbf{R}$` and top-N
> predictions of the reader `$\mathbf{A}^{[:N]}$`, \ours forms a reranked
> passage list `$\mathbf{R'}$` as follows. \ours scans `$\mathbf{R}$`
> from the beginning of the list and appends to `$\mathbf{R'}$` every
> passage `$p \in \mathbf{R}$` if `$p$` contains any reader prediction
> `$a \in \mathbf{A}^{[:N]}$` after string normalization (removing
> articles and punctuation) and tokenization. Then, the remaining
> passages are appended to `$\mathbf{R'}$` according to their original
> order. Intuitively, if the reader prediction is perfect, the retrieval
> accu...

**Heuristische Indikation:** Der Chunk ist methodisch (Passage-Reranking-
Algorithmus). Eigennamen wie `\ours` (oft ein Systemname in der
Publikation) sind im Auszug Platzhalter und benennen ein konkretes
System — diese Selbstreferenz ist ohne Quellkontext nicht als Mention
eindeutig auflösbar. Eine verblindete Fachbewertung wäre nötig, um zu
entscheiden, ob eine `\ours`-Mention extrahierbar ist.

### 3.2 `c_2a9f84f5fe0ce2ce` (`W2955868579`, Section `Typical Deep-Learning Methods`)

Auszug (verbatim, head 600 Zeichen):
> 			\makecell[l]{Hierarchical Attention Flow\\
> (Zhu et al. 2018) [[CITE:zhu2018hierarchical]]}&Conventional&RNN&\makecell[c]{Bidirectional\\
> One-Hop}&\makecell[c]{Option\\ Selector}\\
> 			\midrule
> 			\makecell[l]{ELMo + improved Bi-DAF\\
> (Peters et al. 2018) [[CITE:peters2018deep]]}& Contextual & RNN &\makecell[c]{Bidirectional\\
> One-Hop}&\makecell[c]{ Span\\ Extractor}\\
> 			\midrule
> 			GPT (Radford et al. 2018)[[CITE:radford2018improving]]& Contextual&/ &/ &\makecell[c]{Option\\ Selector}\\
> 			\midrule
> 			BERT (Devlin et al. 2018) [[CITE:devlin2019bert]]& Contextual&/ &/ &\makecell[c]{Span\\ Extr...

**Heuristische Indikation (nicht wissenschaftlich bestätigt):** Im Auszug
enthalten sind diverse Eigennamen, die nach üblicher Ontologie
extrahierbar wären:

| Kandidat | Typ |
|---|---|
| `Hierarchical Attention Flow` | method |
| `ELMo` | method |
| `Bi-DAF` | method |
| `GPT` | method |
| `BERT` | method |

Ein Großteil dieser Kandidaten kommt in der wissenschaftlichen Literatur
als Methoden-Mentions vor. Die LaTeX-Tabellenstruktur (`\makecell`,
`\midrule`) macht die Extraktion technisch aufwändig; das Modell hat
keine Mention emittiert.

### 3.3 `c_c0de5074475c9c0f` (`W3119866685`, Section `Data Parallelism`)

Auszug (verbatim, head 600 Zeichen):
> When training data parallel models, which is the standard for
> distributed training, then all cores are allocated to the
> data-parallel dimension or `$n = N, m = 1$`. This has the advantage
> that no communication is needed until the entire forward and backward
> pass is finished and the gradients need to be then aggregated across
> all cores. This corresponds to the left-most column of Figure
> \ref{fig:core_layouts}.

**Heuristische Indikation (nicht wissenschaftlich bestätigt):** Der
Chunk ist methodisch (Datenparallelismus). Er enthält keine
eigenständigen Methoden- oder Datensatz-Eigennamen im Auszug. Eine
Extraktion von benannten Konzepten ist aus dem Auszug nicht erkennbar;
für den weiteren Chunk-Verlauf wäre eine vollständige Sichtung nötig.

### 3.4 `c_61090bb37bc1e7a9` (`W3119866685`, Section `Pseudo Code for Switch Transformers`)

Auszug (verbatim, head 600 Zeichen):
>
>       \end{lstlisting}
>       \caption{Pseudo code of the Switch Transformer layer in Mesh
> Tensorflow.}
>       \end{figure}
>
>
>       \clearpage
>       \bibliography{references}
>
>
>       \end{document}

**Heuristische Indikation (nicht wissenschaftlich bestätigt):** Der
Auszug enthält zwei klare Eigennamen: `Switch Transformer` und
`Mesh Tensorflow`. Beide sind eigenständige Methoden- bzw. Framework-
Namen. Eine Extraktion als Mentions wäre fachlich plausibel; das
Modell hat im Entity-Schritt keine Mention emittiert.

### 3.5 `c_d285f0a62ff9479b` (`W3022373106`, Section `Residual-based Learning`)

Auszug (verbatim, head 600 Zeichen):
>
>       Through the course of training, the neural model learns to
> encode the semantic patterns that are not captured by text surface
> forms. When training finishes, the two models will  work together,
> as \CLEAR.

**Heuristische Indikation (nicht wissenschaftlich bestätigt):** Der
Chunk referenziert ein benanntes System `\CLEAR` (LaTeX-Macro). Eine
Mention-Extraktion wäre plausibel; das Modell hat im Entity-Schritt
keine Mention emittiert.

**Übergreifend:** Bei **3 von 5** leeren Chunks
(`c_2a9f84f5fe0ce2ce`, `c_61090bb37bc1e7a9`, `c_d285f0a62ff9479b`)
enthält der Auszug heuristisch klar extrahierbare Eigennamen. Bei
`c_7fdad3ec8f3e783b` ist eine Extraktion denkbar, aber an
Platzhalter-Bindungen geknüpft. Bei `c_c0de5074475c9c0f` ist im Auszug
keine offensichtliche Eigenname-Erwähnung erkennbar. Diese Einschätzung
ist **explizit nicht wissenschaftlich bestätigt** und erfordert für
belastbare Aussagen ein Gold Set oder eine verblindete Fachbewertung.

## 4. Experimentelle Whitespace-Reparatur: Bewertung des `ZERO_OCCURRENCES`-Falls

Die bestehende experimentelle Reparatur
(`src/extract/experimental_quote_repair.repair_quote`) trifft für
`c_ca129edd34052284` eine eindeutige Entscheidung:

| Kriterium | Wert |
|---|---|
| `status` | `repaired` |
| `repaired_quote` | `with the\nAttentive and Impatient Readers performing best across both datasets.` |
| `repaired_span` | `[1201, 1279)` |
| Präfix-Übereinstimmung mit Literal | 8 Zeichen |
| Suffix-Übereinstimmung mit Literal | 69 Zeichen |
| Mittlere Abweichung | `" "` vs `"\n"` (single-character whitespace swap) |
| Treffer im Chunk | exakt 1 (`exact_occurrences = [1201]`) |
| Absatzgrenze im rekonstruierten Span | nein |
| Ergebnis der unveränderten V2-Pipeline | Alignment offline prüfbar und bestanden; Endpoint-/Ontologie-/Evidenzgate ohne vollständigen Relationskandidaten nicht prüfbar |

**Bewertung:** Die Reparatur ist eineindeutig und semantikneutral. Der
rekonstruierte Beleg `[1201,1279)` wird exakt aligniert und enthält die
akzeptierte Mention `Attentive and Impatient Readers` an `[1210,1241)`.
Endpoint-Referenzen und `relation_type` des verworfenen Kandidaten sind nicht
persistiert; eine Aussage zur Gesamtakzeptanz durch sämtliche V2-Gates wäre
daher unbelegt.

**Kontrollierte Probe offline (lesend, ohne Mutation der Live-Daten):**

```python
from src.extract.experimental_quote_repair import repair_quote
result = repair_quote(chunk_text, model_literal)
assert result.status == "repaired"
assert result.quote == "with the\nAttentive and Impatient Readers performing best across both datasets."
assert (result.char_start, result.char_end) == (1201, 1279)
```

Ergebnis: bestätigt. **Keine** mehrdeutige und **keine** inhaltliche
Änderung — die Reparatur ist auf einen einzelnen Whitespace-Swap
begrenzt. Eine Übernahme in den finalen Knowledge Graph erfolgt im
Rahmen dieses Auftrags **nicht**; eine solche Übernahme wäre eine
separate, vom Menschen zu autorisierende Entscheidung und liegt
außerhalb des Offline-Diagnose-Auftrags.

## 5. Belegte Ursachen vs. Hypothesen

### 5.1 Belegt

| Chunk | Phase | Grund | Beleg |
|---|---|---|---|
| `c_ca129edd34052284` | relation | `ZERO_OCCURRENCES` | Originalspan `[1201,1279)` mit unique repair; einzige Abweichung ist ` `\→`\n` an Offset 8 |
| `c_6914ed3f4c5362fb` | relation | `ENDPOINT_NOT_CONTAINED` | Endpunkte `[15,31)` und `[37,53)` außerhalb Beleg `[165,279)`; Endpunkte kommen im Beleg nicht vor |
| `c_8a84177f479aea1f` | relation | `ENDPOINT_NOT_CONTAINED` | Mention-Id `m_28e70c94eb305208` → Mention-Span `[540,543)` außerhalb Beleg `[258,312)`; DPR-String existiert im Beleg an `[302,305)`, ist aber nicht die referenzierte Mention-Id |

### 5.2 Hypothesen (Rohantwort zur Klärung erforderlich)

| Chunk | Phase | Grund | Hypothese |
|---|---|---|---|
| `c_9c4a6e0f2cab05e3` | entity | `INVALID_OCCURRENCE_INDEX` (×2) | Modell emittierte Literal mit `occurrence_index` jenseits der Trefferanzahl; mögliche Auslöser: 1-basiertes Zählen (statt 0-basiert) **oder** Literal mit Mehrfachvorkommen und zu hohem Index. Rohantwort ist nicht persistiert. |
| `c_abdfc58c76ee1139` | entity | `INVALID_OCCURRENCE_INDEX` (×2) | Modell emittierte Literal mit `occurrence_index=None`, wobei das Literal im Chunk mehrfach vorkommt; einziger plausibler Wort-Treffer im Chunk ist `multi-passage` (2 Vorkommen). Rohantwort ist nicht persistiert. |

## 6. Ergänzte Benchmark-Kennzahlen

Die folgenden **neuen** Felder werden ab diesem Commit vom Benchmark-
Evaluator ohne Änderung der bestehenden Felder unter
`metrics_<label>.json` ausgegeben. Bestehende Metrikfelder
(`chunks_total`, `chunks_complete`, `chunks_with_entity_ok`,
`chunks_with_relation_ok`, `mentions_accepted`, `assertions_accepted`,
`candidates_rejected`, `schema_errors`, `evidence_errors`,
`ontology_errors`, `alignment_errors`, `total_duration_seconds`,
`total_input_tokens`, `total_output_tokens`, `per_chunk`,
`abort_reasons`) und die wissenschaftlichen Bewertungsgrenzen sind
unverändert.

| Neues Feld | Typ | Bedeutung |
|---|---|---|
| `chunks_complete_with_accepted_records` | `int` | Anzahl Chunks mit `extraction_complete=true` und ≥ 1 akzeptierter Mention **oder** ≥ 1 akzeptierter Assertion |
| `chunks_complete_empty` | `int` | Anzahl Chunks mit `extraction_complete=true` und 0 akzeptierten Mentions **und** 0 akzeptierten Assertions |
| `chunks_incomplete_partial` | `int` | Anzahl Chunks mit `extraction_complete=false` und ≥ 1 akzeptiertem Record |
| `rejects_by_phase` | `tuple[(phase, count), ...]` | Rejects aggregiert nach Phase |
| `rejects_by_phase_and_reason` | `tuple[(phase/reason, count), ...]` | Rejects aggregiert nach Phase und Grund |

Diese Kennzahlen sind rein **technisch** — sie spiegeln die unveränderte
V2-Validierung wider und treffen keine Aussage zur wissenschaftlichen
Korrektheit. Code: `src/benchmark/evaluate.py`. Tests:
`tests/benchmark/test_evaluate.py::test_compute_metrics_separate_bucket_counters`
und `::test_compute_metrics_counters_zero_when_no_runs`.

## 7. Maßnahmenplan (priorisiert)

1. **Vollständigen Relationskandidaten sichern** (technisch): Falls die
   ursprüngliche Rohantwort oder ein passender Offline-Cache noch erreichbar
   ist, den reparierten Span `[1201,1279)` zusammen mit `relation_type`,
   Endpunkten und Confidence durch die unveränderte V2-Pipeline replayen und
   Alignment-, Endpoint-, Ontologie- und Evidenzergebnis protokollieren. Ohne
   diesen Input bleibt nur die belegte Aussage: eindeutige Whitespace-
   Rekonstruktion plus bestandene Alignment-Prüfung.
2. **Rohantworten persistieren** (Verfahrensfrage): Der leere
   `raw_responses/`-Ordner in zukünftigen Läufen verhindert die
   Klärung der `INVALID_OCCURRENCE_INDEX`-Ursachen. Empfehlung:
   zukünftige Autopilot-Läufe mit Persistenz der Modellantwort
   konfigurieren, damit die Hypothesen in Abschnitt 5.2 eingeordnet
   werden können. *Aufwand:* gering; *Risiko:* Speicherverbrauch
   steigt.
3. **Modell-Output für `c_9c4a6e0f2cab05e3` und `c_abdfc58c76ee1139`
   erneut erzeugen oder aus vorhandenem Cache beziehen** (sofern
   verfügbar): erlaubt, die Hypothesen zu falsifizieren oder zu
   bestätigen. *Aufwand:* gering, sofern Rohantworten überhaupt
   rekonstruierbar sind; *Risiko:* Modellverhalten kann sich bei
   neuen Anfragen ändern.
4. **Manuelle Belegwahl-Korrektur diskutieren** (wissenschaftlich):
   Die beiden `ENDPOINT_NOT_CONTAINED`-Fälle sind Modell-seitige
   Fehlwahlen eines Belegspans. Eine automatische Korrektur würde
   Modellaussagen modifizieren und ist außerhalb dieses Auftrags.
   Empfehlung: in einer verblindeten Review-Runde überprüfen, ob der
   jeweils bessere Beleg (z. B. der `DrQA`-Beleg `[85,220)` für
   `c_abdfc58c76ee1139` oder der `Impatient Reader`-Beleg `[11,31)`
   für `c_6914ed3f4c5362fb`) eine vollständige und korrekte Relation
   tragen würde.
5. **Leere Chunks nicht stillschweigend akzeptieren** (Verfahrens-
   grundsatz): Die fünf `chunks_complete_empty`-Chunks (Abschnitt 3)
   sind aus technischer Sicht vollständig (keine Rejects), aber
   inhaltlich ungeprüft. **Keine** automatische Übernahme in den
   finalen Knowledge Graph; Empfehlung: heuristische Markierung als
   "inhaltlich zu prüfen" im Review-Paket. Eine Umsetzung wäre eine
   Erweiterung des Review-Pakets (`processed=true`, aber mit
   `empty_content=true`-Hinweis) und liegt außerhalb dieses Auftrags.
6. **Gold-Set-Erstellung zurückstellen** (offene wissenschaftliche
   Frage): Eine wissenschaftliche Beurteilung der fünf leeren Chunks
   und der zwei endpoint-fehlplatzierten Chunks erfordert ein
   gelabeltes Gold Set. Diese Arbeit ist explizit nicht Teil des
   Offline-Diagnose-Auftrags.

## 8. Pfade

| Artefakt | Pfad |
|---|---|
| Run-Verzeichnis | `~/migration-incoming/manifest-auto-autopilot-live/20260930T083857Z-c69654/` |
| Strukturierte Rohdiagnose (JSON) | `docs/experiments/manifest-auto-review/offline_diagnosis_20260930T091300Z.json` |
| Diagnoseskript | `scripts/experiments/manifest_auto_offline_review/diagnose_20260930T083857Z.py` |
| Erweiterte Benchmark-Kennzahlen | `src/benchmark/evaluate.py` (neue Felder) |
| Neue Tests | `tests/benchmark/test_evaluate.py` (+2 Tests) |
| Bestehende V2-Aligner | `src/extract/align.py` (unverändert) |
| Bestehende experimentelle Whitespace-Reparatur | `src/extract/experimental_quote_repair.py` (unverändert) |

## 9. Was außerhalb dieses Auftrags liegt

- Vergebene wissenschaftliche Review-Labels (`supported` / `unsupported`
  / `unclear` / `schema_error`) für irgendeine Relation oder Mention
  dieses Laufs.
- Eine automatische Übernahme des rekonstruierten Spans aus
  Abschnitt 2.1 in den finalen Knowledge Graph.
- Eine Modell-seitige Belegwahl-Korrektur für die beiden
  `ENDPOINT_NOT_CONTAINED`-Fälle.
- Eine semantische Anpassung des `occurrence_index`-Algorithmus
  (z. B. 1-basiert ↔ 0-basiert) ohne Rohantwort-Klärung.
- Eine neue Modellanfrage für einen der zehn Chunks dieses Laufs.
- Eine Änderung der kanonischen V2-Validierung oder der
  `extractions.jsonl`, `extractions.rejects.jsonl`, `run_summary.json`,
  `metrics.jsonl` dieses Laufs.
- Eine Änderung der wissenschaftlichen Gold-Sets.
