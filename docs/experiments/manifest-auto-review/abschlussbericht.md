# Abschlussbericht: Wissenschaftliche Auswertung Manifest-Auto-Lauf `20260929T160008Z-9dbc3c`

## 1. Anlass und Reichweite

Der vollständige Manifest-Auto-Lauf
`~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/`
(`run_summary.json` mit `n_chunks: 2`, `n_extraction_complete: 2`,
Modell-Routing `auto`, `model_identity_verified: false`) produzierte
4 HTTP-POST-Antworten für 2 Smoke-Chunks (`c_0001921855a8156b`,
`c_001452fd142ce075`). Der bestehende kanonische V2-Pfad blieb
unverändert. Dieser Bericht fasst die wissenschaftliche Auswertung
zusammen, die ohne neue Modellanfragen aus den vorhandenen Artefakten
möglich ist.

**Was dieser Bericht nicht leistet:** Es werden keine Precision-/
Recall-Aussagen, keine Gold-Labels und keine wissenschaftlichen
Wahrheitsurteile vergeben. Der Fokus liegt auf (a) der Aufdeckung
und Korrektur eines Review-Generator-Bugs, (b) dem transparenten
Offline-Vergleich der Records mit dem vorhandenen Qwen-Quote-Repair-
Experiment, (c) der offenen Markierung der verbleibenden
wissenschaftlichen Unsicherheiten.

## 2. Bug in `write_blinded_review_package` und Korrektur

### Befund

Im ursprünglichen Review-Paket unter
`~/migration-incoming/manifest-auto-runs/evaluation/review_records.jsonl`
enthält der Record `c_001452fd142ce075` (`rev_manifest_auto_54dfd813`)
**8** Einträge im Feld `mentions`: 4 echte Mentions (`kind=mention`)
plus 4 Assertions (`kind=assertion`). Das Feld `assertions` enthält
ebenfalls diese 4 Assertions. Damit sind die vier Manifest-Auto-
Assertions im `mentions`-Feld dupliziert.

**Root Cause:** In
`src/benchmark/evaluate.py:283-315` (vor Patch) wurde das `mentions`-
Feld direkt aus `records_by_chunk.get(chunk_id, [])` befüllt — diese
Liste enthält sowohl `kind=mention` als auch `kind=assertion`. Nur
das `assertions`-Feld wurde explizit per `kind == "assertion"`
gefiltert. Der Fehler ist ein klassisches Refactor-Übersehen, kein
Schema-Bruch.

### Korrektur

Patch in `src/benchmark/evaluate.py`:

- `mentions`-Feld wird jetzt durch `[r for r in chunk_records if r.get("kind") == "mention"]` gefiltert.
- `assertions`-Feld wird symmetrisch gefiltert.
- Pro Record wird zusätzlich `processed: bool` gesetzt (`true`, falls
  mindestens eine Mention, Assertion, ein Reject oder ein
  `chunk_status` für den Chunk existiert).
- Der `reviewer_prompt` enthält jetzt den Hinweis, dass Chunks mit
  `processed: false` nicht zu labeln sind.

Die kanonische V2-Validierung, `metrics_<label>.json`, `extractions.jsonl`
und alle wissenschaftlichen Datensätze sind **nicht** verändert. Die
Korrektur betrifft ausschließlich den Review-Export-Pfad.

### Regressionstests

`tests/benchmark/test_evaluate.py`:

- `test_write_blinded_review_package_separates_mentions_and_assertions`
  — prüft: `mentions`-Feld enthält ausschließlich `kind=mention`;
  `assertions`-Feld ausschließlich `kind=assertion`; keine ID-
  Überschneidung; Zählung matcht die Eingabe; unbearbeitete Chunks
  sind `processed: false` mit leeren Listen.
- `test_write_blinded_review_package_marks_processed_chunks`
  — prüft das `processed`-Flag und den neuen Hinweis im
  `reviewer_prompt`.

Beide Tests sind deterministisch, offline und benötigen keinen
Modellaufruf.

## 3. Verifikation der technischen Auswertung

Aus `metrics_manifest-auto.json` (Original und korrigierter Lauf
identisch, da `metrics_<label>.json` unabhängig vom Review-Generator-
Patch ist):

| Kennzahl | Wert |
|---|---:|
| `chunks_total` | 2 |
| `chunks_complete` | 2 |
| `chunks_with_entity_ok` | 2 |
| `chunks_with_relation_ok` | 2 |
| `mentions_accepted` | 6 (2 + 4) |
| `assertions_accepted` | 4 (0 + 4) |
| `candidates_rejected` | 0 |
| `schema_errors`, `evidence_errors`, `ontology_errors`, `alignment_errors` | 0 |
| `model_identity_verified` | false |

Damit bestätigt die technische Auswertung weiterhin **2 vollständige
Chunks, 6 Mentions, 4 Assertions, 0 Rejects** — unverändert vor und
nach dem Patch.

## 4. Warum das Review-Paket 40 Einträge enthält

`scripts/benchmark/evaluate.py` ruft `write_blinded_review_package`
mit `--chunks src/benchmark/dataset/benchmark_chunks.jsonl`. Diese
Datei enthält **40** Chunks (siehe `wc -l
src/benchmark/dataset/benchmark_chunks.jsonl`). Der Generator
iteriert über alle 40 Chunks und legt für jeden Chunk einen Review-
Record an. Bei dem vorliegenden Smoke-Lauf wurden aber nur 2 Chunks
prozessiert; 38 Einträge sind strukturelle Stub-Records mit leeren
`mentions`/`assertions`/`rejects` und ohne `chunk_status`.

**Mitigation:** Der Patch fügt pro Record `processed: bool` hinzu.
Im korrigierten Export tragen 2 Records `processed: true`, 38 Records
`processed: false`. Der `reviewer_prompt` enthält den expliziten
Hinweis, dass nur Records mit `processed: true` zu labeln sind. Die
Stub-Records bleiben im Export enthalten, damit der Reviewer die
Gesamtmenge der 40 Chunks sehen und begründet ausschließen kann.

## 5. Korrigierte Review-Ausgabe

**Pfad:** `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/evaluation-corrected/`

Inhalt:

- `metrics_manifest-auto.json` — identisch zum Original.
- `review_records.jsonl` — 40 Records, davon 2 mit
  `processed: true` und korrekt nach `kind` getrennten
  `mentions`/`assertions`.
- `review_key.json` — Mapping von `review_label` →
  `(provider_label, chunk_id)`, deterministisch via
  `review_seed: 2971328237`.

Erzeugt durch:

```bash
python scripts/benchmark/evaluate.py \
  --run ~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c \
  --provider-label manifest-auto \
  --chunks src/benchmark/dataset/benchmark_chunks.jsonl \
  --prompts-dir prompts \
  --output-dir ~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/evaluation-corrected
```

Die bestehende kanonische V2-Validierung, das Original-`extractions.jsonl`,
das `metrics.jsonl`, das `run_summary.json`, die
`docs/experiments/quote-repair-v1/`-Bestände und alle wissenschaftlichen
Gold-Sets sind unverändert.

## 6. Offline-Vergleich Qwen vs. Manifest-Auto

Vollständige Tabelle und Methodikvergleich siehe
`docs/experiments/manifest-auto-review/offline-comparison-c_001452fd142ce075.md`.

Kernpunkte:

- **Drei Qwen-Quote-Repair-Kandidaten** (alle mit rekonstruiertem
  Caption-Span `[34, 190)`, Original-Whitespace): `a_44999b3871c3f58f`
  (`REPORTS_METRIC` → Answer Coverage), `a_0a775699fcc8a5ce`
  (`EVALUATES_ON` → NQ), `a_2814c002a3ebd912` (`EVALUATES_ON` → TQA).
- **Vier Manifest-Auto-Assertions:** `a_186ebfe9b1cd3758`,
  `a_48f1cfb9f80bca7b` (beide `USES_METHOD` auf DR bzw. ANCE,
  Evidence `[649, 722)`); `a_4fd1879b3d8f18b7`, `a_c690e0b1f2dfec3a`
  (beide `EVALUATES_ON` auf TQA bzw. NQ, Evidence `[25, 162)`).
- **Qwen-Ergebnisse sind experimentell** (`canonical_v2: false`),
  **Manifest-Auto-Ergebnisse sind kanonisch akzeptiert durch die
  unveränderten V2-Gates**. Die beiden Datensätze sind nicht als
  direkter Modellvergleich zu lesen — sie stammen aus
  unterschiedlichen methodischen Pfaden (Cache-Replay vs. Live-Routing).

## 7. Wissenschaftliche Unsicherheiten

### 7.1 Fehlende REPORTS_METRIC für Answer Coverage

Der Manifest-Auto-Lauf hat für `c_001452fd142ce075` **keine**
`REPORTS_METRIC`-Relation zwischen Dokument und der Metrik
`Answer Coverage at Top-20/100` extrahiert. Die Metrik steht im
Caption-Text (`chunk.text[53:82]`); das Modell hat sie aber weder als
Mention noch als Relation emittiert. Im Qwen-Quote-Repair-Experiment
ist genau diese Relation vorhanden
(`a_44999b3871c3f58f`). Mögliche Ursachen [INFERENCE]: das Modell
hat die Metrik nicht als eigenständige Mention erkannt oder der
Relations-Prompt hat die Relation nicht ausgegeben. Ohne das
gespeicherte Roh-Prompt-Response-Paar ist eine zwingende
Ursachenanalyse nicht möglich.

### 7.2 Normalisierung `Trivial QA (TQA)` ↔ `Trivia QA`

- **Verbatim-Quelle (LaTeX-Caption):** `Trivial QA (TQA)` in
  `Tables/joint.tex` (Tarball SHA-256
  `46288f02699612de50ae6f7530080bced7e1edc5e3ffb2225b80e5d6742b58f0`).
- **Vorherige V2-Läufe (`v2-multichunk-01`, `v2-pilot-03`,
  `v2-pilot-04`):** `canonical_form="Trivial QA"`,
  `surface_form="Trivial QA (TQA)"`.
- **Manifest-Auto-Lauf:** `canonical_form="Trivia QA"`,
  `surface_form="Trivial QA (TQA)"`.
- **Offizieller Datensatzname (Joshi et al. 2017):** `Trivia QA` /
  `TriviaQA`. Die Caption-Schreibweise `Trivial QA` ist eine
  Tippvariante im LaTeX-Quelltext.
- **Projektinterne Alias-/Ontologie-Regel:** **nicht vorhanden**.
  `data/aliases.json` enthält nur `work_id`-Mappings; die Ontologie
  deklariert keine Mention-Aliase; `entity_resolve.py` und
  `src/ontology/__init__.py` enthalten keine `Trivial QA` /
  `Trivia QA`-Regel.

**Folgerung:** Die Diskrepanz ist eine Modell-Normalisierungsdivergenz,
keine projektintern definierte Regel. **Keine** automatische
Korrektur vorgenommen. Eine Anpassung der `canonical_form` wäre eine
modellaussage-modifizierende Entscheidung außerhalb dieses Auftrags.

### 7.3 Zwei USES_METHOD-Relationen ohne wissenschaftliche Labels

Verbatim-Beleg für beide Assertions:
`chunk.text[649:722]` =
`The gains are from changing the training of a production DR model to ANCE`.

- `a_186ebfe9b1cd3758` (`USES_METHOD` → `Dense Retrieval`): Mention
  `[706, 708)` vollständig im Evidence-Quote enthalten; im lokalen
  Chunk fehlt der definierende Satz für `DR` (`Evaluation.tex:19`
  liegt außerhalb des Chunk-Spans).
- `a_48f1cfb9f80bca7b` (`USES_METHOD` → `ANCE`): Mention
  `[718, 722)` vollständig im Evidence-Quote enthalten; der Satz
  beschreibt eine Trainingsänderung *zu* ANCE, nicht eine
  klassische `USES_METHOD`-Anwendung.

**Reviewer-Tabelle (offene Felder):**

| Assertion | Relation | Objekt | supported | unsupported | unclear | schema_error |
|---|---|---|:---:|:---:|:---:|:---:|
| `a_186ebfe9b1cd3758` | `USES_METHOD` | `Dense Retrieval` | ☐ | ☐ | ☐ | ☐ |
| `a_48f1cfb9f80bca7b` | `USES_METHOD` | `ANCE` | ☐ | ☐ | ☐ | ☐ |

Die Tabelle enthält **keine** Vorbelegung. Eine Entscheidung
`supported` / `unsupported` / `unclear` ist ohne externes Wissen
oder ein Gold Set nicht möglich.

### 7.4 Caption-Span `[25, 162)` vs. `[34, 190)`

Der Manifest-Auto-Span endet **vor** `[[CITE:karpukhin2020dense]]`
und vor `\label{tab:openqa}}`. Der Qwen-Repair-Span enthält
diese Bestandteile. Der Reviewer muss entscheiden, ob der Verlust
der Citation die wissenschaftliche Tragfähigkeit verändert.

### 7.5 Modellidentität Manifest-Auto

`run_summary.json` deklariert
`model_identity_verified: false` und `reported_model: null`. Die
`attempt_reported_models` enthalten `MiniMax-M3`, aber ohne
Verifikation. Der Vergleich der Records ist daher kein direkter
Modellvergleich, sondern ein Vergleich unterschiedlicher
methodischer Pfade.

## 8. Tests und Reproduzierbarkeit

### Neue Regressionstests in `tests/benchmark/test_evaluate.py`

1. `test_write_blinded_review_package_separates_mentions_and_assertions`
2. `test_write_blinded_review_package_marks_processed_chunks`

Beide Tests sind offline, deterministisch, ohne Modellanbindung.

### Lauf

```bash
cd ~/migration-incoming/model-benchmark-exp
python3 -m pytest tests/benchmark/test_evaluate.py -q
```

Ergebnis: `11 passed in 0.19s` (vorher 9, jetzt 11 inkl. der 2
neuen Regressionstests).

### Reproduzierbare Erzeugung der korrigierten Review-Ausgabe

```bash
python scripts/benchmark/evaluate.py \
  --run ~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c \
  --provider-label manifest-auto \
  --chunks src/benchmark/dataset/benchmark_chunks.jsonl \
  --prompts-dir prompts \
  --output-dir ~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/evaluation-corrected
```

## 9. Pfade zur korrigierten Review-Ausgabe

| Datei | Pfad |
|---|---|
| Korrigierter Review-Export | `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/evaluation-corrected/review_records.jsonl` |
| Korrigierter Review-Key | `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/evaluation-corrected/review_key.json` |
| Metriken (unverändert) | `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/evaluation-corrected/metrics_manifest-auto.json` |
| Original-Review-Paket (unverändert) | `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/evaluation/review_records.jsonl` |
| Offline-Vergleich Qwen ↔ Manifest-Auto | `docs/experiments/manifest-auto-review/offline-comparison-c_001452fd142ce075.md` |
| Patch (Code) | `src/benchmark/evaluate.py` |
| Patch (Tests) | `tests/benchmark/test_evaluate.py` |

## 10. Was außerhalb dieses Auftrags liegt

- Vergebene wissenschaftliche Review-Labels (`supported` /
  `unsupported` / `unclear` / `schema_error`) für die vier
  Manifest-Auto-Assertions oder für die zwei USES_METHOD-Relationen.
- Eine Entscheidung über die fachliche Korrektheit der
  `Trivial QA (TQA)` ↔ `Trivia QA`-Normalisierung.
- Eine Aussage zur wissenschaftlichen Korrektheit der Caption-
  Span-Wahl `[25, 162)` vs. `[34, 190)`.
- Eine Modellaussage-modifizierende Korrektur der
  `canonical_form` für die Trivia-QA-Mention.
- Neue Modellaufrufe, Gold-Set-Erstellung, Precision/Recall-
  Messungen.

## 11. Geänderte Dateien (Commit-Übersicht)

Geändert in `~/migration-incoming/model-benchmark-exp`:

- `src/benchmark/evaluate.py` — `write_blinded_review_package`:
  Filterung von `mentions` nach `kind`, Filterung von `assertions`
  nach `kind` (vorher nur Assertions gefiltert), Hinzufügen
  `processed: bool` und Reviewer-Hinweis.
- `tests/benchmark/test_evaluate.py` — 2 neue Regressionstests
  (`…_separates_mentions_and_assertions`,
  `…_marks_processed_chunks`).
- `docs/experiments/manifest-auto-review/offline-comparison-c_001452fd142ce075.md`
  — neue Dokumentation.
- `docs/experiments/manifest-auto-review/abschlussbericht.md` —
  diese Datei.

Nicht geändert: kanonische V2-Validierung, `extractions.jsonl`,
`metrics.jsonl`, `run_summary.json`, Original-Review-Paket unter
`evaluation/`, wissenschaftliche Gold-Sets, eingefrorene Datensätze.

## 12. Verweise

- Offline-Vergleich (Qwen ↔ Manifest-Auto):
  `docs/experiments/manifest-auto-review/offline-comparison-c_001452fd142ce075.md`
- Vorhandene Qwen-Repair-Dokumentation:
  `docs/experiments/quote-repair-v1.md`
- Vorhandener Quellenkontext (Caption-zu-Tabellen-Beziehung):
  `docs/experiments/quote-repair-v1-review/context-report.md`
- Manifest-Auto-Run-Summary:
  `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/run_summary.json`
- Manifest-Auto-Records:
  `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/extractions.jsonl`
- Ontologie: `ontology.yaml:169-218` (`EVALUATES_ON`, `USES_METHOD`,
  `REPORTS_METRIC`).
- Strukturelle Validierungs-Gates:
  `docs/specs/extraction-contract.md:319-353`
- Original-LaTeX-Tarball: `data/raw/latex/2007.00808.tar.gz`
  (SHA-256 `46288f02699612de50ae6f7530080bced7e1edc5e3ffb2225b80e5d6742b58f0`).
