# Autopilot-Plan für den nächsten Manifest-Auto-Lauf

**Zweck:** Dokumentiert die Auswahl der nächsten 10 Chunks, die nach
dem Smoke-Lauf `20260929T160008Z-9dbc3c` und nach dem ersten
menschlichen Review-Pass verarbeitet werden sollen. Der Plan ist
**ein Vorbereitungsschritt** — es werden **keine** Live-Anfragen an
den Provider gesendet. Die Auswahl ist deterministisch, offline und
reproduzierbar.

## 1. Auswahlkriterien (dokumentiert)

| # | Kriterium | Wert | Begründung |
|--:|---|---|---|
| 1 | Ausgeschlossene Chunks (Smoke-Lauf) | `c_0001921855a8156b`, `c_001452fd142ce075` | Diese zwei Chunks wurden im Smoke-Lauf bereits verarbeitet; ein Re-Send würde das Benchmark-Ergebnis verzerren. |
| 2 | Tercile-Balance (Ziel-Allokation) | 3 × `long`, 3 × `medium`, 4 × `short` | Die 40 Benchmark-Chunks verteilen sich nach Zeichenlänge auf die drei Terciles (Bounds `[445, 843]` Zeichen, berechnet via `select._compute_tercile_bounds`). Eine ausgewogene Allokation stellt sicher, dass sowohl informationsdichte als auch knappe Chunks geprüft werden. |
| 3 | Pro-Dok-Cap | `2` | Vermeidet Übergewichtung einzelner Dokumente. Im Smoke-Lauf wurden 2 Dokumente (W3038572442, W3137305332) bereits vollständig durch jeweils einen Chunk abgedeckt; die übrigen 7 Dokumente tragen je 2–6 Chunks. |
| 4 | Tag-Mindestanforderungen | `prose_min = 1`, `table_min = 1` | Stellt sicher, dass der nächste Lauf sowohl Fließtext als auch Tabellen-Chunks trifft; reine LaTeX-Gerüste sind im Datensatz überrepräsentiert (40 / 40) und brauchen keinen Backfill. |
| 5 | Deterministische Ordnung | `(length_bucket, doc_rank, chunk_index, chunk_id)` | Erlaubt bit-genaue Reproduktion des Plans; `doc_rank` ist `int(sha256(doc_id)[:4])`, `chunk_index` ist die Originalposition im geteilten Chunk-Set. |
| 6 | Seed | `0xFEEDFACE` (`4277009102`) | Eingefroren. Eine Änderung erfordert einen Algorithmus-Bump (`ALGORITHM_VERSION` → `v2`). |

**Algorithmus-Version:** `manifest-auto-autopilot-selector/v1`.
**Implementierung:**
`scripts/experiments/manifest_auto_review_20260929T160008Z/autopilot_select_10.py`.

## 2. Geplanter Aufrufumfang (10 Chunks)

| Rang | `chunk_id` | `doc_id` | Sektion | Zeichen | Tokens | Bucket | Tags (prose / latex / table) |
|--:|---|---|---|---:|---:|---|---|
| 1 | `c_7fdad3ec8f3e783b` | W3120014137 | Method / Passage Reranking | 1448 | 362 | long | – / ✓ / – |
| 2 | `c_6914ed3f4c5362fb` | W2949615363 | Models / Neural Network Models | 1460 | 365 | long | – / ✓ / – |
| 3 | `c_ca129edd34052284` | W2949615363 | Empirical Evaluation | 1283 | 320 | long | – / ✓ / – |
| 4 | `c_2a9f84f5fe0ce2ce` | W2955868579 | Deep-Learning-Based Methods / Typical Deep-Learning Methods | 653 | 163 | medium | – / ✓ / – |
| 5 | `c_9c4a6e0f2cab05e3` | W3214049768 | Conclusion | 571 | 142 | medium | – / ✓ / – |
| 6 | `c_8a84177f479aea1f` | W3120014137 | Experiment Results / Runtime Efficiency | 585 | 146 | medium | ✓ / ✓ / – |
| 7 | `c_c0de5074475c9c0f` | W3119866685 | Designing Models with Data, Model, and Expert-Parallelism / Data Parallelism | 411 | 102 | short | ✓ / ✓ / – |
| 8 | `c_61090bb37bc1e7a9` | W3119866685 | Pseudo Code for Switch Transformers | 162 | 40 | short | – / ✓ / ✓ |
| 9 | `c_abdfc58c76ee1139` | W2955868579 | New Trends / Multi-Passage Machine Reading Comprehension | 413 | 103 | short | ✓ / ✓ / – |
| 10 | `c_d285f0a62ff9479b` | W3022373106 | Proposed Method / Residual-based Learning | 206 | 51 | short | ✓ / ✓ / – |

### Aggregat-Kennzahlen

| Größe | Wert |
|---|---:|
| Chunks ausgewählt | 10 |
| `long`-Chunks | 3 |
| `medium`-Chunks | 3 |
| `short`-Chunks | 4 |
| `prose`-Chunks | 4 |
| `table`-Chunks | 1 |
| Eindeutige Dokumente | 6 (von 7 verfügbaren ohne Smoke-Run-Ausschluss) |
| Σ Zeichen | 7192 |
| Σ `token_estimate` | 1794 |

### Pro-Dok-Allokation

| `doc_id` | Chunks im Plan | Smoke-Überlappung |
|---|---:|:---:|
| W3120014137 | 2 | – |
| W2949615363 | 2 | – |
| W2955868579 | 2 | – |
| W3119866685 | 2 | – |
| W3214049768 | 1 | – |
| W3022373106 | 1 | – |

W2133564696 trägt 6 Kandidaten-Chunks, wird aber durch den
Pro-Dok-Cap (2) und die frühe Selektion der längeren Chunks aus
anderen Dokumenten nicht im Plan bedient. Eine Erweiterung des
Autopiloten auf 20 Chunks würde W2133564696 mit hoher Wahrscheinlichkeit
einbeziehen.

## 3. Was dieser Plan *nicht* tut

* **Keine Live-Anfragen.** Es werden keine HTTP-POSTs an den
  Manifest-Provider gesendet; keine `extractions.jsonl`,
  `extractions.rejects.jsonl` oder `metrics.jsonl` werden erzeugt.
* **Keine Änderungen an Originaldaten.** Der
  `benchmark_chunks.jsonl`-Eingabedatensatz wird nicht mutiert.
* **Keine Änderung der V2-Validierung.** Der kanonische
  Pipeline-Pfad bleibt unverändert.
* **Kein Gold-Set.** Die eingefrorenen Gold-Daten unter
  `gold_set/` und `data/gold_set/` werden nicht angefasst.
* **Kein Push, kein Merge.** Der Plan wird ausschließlich lokal
  committet.

## 4. Reproduzierbarkeit

```bash
# Plan ohne Schreiben neu erzeugen
PYTHONPATH=. python3 \
  scripts/experiments/manifest_auto_review_20260929T160008Z/autopilot_select_10.py

# Plan deterministisch persistieren
PYTHONPATH=. python3 \
  scripts/experiments/manifest_auto_review_20260929T160008Z/autopilot_select_10.py \
  --output docs/experiments/manifest-auto-review/autopilot_plan_20260929T160008Z.json

# Andere Auswahl erzwingen (z. B. zur Exploration weiterer Konfigurationen)
PYTHONPATH=. python3 \
  scripts/experiments/manifest_auto_review_20260929T160008Z/autopilot_select_10.py \
  --exclude c_0001921855a8156b,c_001452fd142ce075,c_7fdad3ec8f3e783b \
  --output /tmp/autopilot_plan_excluding_r1.json
```

Eine Änderung der Auswahlkriterien oder des Seeds erfordert einen
Algorithmus-Bump (`ALGORITHM_VERSION` und ein neuer
`AUTOPILOT_SEED`).

## 5. Nächste Schritte (außerhalb dieses Plans)

1. Menschlicher Review der Tabelle
3. [INFERENCE: möglicherweise] Anpassung der `mention_labels` /
   `assertion_labels` durch Reviewer (bleibt in den JSON-Dateien
   unverändert; eine separate `reviewer_decisions.json` ist die
   vorgesehene Senke).
4. Erst nach Review-Entscheidung: tatsächliches Dispatching der
   ausgewählten 10 Chunks via
   `scripts/benchmark/run_smoke.py` (oder ein gleichwertiger
   Benchmark-Runner), Aufzeichnung der Live-Metriken und ein
   zweiter Review-Pass.

## 6. Pfade

| Datei | Pfad |
|---|---|
| Selektor-Implementierung | `scripts/experiments/manifest_auto_review_20260929T160008Z/autopilot_select_10.py` |
| Selektor-Ausgabe (Plan JSON) | `docs/experiments/manifest-auto-review/autopilot_plan_20260929T160008Z.json` |
| Eingabedatensatz (Chunks) | `src/benchmark/dataset/benchmark_chunks.jsonl` |
| Selektor-Hilfsfunktionen | `src/benchmark/dataset/select.py` (`_compute_tercile_bounds`, `_length_bucket`, `_tag_chunk`) |
| Review-Tabelle (OMP-Vorschläge) | `docs/experiments/manifest-auto-review/review_table_20260929T160008Z.md` |
| Bestehende Annotationen | `docs/experiments/manifest_auto_review_20260929T160008Z/annotations.json`, `annotations_chunk1.json` |