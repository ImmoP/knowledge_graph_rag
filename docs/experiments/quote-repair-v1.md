# Quote Repair v1: reproduzierbarer Offline-Vergleich

## Status

**Technisch erfolgreich, wissenschaftlich-inhaltlich noch nicht validiert.**

Der reproduzierte Vergleich umfasst ausschließlich die zwei Chunks, für die jeweils eine gültige gespeicherte Entity- und Relationsantwort vorhanden ist. Baseline und Experiment lesen dieselben vier content-adressierten Cache-Einträge. Ein fehlender, unlesbarer, falsch adressierter oder schema-inkompatibler Eintrag bricht den Lauf ab; der Runner besitzt keinen Modell- oder Netzwerk-Fallback.

Die Whitespace-Rekonstruktion behebt in diesem Datensatz drei `ZERO_OCCURRENCES`-Zurückweisungen. Danach akzeptiert die unveränderte offizielle V2-Prüfkette drei zusätzliche Assertions. Das zeigt technische Kompatibilität mit Alignment, Endpoint-Containment, Ontologie und Evidenzprüfung. Es zeigt **nicht**, dass die rekonstruierten Assertions fachlich richtig, vollständig oder nützlich sind.

## Experimentaufbau

### Lokale Voraussetzungen

Die Pilotdaten werden nicht ins Repository kopiert. Die folgenden Pfade sind relativ zum experimentellen Worktree und entsprechen der lokalen Ablage:

- Chunks und gespeicherte Baseline: `../v2-multichunk-01/`
- Originalprompts und Modellcache: `../v2-pilot-03/`
- Python-Umgebung des kanonischen Projekts: `../../projects/knowledge_graph_rag/.venv/bin/python`

Erwartete Eingaben:

```text
../v2-multichunk-01/three_chunks.jsonl
../v2-multichunk-01/baseline.v2.jsonl
../v2-multichunk-01/baseline.v2.rejects.jsonl
../v2-pilot-03/prompts/extract_entities_v2.txt
../v2-pilot-03/prompts/extract_relations_v2.txt
../v2-pilot-03/cache/llm/*.json
```

### Verglichene Chunks

| Chunk | Cache-Abdeckung | Ausgangszustand |
|---|---:|---|
| `c_001452fd142ce075` | Entity + Relation | 3 Mentions akzeptiert; 3 Relationskandidaten mit `ZERO_OCCURRENCES` zurückgewiesen; unvollständig |
| `c_0001921855a8156b` | Entity + Relation | 2 Mentions und 2 Assertions akzeptiert; vollständig |
| `c_012d2ff7efe77f40` | keine gültigen Cache-Einträge | Nur offline analysiert; nicht im A/B-Vergleich |

### Kontrollierte Variable

Beide Arme führen `src.extract.pipeline_v2.run_extraction_v2` mit identischen Chunks, Prompts, Ontologie, Parametern und gespeicherten JSON-Antworten aus.

- **Baseline:** gespeicherte Payload unverändert an die V2-Pipeline.
- **Experiment:** nur bei Relationspayloads wird für jedes `evidence_quote` `src.extract.experimental_quote_repair.repair_quote` aufgerufen. Ausschließlich Status `repaired` ersetzt das Zitat durch den eindeutig gefundenen Originalspan. `exact`, `ambiguous`, `not_found`, `paragraph_boundary` und `invalid` bleiben unverändert.
- **Unverändert:** offizieller V2-Aligner, Occurrence-Regeln, Endpoint-Containment, Ontologieprüfung, Evidenzprüfung und ID-Bildung.

Der Runner `scripts/experiments/compare_quote_repair_v1.py` rekonstruiert die produktiven Cache-Schlüssel einschließlich des aktiven sechs-dimensionalen Extraction-Scopes. Er nutzt absichtlich weder `make_llm()` noch `OllamaLLM`: deren Read-through-Cache würde bei einem Miss einen Modellaufruf erlauben.

### Nachweis identischer Antworten

Der Bericht enthält pro Provideraufruf:

- Chunk und Phase,
- reproduzierten Cache-Key,
- SHA-256 der Cache-Datei,
- SHA-256 der kanonisierten unveränderten Payload.

Vor der Ergebnisbildung wird die vollständige Aufrufliste beider Arme verglichen. Jede Abweichung bricht ab. Zusätzlich muss die reproduzierte Baseline exakt den gespeicherten Baseline-Records und -Rejects für die beiden Chunks entsprechen.

## Reproduzierbare Befehle

Vom Repository-Root:

```bash
PYTHON=../../projects/knowledge_graph_rag/.venv/bin/python

"$PYTHON" scripts/experiments/compare_quote_repair_v1.py \
  --chunks ../v2-multichunk-01/three_chunks.jsonl \
  --cache-dir ../v2-pilot-03/cache \
  --prompts-dir ../v2-pilot-03/prompts \
  --baseline-jsonl ../v2-multichunk-01/baseline.v2.jsonl \
  --baseline-rejects ../v2-multichunk-01/baseline.v2.rejects.jsonl \
  --output /tmp/quote-repair-v1-report.json
```

Fokussierte Tests:

```bash
"$PYTHON" -m pytest -q \
  tests/test_experimental_quote_repair.py \
  tests/test_quote_repair_comparison.py
```

Relevante bestehende V2-Regressionstests:

```bash
"$PYTHON" -m pytest -q \
  tests/test_extract_align.py \
  tests/test_extract_v2_pipeline.py \
  tests/test_extract_v2_canonical.py \
  tests/providers/test_cache_key.py \
  tests/providers/test_cache_io.py \
  tests/providers/test_extract_provider_scope.py
```

Keiner dieser Befehle benötigt oder erlaubt einen Modellaufruf. Der Vergleichsrunner beendet einen Cache-Miss mit Exit-Code 2.

## Ergebnisse

### Aggregate für die zwei vergleichbaren Chunks

| Kennzahl | Baseline | Experiment | Delta |
|---|---:|---:|---:|
| Chunks | 2 | 2 | 0 |
| vollständige Chunks | 1 | 2 | +1 |
| akzeptierte Mentions | 5 | 5 | 0 |
| akzeptierte Assertions | 2 | 5 | +3 |
| zurückgewiesene Kandidaten | 3 | 0 | -3 |

Rekonstruktionsentscheidungen:

- `repaired`: 3
- `exact`: 2
- alle anderen Status: 0

### Pro Chunk

#### `c_001452fd142ce075`

- Ausgang: 3 akzeptierte Mentions, 0 akzeptierte Assertions, 3 `ZERO_OCCURRENCES`-Rejects, unvollständig.
- Alle drei Relationskandidaten verwenden dasselbe inhaltliche Zitat mit normalisiertem Whitespace.
- Die Reparatur findet für jeden Kandidaten denselben eindeutigen Originalspan bei `[34, 190)` und stellt dessen Zeilenumbrüche/Einrückung wieder her.
- Danach: 3 akzeptierte Assertions, 0 Rejects, vollständig.

#### `c_0001921855a8156b`

- Ausgang: 2 akzeptierte Mentions, 2 akzeptierte Assertions, 0 Rejects, vollständig.
- Beide Evidence Quotes sind bereits exakte Quellspans; Entscheidung jeweils `exact`.
- Experiment identisch zur Baseline. Keine unnötige Umschreibung.

## Dritter Chunk: Offline-Fehleranalyse

`c_012d2ff7efe77f40` ist kein Bestandteil des Vergleichs, weil im verfügbaren Cache keine gültige Entity-/Relationsantwort für diesen Chunk liegt. Seine gespeicherten Fehlermeldungen in `../v2-multichunk-01/baseline.v2.rejects.jsonl` zeigen zwei andere Fehlerklassen:

1. **Entity-Antwort:** Die nach dem einmaligen JSON-Korrekturversuch gespeicherte Modellantwort endet nach 832 Zeichen innerhalb des Strings `"REAL...` und ist syntaktisch unvollständig. `json.loads` meldet `Unterminated string`.  
   **Mögliche Ursache [INFERENCE]:** Die Antwort wurde durch ein Ausgabe-/Generierungslimit oder einen abgebrochenen Modellstream abgeschnitten. Die Daten belegen die Trunkierung, nicht deren Transportursache. Whitespace-Zitatreparatur kann vor erfolgreichem JSON-Parsing und Schemagate nicht wirken.
2. **Relationsantwort:** Die gespeicherte Antwort ist das gültige JSON-Objekt `{}`, verletzt aber das V2-Schema, weil `assertions` erforderlich ist.  
   **Mögliche Ursache [INFERENCE]:** Nach der fehlgeschlagenen Entity-Phase hatte das Modell keine validierten Mention-Endpunkte oder lieferte trotz Prompt ein leeres Objekt. Die Artefakte belegen nicht, welche dieser Ursachen dominierte. Auch dies ist kein Alignment- oder Whitespace-Problem.

Das Schema bleibt strikt. Es erfolgen weder Reparatur von JSON noch Defaulting von `assertions` noch neue Modellaufrufe.

Hinweis zur Provenienz: Das Manifest in `../v2-pilot-03/` beschreibt den dort gespeicherten Ein-Chunk-Lauf. Der Cache enthält zusätzlich Antworten für `c_0001921855a8156b`, während der separate Drei-Chunk-Baseline-Satz unter `../v2-multichunk-01/` liegt. Das Pilotmanifest authentifiziert daher nicht den vollständigen Vergleichsdatensatz. Der Vergleich weist seine Inputs stattdessen über reproduzierte Cache-Keys, Datei-/Payload-Hashes und den exakten Abgleich gegen die separat gespeicherte Baseline nach.

## Technische versus wissenschaftlich-inhaltliche Validierung

### Technisch validiert

- gleicher gespeicherter Modelloutput in beiden Armen,
- fail-closed Offline-Replay ohne Provider-Fallback,
- strikt begrenzte Transformation: Whitespace-Layout eines eindeutigen, absatzsicheren Spans,
- unveränderte offizielle V2-Gates nach der Transformation,
- Baseline reproduziert gespeicherte Records und Rejects exakt,
- genaue Auditspur pro Chunk, Kandidat und Cache-Eintrag.

### Nicht wissenschaftlich validiert

- ob jede Relation semantisch korrekt ist,
- ob Dokument-zu-Mention-Relationen die beabsichtigte Ontologiesemantik treffen,
- Precision/Recall auf repräsentativen Daten,
- Generalisierung über die zwei Chunks hinaus,
- Auswirkungen auf nachgelagerte Retrieval- oder Antwortqualität.

Für diese Aussagen sind ein gelabeltes Gold Set und eine verblindete fachliche Bewertung nötig. Technisches Bestehen der Gates ist kein Ersatz dafür.

## Risiken und offene Fragen

1. **Sehr kleine Stichprobe:** Zwei Chunks und fünf Relationskandidaten erlauben keine belastbare Qualitäts- oder Generalisierungsaussage.
2. **Selektionsbias:** Verglichen werden nur Chunks mit vollständig verfügbaren, schema-validen Cache-Antworten.
3. **Duplizierter Evidence Span:** Drei unterschiedliche Assertions teilen denselben rekonstruierten Span. Das ist technisch zulässig; die fachliche Granularität muss geprüft werden.
4. **Regex-Kosten:** Die Komponente escaped Nicht-Whitespace vollständig und akzeptiert nur einen Treffer. Für extrem lange Zitate sollte Laufzeit separat gemessen werden, bevor produktive Nutzung erwogen wird.
5. **Absatzdefinition:** Zwei oder mehr Newline-Gruppen gelten als Absatzgrenze. Ob LaTeX-Layout weitere semantische Grenzen benötigt, ist offen.
6. **Cache-Format-Provenienz:** Die Pilotcache-Schlüssel verwenden Cache-Format-Version `1`; aktuelle produktive Ollama-Adapter können eine andere Namespace-Version verwenden. Der Runner verlangt die Version explizit und dokumentiert sie im Bericht.
7. **Manifestinkonsistenz:** Das Pilotmanifest gehört nicht zum vollständigen Drei-Chunk-Artefaktsatz; künftige Piloten sollten immutable, gemeinsam versionierte Input-/Output-/Cache-Manifeste schreiben.

## Nächste Schritte

1. Die drei rekonstruierten Assertions verblindet fachlich bewerten; Entscheidung und Begründung pro Assertion festhalten.
2. Ein kleines, repräsentatives Gold Set mit exakten, whitespace-normalisierten, mehrdeutigen, absatzüberschreitenden und semantisch falschen Zitaten erstellen.
3. Precision und Recovery Rate getrennt berichten: keine Akzeptanz eines falschen Spans versus Rückgewinnung technisch gültiger Spans.
4. Erst nach positiver fachlicher Bewertung eine explizit opt-in Integration entwerfen; V2-Vertrag und Standardverhalten unverändert lassen.
5. Einen neuen, konsistent manifestierten Offline-Pilotlauf erzeugen, sobald dies ohne neue Modellaufrufe aus vorhandenen Antworten möglich ist.

## Versionierte Provenienz

Ein versionsübergreifendes Provenienzmanifest bündelt alle für den vollständigen Zwei-Chunk-Vergleich verwendeten Artefakte (Chunks, Prompts, Cache-Einträge, Baseline, Rekonstruktions-Audit, experimentelle Records, Implementierung):

- Manifest: `docs/experiments/quote-repair-v1-review/provenance/manifest.v1.json`
- Geteilte Originalquelle: `data/raw/latex/2007.00808.tar.gz` (SHA-256 `46288f02699612de50ae6f7530080bced7e1edc5e3ffb2225b80e5d6742b58f0`).
- Caption-Ursprung in der Quelle: `Tables/joint.tex` im Tarball (Caption-Bytes im bereinigten Volltext an Position `[37099, 37321)`; Evidenzspan `[37108, 37264)`).
- Reproduzierte Cache-Schlüssel sind byte-genau die in den gespeicherten Cache-Dateinamen verwendeten Hashes.

Das Manifest deklariert explizit fehlende oder außerhalb dieses Pakets liegende Artefakte (Gold-Annotation, entpackter Quellbaum, gemeinsames Multichunk-Manifest). Es vergibt keine wissenschaftlichen Review-Labels.

## Wissenschaftlicher Quellenkontext

Ein gemeinsamer Bericht für die drei Review-Einträge dokumentiert:

- pro Relation (`EVALUATES_ON`, `REPORTS_METRIC`) die exakten Anforderungen aus `ontology.yaml:169-218` und `docs/specs/extraction-contract.md:319-353`;
- welche Evidenz pro Reaktion **vorhanden** ist (Verbatim-Caption aus `Tables/joint.tex`);
- welche Evidenz **fehlt** (kein Systemname, keine Ergebniswerte im Chunk, keine assertion-spezifische Trennung; der angrenzende Tabellenkörper `Tables/prod.tex` enthält andere Spalten; die numerischen Top-20/100-Werte für NQ/TQA in `Tables/openqa.tex` enden vor dem Chunkstart; der einzige System-bindende Satz in `Evaluation.tex:19` liegt außerhalb des Chunk-Spans);
- die ausdrückliche Trennung zwischen struktureller Validierung (Ontologiekonformität, Endpoint-Containment, Evidenzintegrität) und wissenschaftlicher Substanzierung.

Bericht: `docs/experiments/quote-repair-v1-review/context-report.md`.

### Automatisierte Tests

Die folgenden fokussierten Tests sichern Manifestintegrität, Assertion-Zuordnung und Quellenbeleg-Konsistenz:

```bash
python -m pytest -q \
  tests/test_experimental_quote_repair.py \
  tests/test_quote_repair_comparison.py \
  tests/test_quote_repair_provenance.py
```

`tests/test_quote_repair_provenance.py` prüft insbesondere: (a) die im Manifest deklarierten SHA-256-Werte der versionierten In-Repo-Artefakte, (b) die Übereinstimmung der drei Assertion-IDs zwischen Manifest, Review-Dateien und Datensatz-Inhalt, (c) die Endpunkt-/Mention-Charakteristika jeder Assertion, (d) den gemeinsamen Caption-Span `[34,190)`, (e) den Verbatim-Caption-String in `Tables/joint.tex` und im Chunk-Text, (f) dass die reproduzierten Cache-Schlüssel den gespeicherten Cache-Dateinamen entsprechen, und (g) dass das Manifest keine Review-Labels vergibt.

### Verbleibende Grenzen der wissenschaftlichen Bewertung

- Die drei Assertions teilen denselben Caption-Span; der lokale Chunk enthält weder Systemnamen noch Ergebniswerte und keine assertion-spezifische Trennung. Eine fachliche Entscheidung `supported` / `unsupported` / `unclear` ist ohne externe Evidenz oder ein Gold Set nicht möglich.
- Die numerischen Ergebnisse für `Answer Coverage at Top-20/100` auf NQ und TQA liegen in `Tables/openqa.tex`, das im bereinigten Volltext bei Offset `37059` endet, also **vor** dem Chunkstart `37074`. Eine Zuordnung von Caption zu *ihrem* Tabellenkörper gelingt nur über den ausgeschnittenen Quellkontext.
- Der einzige Satz, der Caption-Datensätze an ein System bindet (`Evaluation.tex:19`), liegt außerhalb des Chunk-Spans und ist damit kein gültiges `evidence_quote` im V2-Vertrag.
- Der Vergleich umfasst weiterhin nur zwei Chunks und fünf Relationskandidaten. Precision/Recall, Generalisierung und Retrieval-/Antwortqualität bleiben unbewiesen.
- Der `confidence: "stated"`-Wert in den gespeicherten Records ist eine Modellaussage, kein menschliches Review-Label.
