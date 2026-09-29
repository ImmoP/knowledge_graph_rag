# Wissenschaftliches Review-Paket: Quote Repair v1

## Zweck und Abgrenzung

Dieses Paket legt die drei durch `experimental-whitespace-v1` technisch akzeptierten Assertions einem menschlichen wissenschaftlichen Review vor. Es ist **keine Gold-Set-Annotation**, keine ann-B-Bewertung und keine wissenschaftliche Qualitätsentscheidung.

Die maschinelle Aussage ist begrenzt: Der Offline-Vergleich rekonstruierte bei drei Modellzitaten ausschließlich originales Whitespace-Layout. Danach passierten die drei Kandidaten die unveränderten V2-Gates für exaktes Alignment, Endpoint-Containment, Ontologie und Evidenzintegrität. Daraus folgt nicht, dass eine Assertion semantisch unterstützt, vollständig oder wissenschaftlich nützlich ist.

## Review-Einträge

| Assertion | Relation | Objekt | Protokoll |
|---|---|---|---|
| `a_44999b3871c3f58f` | `REPORTS_METRIC` | `Answer Coverage at Top-20/100` | [review-a_44999b3871c3f58f.md](review-a_44999b3871c3f58f.md) |
| `a_0a775699fcc8a5ce` | `EVALUATES_ON` | `Natural Questions` | [review-a_0a775699fcc8a5ce.md](review-a_0a775699fcc8a5ce.md) |
| `a_2814c002a3ebd912` | `EVALUATES_ON` | `Trivial QA` | [review-a_2814c002a3ebd912.md](review-a_2814c002a3ebd912.md) |

Jedes Protokoll enthält leere, mutually exclusive Review-Felder `supported`, `unsupported`, `unclear` sowie ein Kommentarfeld. Das gespeicherte Modellfeld `confidence: stated` ist kein Review-Label.

## Gemeinsame Quellenbasis

- Dokument: `W3038572442`, *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (Manifesttitel), Chunk `c_001452fd142ce075`, Abschnitt `Experimental Methodologies`, Chunk-Index 23.
- Originaler Chunk: `../v2-multichunk-01/three_chunks.jsonl`, Zeile 1.
- Ursprünglicher Relationspayload: `../v2-pilot-03/cache/llm/5d692f6660892c61240fcbd6f208ce8af706a450b65c0140281de208df4c65cb.json`, Zeile 1.
- Dokumentierte Rekonstruktion: `../v2-pilot-04/repair.audit.json`, Zeilen 4–25.
- Akzeptierte experimentelle Records: `../v2-pilot-04/experimental.jsonl`, Zeilen 4–6.
- Reproduzierter Auditbericht: `/tmp/quote-repair-v1-review-report.json` (lokal, nicht versioniert).

Zeichenpositionen sind Python-/JSON-String-Indizes im Chunktext, als halboffenes Intervall `[char_start, char_end)`. Für alle drei Einträge gilt: `chunk.text[34:190] == evidence_quote`.

## Kontextprüfung der dreifach verwendeten Tabellenbeschriftung

Alle drei Assertions verwenden denselben Span `[34,190)` derselben LaTeX-Beschriftung. Die Beschriftung nennt:

- die Metrik `Answer Coverage at Top-20/100`,
- die Datensätze `Natural Questions (NQ)` und `Trivial QA (TQA)`,
- den Ausdruck `Retrieval results ... on ...`,
- den Cite-Key `karpukhin2020dense`.

Der unmittelbare Quellkontext zeigt dagegen keine zum Caption-Text gehörenden Tabellenwerte: Nach Caption und Label folgt eine zweite Tabelle mit Spalten `Corpus Size`, `Dim`, `Search`, `Gain` und anschließend eine andere Caption zu relativen Gewinnen einer kommerziellen Suchmaschine. Im verfügbaren Chunk fehlen für die drei Assertions insbesondere:

- Ergebniswerte für `Answer Coverage at Top-20/100`, NQ oder TQA,
- ein explizit benanntes evaluiertes System bzw. Verfahren,
- erläuternder Fließtext zur Versuchsanordnung,
- zusätzlicher assertion-spezifischer Beleg, der die drei verschiedenen Relationen voneinander trennt.

**Kontextbefund:** Die identische Caption enthält die drei Objekt-Mentions und relationale Sprache, liefert aber nicht für jede Assertion eigenen oder vollständigen lokalen Kontext. Ob der vorhandene Wortlaut trotzdem eine konkrete Assertion wissenschaftlich trägt, entscheidet ausschließlich der menschliche Reviewer in den Einzelprotokollen.

## Reproduzierbarkeit

Der Offline-Vergleich wurde mit dem dokumentierten Befehl erfolgreich erneut ausgeführt; er erzeugte ohne stdout/stderr den Bericht `/tmp/quote-repair-v1-review-report.json`. Beobachtetes Ergebnis:

- Baseline: 2 akzeptierte Assertions, 3 Rejects, 1 vollständiger Chunk.
- Experiment: 5 akzeptierte Assertions, 0 Rejects, 2 vollständige Chunks.
- Delta: +3 akzeptierte Assertions, −3 Rejects, +1 vollständiger Chunk.
- Rekonstruktionen: 3 × `repaired`, 2 × `exact`.
- Gespeicherte Baseline: exakte Record-/Reject-Übereinstimmung.
- Rohantworten beider Arme: identische vier Cache-Einträge und Payload-Hashes.

Befehl:

```bash
PYTHON=../../projects/knowledge_graph_rag/.venv/bin/python

"$PYTHON" scripts/experiments/compare_quote_repair_v1.py \
  --chunks ../v2-multichunk-01/three_chunks.jsonl \
  --cache-dir ../v2-pilot-03/cache \
  --prompts-dir ../v2-pilot-03/prompts \
  --baseline-jsonl ../v2-multichunk-01/baseline.v2.jsonl \
  --baseline-rejects ../v2-multichunk-01/baseline.v2.rejects.jsonl \
  --output /tmp/quote-repair-v1-review-report.json
```

### Reproduzierbarkeitsstatus

**Lokal vollständig reproduzierbar mit dem vorhandenen externen Artefaktsatz; aus dem Git-Repository allein nicht vollständig reproduzierbar.**

Der Runner ist fail-closed und enthält keinen Modell- oder Netzwerk-Fallback. Die vier Cache-Einträge, Chunks, Baseline, Rejects und Originalprompts liegen jedoch in benachbarten, nicht versionierten Verzeichnissen. Ihre Pfade sind Teil des lokalen Aufbaus und nicht durch einen gemeinsamen unveränderlichen Datensatz-Manifestdatensatz gebunden.

Wichtige Input-Hashes des beobachteten Replay-Laufs:

| Input | SHA-256 |
|---|---|
| `three_chunks.jsonl` | `b15a5571acd61ed998e19bb2463e87315db4c4321c400ca4317e7a7b890bbf55` |
| V2-Entity-Prompt | `1cbbc0576f5768fb661c6608709f3cc6903cac82a69b6fac81305b2bb9f5e381` |
| V2-Relationsprompt | `cb41416971e35c07a6fbf13ed09d9147e7713662e90787e30996d5aed0c10afd` |
| `ontology.yaml` | `f9a3e203d4c640b91966e077b02f09a547d49b51595cf33e040a3e278cd2325e` |
| relevanter Relationscache | `7b1d391e685940b21ded91a6cf0fb0c448937f26b629245e645562ce4d4b0d97` |

Historische Referenzartefakte für die drei Review-Einträge; sie sind keine Eingaben des Replay-Runners:

| Referenzartefakt | SHA-256 |
|---|---|
| Pilot-04-Reparaturaudit | `7d6ca3cabcf1b061edbb20219e1e30844dddd3d7d77d3c6580cebb4abdff6fa1` |
| Pilot-04-Experimentalrecords | `34fda936f0a4e894e10210205a53855b05ac3740fc43dc969e5b2ea472734b77` |

Beobachtete Laufzeitumgebung: Python 3.12.3, `jsonschema` 4.25.1, PyYAML 6.0.3; die letzten beiden Versionen sind in `requirements.txt` fixiert. Pytest ist nicht in `requirements.txt` fixiert; verwendet wurde 9.1.1.

## Verbleibende Unsicherheiten und Review-Fragen

1. Trägt die Caption allein jede der beiden `EVALUATES_ON`-Relationen zwischen dem Dokument und NQ bzw. TQA, obwohl kein evaluiertes System im lokalen Span genannt ist?
2. Trägt die Nennung `Answer Coverage at Top-20/100` allein `REPORTS_METRIC`, obwohl im verfügbaren Kontext kein Wert dieser Metrik erscheint?
3. Ist ein einziger, identischer Caption-Span für drei semantisch unterschiedliche Assertions hinreichend granular, oder ist zusätzlicher Quellkontext nötig?
4. Gehört der im Chunk folgende Tabellenkörper zur ersten Caption? Die sichtbaren Spalten und die zweite Caption sprechen gegen eine eindeutige lokale Zuordnung; die Originalstruktur vor Chunkgrenze fehlt.
5. Das Pilotmanifest unter `../v2-pilot-03/` beschreibt einen Ein-Chunk-Lauf und authentifiziert nicht den vollständigen Zwei-Chunk-A/B-Vergleich. Der Runner kompensiert dies technisch durch Cache-Key-/Hashnachweise und exakten Baseline-Abgleich, aber nicht durch ein gemeinsames unveränderliches Provenienzmanifest.
6. Der lokale Artefaktsatz ist selektiv: Nur zwei Chunks besitzen je eine gültige Entity- und Relationsantwort. Aussagen zu Generalisierung, Precision/Recall oder Retrievalqualität bleiben unbelegt.
