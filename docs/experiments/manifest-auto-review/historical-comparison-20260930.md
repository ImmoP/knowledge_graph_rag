# Offline Vergleichsanalyse Manifest-Auto-Extraktionen

**Status:** unabhängige Review-Vorlage; keine Gold-Labels, keine neue Modellanfrage.

## Quellen und Grenzen

Verglichen werden ausschließlich `c_ca129edd34052284` und `c_8a84177f479aea1f`.

| Lauf | Quelle | Herkunft/Integrität | Ergebnis |
|---|---|---|---|
| historisch | `~/migration-incoming/manifest-auto-autopilot-live/20260930T083857Z-c69654/` | Datensatz `38182c6c...f84985`, Ontologie `f9a3e203...cd2325e` laut `run_summary.json`; historische Rohantworten fehlen (`raw_responses/` leer) | 10 Chunks, davon 5 technisch complete, 5 incomplete; 13 Mentions, 9 Assertions, 7 Rejects |
| aufgezeichnet | `/home/immo/manifest-two-chunk.WB08yOG3/recording/` | `recording.json`, Input-Snapshot, Prompts, Schemas und Ontologie eingefroren; Recording-Manifest-SHA256 `9d6a91f7f879ac550abf45c07f41f436df4b46bdc38b7bae301c274b02c9bfb8`; Replay matched exakt | 2/2 complete, 8 Mentions, 7 Assertions, 0 Rejects |

Die gemeinsame Datensatzversion ist die eingefrorene Benchmark-Version mit SHA-256 `38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985`; die Ontologie ist V2-Schema 2-kompatibel im Output, aber die fachlichen Definitionen stammen aus der eingefrorenen `ontology.yaml`, Version 1 / Schema 1. Aufgezeichnet sind Prompt- und Schema-Dateien sowie Implementierungs-/Dependency-Hashes. Temperatur und angefordertes Modell sind aus der Aufzeichnung zu übernehmen; ein angezeigter Response-Modellname ist keine unabhängige Modellverifikation. Für den historischen Lauf sind Prompt-, Schema-, Temperatur- und Modellinformationen nur insoweit verfügbar, wie `run_summary.json` sie nennt; die historische Rohantwort fehlt. Daher keine Rekonstruktion.

## Technischer Vergleich

| Chunk | Historisch | Aufgezeichnet | Fachlicher Unterschied |
|---|---|---|---|
| `c_ca129edd34052284` | 2 Mentions: `Deep LSTM Reader` `[110,126)`, `Attentive Reader / Impatient Reader` `[1210,1241)`; 1 akzeptierte `USES_METHOD` zu Deep LSTM `[102,158)`; 1 Relation `ZERO_OCCURRENCES` verworfen (Zitat mit Leerzeichen statt Newline) | 3 Mentions: zusätzlich `RmsProp` `[975,982)`; kanonische Form `Attentive and Impatient Readers`; 3 `USES_METHOD`, zu Deep LSTM `[102,158)`, Attentive/Impatient `[1201,1279)`, RMSProp `[939,982)` | Neue Aufzeichnung akzeptiert zwei fachlich plausible zusätzliche Aussagen; historische Antwort/Reject-Kandidat ist nicht vollständig beobachtbar. Die Quote-Reparaturdiagnose belegt nur technische Rekonstruierbarkeit, nicht fachliche Richtigkeit. |
| `c_8a84177f479aea1f` | 3 Mentions: `NQ` `[490,492)`, `DPR` `[540,543)`, `TriviaQA` `[493,499)`; 2 `EVALUATES_ON` `[436,519)`; `USES_METHOD` zu DPR verworfen, weil Mention `[540,543)` nicht im Quote `[258,312)` enthalten war; technisch incomplete | 5 Mentions: zusätzlich `DPR reader` `[302,312)` und `reranking` `[4,13)`; 2 `EVALUATES_ON` `[342,520)`; zwei `USES_METHOD` zu DPR `[182,342)` und `[521,585)`; complete | Gegenüber den gleichen Aussagen hat die Aufzeichnung eine zweite DPR-Mention mit passenderem Span, akzeptiert aber weiterhin DPR als Laufzeitvergleich. Evidenzgrenzen und Relationstyp bleiben fachlich zu prüfen. |

IDs werden nicht als fachliches Matching verwendet; verglichen werden Surface Forms, Spans, Quotes und Aussageinhalt.

## Relation-Review (Vorschläge, nicht bestätigt)

| Chunk / Relation | exakter Beleg | Ontologieprüfung | Vorschlag / Begründung |
|---|---|---|---|
| ca129 / `USES_METHOD` → Deep LSTM Reader | `[102,158)`: “For the Deep LSTM Reader, we consider hidden layer sizes” | Dokument → Methode; explizite Untersuchung/Anwendung | **durch Chunk belegbar**: klare Anwendung im Hyperparameter-Tuning. |
| ca129 / `USES_METHOD` → Attentive and Impatient Readers | `[1201,1279)`: “with the\nAttentive and Impatient Readers performing best across both datasets.” | Dokument → Methode; Modellvergleich und Ergebnis | **durch Chunk belegbar**, sofern die Ontologie “uses” auch evaluierte Modelle umfasst; fachlich mögliche Ontologiegrenze, weil “performing best” nicht zwingend “applies/builds on” bedeutet. |
| ca129 / `USES_METHOD` → RMSProp | `[939,982)`: “For all models we used asynchronous RmsProp” | Dokument → Methode; explizite Verwendung | **möglicher Ontologiefehler/fachlich unklar**: RMSProp ist ein Optimierer, während `method` Retrieval/Generation/Repräsentation definiert. Chunk belegt Nutzung, aber nicht die Zulässigkeit des Entity-Typs. |
| 8a / `USES_METHOD` → DPR (Training comparison) | `[182,342)`: “...12 to 16 hours with 8 GPUs for the DPR reader...” | Dokument → Methode; DPR wird als Vergleich erwähnt | **fachlich unklar**: der Satz belegt Laufzeitvergleich, nicht eindeutig, dass dieses Dokument DPR anwendet. Ontologie verlangt applies/builds on; mindestens nur mit weiterem Kontext belegbar. |
| 8a / `USES_METHOD` → DPR (inference comparison) | `[521,585)`: “In comparison, the DPR reader takes about 14/40 min...” | Gleiches Problem, reine Vergleichsreferenz | **nicht ausreichend belegt** für `USES_METHOD`; “in comparison” ist gerade ein Gegenbeleg gegen Anwendung durch das Dokument. |
| 8a / `EVALUATES_ON` → Natural Questions | historisch `[436,519)`, neu `[342,520)`; “...answers on the NQ/Trivia test set...” | Dokument → Dataset; Resultat-/Testset-Aussage | **durch Chunk belegbar**, wenn generative reader-Ergebnisse dem Dokument zugerechnet werden. |
| 8a / `EVALUATES_ON` → TriviaQA | gleicher Span | gleicher Befund | **durch Chunk belegbar**; `Trivia` ist Surface Form, kanonische `TriviaQA`-Zuordnung bleibt Ontologie-/Corpus-Normalisierung und sollte unabhängig geprüft werden. |
| 8a / fehlende `USES_METHOD` → generative reader / `\ours` | im Chunk genannt, aber keine extrahierte Methode | Chunk belegt nur generative reader als generische Beschreibung; `\ours` ist Makro | **Review-Kandidat**, nicht automatisch Label: exakte Entität und Ontologiezuordnung benötigen Dokumentkontext. |

Für jede Relation sind Reviewer-Felder separat auszufüllen. `EVALUATES_ON` ist nicht aus bloßer Ko-Erwähnung abgeleitet; der Testset-Satz ist der konkrete Beleg. `EVALUATED_ON` wird hier nicht als extrahierte Relation gezählt, da es laut Ontologie derived ist.

## Rejects

Historisch: `relation/ENDPOINT_NOT_CONTAINED` ×2, `relation/ZERO_OCCURRENCES` ×1, `entity/INVALID_OCCURRENCE_INDEX` ×4. Aufgezeichnet: keine Rejects. Die historische `ZERO_OCCURRENCES`-Quote ist technisch durch Newline-Normalisierung rekonstruierbar, aber die fehlende Rohantwort verhindert sichere Zuordnung des ursprünglichen Kandidaten. Keine Reject-Ursache wird als semantische Fehlerquote interpretiert.

## Messdefinitionen

**Technisch:** `complete_chunks / total_chunks`; `empty_chunks / total_chunks` anhand `extraction_complete=true` und null akzeptierter Mentions/Assertions; akzeptierte Mentions/Assertions; Reject-Zähler nach `phase` und `reason`; Attempt-/Failure-Zähler. Für den historischen Lauf: 5/10 complete, 5/10 vollständig leer, 13 Mentions, 9 Assertions, 7 Rejects. Für den aufgezeichneten Lauf: 2/2 complete, 0/2 vollständig leer, 8 Mentions, 7 Assertions, 0 Rejects.

**Wissenschaftlich:** Relation-Precision erst nach unabhängiger Prüfung aller entsprechenden Predictions; Entity-/Relation-Recall nur für Chunks mit vollständig geprüfter Referenzannotation. Die vorliegenden fünf leeren Chunks besitzen keine solche Referenz. Deshalb werden hier keine Precision-/Recall-Werte und kein Qualitätsranking berichtet.

### Reviewer-Protokoll

| Kandidat | Reviewer 1 Urteil/Begründung | Reviewer 2 Urteil/Begründung | Konfliktauflösung |
|---|---|---|---|
| jede Mention/Relation | *(leer)* | *(leer)* | *(leer)* |

## Nächste kontrollierte Versuchsreihe

1. Dieselben zwei Chunks plus die fünf leeren Chunks in einer neuen, vollständig aufgezeichneten Offline-replay-fähigen Aufnahme; keine Rohdaten in Git.
2. Vorab beide Prompt-/Schema-/Ontology-Hashes, Temperatur, angefordertes Modell und Response-Modellidentität als getrennte Felder einfrieren; unbekannt explizit `unknown`.
3. Mindestens zwei unabhängige menschliche Reviewer bewerten zuerst Originaltext und Spans, danach blind die Outputs; keine automatische Gold-Übernahme.
4. DPR-Vergleichssätze, RMSProp, `\ours`, `\CLEAR`, Switch Transformer und Mesh TensorFlow als gezielte Ontologie-/Span-Fälle markieren.
5. Konflikte adjudizieren und erst danach Relation-Precision bzw. Recall für vollständig annotierte Chunks berechnen. Zwei Modellstichproben bleiben deskriptiv und sind nicht statistisch signifikant.
