# Evaluationsbericht

## Ergebnis

Die Läufe sind technisch vergleichbar, aber nicht wissenschaftlich als Qualitätsvergleich auswertbar. Historisch fehlen Rohantworten; der neue Lauf ist exakt replaybar. Die zwei gemeinsamen Chunks zeigen deskriptive Unterschiede bei Mentions, Evidence-Spans, Relationstypen und technischer Vollständigkeit. Daraus folgt kein statistisch signifikanter Modell- oder Qualitätsunterschied.

## Kennzahlen

| Lauf | Chunks | technisch vollständig | vollständig leer | akzeptierte Mentions | akzeptierte Relationen | Rejects |
|---|---:|---:|---:|---:|---:|---:|
| historisch, 10 Chunks | 10 | 5/10 | 5/10 | 13 | 9 | 7 |
| neuer Recording-Lauf, 2 gemeinsame Chunks | 2 | 2/2 | 0/2 | 8 | 7 | 0 |

Historische Reject-Ursachen: `entity/INVALID_OCCURRENCE_INDEX` 4, `relation/ENDPOINT_NOT_CONTAINED` 2, `relation/ZERO_OCCURRENCES` 1. Der neue Lauf hat keine Rejects. Technische Vollständigkeit bedeutet validierte Verarbeitung, nicht semantische Richtigkeit.

## Nicht berechnete Qualitätsmetriken

Relation-Precision bleibt offen, bis jede entsprechende Vorhersage durch zwei unabhängige Reviewer geprüft und Konflikte aufgelöst sind. Entity- und Relation-Recall bleiben offen, weil keine der betrachteten Chunkgruppen eine vollständig geprüfte Referenzannotation besitzt. Die fünf leeren Chunks sind Review-Kandidaten, keine Negativ-Goldmenge.

## Grenzen

- Historische Rohantworten fehlen; fehlende Kandidaten können nicht rekonstruiert werden.
- Response-Modellnamen sind nicht unabhängig verifiziert.
- Zwei Modellstichproben sind keine statistische Stichprobe.
- Chunklokaler Beleg unterscheidet sich von dokumentweitem Kontext.
- LaTeX-Makros (`\ours`, `\CLEAR`) und Tabellen erschweren Typisierung.
- `RMSProp` passt möglicherweise nicht in die definierte `method`-Population.
- DPR-Laufzeitvergleiche belegen nicht automatisch `USES_METHOD`.
- `EVALUATED_ON` ist derived und wird nicht als direkte Modellrelation gewertet.

## Review-Artefakte

- Historischer Vergleich und Relationsreview: `historical-comparison-20260930.md`
- Fünf vollständige Originaltexte und leere Reviewer-Felder: `empty-chunks-review-20260930.md`

## Kontrollierter nächster Versuch

Neue vollständig aufgezeichnete Aufnahme mit denselben zwei Vergleichschunks plus fünf leeren Chunks; vorab Hashes von Dataset, Ontologie, Prompts und Schemas einfrieren; angefordertes und gemeldetes Modell getrennt speichern; zwei Reviewer blind und span-genau annotieren; anschließend DPR-, RMSProp-, Makro- und Tabellenfälle adjudizieren. Erst dann Precision berechnen; Recall nur für vollständig referenzannotierte Chunks.
