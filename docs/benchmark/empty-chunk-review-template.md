# Wissenschaftliches Review vollständig leerer Extraktions-Chunks

## Zweck und Auswahl

Diese Vorlage dient der **menschlichen** Bewertung von Chunks mit **null akzeptierten Mentions und null akzeptierten Assertions**. Das Offline-Replay erzeugt dafür `empty_chunk_review.json` ausschließlich im externen Ausgabeordner. Alle Urteilsfelder bleiben zunächst `null`. Weder die Auswahl noch eine technisch vollständige Extraktion ist ein Gold-Label. Review-Ergebnisse werden nicht automatisch in ein Gold-Set übernommen.

Drei Beobachtungen strikt unterscheiden:

1. **Quellabschnitt ohne relevante Inhalte:** Keine im verwendeten Ontologieumfang belegbare Entität oder Relation. Das ist eine wissenschaftliche Review-Entscheidung, nicht die Bedeutung einer leeren Modellliste.
2. **Mögliche Auslassung:** Der Quelltext enthält plausible Entitäten oder Relationen, aber die Modellantwort oder die akzeptierte Extraktion nicht. Entitäten und Relationen getrennt prüfen.
3. **Technisch leer/unvollständig:** Rejects, abgeschnittene Antwort, ungültiges JSON, Transportfehler oder Aufruflimit können null akzeptierte Records verursachen. Kein Nachweis einer tatsächlich leeren Quelle. Fehlende Rohantworten dürfen nicht aus Rejects oder akzeptierten Records rekonstruiert werden.

## Identität und Provenienz

- Reviewer / Datum / unabhängiger Zweitreviewer:
- Recording-ID und SHA-256 von `recording.json`:
- Chunk-ID / Dokument-ID / Abschnittspfad:
- SHA-256 des unveränderten Chunktexts:
- Ontologieversion und SHA-256; Entity-/Relationsprompt und Schema-Versionen/Hashes:
- SHA-256 der kanonischen Extractions- und Reject-Dateien:
- Referenzen auf Request-/Response-Dateien und zugehörige HTTP-Versuche:
- Angefordertes Modell / tatsächlich gemeldete Modelle (fehlend = unbekannt):
- Kanonischer Status / Reject-Gründe / Retries / Abbruchgrund:

## Review-Ablauf

1. **Quelltext zunächst unabhängig lesen.** Originale Zeilenumbrüche, Tabellen, Captions, LaTeX und Chunkgrenzen beachten. Nicht allein anhand des Modelloutputs urteilen.
2. Ontologieumfang prüfen: Welche Entity-Typen und Relationsrichtungen sind erlaubt? Strukturelle Dokumententitäten nicht als ausgelassene extrahierte Mentions zählen.
3. Mögliche Mentions mit wörtlichem Quellbeleg und Zeichenintervall `[start, end)` erfassen. Indizes beziehen sich auf den unveränderten Python-/JSON-Chunkstring, nicht auf UTF-8-Bytes. Ontologietyp, Mehrfachvorkommen und unsichere Zuordnung notieren.
4. Relationen getrennt prüfen: Subjekt/Objekt, zulässiger Relationstyp, explizite Aussage, wörtlicher Beleg, Endpoint-Containment, Dokument- versus Mention-Referenz. Reine Ko-Erwähnung ist kein Relationsbeweis. Fehlende Entitäten können Relationsauslassungen erklären.
5. Erst danach vollständige Rohantworten, Korrekturantworten, Rejects und Fehler vergleichen. Unterscheiden: vom Modell nicht vorgeschlagen, vorgeschlagen aber technisch verworfen, wegen Abbruch nicht beobachtbar. Leere Arrays nur in echten gespeicherten Modellantworten feststellen.
6. Benötigten Nachbarkontext als separate Quelle mit Identität/Hash dokumentieren; keinen Kontext in den ursprünglichen Chunk hineininterpretieren. Fehlender Kontext führt zu Unsicherheit, nicht zu einem negativen Gold-Label.
7. Optionalen Whitespace-Vergleich separat lesen: technische Akzeptanz nach Reparatur beweist keine semantische Richtigkeit. Er darf das Urteil über die ursprüngliche kanonische V2-Ausgabe nicht ersetzen.

## Unausgefülltes Review-Protokoll

Genau eine Gesamtbewertung nach menschlicher Prüfung auswählen; bis dahin alle Optionen offen lassen:

- [ ] Im Ontologieumfang tatsächlich leer (`truly_empty_for_ontology`).
- [ ] Möglicherweise übersehene Entitäten (`possible_missed_entities`).
- [ ] Möglicherweise übersehene Relationen (`possible_missed_relations`).
- [ ] Beides (`possible_missed_entities_and_relations`).
- [ ] Nicht entscheidbar: Kontext fehlt oder technischer Fehler (`insufficient_context_or_technical_failure`).

### Mögliche Entitäten

| Wörtlicher Span / Intervall | Ontologietyp | Beobachtung in Rohantwort/Rejects | Begründung / Unsicherheit |
|---|---|---|---|
| | | | |

### Mögliche Relationen

| Subjekt / Objekt | Relationstyp | Wörtliche Evidenz / Intervall | Beobachtung in Rohantwort/Rejects | Begründung / Unsicherheit |
|---|---|---|---|---|
| | | | | |

### Wissenschaftliche Begründung

- Warum ist der Abschnitt tatsächlich leer oder warum besteht ein Auslassungsverdacht?
- Welche Gegenbelege, Grenzfälle, Chunkgrenzen oder Ontologieausschlüsse wurden geprüft?
- Wurde nur die verfügbare Quelle beurteilt? Welche Kontextinformationen fehlen?
- Abweichendes Zweitreview / Adjudikation:

**Keine automatische Gold-Vergabe.** Auch ein abgeschlossenes Review ist zunächst ein nachvollziehbares Review-Protokoll; eine Gold-Set-Übernahme benötigt einen gesonderten, kontrollierten Annotationsprozess.
