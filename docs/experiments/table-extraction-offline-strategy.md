# V2-Extraktion: Offline-Strategie für Leerantworten und überlappende Mentions

**Status:** technischer Entwurf und reproduzierbarer Offline-Test. Keine Provider-Aufrufe, keine Änderung an Ontologie, Prompts, kanonischer Pipeline oder Gold-Daten.

## 1. Problem und Entscheidung

Die V2-Pipeline behandelt `{"mentions": []}` als schema-valide, erfolgreiche Entity-Phase. Das ist technisch korrekt, unterscheidet aber nicht zwischen einer fachlich wirklich leeren Passage und einer erfolgreichen Nullantwort mit möglicher Auslassung. Gleichzeitig akzeptiert die Pipeline überlappende Mentions unabhängig voneinander. Bei gleicher Kombination aus Chunk, Typ und Span erzeugt V2 dieselbe `mention_id`, auch wenn `canonical_form` variiert.

**Entscheidung:** keine produktive Heuristik in den Validierungs- oder Extraktionspfad einbauen. Stattdessen ein isoliertes, deterministisches Diagnosemodul an der aufgezeichneten Ergebnis-Seam:

```text
aufgezeichnete Antworten / V2-Records
                 │
                 ▼
    eingefrorene, kleine Fixture
                 │
                 ▼
 diagnose_recordings.py  ──► technische Beobachtungen
                 │                - successful empty
                 │                - Span-Überlappung
                 │                - canonical conflict
                 ▼
       offene fachliche Fragen
       (kein automatisches Urteil)
```

Diese Seam bleibt tief: `diagnose(fixture) -> report` verbirgt Sortierung, Paarbildung, Konfliktprojektion und Klassifikation. Tests und CLI verwenden dieselbe Schnittstelle. Die kanonische Pipeline bleibt unverändert.

## 2. Untersuchte Pfade und Aufzeichnungen

### V2-Pfad

1. `src/providers/adapters/manifest_openai.py`: Assistant-Inhalt wird geparst und gegen das V2-JSON-Schema validiert. Eine leere `mentions`-Liste ist erlaubt.
2. `src/extract/pipeline_v2.py::_run_entity_phase_v2`: `mentions_raw` wird an `align_mentions` übergeben. Bei leerer Liste entstehen keine Rejects und `status="validated"`, `mention_count=0`.
3. `src/extract/align.py::align_mentions`: jede Mention wird unabhängig exakt im Text ausgerichtet. Unterschiedliche Kandidaten dürfen überlappen; die Occurrence-Suche innerhalb eines einzelnen Literals bleibt links-nach-rechts und nicht überlappend.
4. `src/extract/validation.py`: Evidence- und Ontology-Gates prüfen Bytegleichheit, SHA und Typmitgliedschaft; sie entscheiden keine Entitätsgranularität.
5. `src/extract/ids.py::mention_id_v2`: Identität verwendet Chunk, Typ, Span, Text-SHA und Schema-Version; `surface_form`, `canonical_form` und Confidence sind nicht Teil der ID.
6. `src/extract/pipeline_v2.py`: auch nach leerer Entity-Phase läuft die Relation-Phase mit `(no same-chunk mentions available)` weiter. Leere Entity- und Relation-Arrays ergeben einen vollständigen Chunk.
7. `src/benchmark/manifest_replay.py`: `ReplayClient` ist die vorhandene Offline-Seam für vollständige aufgezeichnete Runs. Die neue Diagnostik sitzt daneben und verändert Replay oder kanonische Bytes nicht.

### Fünf Tabellen-Aufzeichnungen

Fixture: `tests/fixtures/table_extraction_v2_recordings.json`. Quelle: `omp-repeat-probe/runs_target/run_01` bis `run_05`; alle fünf haben denselben Request-SHA `df227ff9…257e`, HTTP 200, `finish_reason=stop`, einen Versuch und null Rejects.

| Record | Mentions | charakteristische technische Beobachtung |
|---|---:|---|
| target-01 | 6 | `[210,232)` enthält ELMo `[210,214)` und improved Bi-DAF `[217,232)` |
| target-02 | 5 | stabiler Basissatz, keine zusätzliche Teilmention |
| target-03 | 6 | Bi-DAF `[226,232)` → `canonical_form="Bi-DAF"` |
| target-04 | 6 | identischer Typ/Surface/Span → `canonical_form="BiDAF"` |
| target-05 | 5 | semantisch identisch zu target-02 |

Die fünf Control-Aufzeichnungen sind erfolgreiche Leerantworten für `c_c0de5074475c9c0f`. Der historische Target-Lauf enthält ebenfalls eine erfolgreiche Null-Extraktion für `c_2a9f84f5fe0ce2ce`; mangels historischem Raw-Request ist dessen Request-Byteidentität ausdrücklich `not_recorded`.

## 3. Deterministische Diagnostik

`scripts/experiments/table_extraction_offline/diagnose_recordings.py` erzeugt vier getrennte Flächen:

- `successful_empty_observations`: nur technisch erfolgreiche Leeren — HTTP 200, `stop`, parsebar, null Rejects, leere Mention-Liste. `domain_conclusion` bleibt `null`.
- `overlap_groups`: exakte paarweise Spanrelationen (`contains` oder `partial_overlap`) samt Record-IDs. Überlappung wird beobachtet, weder verworfen noch als eigenständige Entität bestätigt.
- `canonical_conflicts`: gleiche `(entity_type, surface_form, char_start, char_end)` mit mehr als einem `canonical_form`. Der beobachtete Bi-DAF-Konflikt ist eine technische Anomalie, keine automatische Normalisierungsentscheidung.
- `domain_questions`: ausdrücklich ungelöste fachliche Fragen. Insbesondere bleibt offen, ob **ELMo + improved Bi-DAF** zusätzlich zu enthaltenen Namen eine eigenständige Entität ist.

Alle Mengen werden kanonisch sortiert. Eingabereihenfolge von Aufzeichnungen und Mentions beeinflusst den Bericht nicht. Externe absolute Rohdatenpfade werden nicht in den erzeugten Bericht kopiert.

Ausführung:

```bash
python3 scripts/experiments/table_extraction_offline/diagnose_recordings.py \
  --fixture tests/fixtures/table_extraction_v2_recordings.json \
  --output /tmp/table-extraction-diagnostics.json
```

Der Befehl liest ausschließlich die Fixture und schreibt einen JSON-Bericht. Er importiert keinen Provider und kennt keine Zugangsdaten.

## 4. Keine Wiederholungen oder Provider-Aufrufe

Der integrierte Shadow-Modus plant und startet keine Wiederholungen. Das
Diagnosemodul importiert keinen Provider; der Bericht setzt
`provider_calls=0` und `retries_enabled=false`. Erfolgreiche Leerantworten
werden ausschließlich aus dem verifizierten Attempt-Index, der zugehörigen
Raw-Antwort und den bereits kanonisch validierten Chunk-Status-Daten erkannt.
Fehlen historische Raw-Antworten oder stimmen ihre Prüfsummen nicht, bricht die
Diagnose geschlossen ab und schreibt keinen Bericht.

Damit betragen zusätzliche Provider-Aufrufe, Token und Kosten exakt null.
Eine Retry-Policy oder ein Retry-Planer ist ausdrücklich nicht Teil dieses
Experiments.

## 5. Wissenschaftliche Grenzen

| Risiko | Mechanismus | Gegenmaßnahme |
|---|---|---|
| Survivorship bias | nur erfolgreiche gespeicherte Runs könnten analysiert werden | terminale/inkomplette Versuche in der Run-Provenienz behalten |
| Granularitätsbias | verschachtelte Mentions könnten als Dubletten missverstanden werden | Overlaps nur diagnostizieren, niemals filtern oder vereinigen |
| Canonicalization bias | eine automatische Mehrheitswahl könnte seltene, richtige Formen unterdrücken | Konflikte offen ausgeben; Fachreview getrennt halten |
| Leakage / adaptive tuning | Gold-Information könnte technische Beobachtungen verzerren | ausschließlich technische Run-Artefakte lesen; keine Gold-Labels erzeugen |

Fünf Wiederholungen erlauben keine belastbare Leerraten- oder
Stabilitätswahrscheinlichkeit. Die Diagnose ist deskriptiv, nicht
inferenzstatistisch.

## 6. Integration in den Benchmark-Workflow

`scripts/benchmark/manifest_recording.py replay --offline-diagnostics` aktiviert
den Shadow-Modus. Erst verifiziert `load_recording(...)` Manifest, Inputs,
Attempt-Index, Raw-Antworten und kanonische Artefakte. Danach fordert
`replay_run(...)` byteidentische kanonische Replay-Ausgaben. Nur bei Erfolg
schreibt `src/benchmark/offline_diagnostics.py` die separate Datei
`offline_diagnostics.json`. Der kanonische Output wird weder ersetzt noch
ergänzt.

Der Bericht enthält Manifest-, Input-, Attempt- und kanonische
Artefakt-Prüfsummen. Absolute externe Pfade werden nicht übernommen.

### 6.1 Historical recordings and original implementation

Historical V2 recordings (Manifest Auto, Qwen, ältere Pilot- oder
Diagnoseläufe) werden ausschließlich mit **ihrer ursprünglichen
Implementierung** verifiziert. Replay bündelt Eingaben, Prompt- und
Schema-Snapshots, Implementierungs-Hashes und Plattform-Provenienz im
Recording-Manifest. Stimmt eines dieser Felder nicht, schlägt
`load_recording(...)` mit `RecordingManifestError` fehl. Eine
automatische Migration auf neue Schemata, Prompt-Versionen oder
Ontologie-Revisionen findet **nicht** statt; Anpassungen sind
ausschließlich über eine neue, separat erstellte Aufzeichnung zulässig.
Der hier vorgestellte Diagnosebericht verändert weder Aufzeichnungen
noch kanonische Outputs.

## 7. Akzeptanzkriterien

- Dieselben verifizierten Run-Artefakte erzeugen byteidentisches Diagnose-JSON.
- Erfolgreiche Leeren erfordern HTTP 200, `finish_reason=stop`, eine parsebare
  echte Raw-Antwort mit `mentions=[]` und einen validierten, rejectfreien
  Entity-Chunk-Status.
- Overlaps und `canonical_form`-Konflikte werden aus kanonischen Mention-Records
  berichtet; `domain_conclusion` bleibt `null`.
- Manifest, Inputs, Attempt-Index und kanonische Artefakte sind durch Pfad,
  Größe und SHA-256 gebunden.
- Fehlende oder veränderte historische Rohdaten brechen geschlossen ab.
- Diagnose und Tests führen keine Netzwerk-, Provider- oder Retry-Aufrufe aus.
- `ontology.yaml`, `prompts/`, `src/extract/pipeline_v2.py`, kanonische
  Extraktionsdateien und Gold-Daten bleiben unverändert.

## 8. Reproduzierbare Tests

```bash
env -u MANIFEST_API_KEY PYTHONPATH=. \
  /home/immo/projects/knowledge_graph_rag/.venv/bin/python -m pytest -q \
  tests/benchmark/test_offline_diagnostics.py \
  tests/experiments/test_table_extraction_offline_diagnostics.py
```

Die Tests verwenden nur technische Mock-Responses, die eingefrorene Fixture
und temporäre Ausgabeverzeichnisse. Ein Netzwerkzugriff ist blockiert.
