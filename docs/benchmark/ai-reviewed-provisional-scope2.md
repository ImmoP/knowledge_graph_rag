# Scope-2 Pilot – KI-geprüfter Zwei-Chunk-Vorabgleich (2026-10-01)

## Geltungsbereich

Pilotumfang exakt die zwei Chunks `c_61090bb37bc1e7a9` (positiv) und
`c_c0de5074475c9c0f` (negativ).  Die drei zurückgestellten Chunks
`c_2a9f84f5fe0ce2ce`, `c_7fdad3ec8f3e783b`, `c_d285f0a62ff9479b` sind in
keiner Gate-Ausgabe zulässig.  Scope ist `ai_reviewed_provisional_scope2`,
`human_gold=false`, kein automatischer Goldimport, keine globalen
P/R/F1-Werte, keine Modellrangliste.

## Historische Quelle

`20260930T083857Z-c69654` enthält die vier kanonischen V2-Artefakte:

| Datei | SHA-256 |
|---|---|
| `extractions.jsonl` | `6fa92e307a25fe6a2440e591686e6d901ce5da9b82450583f00c8c68b462404e` |
| `extractions.rejects.jsonl` | `cb29588ad3c0fb25b60d5b7c6d0f6033f7aba86092192ffd3e968831fd8a2292` |
| `metrics.jsonl` | `b69194b5b79893733cc534b3966bc7be29bf8882e88cd9200be96ba677342bb4` |
| `run_summary.json` | `7e0930240d37752e894db0644bf05f43734ba4109545ff692f205b395b6faf25` |

Eingefrorener Datensatz: `38182c6cdbb9f86abc6ce20ece7a3d9e1cb69ccac2576e2d6274e727c2f84985`.
Ontologie: `f9a3e203d4c640b91966e077b02f09a547d49b51595cf33e040a3e278cd2325e`.
Pilot-Manifest: `2a96a3cf259ab8f65e2698d33c162cd1415ccde73b3b3fb5777c378149ed4886`.
Blind-Review-ZIP: `04081e1e1adf2517017d758b468ac30d4623bad881a213b71c7ab8d6656ed702`.

Die historische Aufzeichnung berichtet `requested_model="auto"`,
Provider `manifest_auto_benchmark`, pro Phase `MiniMax-M3`,
aber `model_identity_verified=false`.  Rohe Anfragen und rohe
Antworten sind im Aufzeichnungsverzeichnis nicht vorhanden.  Die
Scope-2-Auswertung behandelt den Lauf daher ausschließlich als
**historische Beobachtung**, nicht als neue Live-Messung.

## Historische Beobachtungen (Scope 2)

| Chunk | Rolle | `extraction_complete` | entity / relation Status | Mention / Assertion | Scope-2 Treffer |
|---|---|---|---|---|---|
| `c_61090bb37bc1e7a9` | positiv | `true` | beide Phasen `validated` | 0 / 0 | exakte `[46, 64)` Switch-Transformer-Mention erwartet |
| `c_c0de5074475c9c0f` | negativ | `true` | beide Phasen `validated` | 0 / 0 | keine Mentions, keine Relations erwartet |

Laut `run_summary.json` war der Gesamtlauf `5/10` vollständig.  Nur
die zwei Scope-2-Zielchunks werden in den Pilot überführt; die fünf
anderen unvollständigen Run-Chunks werden **niemals** als
Scope-2-Ausgaben übernommen und das Pilot-Gate verweigert bei
unvollständigem Zielchunk/fehlender Phase.

## Methodische Limits

- `no_global_quality_claims`: keine Precision, Recall, F1.
- `no_model_ranking`: keine Modellrangliste.
- `no_human_gold`: kein menschlich freigegebener Goldstandard.
- `selection_bias_targeted_empty_chunks`: gezielte Auswahl historisch
  leerer Chunks, kein Zufallssample.
- `raw_responses_unavailable`: rohe Antworten sind nicht vorhanden;
  keine Rekonstruktion.
- `model_identity_unverified`: pro Phase wurde ein Modell berichtet,
  aber `model_identity_verified=false`.

## P01 / P03

- P01 `chunk_only`: ausschließlich `text` des ursprünglichen Chunks;
  keine Belege aus Paperkontext oder `section_path`.
- P03 `include_named_architecture_single_caption`: `Switch Transformer`
  aus der Caption wird als `method` gezählt.

## Synthetic vs. Historisch

Die unter `tests/benchmark/test_scope2_pilot.py` enthaltenen synthetischen
Fixtures werden ausschließlich zur Funktionsprüfung verwendet.  Sie
werden niemals als historische Ergebnisse behandelt und niemals in
`pilot_smoke_pass` einbezogen.

## Befehle (offline)

```bash
PYTHONPATH=. python scripts/benchmark/report_scope2_pilot.py \
  --historical-run /home/immo/migration-incoming/manifest-auto-autopilot-live/20260930T083857Z-c69654 \
  --pilot-root /home/immo/migration-incoming/pilot/kg_rag_ai_pilot_scope2_20261001 \
  --report /tmp/scope2-report.json
```

Die Tests wurden noch nicht ausgeführt.  Der Hauptagent wird die
gesamte Suite einmal ausführen und die Ergebnisse nachliefern.

## Verbleibende wissenschaftliche / technische Arbeiten

1. Rohe Anfragen/Antworten aus einem zukünftigen, reproduzierbaren
   Lauf aufnehmen und im Pilot verifizieren.
2. Tatsächlich geroutetes Modell pro Phase bestätigen.
3. Einen deterministischen, nicht-leeren Kontrollchunk einbinden.
4. Pilot ausschließlich nach einem separaten, vollständig
   fachlich begutachteten Goldset erweitern.
5. Nächster wissenschaftlicher Schritt: den Zwei-Chunk-Pilot durch
   eine stratifizierte, vorab versiegelte Auswahl zulässiger Ziele
   ersetzen, die von Fachgutachtern adjudiziert und gegen den
   persistierten Snippet-Bestand geroutet wird.

## Sicherheitsmerkmale

- `human_gold=false`, `automatic_gold_import=false`,
  `allow_global_quality_claims=false` sind unveränderlich.
- Drei withheld Chunks werden in `excluded_chunks` gehasht und in
  keiner Ausgabe propagiert.
- Keine Netzwerk-, Provider- oder Manifest-API-Aufrufe im Pilot.
- Keine Änderungen an Ontologie, Prompts, eingefrorenem Datensatz,
  bestehendem Fünf-Chunk-Workflow oder historischen Recordings.