# Offline-Vergleich: Qwen-Quote-Repair vs. Manifest-Auto für `c_001452fd142ce075`

## Zweck und Abgrenzung

Dieser Bericht vergleicht **ausschließlich offline** die bereits
vorhandenen experimentellen Ergebnisse aus dem Qwen-Quote-Repair-Lauf
(`~/migration-incoming/v2-pilot-04/experimental.jsonl`,
`~/migration-incoming/v2-pilot-04/repair.audit.json`) mit den kanonisch
akzeptierten Records aus dem Manifest-Auto-Lauf
`20260929T160008Z-9dbc3c`
(`~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/extractions.jsonl`).

- **Qwen-Experiment** = experimentelles Whitespace-Rekonstruktionsverfahren
  aus dem dokumentierten v2-Pilot-04-Lauf. Die Records sind mit
  `canonical_v2: false` markiert (`repair.audit.json:2`); die
  V2-Gates wurden nach der Whitespace-Reparatur angewandt.
- **Manifest-Auto-Lauf** = kanonischer V2-Lauf über das
  Manifest-Auto-Routing mit dem Modell `auto` (Realmodell nicht
  verifiziert; `model_identity_verified: false`). Bestehende
  kanonische V2-Validierung bleibt unverändert.

Es werden **keine** wissenschaftlichen Review-Labels vergeben. Der
Bericht dokumentiert ausschließlich vorhandene Belege und die exakten
Evidenzspannen.

## Geteilte Quellenlage

| Aspekt | Wert |
|---|---|
| Dokument | `W3038572442` — *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (`data/manifest.jsonl:108`) |
| arXiv-ID | `2007.00808` (DOI `10.48550/arxiv.2007.00808`) |
| Original-LaTeX-Tarball | `data/raw/latex/2007.00808.tar.gz`, SHA-256 `46288f02699612de50ae6f7530080bced7e1edc5e3ffb2225b80e5d6742b58f0` |
| Chunk | `c_001452fd142ce075`, `section_path=["Experimental Methodologies"]`, `chunk_index=23` |
| Chunk-Quelltextsha256 | `4a2900139582bcea20c1200f509398ddb91b1d791b098505ca459724f081d335` |
| Caption-Quelle | `Tables/joint.tex` (SHA-256 `3ac38b93f8e7b7c4af81d5c291d0571854b7ea5b762941d2b036fc6107bf72ba`) |

## Drei Qwen-Quote-Repair-Kandidaten

Alle drei Qwen-Quote-Repair-Kandidaten teilen dieselbe
Relationsmodell-Antwort (`qwen3:8b`-Cache
`5d692f6660892c61240fcbd6f208ce8af706a450b65c0140281de208df4c65cb.json`)
und denselben rekonstruierten Evidenzspan.

| Assertion | Relation | Objekt-Mention (canonical_form / span) | evidence_quote (exakt) | char_start..char_end |
|---|---|---|---|---|
| `a_44999b3871c3f58f` | `REPORTS_METRIC` | `Answer Coverage at Top-20/100` / `[53, 82)` | `Retrieval results (Answer Coverage at Top-20/100) on \n         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]].` | `[34, 190)` |
| `a_0a775699fcc8a5ce` | `EVALUATES_ON` | `Natural Questions` / `[97, 119)` | identisch zu oben | `[34, 190)` |
| `a_2814c002a3ebd912` | `EVALUATES_ON` | `Trivial QA` / `[124, 140)` | identisch zu oben | `[34, 190)` |

Rekonstruktionsentscheidungen aus `repair.audit.json`:

| Relation | `model_quote` | `restored_quote` | Status |
|---|---|---|---|
| `REPORTS_METRIC` | normalisiert (Whitespace kollabiert) | `[34, 190)` mit originalen Zeilenumbrüchen/Einrückungen | `repaired` |
| `EVALUATES_ON` (NQ) | identisch normalisiert | identisch | `repaired` |
| `EVALUATES_ON` (TQA) | identisch normalisiert | identisch | `repaired` |

## Vier Manifest-Auto-Assertions (MiniMax-Realmodell, unverifiziert)

Aus `extractions.jsonl` (Manifest-Auto-Lauf), Chunk `c_001452fd142ce075`.
Modell: `auto`/Routing; konkretes Realmodell laut
`run_summary.json` `provider.reported_model = null` und
`provider.model_identity_verified = false`. Die Manifest-Antwort
referenziert `MiniMax-M3` in `attempt_reported_models`, ohne
Verifikation. Der Lauf selbst unterscheidet sich methodisch vom Qwen-
Experiment (kein Cache-Replay, frische Modellantwort).

| Assertion | Relation | Objekt-Mention (canonical_form / span) | evidence_quote (exakt) | char_start..char_end |
|---|---|---|---|---|
| `a_186ebfe9b1cd3758` | `USES_METHOD` | `Dense Retrieval` (`m_da34cbe93057b973`) / `[706, 708)` | `The gains are from changing the training of a production DR model to ANCE` | `[649, 722)` |
| `a_48f1cfb9f80bca7b` | `USES_METHOD` | `ANCE` (`m_b0dfffc9fd61a4d2`) / `[718, 722)` | identisch zu oben | `[649, 722)` |
| `a_4fd1879b3d8f18b7` | `EVALUATES_ON` | `Trivia QA` (`m_0045e51965d8059f`) / `[124, 140)` | `\caption{Retrieval results (Answer Coverage at Top-20/100) on \n         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~` | `[25, 162)` |
| `a_c690e0b1f2dfec3a` | `EVALUATES_ON` | `Natural Questions` (`m_9e85ca5de092683`) / `[97, 119)` | identisch zu oben | `[25, 162)` |

Mentions aus dem Manifest-Auto-Lauf (4 Records):

| Mention | entity_type | canonical_form | surface_form | span |
|---|---|---|---|---|
| `m_0045e51965d8059f` | dataset | `Trivia QA` | `Trivial QA (TQA)` | `[124, 140)` |
| `m_9e85ca5fde092683` | dataset | `Natural Questions` | `Natural Questions (NQ)` | `[97, 119)` |
| `m_b0dfffc9fd61a4d2` | method | `ANCE` | `ANCE` | `[718, 722)` |
| `m_da34cbe93057b973` | method | `Dense Retrieval` | `DR` | `[706, 708)` |

`chunk_status`: 4 Mentions, 4 Assertions, 1 Attempt je Phase, 0 Failures.

## Gegenüberstellung der Methodik

| Aspekt | Qwen-Quote-Repair (v2-Pilot-04) | Manifest-Auto (20260929T160008Z-9dbc3c) |
|---|---|---|
| Modell | `qwen3:8b` (lokal, gespeicherte Cache-Antwort) | `auto`-Routing, Realmodell nicht verifiziert |
| Pipeline-Pfad | `experimental_whitespace_v1` (siehe `repair.audit.json:1-2`) | unveränderte kanonische V2-Validierung |
| Datenquelle | reproduzierter Cache-Hit, keine Modellaufrufe | 4 HTTP POSTs an den Manifest-Provider |
| Evidenzspan-Disziplin | rekonstruierter Span `[34, 190)` (Original-Whitespace) | zwei verschiedene Spans: `[25, 162)` für Caption-Assertions, `[649, 722)` für USES_METHOD |
| Record-Set | 3 Mentions + 3 Assertions + `chunk_status` | 4 Mentions + 4 Assertions + `chunk_status` |
| Markierung | `canonical_v2: false` (experimentell) | kanonisch akzeptiert durch V2-Gates |

## Strukturelle Differenzen

1. **Caption-Span-Differenz:** Beide Läufe adressieren dieselbe
   Caption, aber mit unterschiedlichen Spans. Der Qwen-Quote-Repair-
   Span `[34, 190)` enthält die volle Caption inkl. `[[CITE:...]]`
   und `\\label{tab:openqa}}` (siehe Verbatim-Vergleich in
   `docs/experiments/quote-repair-v1-review/context-report.md:17`).
   Der Manifest-Auto-Span `[25, 162)` beginnt beim Caption-Start
   (`\\caption{`) und endet **vor** `[[CITE:...]]` — der
   zitationsbindende Teil fehlt im Evidence-Quote.
2. **Relationssatz:** Qwen liefert `REPORTS_METRIC` für
   `Answer Coverage at Top-20/100` (Mention `m_beb3b4ae863a80bd`,
   Span `[53, 82)`). Der Manifest-Auto-Lauf hat **keine** Metrik-
   Mention und damit **keine** `REPORTS_METRIC`-Assertion extrahiert
   (siehe unten *Fehlende Relation*).
3. **Zusatzrelationen:** Der Manifest-Auto-Lauf liefert zwei
   `USES_METHOD`-Assertions (DR, ANCE auf den Chunk bzw. das Dokument
   als Subjekt). Diese Methode-zu-Methode- bzw.
   Dokument-zu-Methode-Relationen sind im Qwen-Quote-Repair-Lauf
   **nicht** vorhanden — der Qwen-Cache enthält nur die drei
   Caption-getragenen Kandidaten, von denen alle drei denselben
   Evidenzspan teilen.
4. **Datensatz-Normalisierung:** Siehe nächster Abschnitt.

## Fehlende Relation: REPORTS_METRIC für Answer Coverage

Im Manifest-Auto-Lauf fehlt für `c_001452fd142ce075` eine
`REPORTS_METRIC`-Relation zwischen Dokument und der Metrik
`Answer Coverage at Top-20/100`. Die Ontologie definiert diese
Relation in `ontology.yaml:197-207`:

```yaml
- name: REPORTS_METRIC
  from: document
  to: metric
  definition: Document reports a metric in its results.
  population_mode: extracted
```

Die Metrik steht im Caption-Text (`chunk.text[53:82]` =
`Answer Coverage at Top-20/100`), aber das Manifest-Auto-Modell hat
keine entsprechende Mention und keine Relation emittiert. Im Qwen-
Experiment ist genau diese Relation vorhanden
(`a_44999b3871c3f58f` mit Mention `m_beb3b4ae863a80bd`).

**Mögliche Ursachen [INFERENCE]:** (a) das Modell hat den Metrik-Span
nicht als eigenständige Mention erkannt; (b) der
Relations-Prompt hat `REPORTS_METRIC` nicht ausgegeben, obwohl die
Metrik im Chunk sichtbar ist; (c) das Realmodell hat eine andere
Relation bevorzugt oder den Caption-Block insgesamt verworfen. Die
Manifest-Auto-Antwort ist im gespeicherten `metrics.jsonl` nicht
transparent gemacht; ein zwingender Rückschluss auf eine dieser
Ursachen ist nicht möglich.

**Konsequenz für den wissenschaftlichen Review:** Die Caption enthält
die Metrik, aber das Modell hat sie nicht in eine `REPORTS_METRIC`-
Relation übersetzt. Damit fehlt im Manifest-Auto-Lauf eine Aussage,
die im Qwen-Experiment vorhanden ist. Das ist **keine** Aussage über
wissenschaftliche Korrektheit, sondern ein quantitativer Unterschied
im extrahierten Relationssatz.

## Normalisierung: `Trivial QA (TQA)` ↔ `Trivia QA`

Die Verbatim-Caption (`chunk.text[124:140]`) lautet `Trivial QA (TQA)`.
Die Beobachtungen in den vorhandenen Datensätzen:

| Quelle | `canonical_form` | `surface_form` |
|---|---|---|
| `v2-multichunk-01/baseline.v2.jsonl:6` | `Trivial QA` | `Trivial QA (TQA)` |
| `v2-pilot-03/extractions.v2.jsonl:1` | `Trivial QA` | `Trivial QA (TQA)` |
| `v2-pilot-03/cache/llm/ccebfd17….json` (qwen3:8b) | `Trivial QA` | `Trivial QA (TQA)` |
| `v2-pilot-04/experimental.jsonl:1` | `Trivial QA` | `Trivial QA (TQA)` |
| `manifest-auto-runs/20260929T160008Z-9dbc3c/extractions.jsonl` | `Trivia QA` | `Trivial QA (TQA)` |

**Diskrepanz:** Alle vorherigen Läufe und der zugrundeliegende Qwen-
Modellcache liefern `canonical_form="Trivial QA"`. Der Manifest-Auto-
Lauf normalisiert abweichend zu `"Trivia QA"`. Der `surface_form`
bleibt in allen Fällen identisch zum Caption-Verbatim
(`Trivial QA (TQA)`).

### Originalquelle

Der kanonische Name des Datensatzes (Joshi et al., 2017, *Trivia
QA: A Large Scale Distantly Supervised Challenge Dataset for Reading
Comprehension*) ist **Trivia QA** mit Bindestrich-Variante
**TriviaQA**. Die im Caption-Text des Chunks verwendete Schreibweise
`Trivial QA` ist eine **Tippvariante im LaTeX-Quelltext** und keine
offizielle Schreibweise des Datensatzes.

**Beleg (LaTeX-Quelle):** `Tables/joint.tex`, Caption-Zeile
`\caption{Retrieval results (Answer Coverage at Top-20/100) on
Natural Questions (NQ) and Trivial QA (TQA) ...}`. Der Tarball
`2007.00808.tar.gz` (SHA-256 oben) enthält genau diese Schreibweise
in der Beschriftung.

### Projektinterne Alias-/Ontologie-Regel

- `data/aliases.json` enthält ausschließlich `work_id`-Mappings (z. B.
  `W1544827683 → W2949615363`); **keine** Mention- oder
  Datensatz-Aliase. Eine projektinterne Regel, die `Trivial QA` ↔
  `Trivia QA` verknüpft, ist im Repository **nicht vorhanden**.
- Die Ontologie (`ontology.yaml`) deklariert ausschließlich
  Relationsdefinitionen; Mention-Aliase werden dort nicht
  spezifiziert.
- Die `entity_resolve.py` / `src/ontology/__init__.py` enthalten
  keine Erwähnung von `Trivial QA` oder `Trivia QA`.

### Folgerung (keine automatische Korrektur)

Die Diskrepanz zwischen `Trivia QA` (Manifest-Auto-Lauf) und
`Trivial QA` (alle vorherigen Läufe, inklusive Qwen-Quote-Repair) ist
eine **Normalisierungsdivergenz innerhalb der Modelle**, nicht eine
projektintern definierte Aliasregel. Die im Caption-Text stehende
Schreibweise `Trivial QA (TQA)` ist die geteilte Verbatim-Quelle.
Welche Schreibweise fachlich korrekt ist, hängt von der
Reviewer-Entscheidung über die Identität mit dem offiziellen Datensatz
ab (Trivia QA / TriviaQA).

**Dieser Bericht nimmt keine Korrektur vor.** Eine automatische
Überschreibung der Manifest-Auto-`canonical_form` wäre eine
Modellaussage-modifizierende Entscheidung, die außerhalb des Auftrags
dieses Vergleichs liegt.

## Belegprüfung der zwei USES_METHOD-Relationen (ohne Labels)

**Span und Verbatim-Quelle:** Beide Assertions referenzieren denselben
Evidence-Quote `chunk.text[649:722]` =
`The gains are from changing the training of a production DR model to ANCE`.

**Mention-Spans innerhalb des Evidence-Quotes:**

- `m_da34cbe93057b973` (canonical_form `Dense Retrieval`,
  surface_form `DR`) → Span `[706, 708)`, **vollständig im
  Evidence-Quote enthalten**.
- `m_b0dfffc9fd61a4d2` (canonical_form `ANCE`, surface_form `ANCE`)
  → Span `[718, 722)`, **vollständig im Evidence-Quote enthalten**.

**Subjekt:** Beide Relations referenzieren
`subject_ref.id = "W3038572442"` (Dokument), gemäß
`USES_METHOD`-Definition `from: document, to: method` in
`ontology.yaml:185-188`.

**Vorhandene Evidenz im lokalen Chunk:**

| Vorhanden | Beleg |
|---|---|
| Verbatim-Satz im Chunk | `The gains are from changing the training of a production DR model to ANCE` |
| Dokumentreferenz | `subject_ref.id` = `W3038572442` |
| Method-Mentions in der Ontologie | `USES_METHOD` ist in `ontology.yaml:185-188` deklariert |
| `source_text_sha256` der Evidenz | `4a2900139582bcea20c1200f509398ddb91b1d791b098505ca459724f081d335` = Chunksha256 |

**Was im lokalen Chunk fehlt:**

| Fehlt lokal | Konsequenz |
|---|---|
| Definierte Bedeutung von `DR` als Methodenname | Der Caption-Block der Veröffentlichung definiert `DR` als Abkürzung; der Chunk enthält den definierenden Satz nicht (`Evaluation.tex:19` mit *It also outperforms DPR in passage retrieval in OpenQA (Table~\ref{tab:openqa}).* liegt außerhalb des Chunk-Spans). |
| Eindeutige Bindung von `ANCE` an ein System vs. Methode | `ANCE` ist eine Methode; die Beziehung `ANCE uses DR` oder `DR builds on ANCE` ist im Satz nicht eindeutig, der Satz beschreibt eine *Änderung des Trainings* eines `production DR model` zu `ANCE`. |

### Manuelle Review-Tabelle (offene Felder)

Die folgenden Felder sind **noch nicht belegt**. Sie sind keine
wissenschaftlichen Wahrheitslabels dieses Berichts, sondern
Reviewer-Eingabefelder.

| Assertion | Objekt-Mention | Relation | supported | unsupported | unclear | schema_error | Kommentar |
|---|---|---|:---:|:---:|:---:|:---:|---|
| `a_186ebfe9b1cd3758` | `Dense Retrieval` (`m_da34cbe93057b973`) | `USES_METHOD` | ☐ | ☐ | ☐ | ☐ | Evidence-Quote nennt `DR` ohne definierenden Satz im Chunk. |
| `a_48f1cfb9f80bca7b` | `ANCE` (`m_b0dfffc9fd61a4d2`) | `USES_METHOD` | ☐ | ☐ | ☐ | ☐ | Satz beschreibt Trainingsänderung *zu* ANCE, nicht eine klassische `USES_METHOD`-Anwendung. |

**Hinweis:** Die Bewertung erfordert externes Wissen über die
Veröffentlichung und/oder ein Gold Set. Dieser Bericht vergibt keine
Labels.

## Quantitative Differenzen im Überblick

| Kennzahl | Qwen-Quote-Repair (v2-Pilot-04) | Manifest-Auto (20260929T160008Z-9dbc3c) |
|---|---:|---:|
| Chunks verarbeitet | 1 (`c_001452fd142ce075` im direkten Repair-Audit) | 2 |
| Mentions akzeptiert | 3 | 4 (1 zusätzliche: `DR` als Methode) |
| Assertions akzeptiert | 3 | 4 (1 zusätzliche: `USES_METHOD` auf ANCE) |
| Relationsarten | `EVALUATES_ON` × 2, `REPORTS_METRIC` × 1 | `EVALUATES_ON` × 2, `USES_METHOD` × 2 |
| Rejects | 0 (nach Reparatur) | 0 |
| Status | experimentell (`canonical_v2: false`) | kanonisch akzeptiert |

## Reihenfolge der wissenschaftlichen Unsicherheiten

1. **`Trivia QA` vs. `Trivial QA`:** Keine projektinterne
   Alias-/Ontologie-Regel vorhanden; Diskrepanz ist eine
   Modell-Normalisierungsdivergenz. **Keine** automatische Korrektur
   vorgenommen.
2. **Fehlende `REPORTS_METRIC` im Manifest-Auto-Lauf:** Caption
   enthält die Metrik, aber das Modell hat sie nicht in eine Relation
   übersetzt. Mögliche Ursachen sind Inferenz; das gespeicherte
   Roh-Prompt-Response-Paar enthält sie nicht ohne weiteres.
3. **Caption-Span `[25, 162)` vs. `[34, 190)`:** Der Manifest-Auto-
   Span endet vor `[[CITE:...]]` und vor `\label{tab:openqa}`; ein
   Reviewer muss entscheiden, ob der Verlust der Citation die
   wissenschaftliche Tragfähigkeit verändert.
4. **`USES_METHOD`-Bindung an `DR`:** Der definierende Satz
   `Evaluation.tex:19` liegt außerhalb des Chunk-Spans; eine
   verblindete Entscheidung `supported` ist ohne externes Wissen
   nicht möglich.
5. **Modellidentität Manifest-Auto:** Realmodell nicht verifiziert
   (`model_identity_verified: false`); ein Vergleich der
   Qwen-Quote-Repair-Kandidaten mit den Manifest-Auto-Records ist
   nicht ein direkter Modellvergleich, sondern ein Vergleich
   unterschiedlicher methodischer Pfade.

## Verweise

- Qwen-Repair-Audit: `~/migration-incoming/v2-pilot-04/repair.audit.json`
- Qwen-Repair-Records: `~/migration-incoming/v2-pilot-04/experimental.jsonl`
- Qwen-Cache: `~/migration-incoming/v2-pilot-03/cache/llm/5d692f6660892c61240fcbd6f208ce8af706a450b65c0140281de208df4c65cb.json`
- Qwen-Entity-Cache: `~/migration-incoming/v2-pilot-03/cache/llm/ccebfd17a77681c2b863cb5beca3010ae066f1b157274519217c91ede9818ced.json`
- Manifest-Auto-Records: `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/extractions.jsonl`
- Manifest-Auto-Run-Summary: `~/migration-incoming/manifest-auto-runs/20260929T160008Z-9dbc3c/run_summary.json`
- Original-Caption-Span-Vergleich (Caption-zu-Tabellen-Beziehung):
  `docs/experiments/quote-repair-v1-review/context-report.md`
- Ontologie-Definitionen: `ontology.yaml:169-218`
- Strukturelle Validierung: `docs/specs/extraction-contract.md:319-353`
- Originalquelle (TeX-Tarball): `data/raw/latex/2007.00808.tar.gz`
  (SHA-256 `46288f02699612de50ae6f7530080bced7e1edc5e3ffb2225b80e5d6742b58f0`)
