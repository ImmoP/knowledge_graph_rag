# Gemeinsamer Quellenkontext für das wissenschaftliche Review

Dieser Bericht fasst die drei Review-Einträge `a_44999b3871c3f58f` (REPORTS_METRIC), `a_0a775699fcc8a5ce` (EVALUATES_ON) und `a_2814c002a3ebd912` (EVALUATES_ON) zusammen. Er ergänzt die bestehenden Review-Protokolle um eine gemeinsame Quellenbasis und macht transparent, welche Evidenz die Ontologie für `EVALUATES_ON` und `REPORTS_METRIC` verlangt, welche Belege im lokalen Quellenmaterial vorhanden sind und welche fehlen. Es werden keine wissenschaftlichen Review-Labels vergeben.

## 1. Geteilte Quellenlage

| Aspekt | Wert |
|---|---|
| Dokument | `W3038572442` — *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* (OpenAlex-Titel; `data/manifest.jsonl:108`) |
| arXiv-ID | `2007.00808` (DOI `10.48550/arxiv.2007.00808`) |
| Original-LaTeX-Tarball | `data/raw/latex/2007.00808.tar.gz`, SHA-256 `46288f02699612de50ae6f7530080bced7e1edc5e3ffb2225b80e5d6742b58f0` (siehe `data/fulltext_manifest.jsonl:42`) |
| Chunk | `c_001452fd142ce075`, `section_path=["Experimental Methodologies"]`, `chunk_index=23` |
| Chunk-Quelltextsha256 | `4a2900139582bcea20c1200f509398ddb91b1d791b098505ca459724f081d335` |
| Originaldatei der Caption | `Tables/joint.tex` (SHA-256 `3ac38b93f8e7b7c4af81d5c291d0571854b7ea5b762941d2b036fc6107bf72ba`); Caption-Bytes im bereinigten Volltext an Positionen `[37099, 37321)` |
| Vorgelagerter Fließtext | `Evaluation.tex:19` (`It also outperforms DPR in passage retrieval in OpenQA (Table~\ref{tab:openqa}).`) — **liegt außerhalb des Chunk-Spans `[37074, 37909)`** |
| Angrenzende Tabellen | `Tables/openqa.tex` (offene Tabelle mit den NQ/TQA-Top-20/100-Werten; endet bei bereinigtem Volltext-Offset `37059`, also **vor** dem Chunkstart) und `Tables/prod.tex` (Tabelle zu kommerzieller Suche, andere Spalten und andere Caption) |
| Evidenzspan (alle drei Einträge) | `chunk.text[34:190]` = bereinigter Volltext `[37108:37264]`. Verbatim aus `Tables/joint.tex`. |

## 2. Ontologiedefinitionen und Evidenzanforderungen

### 2.1 `EVALUATES_ON` (Ontologie v1, Zeilen 169–183 in `ontology.yaml`)

- Endpunkte: `from: document`, `to: dataset`.
- Definition: *Document reports experimental results on a dataset.*
- Strukturpflichten aus `docs/specs/extraction-contract.md` §8.2 (Ontologiekonformität) und §8.3 (Evidenzintegrität): `subject_ref.kind="document"`, `object_ref.kind="mention"` (Datensatz-Mention desselben Chunks), Evidenzspan liegt im Chunktext und enthält den Mention-Span, `source_text_sha256` matcht. Keine Kardinalitäts- oder Mehrfachverwendungsregel in der Ontologie v1.

Wissenschaftliche Substanzierung gemäß `gold_set/GUIDELINES.md` §4.1.3 und §6.1: Eine `EVALUATES_ON`-Relation ist im Gold-Set üblicherweise `stated` (Satz-/Caption-Formulierung wie *evaluate DPR on Natural Questions*) oder `inferred` (Tabellenzeile wie *DPR NQ 41.5*). Ein **rein caption-getragener Hinweis ohne Ergebniswert und ohne benanntes System** ist im lokalen Material **nicht** eindeutig einem der drei Confidence-Stufen zuzuordnen.

### 2.2 `REPORTS_METRIC` (Ontologie v1, Zeilen 197–207 in `ontology.yaml`)

- Endpunkte: `from: document`, `to: metric`.
- Definition: *Document reports a metric in its results.*
- Strukturpflichten identisch zu `EVALUATES_ON`. Keine Kardinalitätsregel.

Wissenschaftliche Substanzierung gemäß `gold_set/GUIDELINES.md` §4.1.5 und §6.1: Standardformulierungen sind `stated` (z. B. *report Exact Match on the development split*) oder `inferred` (Wert in einer Tabellenzeile). Die reine Erwähnung des Metric-Namens ohne zugehörigen Ergebniswert ist im lokalen Material **nicht** eindeutig zuordenbar.

### 2.3 Modell-`confidence` vs. wissenschaftliches Review

`gold_set/GUIDELINES.md` §6 untersagt explizit, die Modell-Konfidenz (`stated` aus `confidence: "stated"` in den Record-Dateien) als Gold-Label zu übernehmen. Das `stated`-Feld im gespeicherten Relationspayload ist eine Modellaussage, **kein** menschliches Review-Label.

## 3. Per-Assertion-Inventur

Alle drei Einträge teilen denselben Evidenzspan und dieselbe Caption. Die nachfolgende Tabelle zeigt pro Eintrag, welche Belege vorhanden sind und welche fehlen. *Keine* Spalte vergibt ein Review-Label.

| Assertion | Relation | Objekt-Mention | Vorhanden | Fehlt lokal |
|---|---|---|---|---|
| `a_44999b3871c3f58f` | `REPORTS_METRIC` | `Answer Coverage at Top-20/100` (metric, `[53,82)`) | Metrik-Mention steht im Caption-Text; Verbatim-Substring aus `Tables/joint.tex`. | Numerischer Wert der Metrik im Chunk (kein Top-20/100-Wert; angrenzender Tabellenkörper ist `prod.tex`); benanntes System im Span; assertion-spezifische Trennung gegenüber den beiden `EVALUATES_ON`-Einträgen. |
| `a_0a775699fcc8a5ce` | `EVALUATES_ON` | `Natural Questions (NQ)` (dataset, `[97,119)`) | Datensatz-Mention steht im Caption-Text; Formulierung `Retrieval results on Natural Questions` im Span; Verbatim-Substring. | Numerisches NQ-Ergebnis im Chunk (Werte liegen in `Tables/openqa.tex` vor dem Chunk); benanntes System im Span; prose-Bindung (Evaluation.tex:19) liegt außerhalb des Chunk-Spans. |
| `a_2814c002a3ebd912` | `EVALUATES_ON` | `Trivial QA (TQA)` (dataset, `[124,140)`) | Datensatz-Mention steht im Caption-Text; Formulierung `... and Trivial QA (TQA)` im Span; Verbatim-Substring. | Numerisches TQA-Ergebnis im Chunk (Werte liegen in `Tables/openqa.tex`); benanntes System im Span; prose-Bindung außerhalb des Chunk-Spans. |

## 4. Caption-zu-Tabellen-Beziehung — nur lokal nachvollziehbare Aussagen

Aus dem lokalen Quellenmaterial (`data/raw/latex/2007.00808.tar.gz`, extrahiert in `/tmp/w3038572442-source` und der bereinigte Volltext `/tmp/w3038572442-source-cleaned.txt`) lassen sich ausschließlich die folgenden Beziehungen zwischen der Caption `\caption{Retrieval results (Answer Coverage at Top-20/100) on ... \label{tab:openqa}}` und Tabellenkörpern feststellen:

1. **Caption-zu-Tabellenkörper-Verknüpfung:** `Tables/joint.tex` bindet über `\ttabbox{\resizebox{0.6\textwidth}{!}{\input{Tables/openqa}}}{ ... \label{tab:openqa}}` die Caption `tab:openqa` an `Tables/openqa.tex` (Top-20/100-Werte für NQ und TQA). Die Caption adressiert damit numerisch genau die in den drei Assertions genannten Datensätze und die Metrik.
2. **Caption-zu-Caption-Disjunktion:** Derselbe `floatrow` enthält unmittelbar nach der ersten Caption eine **zweite** Caption (`\caption{Relative gains in the first stage retrieval of a commercial search engine ... \label{tab.prod}}`), die an die separate Tabelle `Tables/prod.tex` (Spalten `Corpus Size, Dim, Search, Gain`) gebunden ist. Die zweite Tabelle enthält **keine** `Answer Coverage @ Top-20/100`-Werte und **keine** NQ/TQA-Zeilen; sie ist damit **nicht** als Ergebnistabelle für die genannte Metrik interpretierbar.
3. **Caption-zu-Block-Grenze:** Der Chunk enthält sowohl die erste Caption inkl. Label `\label{tab:openqa}}}` als auch die `Tables/prod.tex`-Tabellenzeilen und den Beginn der zweiten Caption. Die numerischen `Tables/openqa.tex`-Inhalte enden **vor** dem Chunkstart (`cleaned_text` Offset `37059` vs. Chunkstart `37074`). Eine lokale Zuordnung der Caption zu ihrem *eigenen* Tabellenkörper gelingt im Chunk **nicht** ohne Kontext aus dem angrenzenden, ausgeschnittenen Quelltext.
4. **Erwähnung im Fließtext:** `Evaluation.tex:19` enthält den einzigen Satz, der die in der Caption genannten Datensätze explizit an ein System bindet (*It also outperforms DPR in passage retrieval in OpenQA (Table~\ref{tab:openqa}).*). Dieser Satz liegt **außerhalb** des Chunk-Spans und ist daher nicht Bestandteil der drei Evidence Quotes.

## 5. Nicht im Bericht enthaltene Zusicherungen

- Es wird keine Precision-/Recall-Aussage getroffen; eine Gold-Set-Bewertung existiert nicht.
- Es wird keine Aussage zur Generalisierung über die zwei Chunks hinaus getroffen.
- Es wird keine Aussage zur Wirkung auf nachgelagerte Retrieval- oder Antwortqualität getroffen.
- Es werden **keine** wissenschaftlichen Review-Labels (`supported`, `unsupported`, `unclear`) vergeben.

## 6. Verweise

- Ontologydefinitionen: `ontology.yaml:149-218`.
- Strukturelle Validierungs-Gates: `docs/specs/extraction-contract.md:319-353`.
- Wissenschaftliche Rubrik: `gold_set/GUIDELINES.md:422-480` und `gold_set/GUIDELINES.md:628-700`.
- Provenienzmanifest: `docs/experiments/quote-repair-v1-review/provenance/manifest.v1.json`.
- Review-Protokolle:
  - `docs/experiments/quote-repair-v1-review/review-a_44999b3871c3f58f.md`
  - `docs/experiments/quote-repair-v1-review/review-a_0a775699fcc8a5ce.md`
  - `docs/experiments/quote-repair-v1-review/review-a_2814c002a3ebd912.md`