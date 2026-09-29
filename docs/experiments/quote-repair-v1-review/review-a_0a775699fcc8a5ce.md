# Review-Protokoll `a_0a775699fcc8a5ce`

## Assertion

| Feld | Wert |
|---|---|
| Technischer Status | `accepted after whitespace reconstruction` |
| Wissenschaftlicher Status | **nicht entschieden** |
| Relationstyp | `EVALUATES_ON` — Ontologiedefinition: „Document reports experimental results on a dataset.“ |
| Subjekt | `document:W3038572442` — *Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval* |
| Objekt | `mention:m_9e85ca5fde092683` — Typ `dataset`, Oberfläche `Natural Questions (NQ)`, kanonische Form `Natural Questions`, Mention-Span `[97,119)` |
| Chunk | `c_001452fd142ce075`, Abschnitt `Experimental Methodologies`, Chunk-Index 23 |
| Modell-Confidence | `stated` — gespeicherte Modellausgabe, **kein** menschliches Review-Label |

## Exaktes Originalzitat

Zeichenpositionen im originalen Chunktext: **`[34,190)`**. Newline und neun folgende Leerzeichen sind Bestandteil des Zitats.

```text
Retrieval results (Answer Coverage at Top-20/100) on 
         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]].
```

Verifizierte Invarianten:

- `chunk.text[34:190] == evidence_quote`
- Objektspan `[97,119)` liegt vollständig im Evidenzspan `[34,190)`.
- `source_text_sha256 = 4a2900139582bcea20c1200f509398ddb91b1d791b098505ca459724f081d335`

## Unmittelbar umgebender Originalquelltext

```latex
}}
         {  
         \caption{Retrieval results (Answer Coverage at Top-20/100) on 
         Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]]. \vspace{-0.35cm}
        
          \label{tab:openqa}}}
    
        \ttabbox{\resizebox{0.35\textwidth}{!}
     {
\begin{tabular}{rrr|r} \hline \hline
\textbf{Corpus Size} & \textbf{Dim} & \textbf{Search} &
\textbf{Gain}  
\\ \hline	
250 Million & 768 & KNN & +18.4\% \\
8 Billion & 64 & KNN & +14.2\% \\
8 Billion & 64 & ANN & +15.5\% \\ \hline 

\end{tabular}
}}
      {\caption{Relative gains in the first stage retrieval of a commercial search engine. The gains are from changing the training of a production DR model to ANCE. \vspace{-0.123cm}
```

## Ursprüngliches Modellzitat

```text
Retrieval results (Answer Coverage at Top-20/100) on Natural Questions (NQ) and Trivial QA (TQA) in the setting from ~[[CITE:karpukhin2020dense]].
```

Ursprünglicher Kandidat: Index 1 im gespeicherten Relationspayload, `EVALUATES_ON(document:W3038572442, mention:m_9e85ca5fde092683)`.

## Dokumentierte Rekonstruktion

- Reparaturversion: `experimental-whitespace-v1`
- Status: `repaired`
- Änderung: ausschließlich Wiederherstellung von `\n         ` zwischen `on` und `Natural`; keine Zeichen außerhalb des Whitespace geändert.
- Rekonstruierter eindeutiger Originalspan: `[34,190)`.
- Baseline: `ZERO_OCCURRENCES`.
- Nach Rekonstruktion: akzeptiert durch unverändertes V2-Alignment, Endpoint-Containment, Ontologie- und Evidenzgate.

## Kontextinventur für das wissenschaftliche Review

Vorhanden:

- Die Natural-Questions-Mention steht exakt im Caption-Text.
- Die Formulierung `Retrieval results ... on Natural Questions` ist vorhanden.
- Die Caption nennt die Metrik und zusätzlich TQA sowie einen Cite-Key.

Fehlend oder nicht eindeutig zuordenbar:

- Kein explizit benanntes evaluiertes System bzw. Verfahren im Evidenzspan.
- Keine Ergebniswerte für Natural Questions im verfügbaren lokalen Kontext.
- Der folgende Tabellenkörper enthält andere Spalten und erhält danach eine andere Caption; er liefert keinen eindeutig zugeordneten NQ-Beleg.
- Kein zusätzlicher assertion-spezifischer Beleg gegenüber `EVALUATES_ON(..., Trivial QA)` und `REPORTS_METRIC(...)`.

Diese Inventur ist **keine** Entscheidung, ob `EVALUATES_ON` unterstützt ist.

## Menschliche Bewertung

Genau eine Option markieren:

- [ ] `supported`
- [ ] `unsupported`
- [ ] `unclear`

Kommentar:

> 

Reviewer-ID/Kürzel: ____________________

Review-Datum: ____________________

## Provenienz

- Originalchunk: `../v2-multichunk-01/three_chunks.jsonl:1`
- Dokumentmetadaten: `data/manifest.jsonl:108`
- Mention-Auflösung: `../v2-multichunk-01/baseline.v2.jsonl:7`
- Originalmodellpayload: `../v2-pilot-03/cache/llm/5d692f6660892c61240fcbd6f208ce8af706a450b65c0140281de208df4c65cb.json:1`
- Rekonstruktionsaudit: `../v2-pilot-04/repair.audit.json:12-18`
- Akzeptierter Experimentalrecord: `../v2-pilot-04/experimental.jsonl:4`
- Ontologiedefinition: `ontology.yaml:169-183`
