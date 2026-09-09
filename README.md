# Graph retrieval vs Vector Retrieval
**Does Graph retrieval outperforms vector retrieval when it comes to questions that require several jumps over related entities?**

## Corpus Construction
This project is based on the idea that graph retrieval is better than vector retrieval for questions requiring two or three hops across related entities. This claim can only be tested if the documents are genuinely related. Before writing any extraction code, I therefore checked whether the corpus supported multi-hop paths.

### Domain
The corpus covers *retrieval-augmented generation literature*, ranging from open-domain question answering to graph-based retrieval. This domain was selected due to three key properties: a dense internal citation structure, an explicit lineage of 'extends_method' running through the papers, and named datasets and benchmarks that provide the ontology with concrete information to extract.

### Method
Seven anchor papers form the basis of the corpus. Expansion runs in both directions: backwards through each anchor's referenced works and forwards through works that cite it. The metadata comes from the OpenAlex API.

OpenAlex ID	Paper	Year
W2962985038	Reading Wikipedia to Answer Open-Domain Questions (DrQA)	2017
W2947497897	Latent Retrieval for Weakly Supervised Open Domain QA (ORQA)	2019
W2988841832	Generalization through Memorization: Nearest Neighbor LMs	2019
W3007672467	REALM: Retrieval-Augmented Language Model Pre-Training	2020
W3015883388	Dense Passage Retrieval for Open-Domain QA (DPR)	2020
W3039017601	Leveraging Passage Retrieval with Generative Models (FiD)	2021
W3098425262	Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks	2020

The expansion produced 1,598 candidates. However, selecting from this pool based on either citation count or "number of anchors touched" is ineffective: as most candidates touch exactly one anchor, the ranking system becomes arbitrary. Instead, the candidates are treated as a graph, with an undirected edge wherever one candidate cites another, and reduced to its k-core. Works with fewer than k internal neighbours are repeatedly removed until the set is stable, since removing one work can push its neighbours below the threshold. Anchors are exempt.

Core	Works remaining
1-core	1,598
2-core	1,553
3-core	1,459

The one-core figure is meaningless: by construction, every candidate has at least one edge to an anchor. The fact that 1,459 works survive the 3-core process is a clear indication that the domain is densely connected.

The final corpus comprises the 200 works with the highest 4-core degree. Thirteen IDs returned no record as OpenAlex had merged or deleted them upstream; however, they still appear in the reference lists of other works. This leaves 187.

### Result
Metric	Value
Works in corpus	187
Reference edges total	13,764
Edges landing inside the corpus	2,247 (16.3%)
Median internal share per work	20.5%
Works with no internal reference	23

Internal citation density is the deciding factor: it indicates how frequently a reference points to another piece of work within the corpus rather than outside of it. Below 15%, there are too few internal paths for multi-hop questions to have an answer. The 23 works with no outgoing internal references can be accessed, but they are dead ends for traversal.

Every work is recorded in the data/manifest.json file, which contains the SHA-256 of the stored record and the timestamp of its retrieval. This allows a later benchmark run to prove which snapshot it measured against. OpenAlex updates records continuously. Without the manifest, a changed result could not be attributed to the code or the data.

### Full-text availability
The citation graph is derived from the metadata alone. Full text is only needed for the semantic edges,'uses_dataset', 'evaluates_on' and 'extends_method', so the extent of its coverage is a separate issue.

Source	Works	Share
arXiv (LaTeX source available)	75	40%
ACL Anthology PDF	26	14%
Other open PDF	54	29%
Landing page only, no PDF	32	17%
Reachable full text	155	83%

The ingestion process is staged accordingly: first the arXiv subset, since the LaTeX source preserves the section structure and citation keys. Next come the ACL PDFs, which share a single production pipeline and are parsed uniformly. Finally, the remaining PDFs are processed, as these span many publishers and will inevitably produce errors. Developing the ontology against the cleanest subset helps to avoid mistaking parser artefacts for extraction errors.

### What went wrong?
Initially, every anchor resolved to the wrong record. The first pass looked up the anchors by their arXiv DOI (10.48550/arxiv.NNNN.NNNNN). OpenAlex holds both an arXiv preprint record and a merged published record for many papers, and the DOI lookup returns the preprint. These records contain only a fraction of the citations, and in some cases have no reference list at all. QLoRA resolved to a record with zero references, which is impossible for a paper to have.

The consequence is subtle and would have been difficult to trace later on. Works found via citations are attached to the preprint ID, while the reference lists of other corpus members point to the published ID. The same paper exists under two identities with no connection between them, causing the graph to fall apart. This is the entity resolution problem that the project is designed to solve, and it is encountered one level below the application layer.

The solution was to resolve the anchors using a title search instead, with the candidate records printed for manual selection. The first attempt at this also failed because passing sort=cited_by_count:desc overrides OpenAlex's relevance ranking. This means that a title lookup returns every work containing those words, with the most-cited work appearing first. For example, searching for "Dense Passage Retrieval" returned a paper on Raman spectroscopy. Disabling the sort parameter and switching from search to title.search resolved the issue.

The first selection rule was arbitrary. Ranking candidates based on the number of anchors they are connected to becomes problematic when most of them are connected to exactly one. In this case, the code falls back on sorting by ID. This was replaced by the k-core approach, which has the useful property of optimising the same quantity that the density report measures.

Both corrections together:

	Anchors	Works	Density	Median per work	Isolated
First attempt	5 (preprint records)	142	6.1%	4.9%	37
Current	7 (merged records)

This caveat still stands. Ranking measures each work's position within the full set of 1,553 candidate works, rather than within the final selection of 187. The 187 works with the highest degree in the large graph are not necessarily the 187 works with the highest density. Recomputing the degree on the induced subgraph and iterating would be the correct solution, but with a measured density of 16.3%, it was not worth the added complexity.

### Cost
The OpenAlex API has a free metered tier with a daily usage limit of $1.00, and the cost of each response is reported separately. A full corpus build comprises 25 calls at approximately $0.0025 each. The client tracks cumulative spending and stops before the configurable limit is exceeded.

### Reproducing

```
python -m src.ingest.fetch --resolve                    # resolve anchors, pick W-IDs
python -m src.ingest.fetch --dry-run                    # inspect core sizes
python -m src.ingest.fetch --citing-per-anchor 800 \
    --limit 200 --min-degree 4                          # build
python -m src.ingest.inspect_sources --fix-manifest     # full-text audit

```

Raw records and derived artefacts are ignored. However, anchors.txt and data/manifest.json are versioned, meaning that the corpus can be reconstructed without storing several hundred megabytes in the repository.
