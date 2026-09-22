# ERCS GraphRAG for Multi-Hop Evidence Retrieval

An undergraduate research prototype studying whether entity, relation and community signals improve evidence retrieval for multi-hop question answering.

**At a glance:** the project implements ERCS/UCP-ERCS graph-based reranking alongside BM25, dense and hybrid retrieval, using MultiHop-RAG and a converted HotpotQA subset. The archived MultiHop-RAG evaluation covers 500 questions; the main HotpotQA artifact covers **100**, despite its directory suffix. Graph reranking does not consistently outperform BM25 or Hybrid. The contribution represented by this repository is the experimental implementation and comparative analysis; it does not introduce a new foundation model or claim a published paper. [Results](#main-results) · [Reproduction](#reproduction) · [Limitations](#limitations).

## Overview

Multi-hop questions require evidence from several documents. This prototype tests whether a lightweight knowledge graph can help rank evidence beyond lexical or embedding similarity. It compares controlled retrieval variants and examines graph quality, robustness and failure cases. It is a simplified GraphRAG implementation, not a full reproduction of Microsoft GraphRAG.

## Research Objectives

- Compare lexical, dense, hybrid and graph-based evidence retrieval.
- Examine the effect of entity importance, relation weights, community signals and query-conditioned graph signals.
- Assess whether preserving strong Hybrid candidates reduces degradation from graph reranking.

## Method

- Split corpus text into chunks; extract triples with rules or an optional OpenAI-compatible LLM endpoint.
- Construct a weighted undirected graph. Entity importance combines PageRank and degree centrality; communities use greedy modularity.
- Retrieve candidates with BM25, Sentence-Transformers or their hybrid.
- ERCS/UCP-ERCS combine semantic and graph signals; Hybrid and Guarded variants rerank Hybrid candidates and optionally preserve leading candidates.
- Evaluate evidence matching and a simple answer-generation pipeline. LLM extraction, generation and summarization are optional and require separate credentials.

See `ercs_graphrag/retrieval.py`, `graph.py`, `extract.py` and `evaluate.py`. The reusable components and evaluation scripts establish the repository's implementation scope; individual authorship of every inherited component has not been independently verified.

## Data

| Local dataset | Questions | Corpus documents | Archived main evaluation |
| --- | ---: | ---: | --- |
| MultiHop-RAG | 2,556 | 609 | 500 test questions; first 300 documents |
| HotpotQA converted distractor validation subset | 500 | 4,937 | 100 local test questions; document cap 5,000 |

Questions are shuffled with seed 42 and split 60%/20%/remainder into train/dev/test. The main HotpotQA artifact applies this local split to the converted validation subset. This is not an official HotpotQA test result. The MultiHop-RAG parameter file records a 50-question development search. Gold evidence is not injected into the retrieval corpus in the recorded main configurations.

Raw data and derived text are excluded from the intended release. Sources, preparation, licensing references and protocol details are in [data/README.md](data/README.md). This release contains only reviewed source, documentation and aggregate results; the original working archive and its history are not included. See the [release audit](docs/release_audit.md).

## Repository Structure

```text
README.md
requirements.txt
.env.example
configs/                         # Existing supplementary experiment configuration
ercs_graphrag/                    # Data, extraction, graph, retrieval, evaluation, LLM client
scripts/ercs_exp_utils.py         # Shared supplementary-experiment utilities
run_experiment.py                # Main CLI entry point
prepare_external_dataset.py      # External dataset conversion
param_search.py                  # Development parameter search
multi_seed_experiment.py         # Repeated split-seed experiments
significance_test.py             # Paired bootstrap and permutation tests
*_experiment.py, *_analysis.py    # Supplementary experiments and analyses
data/README.md                   # Acquisition and protocol
results/main_results.csv         # Metrics transcribed from archived summaries
results/best_params.json         # Existing development-search parameters
results*/                        # Compact original result summaries
docs/                           # Provenance, reproducibility and release review
```

Research scripts remain in their original locations so imports and relative paths continue to work. Run commands from the repository root. Local thesis documents, generated notebooks, datasets, caches and detailed outputs are not release material.

## Installation

Python 3.10 or newer is required by the source syntax. Only the maintenance environment described below was inspected; a fresh-environment installation has not been validated.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are direct project requirements rather than a full environment dump. Historical versions are unknown. PyTorch/CUDA installation depends on the target hardware; no CUDA build is hard-coded. The TF-IDF path does not need a downloaded embedding model. Neural retrieval uses `sentence-transformers/all-MiniLM-L6-v2`; the main CrossEncoder implementation defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2`. The supplementary reranker config selects `BAAI/bge-reranker-base`.

## Reproduction

These commands create new outputs and do not replace archived results. They reconstruct the documented protocol; exact historical neural metrics are not guaranteed because model revisions and complete runtime metadata were not saved.

1. Obtain the two MultiHop-RAG files as described in [data preparation](data/README.md), then verify the local sample counts before running:

```sh
python -c "import json; from pathlib import Path; p=Path('data/raw'); assert len(json.loads((p/'MultiHopRAG.json').read_text(encoding='utf-8')))==2556; assert len(json.loads((p/'corpus.json').read_text(encoding='utf-8')))==609; print('Data counts verified')"
```

2. Run a small pipeline check (its TF-IDF scores are not the archived neural results):

```sh
python run_experiment.py --data-dir data/raw --split test --seed 42 --max-queries 3 --max-docs 10 --embedding-backend tfidf --methods bm25_rag,hybrid_rag,ucp_ercs_graphrag --processed-dir data/processed_smoke --output-dir outputs/smoke
```

3. Reconstruct the MultiHop-RAG main evaluation protocol:

```sh
python run_experiment.py --data-dir data/raw --split test --seed 42 --max-queries 500 --max-docs 300 --top-k 5 --embedding-backend sbert --params-file results/best_params.json --methods bm25_rag,hybrid_rag,vector_rag,graph_rag,ercs_graphrag,ucp_ercs_graphrag,hybrid_ucp_ercs_graphrag --processed-dir data/processed_reproduction --output-dir outputs/multihop_reproduction
```

For the recorded HotpotQA sampling protocol:

```sh
python prepare_external_dataset.py --dataset hotpotqa --split validation --max-queries 500 --output-dir data/hotpotqa_raw_500
python run_experiment.py --data-dir data/hotpotqa_raw_500 --split test --seed 42 --max-queries 500 --max-docs 5000 --top-k 5 --embedding-backend sbert --methods bm25_rag,hybrid_rag,vector_rag,ucp_ercs_graphrag,hybrid_ucp_ercs_graphrag --processed-dir data/processed_hotpotqa_reproduction --output-dir outputs/hotpotqa_reproduction
```

The second command evaluates 100 questions after splitting. It is a protocol reconstruction, not an assertion that all archived hyperparameters were recovered. Using `--split all` would instead evaluate all 500 converted questions and constitute a different experiment.

Optional LLM runs: copy `.env.example` to local `.env`, set your credentials and model, and pass `--llm-env-file .env` with the chosen LLM options. These calls may incur API charges; no API calls are needed for the commands above. Do not commit credentials. The legacy `reproduce_all.py` configuration uses several bare default commands and can overwrite outputs; it is not the recommended main reproduction path.

## Main Results

Unchanged archived summaries, rounded to four decimals below. EC@5 denotes mean evidence coverage, using the implementation's title-or-text match. `answer_accuracy` is normalized gold-answer substring containment, not official exact match or F1.

| Dataset (questions) | Method | EC@5 | MRR | Answer accuracy |
| --- | --- | ---: | ---: | ---: |
| MultiHop-RAG (500) | BM25 | 0.4193 | 0.6265 | 0.1860 |
| MultiHop-RAG (500) | Hybrid | 0.3858 | 0.6236 | 0.1940 |
| MultiHop-RAG (500) | Vector | 0.2977 | 0.5065 | 0.1660 |
| MultiHop-RAG (500) | ERCS | 0.3028 | 0.4749 | 0.1820 |
| MultiHop-RAG (500) | UCP-ERCS | 0.3113 | 0.4621 | 0.1820 |
| MultiHop-RAG (500) | Hybrid UCP-ERCS | 0.3823 | 0.5840 | 0.1860 |
| HotpotQA subset (100) | BM25 | 0.7433 | 0.8995 | 0.2500 |
| HotpotQA subset (100) | Hybrid | 0.8217 | 0.9250 | 0.2500 |
| HotpotQA subset (100) | Vector | 0.7467 | 0.8867 | 0.2200 |
| HotpotQA subset (100) | ERCS | 0.7550 | 0.9003 | 0.2300 |
| HotpotQA subset (100) | UCP-ERCS | 0.7667 | 0.9028 | 0.2400 |
| HotpotQA subset (100) | Hybrid UCP-ERCS | 0.8200 | 0.9265 | 0.2500 |

Sources: [MultiHop-RAG summary](results_clean_test_500/summary.json), [HotpotQA summary](results_hotpotqa_500/summary.json), and [machine-readable table](results/main_results.csv). [Provenance hashes](docs/result_provenance.json) identify the original summary and detail files used for this transcription. Original detailed files remain local.

BM25 has the highest evidence coverage among the listed MultiHop-RAG methods. Hybrid is strongest on HotpotQA evidence coverage; Hybrid UCP-ERCS has a slightly higher MRR but slightly lower coverage. These results do not support a claim of universal graph-retrieval improvement. Separate CrossEncoder artifacts have different protocols or evaluators and are not combined into this table.

## Ablation / Robustness

- Archived main HotpotQA summaries include incremental graph-signal ablations. `ablation_experiment.py` also aggregates existing result files; aggregation alone is not a new controlled experiment.
- [Multi-seed summary](results_multi_seed_500/multi_seed_summary.csv) reports three seeds (42, 2024, 2025). Seeds change question partitions; these are not three independent neural training runs.
- `significance_test.py` implements paired bootstrap intervals and paired permutation tests. Archived significance files are retained without promoting unverified statistical claims.
- Recorded configurations disable gold-evidence injection. This documents a safeguard, not a comprehensive proof of no leakage or test-set tuning.
- Triple-quality summaries include automatic heuristics and a separate manual-labelled artifact. The automatic HotpotQA summary explicitly reports zero manual labels; it must not be described as human validation.

## Limitations

- Small question subsets and truncated/pooled corpora limit generalization. Baselines can outperform graph variants.
- Rule-extracted graphs contain noisy entities and relations; graph scores may amplify extraction errors.
- Evidence matching accepts document-title matches, which can overestimate sentence-level support. Answer accuracy and lexical faithfulness are proxies. Archived faithfulness values of 1.0 do not establish factual correctness.
- Historical model revisions, full commands and package versions are incomplete. Cache reuse does not validate that cached triples match a new corpus/configuration; use new cache directories.
- The legacy downloader can silently substitute toy data on failure. Verify data before running. Optional CrossEncoder/LLM paths also have fallback behavior that must be checked before interpreting results.
- No full-scale neural reproduction, paid LLM run or fresh-environment installation was performed during repository maintenance.
- Code ownership and a redistribution license require author confirmation; no LICENSE has been invented.

## Reproducibility

Default split seed: 42. Existing configuration: `configs/ercs_hotpotqa_500.yaml`; direct main-CLI flags are documented above. No trained checkpoints are distributed. Third-party model downloads and upstream data access are required for neural runs. [Reproducibility notes](docs/reproducibility.md) distinguish the observed maintenance environment from the unknown historical experiment environment.

## Citation

This repository contains an undergraduate research/course project by Haoming Luo.
No formal publication or project BibTeX citation is claimed. Please acknowledge the original dataset and model authors when using their resources.

## Author

Haoming Luo

Zhongnan University of Economics and Law

B.Eng. in Artificial Intelligence & B.Mgt. in Accounting
