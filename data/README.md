# Data preparation and availability

Raw datasets, converted passages and graph caches are retained locally and excluded from the intended release. This release has a separate history and does not include the original working archive. See `docs/release_audit.md`.

## MultiHop-RAG

Source: https://github.com/yixuantt/MultiHop-RAG and https://huggingface.co/datasets/yixuantt/MultiHopRAG . The upstream repository states ODC-BY (checked 2026-09-22); this is not a license grant for this project's code or every underlying news article.

The local files contain 2,556 questions and 609 corpus documents. Obtain `MultiHopRAG.json` and `corpus.json` from the upstream dataset and place them in `data/raw/`. The downloader in `ercs_graphrag/data.py` has a legacy toy-data fallback on download failure; therefore manual acquisition and count verification are used in the README reproduction instructions. Never report that fallback as a benchmark experiment.

The code shuffles questions with seed 42, uses 60% train / 20% dev / the remainder test, then caps each requested run. The test partition contains 512 questions and the main stored run uses the first 500. Its corpus is restricted to the first 300 documents. Development parameter search used 50 questions. Evidence chunks are not injected by default. Corpus documents are not split by question partition.

## HotpotQA

Source: https://huggingface.co/datasets/hotpotqa/hotpot_qa (dataset card lists CC-BY-SA-4.0, checked 2026-09-22). The converter reads the distractor validation split, takes the first 500 questions, and deduplicates context passages by title. Local conversion produced 4,937 documents. Supporting facts remain evaluation annotations.

```sh
python prepare_external_dataset.py --dataset hotpotqa --split validation --max-queries 500 --output-dir data/hotpotqa_raw_500
```

The archived `results_hotpotqa_500` run evaluates only 100 questions: its recorded protocol applies the local 60/20/20 split to the 500 converted validation questions, selecting test. This is a pooled distractor-context experiment, not full-Wikipedia retrieval or the official benchmark protocol.

## Models and caches

No model weights are distributed. Sentence-Transformers and CrossEncoder download third-party models when selected. Review the corresponding model cards and terms before use. Processed JSON and GraphML contain derived source text and must remain local. The 2Wiki converter exists, but no 2Wiki result is claimed here.
