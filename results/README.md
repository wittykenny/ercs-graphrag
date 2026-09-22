# Archived results

`main_results.csv` transcribes selected metrics from the unchanged summaries in `results_clean_test_500/` and `results_hotpotqa_500/`. No experiment was rerun to replace these results. `n_queries` comes from counting the original per-method detail rows. Source hashes are in `docs/result_provenance.json`.

The release retains compact summaries, significance tables, multi-seed aggregates and ablation aggregates. Detailed predictions, extracted triples, case studies and spreadsheet exports remain local because they may reproduce third-party text. These can be regenerated after data preparation.

Directory suffixes are historical names, not reliable sample-size metadata. HotpotQA's main stored run has 100 queries per method; the separate CrossEncoder runs use different samples or implementations and must not be inserted into the main comparison as a controlled baseline. The BGE reranker summary uses a separate evaluator and TF-IDF candidate retrieval.

`results/best_params.json` is a development-search artifact used as input by the main MultiHop-RAG command. Do not overwrite it while reproducing evaluation results.
