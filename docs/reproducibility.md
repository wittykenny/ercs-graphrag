# Reproducibility status

Maintenance host inspected on 2026-09-22: Windows, Python 3.10.11; numpy 1.26.4, pandas 2.3.3, networkx 3.4.2, scikit-learn 1.7.2, sentence-transformers 5.5.1, torch 2.5.1+cu121, matplotlib 3.10.9, PyYAML 6.0.3, openpyxl 3.1.5, datasets 4.8.5. These are installed package metadata, not proof that historical experiments used them or that every optional backend works. CUDA availability has not been validated.

The dependency list includes scipy because NetworkX PageRank and scientific dependencies require it. No evidence supports adding torch-geometric, statsmodels or a separate OpenAI SDK: this code uses urllib for its OpenAI-compatible client. requirements_project3_extra.txt is a retained legacy supplementary list, now covered by requirements.txt. No environment.yml or historical lockfile was present.

Main research code uses relative/configurable data paths; detected personal absolute paths occur in excluded local thesis-editing utilities. These unrelated files are preserved, including a pre-existing syntax error in review_docx_structure_final.py. No model algorithm or original numeric result was changed. One output-path fix adds parents=True when creating the result directory; the README smoke command passed afterwards.

Historical experiment_config.json files record split, seed, query/document caps and evidence inclusion, but not every hyperparameter, model revision or backend. Neural runs and exact numerical reproduction therefore remain unverified. The archived HotpotQA directory holds 100 per-method detail rows and a test split configuration, even though its name ends in 500. The README preserves the actual evidence.

Use fresh output/cache directories. Do not use --reuse-cache across corpus or extraction changes. Do not run the legacy all-in-one runner against archived output directories. Data acquisition must fail visibly or be followed by validation; the existing downloader's toy fallback is documented rather than silently changed.

See release_audit.md for the maintenance checks actually performed.
