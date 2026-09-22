# Clean release audit

This standalone release was prepared from an existing research archive with the author's authorization. Only the 76 files in `release_manifest.txt` were copied. The original repository, its history and all local data were preserved separately. No previous commits or Git metadata were copied.

## Release scope

Included: research Python source, existing configurations, direct dependency declarations, documentation, placeholder environment variables and compact aggregate results.

Excluded: credentials, raw datasets, processed passages and triples, model weights, detailed predictions, duplicate notebooks, thesis documents, private editing reports and runtime caches. No project license has been selected; author confirmation remains necessary before licensing the code.

## Verification and limits

- Files were copied against the audited manifest and verified byte-for-byte before updating release-specific documentation.
- Source results were preserved. `result_provenance.json` identifies archived source summaries and detail files; excluded detail files are retained in the author's original archive.
- Research Python sources passed syntax checks. The small TF-IDF pipeline passed in the original maintenance environment after an output-directory creation fix.
- Main archived evaluations cover 500 MultiHop-RAG questions and 100 HotpotQA questions per method; directory suffixes do not determine sample size.
- Text-pattern scanning covers credentials, private-key markers, email addresses, phone-like strings and personal absolute paths. This is a bounded audit, not a guarantee against all possible sensitive content.
- No fresh-environment installation, full neural reproduction, paid API run or historical environment reconstruction was performed.

## Application review

The English README states the research problem, implementation scope, datasets, sourced results and limitations. It distinguishes proxy metrics from official benchmark scores and does not claim publication or consistent superiority over baselines. Author contribution attribution and licensing still require confirmation.

## Publication

Suggested name: `ercs-graphrag`. Initial visibility: PRIVATE. GitHub creation/upload depends on successful authentication and checking name availability. Use this standalone repository for publication; do not push the original research archive.
