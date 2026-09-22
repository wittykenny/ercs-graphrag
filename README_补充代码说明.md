# Project 3 补充代码包说明

本补丁包用于给 GraphRAG / ERCS 项目补充 6 类论文需要的代码能力：

1. `configs/ercs_hotpotqa_500.yaml`：统一配置文件；
2. `reproduce_all.py`：一键复现实验入口；
3. `ablation_experiment.py`：消融实验/结果汇总；
4. `evaluate_triple_quality.py`：三元组质量自动评估与人工复核表；
5. `export_paper_tables.py`：论文表格自动导出；
6. `reranker_baseline.py`：Cross-Encoder 强基线，默认适配 4070 Laptop。

## 使用方法

把本补丁包里的所有文件复制到 `New project 3` 根目录，目录结构应类似：

```text
New project 3/
  configs/ercs_hotpotqa_500.yaml
  scripts/ercs_exp_utils.py
  reproduce_all.py
  ablation_experiment.py
  evaluate_triple_quality.py
  export_paper_tables.py
  reranker_baseline.py
```

安装补充依赖：

```bash
pip install pyyaml scikit-learn openpyxl
```

如果要运行 Cross-Encoder 强基线，再安装：

```bash
pip install sentence-transformers torch
```

4070 Laptop 建议参数已经写在配置文件中：

```yaml
batch_size: 16
max_length: 512
use_fp16: true
model_name: BAAI/bge-reranker-base
```

## 推荐运行顺序

只生成论文表格、消融汇总、三元组质量审计：

```bash
python reproduce_all.py --config configs/ercs_hotpotqa_500.yaml --steps ablation,triple_quality,export
```

包含强基线：

```bash
python reproduce_all.py --config configs/ercs_hotpotqa_500.yaml --steps ablation,triple_quality,reranker,export
```

调用原项目已有脚本并补充所有分析：

```bash
python reproduce_all.py --config configs/ercs_hotpotqa_500.yaml --steps existing,ablation,triple_quality,reranker,export
```

## 输出结果

主要输出目录：

```text
results_ablation_500/
  ablation_results.csv
  ablation_results.json

results_triple_quality_500/
  triple_quality_sample.csv
  triple_quality_summary.json

results_cross_encoder_hotpotqa_500_new/
  details.csv
  summary.json

results_paper_tables/
  table_main_results_all_summaries.csv
  table_significance_tests.csv
  table_multi_seed_summary.csv
  table_triple_quality_samples.csv
  paper_tables.xlsx
```

## 注意

- `evaluate_triple_quality.py` 的自动评估是启发式评估，论文中应称为“自动预筛 + 人工复核模板”，不要把它直接说成人工准确率。
- `reranker_baseline.py` 如果无法加载 `sentence-transformers`，会自动退化为 TF-IDF fallback，并在 `summary.json` 中记录。
- `reproduce_all.py` 中对原有脚本的调用比较保守。如果你原来的 `run_experiment.py` 支持参数，可以在 `configs/ercs_hotpotqa_500.yaml` 的 `reproduce.commands` 中补充参数。
