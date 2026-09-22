from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewer-oriented diagnostic tables.")
    parser.add_argument("--main-results", default="results_hotpotqa_500")
    parser.add_argument("--multihop-results", default="results_clean_test_500")
    parser.add_argument("--cross-encoder-results", default="results_cross_encoder_hotpotqa_500_new")
    parser.add_argument("--output-dir", default="results_review_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_main_delta_table(Path(args.main_results), out_dir / "main_delta_vs_baselines.csv")
    export_multihop_boundary_table(Path(args.multihop_results), out_dir / "multihop_boundary.csv")
    export_cost_table(Path(args.main_results), Path(args.cross_encoder_results), out_dir / "method_cost_table.csv")
    print(f"saved diagnostics to {out_dir}")


def export_main_delta_table(result_dir: Path, out_path: Path) -> None:
    summary = read_json(result_dir / "summary.json", {})
    methods = [
        "vector_rag",
        "bm25_rag",
        "hybrid_rag",
        "hybrid_ucp_ercs_graphrag",
        "guarded_hybrid_ucp_ercs_graphrag",
        "safe_hybrid_ucp_ercs_graphrag",
        "cross_encoder_rag",
        "ablation_sim_only",
        "ablation_sim_er",
        "ablation_sim_er_rw",
        "ablation_sim_er_rw_c",
        "ablation_sim_er_rw_c_pcs",
        "ablation_hybrid_only",
        "ablation_hybrid_ucp",
        "ablation_guarded_hybrid_ucp",
    ]
    vector = get_metric(summary, "vector_rag", "evidence_coverage@5")
    hybrid = get_metric(summary, "hybrid_rag", "evidence_coverage@5")
    rows = []
    for method in methods:
        if method not in summary:
            continue
        ec = get_metric(summary, method, "evidence_coverage@5")
        rows.append(
            {
                "method": method,
                "evidence_coverage@5": ec,
                "mrr": get_metric(summary, method, "mrr"),
                "precision@5": get_metric(summary, method, "precision@5"),
                "query_time_ms": get_metric(summary, method, "query_time_ms"),
                "delta_vs_vector": ec - vector if vector else "",
                "delta_vs_hybrid": ec - hybrid if hybrid else "",
                "interpretation": interpretation(method, ec, vector, hybrid),
            }
        )
    write_csv(out_path, rows)


def export_multihop_boundary_table(result_dir: Path, out_path: Path) -> None:
    summary = read_json(result_dir / "summary.json", {})
    bm25 = get_metric(summary, "bm25_rag", "evidence_coverage@5")
    hybrid = get_metric(summary, "hybrid_rag", "evidence_coverage@5")
    rows = []
    for method, metrics in summary.items():
        ec = float(metrics.get("evidence_coverage@5", 0.0))
        rows.append(
            {
                "dataset": "MultiHop-RAG",
                "method": method,
                "evidence_coverage@5": ec,
                "mrr": metrics.get("mrr", 0.0),
                "first_two_evidence_hit@5": metrics.get("first_two_evidence_hit@5", 0.0),
                "answer_accuracy": metrics.get("answer_accuracy", 0.0),
                "delta_vs_bm25": ec - bm25 if bm25 else "",
                "delta_vs_hybrid": ec - hybrid if hybrid else "",
                "paper_role": "robustness/boundary evidence, not main positive claim",
            }
        )
    write_csv(out_path, rows)


def export_cost_table(main_dir: Path, cross_dir: Path, out_path: Path) -> None:
    rows = []
    details = read_csv(main_dir / "details.csv")
    by_method: dict[str, list[float]] = {}
    for row in details:
        try:
            by_method.setdefault(row["method"], []).append(float(row.get("query_time_ms") or 0.0))
        except (KeyError, ValueError):
            continue
    summary = read_json(main_dir / "summary.json", {})
    for method, vals in sorted(by_method.items()):
        rows.append(
            {
                "method": method,
                "avg_query_time_ms": mean(vals) if vals else 0.0,
                "evidence_coverage@5": get_metric(summary, method, "evidence_coverage@5"),
                "cost_type": "in-process retrieval/rerank",
                "interpretability": interpretability(method),
            }
        )
    ce_summary = read_json(cross_dir / "summary.json", {})
    if ce_summary:
        rows.append(
            {
                "method": ce_summary.get("backend", "external_cross_encoder"),
                "avg_query_time_ms": ce_summary.get("avg_query_time_ms", ""),
                "evidence_coverage@5": ce_summary.get("evidence_coverage_at_5", ""),
                "cost_type": "separate cross-encoder script",
                "interpretability": "low semantic interaction score; high performance baseline",
            }
        )
    write_csv(out_path, rows)


def interpretation(method: str, ec: float, vector: float, hybrid: float) -> str:
    if method == "hybrid_rag":
        return "main recall gain source over vector"
    if method == "safe_hybrid_ucp_ercs_graphrag":
        return "no-regret hybrid-preserving explanation layer" if ec >= hybrid else "unexpected regression; check preserve_base_top_n"
    if "ucp" in method and hybrid:
        return "graph reranking marginal gain" if ec >= hybrid else "graph reranking does not exceed hybrid"
    if "cross_encoder" in method:
        return "strong high-cost reranking upper baseline"
    return "baseline or ablation"


def interpretability(method: str) -> str:
    if "cross_encoder" in method:
        return "low"
    if "ucp" in method or "ercs" in method or "graph" in method:
        return "high: entity/relation/community/path features"
    if "hybrid" in method:
        return "medium: lexical+dense score components"
    return "medium"


def get_metric(summary: dict, method: str, metric: str) -> float:
    try:
        return float(summary.get(method, {}).get(metric, 0.0))
    except (TypeError, ValueError):
        return 0.0


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
