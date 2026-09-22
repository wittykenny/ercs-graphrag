from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select case studies from experiment details.")
    parser.add_argument("--details", default="results/details.csv")
    parser.add_argument("--output-md", default="results/case_studies.md")
    parser.add_argument("--output-csv", default="results/case_studies.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--baseline", default="vector_rag")
    parser.add_argument("--method", default="ucp_ercs_graphrag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        out_dir = Path(args.output_dir)
        args.output_md = str(out_dir / "case_studies.md")
        args.output_csv = str(out_dir / "case_studies.csv")
    rows = read_rows(Path(args.details))
    grouped = group_by_query(rows)
    cases = []
    for query_id, method_rows in grouped.items():
        baseline = method_rows.get(args.baseline)
        ercs = method_rows.get("ercs_graphrag")
        method = method_rows.get(args.method)
        if not baseline or not method:
            continue
        gain_vs_baseline = metric(method, "evidence_coverage@5") - metric(baseline, "evidence_coverage@5")
        gain_vs_ercs = metric(method, "evidence_coverage@5") - metric(ercs, "evidence_coverage@5") if ercs else 0.0
        multi_hop_gain = metric(method, "first_two_evidence_hit@5") - metric(baseline, "first_two_evidence_hit@5")
        score = gain_vs_baseline + gain_vs_ercs + multi_hop_gain
        if score > 0:
            cases.append(
                {
                    "query_id": query_id,
                    "score": score,
                    "gain_vs_baseline": gain_vs_baseline,
                    "gain_vs_ercs": gain_vs_ercs,
                    "multi_hop_gain": multi_hop_gain,
                    "baseline": baseline,
                    "ercs": ercs,
                    "method": method,
                }
            )
    cases.sort(key=lambda x: x["score"], reverse=True)
    cases = cases[: args.limit]
    write_case_csv(Path(args.output_csv), cases)
    write_case_markdown(Path(args.output_md), cases, args.baseline, args.method)
    print(f"selected {len(cases)} cases")
    print(f"saved {args.output_csv}")
    print(f"saved {args.output_md}")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_by_query(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], {})[row["method"]] = row
    return grouped


def metric(row: dict | None, name: str) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get(name, 0.0) or 0.0)
    except ValueError:
        return 0.0


def write_case_csv(path: Path, cases: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    fields = [
        "query_id",
        "question_type",
        "query",
        "answer",
        "gain_vs_baseline",
        "gain_vs_ercs",
        "multi_hop_gain",
        "baseline_covered",
        "method_covered",
        "baseline_top_titles",
        "method_top_titles",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            baseline = case["baseline"]
            method = case["method"]
            writer.writerow(
                {
                    "query_id": case["query_id"],
                    "question_type": method.get("question_type", ""),
                    "query": method.get("query", ""),
                    "answer": method.get("answer", ""),
                    "gain_vs_baseline": case["gain_vs_baseline"],
                    "gain_vs_ercs": case["gain_vs_ercs"],
                    "multi_hop_gain": case["multi_hop_gain"],
                    "baseline_covered": baseline.get("covered_evidence_ids", ""),
                    "method_covered": method.get("covered_evidence_ids", ""),
                    "baseline_top_titles": baseline.get("top_titles", ""),
                    "method_top_titles": method.get("top_titles", ""),
                }
            )


def write_case_markdown(path: Path, cases: list[dict], baseline_name: str, method_name: str) -> None:
    path.parent.mkdir(exist_ok=True)
    lines = ["# Case Studies", ""]
    for idx, case in enumerate(cases, 1):
        baseline = case["baseline"]
        ercs = case.get("ercs")
        method = case["method"]
        lines.extend(
            [
                f"## Case {idx}: query_id={case['query_id']}",
                "",
                f"- Question type: `{method.get('question_type', '')}`",
                f"- Query: {method.get('query', '')}",
                f"- Gold answer: {method.get('answer', '')}",
                f"- Gain vs {baseline_name}: {case['gain_vs_baseline']:.4f}",
                f"- Gain vs ercs_graphrag: {case['gain_vs_ercs']:.4f}",
                f"- Multi-hop gain: {case['multi_hop_gain']:.4f}",
                "",
                f"### {baseline_name}",
                "",
                f"- Covered evidence ids: `{baseline.get('covered_evidence_ids', '')}`",
                f"- Evidence coverage: `{baseline.get('evidence_coverage@5', '')}`",
                f"- Top titles: {baseline.get('top_titles', '')}",
                "",
            ]
        )
        if ercs:
            lines.extend(
                [
                    "### ercs_graphrag",
                    "",
                    f"- Covered evidence ids: `{ercs.get('covered_evidence_ids', '')}`",
                    f"- Evidence coverage: `{ercs.get('evidence_coverage@5', '')}`",
                    f"- Top titles: {ercs.get('top_titles', '')}",
                    "",
                ]
            )
        lines.extend(
            [
                f"### {method_name}",
                "",
                f"- Covered evidence ids: `{method.get('covered_evidence_ids', '')}`",
                f"- Evidence coverage: `{method.get('evidence_coverage@5', '')}`",
                f"- Top titles: {method.get('top_titles', '')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
