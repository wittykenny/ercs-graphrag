from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and summarize failure cases for thesis analysis.")
    parser.add_argument("--details", default="results/details.csv")
    parser.add_argument("--output-md", default="results/failure_cases.md")
    parser.add_argument("--output-csv", default="results/failure_cases.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--baseline", default="vector_rag")
    parser.add_argument("--method", default="ucp_ercs_graphrag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        out_dir = Path(args.output_dir)
        args.output_md = str(out_dir / "failure_cases.md")
        args.output_csv = str(out_dir / "failure_cases.csv")
    rows = read_rows(Path(args.details))
    grouped = group_by_query(rows)
    failures = []
    for query_id, method_rows in grouped.items():
        baseline = method_rows.get(args.baseline)
        method = method_rows.get(args.method)
        ercs = method_rows.get("ercs_graphrag")
        if not baseline or not method:
            continue
        coverage_delta = metric(method, "evidence_coverage@5") - metric(baseline, "evidence_coverage@5")
        mrr_delta = metric(method, "mrr") - metric(baseline, "mrr")
        answer_delta = metric(method, "answer_accuracy") - metric(baseline, "answer_accuracy")
        score = coverage_delta + 0.5 * mrr_delta + 0.5 * answer_delta
        if score < 0:
            failures.append(
                {
                    "query_id": query_id,
                    "score": score,
                    "coverage_delta": coverage_delta,
                    "mrr_delta": mrr_delta,
                    "answer_delta": answer_delta,
                    "baseline": baseline,
                    "ercs": ercs,
                    "method": method,
                    "reason": infer_reason(baseline, method),
                }
            )
    failures.sort(key=lambda x: x["score"])
    failures = failures[: args.limit]
    write_csv(Path(args.output_csv), failures)
    write_markdown(Path(args.output_md), failures, args.baseline, args.method)
    print(f"selected {len(failures)} failures")
    print(f"saved {args.output_csv}")
    print(f"saved {args.output_md}")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_by_query(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        key = row.get("original_query_id") or row.get("query_id")
        grouped.setdefault(key, {})[row["method"]] = row
    return grouped


def infer_reason(baseline: dict, method: dict) -> str:
    reasons = []
    if repeated_titles(method.get("top_titles", "")) > repeated_titles(baseline.get("top_titles", "")):
        reasons.append("higher duplicate-title concentration")
    if metric(method, "mrr") < metric(baseline, "mrr"):
        reasons.append("relevant evidence ranked later")
    if metric(method, "evidence_coverage@5") < metric(baseline, "evidence_coverage@5"):
        reasons.append("lower supporting-evidence coverage")
    if not method.get("covered_evidence_ids"):
        reasons.append("no supporting evidence retrieved")
    return "; ".join(reasons) or "graph reranking did not improve this case"


def repeated_titles(titles: str) -> int:
    items = [x.strip() for x in titles.split("|") if x.strip()]
    return len(items) - len(set(items))


def write_csv(path: Path, failures: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    fields = [
        "query_id",
        "question_type",
        "query",
        "answer",
        "coverage_delta",
        "mrr_delta",
        "answer_delta",
        "reason",
        "baseline_covered",
        "method_covered",
        "baseline_top_titles",
        "method_top_titles",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for failure in failures:
            baseline = failure["baseline"]
            method = failure["method"]
            writer.writerow(
                {
                    "query_id": failure["query_id"],
                    "question_type": method.get("question_type", ""),
                    "query": method.get("query", ""),
                    "answer": method.get("answer", ""),
                    "coverage_delta": failure["coverage_delta"],
                    "mrr_delta": failure["mrr_delta"],
                    "answer_delta": failure["answer_delta"],
                    "reason": failure["reason"],
                    "baseline_covered": baseline.get("covered_evidence_ids", ""),
                    "method_covered": method.get("covered_evidence_ids", ""),
                    "baseline_top_titles": baseline.get("top_titles", ""),
                    "method_top_titles": method.get("top_titles", ""),
                }
            )


def write_markdown(path: Path, failures: list[dict], baseline_name: str, method_name: str) -> None:
    path.parent.mkdir(exist_ok=True)
    lines = ["# Failure Cases", ""]
    for idx, failure in enumerate(failures, 1):
        baseline = failure["baseline"]
        method = failure["method"]
        lines.extend(
            [
                f"## Failure {idx}: query_id={failure['query_id']}",
                "",
                f"- Question type: `{method.get('question_type', '')}`",
                f"- Query: {method.get('query', '')}",
                f"- Gold answer: {method.get('answer', '')}",
                f"- Coverage delta ({method_name} - {baseline_name}): {failure['coverage_delta']:.4f}",
                f"- MRR delta: {failure['mrr_delta']:.4f}",
                f"- Answer delta: {failure['answer_delta']:.4f}",
                f"- Possible reason: {failure['reason']}",
                "",
                f"### {baseline_name}",
                f"- Covered evidence ids: `{baseline.get('covered_evidence_ids', '')}`",
                f"- Top titles: {baseline.get('top_titles', '')}",
                "",
                f"### {method_name}",
                f"- Covered evidence ids: `{method.get('covered_evidence_ids', '')}`",
                f"- Top titles: {method.get('top_titles', '')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def metric(row: dict | None, name: str) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get(name, 0.0) or 0.0)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    main()
