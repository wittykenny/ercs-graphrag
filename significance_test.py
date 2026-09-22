from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired significance tests for retrieval experiment results.")
    parser.add_argument("--details", default="results/details.csv")
    parser.add_argument("--output-csv", default="results/significance_tests.csv")
    parser.add_argument("--output-json", default="results/significance_tests.json")
    parser.add_argument("--output-dir", default=None, help="Optional directory for default significance outputs.")
    parser.add_argument("--methods", default="vector_rag,ercs_graphrag,ucp_ercs_graphrag")
    parser.add_argument("--metrics", default="evidence_coverage@5,mrr,answer_accuracy,first_two_evidence_hit@5")
    parser.add_argument("--baseline", default="vector_rag")
    parser.add_argument(
        "--pairs",
        default="",
        help=(
            "Optional comma-separated paired comparisons in method:baseline form. "
            "Example: hybrid_ucp_ercs_graphrag:hybrid_rag,guarded_hybrid_ucp_ercs_graphrag:hybrid_rag"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        out_dir = Path(args.output_dir)
        args.output_csv = str(out_dir / "significance_tests.csv")
        args.output_json = str(out_dir / "significance_tests.json")
    rows = read_rows(Path(args.details))
    grouped = group_rows(rows)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rng = random.Random(args.seed)
    results = []
    pairs = parse_pairs(args.pairs)
    if not pairs:
        pairs = [(method, args.baseline) for method in methods if method != args.baseline]
    for method, baseline in pairs:
        for metric in metrics:
            diffs = paired_differences(grouped, method, baseline, metric)
            if not diffs:
                continue
            mean_diff = sum(diffs) / len(diffs)
            ci_low, ci_high = bootstrap_ci(diffs, args.bootstrap_samples, rng)
            p_value = paired_permutation_pvalue(diffs, args.permutation_samples, rng)
            results.append(
                {
                    "baseline": baseline,
                    "method": method,
                    "metric": metric,
                    "n": len(diffs),
                    "mean_diff": mean_diff,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "permutation_p_value": p_value,
                    "significant_0.05": p_value < 0.05,
                }
            )
    write_csv(Path(args.output_csv), results)
    Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output_csv}")
    print(f"saved {args.output_json}")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        key = row.get("original_query_id") or row.get("query_id")
        grouped.setdefault(key, {})[row["method"]] = row
    return grouped


def paired_differences(grouped: dict[str, dict[str, dict]], method: str, baseline: str, metric: str) -> list[float]:
    diffs = []
    for method_rows in grouped.values():
        if method not in method_rows or baseline not in method_rows:
            continue
        diffs.append(to_float(method_rows[method].get(metric)) - to_float(method_rows[baseline].get(metric)))
    return diffs


def parse_pairs(spec: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --pairs item {item!r}; expected method:baseline")
        method, baseline = [part.strip() for part in item.split(":", 1)]
        if not method or not baseline:
            raise ValueError(f"Invalid --pairs item {item!r}; expected method:baseline")
        pairs.append((method, baseline))
    return pairs


def bootstrap_ci(diffs: list[float], samples: int, rng: random.Random) -> tuple[float, float]:
    means = []
    n = len(diffs)
    for _ in range(samples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int(0.025 * (samples - 1))
    high_idx = int(0.975 * (samples - 1))
    return means[low_idx], means[high_idx]


def paired_permutation_pvalue(diffs: list[float], samples: int, rng: random.Random) -> float:
    observed = abs(sum(diffs) / len(diffs))
    if observed == 0:
        return 1.0
    extreme = 0
    for _ in range(samples):
        permuted = [d if rng.random() < 0.5 else -d for d in diffs]
        stat = abs(sum(permuted) / len(permuted))
        if stat >= observed:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


if __name__ == "__main__":
    main()
