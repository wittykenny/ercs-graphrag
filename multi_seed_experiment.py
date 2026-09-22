from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed ERCS-GraphRAG experiments and aggregate results.")
    parser.add_argument("--seeds", default="42,2024,2025", help="Comma-separated random seeds.")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--params-file", default="results/best_params.json")
    parser.add_argument("--embedding-backend", default="sbert", choices=["auto", "sbert", "tfidf"])
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--methods", default="bm25_rag,hybrid_rag,vector_rag,graph_rag,ercs_graphrag,ucp_ercs_graphrag")
    parser.add_argument("--output-root", default="results_multi_seed")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--include-evidence-chunks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        out_dir = output_root / f"seed_{seed}"
        processed_dir = Path("data") / f"processed_seed_{seed}_{args.split}_{args.max_queries}"
        cmd = [
            sys.executable,
            "run_experiment.py",
            "--data-dir",
            args.data_dir,
            "--split",
            args.split,
            "--seed",
            str(seed),
            "--max-queries",
            str(args.max_queries),
            "--max-docs",
            str(args.max_docs),
            "--top-k",
            str(args.top_k),
            "--processed-dir",
            str(processed_dir),
            "--embedding-backend",
            args.embedding_backend,
            "--embedding-model",
            args.embedding_model,
            "--methods",
            args.methods,
            "--output-dir",
            str(out_dir),
        ]
        if args.params_file and Path(args.params_file).exists():
            cmd.extend(["--params-file", args.params_file])
        if args.reuse_cache:
            cmd.append("--reuse-cache")
        if args.include_evidence_chunks:
            cmd.append("--include-evidence-chunks")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    aggregate(output_root, seeds)


def aggregate(output_root: Path, seeds: list[int]) -> None:
    rows: list[dict] = []
    for seed in seeds:
        summary_path = output_root / f"seed_{seed}" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for method, metrics in summary.items():
            for metric, value in metrics.items():
                rows.append({"seed": seed, "method": method, "metric": metric, "value": float(value)})

    raw_path = output_root / "multi_seed_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "method", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["metric"]), []).append(row["value"])

    summary_rows = []
    for (method, metric), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = variance ** 0.5
        else:
            std = 0.0
        summary_rows.append(
            {
                "method": method,
                "metric": metric,
                "mean": mean,
                "std": std,
                "n_seeds": len(values),
            }
        )

    summary_path = output_root / "multi_seed_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "metric", "mean", "std", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
