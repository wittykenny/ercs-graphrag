from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small LLM answer-generation evaluation.")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed_llm_eval")
    parser.add_argument("--params-file", default="results/best_params.json")
    parser.add_argument("--output-dir", default="results_llm_eval")
    parser.add_argument("--llm-model", default=None, help="OpenAI-compatible model. Also accepts LLM_MODEL env var.")
    parser.add_argument("--methods", default="vector_rag,ucp_ercs_graphrag")
    parser.add_argument("--embedding-backend", default="sbert", choices=["auto", "sbert", "tfidf"])
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--data-dir",
        args.data_dir,
        "--split",
        args.split,
        "--seed",
        str(args.seed),
        "--max-queries",
        str(args.max_queries),
        "--max-docs",
        str(args.max_docs),
        "--top-k",
        str(args.top_k),
        "--processed-dir",
        args.processed_dir,
        "--embedding-backend",
        args.embedding_backend,
        "--methods",
        args.methods,
        "--answer-generator",
        "llm",
        "--output-dir",
        args.output_dir,
    ]
    if args.params_file and Path(args.params_file).exists():
        cmd.extend(["--params-file", args.params_file])
    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])
    if args.reuse_cache:
        cmd.append("--reuse-cache")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved LLM answer evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
