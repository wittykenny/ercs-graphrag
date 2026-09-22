from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled LLM-triple extraction experiment.")
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--max-docs", type=int, default=120)
    parser.add_argument("--llm-extract-max-chunks", type=int, default=200)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed_llm_triples")
    parser.add_argument("--output-dir", default="results_llm_triples")
    parser.add_argument("--methods", default="hybrid_rag,hybrid_ucp_ercs_graphrag,guarded_hybrid_ucp_ercs_graphrag")
    parser.add_argument("--params-file", default="results/best_params.json")
    parser.add_argument("--embedding-backend", default="sbert", choices=["auto", "sbert", "tfidf"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--llm-timeout", type=int, default=90)
    parser.add_argument("--validate-llm", action="store_true")
    parser.add_argument("--allow-rule-fallback", action="store_true")
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
        "--max-queries",
        str(args.max_queries),
        "--max-docs",
        str(args.max_docs),
        "--processed-dir",
        args.processed_dir,
        "--output-dir",
        args.output_dir,
        "--embedding-backend",
        args.embedding_backend,
        "--extractor",
        "llm",
        "--llm-extract-max-chunks",
        str(args.llm_extract_max_chunks),
        "--methods",
        args.methods,
    ]
    if args.params_file and Path(args.params_file).exists():
        cmd.extend(["--params-file", args.params_file])
    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])
    if args.llm_base_url:
        cmd.extend(["--llm-base-url", args.llm_base_url])
    if args.llm_api_key_env:
        cmd.extend(["--llm-api-key-env", args.llm_api_key_env])
    if args.llm_env_file:
        cmd.extend(["--llm-env-file", args.llm_env_file])
    if args.llm_timeout:
        cmd.extend(["--llm-timeout", str(args.llm_timeout)])
    if args.validate_llm:
        cmd.append("--validate-llm")
    if args.allow_rule_fallback:
        cmd.append("--llm-extract-allow-rule-fallback")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
