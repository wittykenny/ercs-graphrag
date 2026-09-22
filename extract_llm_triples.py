from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ercs_graphrag.data import build_chunks, load_dataset, split_qa_dataset
from ercs_graphrag.extract import extract_triples_with_llm
from ercs_graphrag.llm import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and save real LLM triples without running retrieval experiments.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--dev-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--max-docs", type=int, default=120)
    parser.add_argument("--max-chunks", type=int, default=200)
    parser.add_argument("--processed-dir", default="data/processed_llm_triples")
    parser.add_argument("--sample-output", default="results_llm_triples/triple_quality_sample.csv")
    parser.add_argument("--summary-output", default="results_llm_triples/triple_quality_summary.json")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--llm-timeout", type=int, default=90)
    parser.add_argument("--validate-llm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_env_file:
        load_env_file(Path(args.llm_env_file))
    llm = LLMClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=os.getenv(args.llm_api_key_env),
        timeout=args.llm_timeout,
    )
    if not llm.enabled:
        raise SystemExit(
            f"LLM triple extraction requested, but {args.llm_api_key_env} and LLM_MODEL are not both set."
        )
    if args.validate_llm:
        llm.validate_json_array()

    qa_all, corpus = load_dataset(args.data_dir)
    qa = split_qa_dataset(
        qa_all,
        split=args.split,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )[: args.max_queries]
    chunks = build_chunks(corpus, qa, max_docs=args.max_docs)
    chunks = chunks[: args.max_chunks]

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    triples_path = processed_dir / "triples.json"
    triples = load_existing_triples(triples_path) if args.resume else []
    completed = {str(tri.get("chunk_id", "")) for tri in triples if tri.get("chunk_id")}
    for idx, chunk in enumerate(chunks, 1):
        chunk_id = chunk["chunk_id"]
        if chunk_id in completed:
            print(f"[{idx}/{len(chunks)}] skip {chunk_id}", flush=True)
            continue
        extracted = extract_chunk_with_retries(chunk, llm, args.retries, args.retry_sleep)
        for tri in extracted:
            tri["extractor"] = "llm"
            tri["llm_model"] = llm.model
        triples.extend(extracted)
        triples_path.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")
        completed.add(chunk_id)
        print(f"[{idx}/{len(chunks)}] {chunk_id}: +{len(extracted)} triples, total={len(triples)}", flush=True)
    (processed_dir / "llm_extraction_config.json").write_text(
        json.dumps(
            {
                "data_dir": args.data_dir,
                "split": args.split,
                "seed": args.seed,
                "max_queries": args.max_queries,
                "max_docs": args.max_docs,
                "max_chunks": args.max_chunks,
                "llm_model": llm.model,
                "llm_url": llm.url,
                "triple_count": len(triples),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "triple_quality_sample.py",
            "--triples",
            str(triples_path),
            "--sample-size",
            "100",
            "--seed",
            str(args.seed),
            "--output",
            args.sample_output,
            "--summary-output",
            args.summary_output,
        ],
        check=True,
    )
    print(f"saved triples: {triples_path}")
    print(f"triples={len(triples)} chunks={len(chunks)}")


def load_existing_triples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def extract_chunk_with_retries(chunk: dict, llm: LLMClient, retries: int, retry_sleep: float) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return extract_triples_with_llm(chunk, llm, allow_rule_fallback=False)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"LLM extraction failed for {chunk.get('chunk_id')}: {last_error}") from last_error


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"LLM env file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    main()
