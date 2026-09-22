from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ercs_graphrag.llm import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OpenAI-compatible LLM settings for triple extraction.")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--llm-timeout", type=int, default=90)
    parser.add_argument("--output", default="results_llm_triples/llm_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_env_file:
        load_env_file(Path(args.llm_env_file))
    client = LLMClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=os.getenv(args.llm_api_key_env),
        timeout=args.llm_timeout,
    )
    if not client.enabled:
        raise SystemExit(
            f"LLM is not configured. Set {args.llm_api_key_env} and LLM_MODEL, "
            "or pass --llm-model plus --llm-env-file."
        )
    rows = client.validate_json_array()
    payload = {
        "ok": True,
        "model": client.model,
        "url": client.url,
        "sample": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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
