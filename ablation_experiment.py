# -*- coding: utf-8 -*-
"""Build an ablation table from existing Project 3 result directories.

The script is schema-tolerant: it can parse common summary.json structures and
also falls back to aggregating details.csv when summary.json is unavailable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from scripts.ercs_exp_utils import ensure_dir, flatten_metrics, looks_like_number, read_config, read_csv_dict, read_json, setup_logger, write_csv, write_json

METRIC_HINTS = [
    "evidence_coverage", "coverage", "recall", "precision", "mrr", "ndcg", "hit", "f1", "accuracy", "map"
]
METHOD_KEYS = ["method", "model", "retriever", "strategy", "variant", "setting", "name"]


def clean_method(name: str, aliases: Dict[str, str]) -> str:
    low = str(name).strip().lower()
    for k, v in aliases.items():
        if k.lower() == low or k.lower() in low:
            return v
    return str(name).strip() or "unknown"


def parse_summary(summary: Any, result_dir: Path, aliases: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(summary, dict):
        return rows

    # Pattern A: {method: {metric: value}}
    for key, val in summary.items():
        if isinstance(val, dict):
            flat = flatten_metrics("", val)
            numeric_metrics = {k: v for k, v in flat.items() if looks_like_number(v) and any(h in k.lower() for h in METRIC_HINTS)}
            if numeric_metrics:
                row = {"result_dir": result_dir.name, "method": clean_method(key, aliases)}
                row.update(numeric_metrics)
                rows.append(row)

    # Pattern B: {"methods": [{"method": ..., "metric": ...}]}
    for list_key in ["methods", "results", "summary", "metrics"]:
        val = summary.get(list_key)
        if isinstance(val, list):
            for item in val:
                if not isinstance(item, dict):
                    continue
                method = None
                for k in METHOD_KEYS:
                    if k in item:
                        method = item[k]
                        break
                if method is None:
                    continue
                flat = flatten_metrics("", item)
                row = {"result_dir": result_dir.name, "method": clean_method(str(method), aliases)}
                for k, v in flat.items():
                    if k not in METHOD_KEYS and looks_like_number(v):
                        row[k] = v
                rows.append(row)

    # Pattern C: flat single summary, use directory name as method.
    if not rows:
        flat = flatten_metrics("", summary)
        numeric_metrics = {k: v for k, v in flat.items() if looks_like_number(v) and any(h in k.lower() for h in METRIC_HINTS)}
        if numeric_metrics:
            row = {"result_dir": result_dir.name, "method": clean_method(result_dir.name, aliases)}
            row.update(numeric_metrics)
            rows.append(row)
    return rows


def parse_details(details_path: Path, aliases: Dict[str, str]) -> List[Dict[str, Any]]:
    rows = read_csv_dict(details_path)
    if not rows:
        return []
    method_col: Optional[str] = None
    for k in METHOD_KEYS:
        if k in rows[0]:
            method_col = k
            break
    if method_col is None:
        for k in rows[0].keys():
            if "method" in k.lower() or "retriever" in k.lower():
                method_col = k
                break
    if method_col is None:
        return []

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        method = clean_method(r.get(method_col, "unknown"), aliases)
        for k, v in r.items():
            lk = k.lower()
            if k == method_col:
                continue
            if any(h in lk for h in METRIC_HINTS) and looks_like_number(v):
                grouped[method][k].append(float(v))
    out: List[Dict[str, Any]] = []
    for method, metrics in grouped.items():
        item: Dict[str, Any] = {"method": method}
        for k, vals in metrics.items():
            if vals:
                item[k] = mean(vals)
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ercs_hotpotqa_500.yaml")
    ap.add_argument("--results", nargs="*", default=None, help="Result directories to parse. Overrides config.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_config(args.config)
    logger = setup_logger("ablation_experiment")
    abl_cfg = cfg.get("ablation", {})
    aliases = abl_cfg.get("method_aliases", {})
    result_dirs = args.results or abl_cfg.get("baseline_result_dirs", [])
    output_dir = Path(args.output_dir or abl_cfg.get("output_dir", "results_ablation_500"))
    ensure_dir(output_dir)

    all_rows: List[Dict[str, Any]] = []
    for d in result_dirs:
        rd = Path(d)
        if not rd.exists():
            logger.warning("Missing result directory: %s", rd)
            continue
        rows = parse_summary(read_json(rd / "summary.json", {}), rd, aliases)
        if not rows:
            rows = parse_details(rd / "details.csv", aliases)
            for r in rows:
                r.setdefault("result_dir", rd.name)
        logger.info("Parsed %d ablation rows from %s", len(rows), rd)
        all_rows.extend(rows)

    # De-duplicate exact same result_dir/method rows.
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for row in all_rows:
        key = (row.get("result_dir"), row.get("method"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    write_csv(output_dir / "ablation_results.csv", dedup)
    write_json(output_dir / "ablation_results.json", dedup)
    logger.info("Saved: %s", output_dir / "ablation_results.csv")


if __name__ == "__main__":
    main()
