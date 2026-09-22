# -*- coding: utf-8 -*-
"""One-command reproduction runner for Project 3.

Usage:
  python reproduce_all.py --config configs/ercs_hotpotqa_500.yaml
  python reproduce_all.py --config configs/ercs_hotpotqa_500.yaml --steps existing,ablation,triple_quality,export,reranker

This script does not require changing your original code. It calls the existing
project scripts when they are present, then runs the supplementary analysis
scripts added in this patch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.ercs_exp_utils import read_config, run_command, setup_logger, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ercs_hotpotqa_500.yaml")
    parser.add_argument("--steps", default="existing,ablation,triple_quality,export",
                        help="Comma-separated: existing,ablation,triple_quality,export,reranker")
    parser.add_argument("--fail-fast", action="store_true", help="Stop if an optional command fails.")
    args = parser.parse_args()

    cfg = read_config(args.config)
    root = Path(cfg.get("project_root", ".")).resolve()
    py = cfg.get("python", sys.executable)
    logger = setup_logger("reproduce_all", root / "logs")
    set_seed(int(cfg.get("seed", 42)))
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}

    logger.info("Project root: %s", root)
    logger.info("Steps: %s", ", ".join(sorted(steps)))

    if "existing" in steps:
        for item in cfg.get("reproduce", {}).get("commands", []):
            name = item.get("name", "unnamed")
            cmd = item.get("cmd", [])
            optional = bool(item.get("optional", True))
            if not cmd:
                continue
            script = root / str(cmd[1]) if len(cmd) > 1 and str(cmd[1]).endswith(".py") else None
            if script is not None and not script.exists():
                logger.warning("Skip %s because script is missing: %s", name, script)
                continue
            real_cmd = [py if x == "python" else str(x) for x in cmd]
            code = run_command(real_cmd, cwd=root, logger=logger, fail_fast=args.fail_fast and not optional)
            if code != 0:
                logger.warning("Command failed but continued: %s", name)

    if "ablation" in steps:
        run_command([py, "ablation_experiment.py", "--config", args.config], cwd=root, logger=logger, fail_fast=args.fail_fast)
    if "triple_quality" in steps:
        run_command([py, "evaluate_triple_quality.py", "--config", args.config], cwd=root, logger=logger, fail_fast=args.fail_fast)
    if "export" in steps:
        run_command([py, "export_paper_tables.py", "--config", args.config], cwd=root, logger=logger, fail_fast=args.fail_fast)
    if "reranker" in steps:
        run_command([py, "reranker_baseline.py", "--config", args.config], cwd=root, logger=logger, fail_fast=args.fail_fast)

    logger.info("Finished. Check results_paper_tables/ and logs/.")


if __name__ == "__main__":
    main()
