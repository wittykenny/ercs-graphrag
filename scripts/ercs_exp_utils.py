# -*- coding: utf-8 -*-
"""Shared utilities for the Project 3 GraphRAG supplementary experiments.

These helpers are deliberately dependency-light so the scripts can run in the
existing project without changing its original modules.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str, log_dir: os.PathLike | str = "logs", level: int = logging.INFO) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_path = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("Log file: %s", log_path)
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike | str, data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_config(path: Optional[os.PathLike | str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Run: pip install pyyaml")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def write_csv(path: os.PathLike | str, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        if fieldnames:
            with p.open("w", encoding="utf-8-sig", newline="") as f:
                csv.DictWriter(f, fieldnames=list(fieldnames)).writeheader()
        return
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dict(path: os.PathLike | str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_command(cmd: Sequence[str], cwd: Optional[os.PathLike | str] = None, logger: Optional[logging.Logger] = None,
                fail_fast: bool = True) -> int:
    if logger:
        logger.info("RUN: %s", " ".join(map(str, cmd)))
    proc = subprocess.Popen(list(cmd), cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        if logger:
            logger.info(line.rstrip())
        else:
            print(line, end="")
    code = proc.wait()
    if code != 0 and fail_fast:
        raise subprocess.CalledProcessError(code, list(cmd))
    return code


def flatten_metrics(prefix: str, obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.update(flatten_metrics(key, v))
            else:
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_metrics(f"{prefix}.{i}", v))
    return out


def looks_like_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False) if isinstance(text, (dict, list)) else str(text)
    return re.sub(r"\s+", " ", text).strip()


def pick_first(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def iter_result_dirs(root: os.PathLike | str = ".") -> List[Path]:
    base = Path(root)
    dirs = []
    for p in base.iterdir() if base.exists() else []:
        if p.is_dir() and p.name.startswith("results"):
            dirs.append(p)
    return sorted(dirs, key=lambda x: x.name)
