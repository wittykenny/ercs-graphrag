# -*- coding: utf-8 -*-
"""Automatic + manual-template triple quality audit for Project 3.

This script samples triples, applies transparent heuristic checks, and creates a
CSV that can be manually corrected for the thesis. It complements, rather than
replaces, human evaluation.
"""
from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.ercs_exp_utils import ensure_dir, normalize_text, pick_first, read_config, read_json, setup_logger, set_seed, write_csv, write_json

ARTIFACT_PATTERNS = re.compile(r"(http|www\.|\.com|\.org|wiki|metadata|title:|id:|\{\}|\[\]|none|null)", re.I)
GENERIC_RELATIONS = {"related_to", "relates_to", "associated_with", "part_of", "is", "has", "in", "of", "about", "mentions"}


def entity_error(entity: str) -> str:
    e = normalize_text(entity)
    if not e:
        return "empty_entity"
    if len(e) > 80 or len(e.split()) > 10:
        return "entity_boundary"
    if ARTIFACT_PATTERNS.search(e):
        return "metadata_artifact"
    if sum(ch.isdigit() for ch in e) > max(6, len(e) * 0.6):
        return "metadata_artifact"
    return ""


def relation_error(rel: str) -> str:
    r = normalize_text(rel).lower().strip()
    if not r:
        return "empty_relation"
    if ARTIFACT_PATTERNS.search(r):
        return "metadata_artifact"
    if r in GENERIC_RELATIONS or len(r) <= 2:
        return "too_generic_relation"
    if len(r.split()) > 8:
        return "relation_boundary"
    return ""


def load_chunk_texts(chunks_obj: Any) -> Tuple[Dict[str, str], List[str]]:
    mapping: Dict[str, str] = {}
    texts: List[str] = []
    if isinstance(chunks_obj, dict):
        iterable = chunks_obj.items()
    elif isinstance(chunks_obj, list):
        iterable = enumerate(chunks_obj)
    else:
        return mapping, texts
    for key, item in iterable:
        if isinstance(item, dict):
            cid = str(pick_first(item, ["id", "chunk_id", "doc_id", "title"], key))
            text = normalize_text(pick_first(item, ["text", "content", "passage", "body", "abstract"], item))
        else:
            cid, text = str(key), normalize_text(item)
        mapping[cid] = text
        if text:
            texts.append(text)
    return mapping, texts


def parse_triple(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        h = pick_first(item, ["head", "head_entity", "subject", "source", "h"], "")
        r = pick_first(item, ["relation", "predicate", "rel", "p"], "")
        t = pick_first(item, ["tail", "tail_entity", "object", "target", "t"], "")
        cid = pick_first(item, ["chunk_id", "doc_id", "source_id", "evidence_id", "id"], "")
        evidence = pick_first(item, ["evidence", "source_text", "text", "sentence", "chunk_text"], "")
        return {"head": h, "relation": r, "tail": t, "chunk_id": cid, "evidence": evidence, "raw": item}
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {"head": item[0], "relation": item[1], "tail": item[2], "chunk_id": "", "evidence": "", "raw": item}
    return {"head": "", "relation": "", "tail": "", "chunk_id": "", "evidence": "", "raw": item}


def evidence_check(head: str, tail: str, evidence: str) -> bool:
    ev = normalize_text(evidence).lower()
    if not ev:
        return False
    h = normalize_text(head).lower()
    t = normalize_text(tail).lower()
    return bool(h and t and h in ev and t in ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ercs_hotpotqa_500.yaml")
    ap.add_argument("--triples", default=None)
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_config(args.config)
    tq = cfg.get("triple_quality", {})
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger = setup_logger("evaluate_triple_quality")

    triples_path = Path(args.triples or tq.get("triples_path", "data/processed_hotpotqa_500/triples.json"))
    chunks_path = Path(args.chunks or tq.get("chunks_path", "data/processed_hotpotqa_500/chunks.json"))
    sample_size = int(args.sample_size or tq.get("sample_size", 200))
    out_dir = ensure_dir(args.output_dir or tq.get("output_dir", "results_triple_quality_500"))

    triples_obj = read_json(triples_path, [])
    chunks_obj = read_json(chunks_path, [])
    if isinstance(triples_obj, dict):
        triples = list(triples_obj.values())
    else:
        triples = list(triples_obj or [])
    chunk_map, all_chunk_texts = load_chunk_texts(chunks_obj)

    logger.info("Loaded triples: %d", len(triples))
    sample = random.sample(triples, min(sample_size, len(triples))) if triples else []

    rows: List[Dict[str, Any]] = []
    err_counter = Counter()
    for i, raw in enumerate(sample, 1):
        tr = parse_triple(raw)
        head, rel, tail = map(normalize_text, [tr["head"], tr["relation"], tr["tail"]])
        evidence = normalize_text(tr.get("evidence"))
        cid = str(tr.get("chunk_id") or "")
        if not evidence and cid in chunk_map:
            evidence = chunk_map[cid]
        if not evidence and all_chunk_texts:
            # Cheap fallback: use the first chunk containing either entity.
            low_h, low_t = head.lower(), tail.lower()
            for txt in all_chunk_texts[:2000]:
                ltxt = txt.lower()
                if (low_h and low_h in ltxt) or (low_t and low_t in ltxt):
                    evidence = txt
                    break

        errors = [e for e in [entity_error(head), entity_error(tail), relation_error(rel)] if e]
        consistent = evidence_check(head, tail, evidence)
        if not consistent:
            errors.append("evidence_mismatch")
        error_type = ";".join(sorted(set(errors))) if errors else "ok"
        err_counter.update(error_type.split(";"))
        rows.append({
            "triple_id": i,
            "head_entity": head,
            "relation": rel,
            "tail_entity": tail,
            "chunk_id": cid,
            "source_text": evidence[:1000],
            "auto_entity_correct": int(not entity_error(head) and not entity_error(tail)),
            "auto_relation_correct": int(not relation_error(rel)),
            "auto_evidence_consistent": int(consistent),
            "auto_error_type": error_type,
            # Manual columns for thesis-grade audit.
            "manual_entity_correct": "",
            "manual_relation_correct": "",
            "manual_evidence_consistent": "",
            "manual_error_type": "",
            "manual_note": "",
        })

    summary = {
        "triples_path": str(triples_path),
        "chunks_path": str(chunks_path),
        "total_triples": len(triples),
        "sample_size": len(rows),
        "auto_entity_accuracy": sum(r["auto_entity_correct"] for r in rows) / len(rows) if rows else 0,
        "auto_relation_accuracy": sum(r["auto_relation_correct"] for r in rows) / len(rows) if rows else 0,
        "auto_evidence_consistency": sum(r["auto_evidence_consistent"] for r in rows) / len(rows) if rows else 0,
        "auto_error_counts": dict(err_counter),
    }
    summary.update(manual_quality_summary(rows))
    write_csv(out_dir / "triple_quality_sample.csv", rows)
    write_json(out_dir / "triple_quality_summary.json", summary)
    logger.info("Saved sample and summary to %s", out_dir)


def manual_quality_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize manual labels if the sample CSV has been filled and reloaded.

    The generated sample starts with blank manual columns. If users later fill
    entity/relation/evidence correctness and error type, this helper can be
    reused by loading those rows in downstream scripts. For the fresh automatic
    run it still exports normalized automatic error ratios for paper tables.
    """
    total = max(1, len(rows))
    auto_counts = Counter()
    for row in rows:
        for err in str(row.get("auto_error_type", "")).split(";"):
            err = err.strip() or "ok"
            auto_counts[err] += 1
    auto_ratios = {key: val / total for key, val in sorted(auto_counts.items())}
    manual_rows = [
        row for row in rows
        if str(row.get("manual_entity_correct", "")).strip()
        or str(row.get("manual_relation_correct", "")).strip()
        or str(row.get("manual_evidence_consistent", "")).strip()
        or str(row.get("manual_error_type", "")).strip()
    ]
    payload: Dict[str, Any] = {
        "auto_error_ratios": auto_ratios,
        "manual_labeled_count": len(manual_rows),
    }
    if not manual_rows:
        return payload
    n = len(manual_rows)
    payload.update(
        {
            "manual_entity_accuracy": avg_binary(manual_rows, "manual_entity_correct"),
            "manual_relation_accuracy": avg_binary(manual_rows, "manual_relation_correct"),
            "manual_evidence_consistency": avg_binary(manual_rows, "manual_evidence_consistent"),
            "manual_error_counts": dict(Counter(normalize_manual_error(row) for row in manual_rows)),
            "manual_error_ratios": {
                key: val / n for key, val in Counter(normalize_manual_error(row) for row in manual_rows).items()
            },
        }
    )
    return payload


def avg_binary(rows: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        raw = str(row.get(key, "")).strip().lower()
        if raw in {"1", "true", "yes", "y", "正确", "是"}:
            vals.append(1.0)
        elif raw in {"0", "false", "no", "n", "错误", "否"}:
            vals.append(0.0)
    return sum(vals) / len(vals) if vals else 0.0


def normalize_manual_error(row: Dict[str, Any]) -> str:
    raw = normalize_text(row.get("manual_error_type", "")).lower()
    if not raw:
        return "ok"
    mapping = {
        "entity_boundary_error": "entity_boundary",
        "boundary": "entity_boundary",
        "wrong_relation": "wrong_relation",
        "relation_error": "wrong_relation",
        "evidence_mismatch": "evidence_mismatch",
        "metadata_noise": "metadata_artifact",
        "metadata": "metadata_artifact",
        "over_generic_relation": "too_generic_relation",
        "generic": "too_generic_relation",
        "title_fragment": "title_fragment",
    }
    return mapping.get(raw, raw)


if __name__ == "__main__":
    main()
