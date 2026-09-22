from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


MANUAL_FIELDS = [
    "entity_correct",
    "relation_correct",
    "evidence_consistent",
    "error_type",
    "manual_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample triples for manual graph extraction quality auditing.")
    parser.add_argument("--triples", default="data/processed_formal_test/triples.json")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/triple_quality_sample.csv")
    parser.add_argument("--summary-output", default="results/triple_quality_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triples_path = Path(args.triples)
    triples = json.loads(triples_path.read_text(encoding="utf-8"))
    sample = sample_triples(triples, args.sample_size, args.seed)
    output = Path(args.output)
    previous_labels = load_previous_labels(output)

    rows = []
    for idx, tri in enumerate(sample, 1):
        key = triple_key(tri)
        row = {
            "sample_id": idx,
            "chunk_id": tri.get("chunk_id", ""),
            "head": tri.get("head", ""),
            "relation": tri.get("relation", ""),
            "tail": tri.get("tail", ""),
            "evidence": tri.get("evidence", ""),
            "auto_entity_surface_hit": int(surface_hit(tri.get("head", ""), tri.get("evidence", "")) and surface_hit(tri.get("tail", ""), tri.get("evidence", ""))),
            "auto_relation_known": int(bool(tri.get("relation")) and tri.get("relation") != "related_to"),
            "auto_evidence_nonempty": int(bool(str(tri.get("evidence", "")).strip())),
            "entity_correct": "",
            "relation_correct": "",
            "evidence_consistent": "",
            "error_type": "",
            "manual_note": "",
        }
        row.update(previous_labels.get(key, {}))
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows)
    summary["manual_instruction"] = (
        "Fill entity_correct, relation_correct, evidence_consistent with 1/0. "
        "Use error_type such as entity_boundary, wrong_relation, unsupported, duplicate, alias, other."
    )
    summary["source_triples"] = str(triples_path)
    summary["sample_size"] = len(rows)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved sample: {output}")
    print(f"Saved summary: {summary_output}")


def sample_triples(triples: list[dict], sample_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(triples) <= sample_size:
        return triples[:]
    return rng.sample(triples, sample_size)


def surface_hit(entity: str, evidence: str) -> bool:
    entity = normalize(entity)
    evidence = normalize(evidence)
    return bool(entity and entity in evidence)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def build_summary(rows: list[dict]) -> dict:
    summary = {
        "auto_entity_surface_hit_rate": mean(int(r["auto_entity_surface_hit"]) for r in rows),
        "auto_relation_known_rate": mean(int(r["auto_relation_known"]) for r in rows),
        "auto_evidence_nonempty_rate": mean(int(r["auto_evidence_nonempty"]) for r in rows),
    }
    for field in ["entity_correct", "relation_correct", "evidence_consistent"]:
        values = [parse_binary(r.get(field, "")) for r in rows]
        values = [v for v in values if v is not None]
        if values:
            summary[f"manual_{field}_rate"] = mean(values)
            summary[f"manual_{field}_n"] = len(values)
    return summary


def load_previous_labels(path: Path) -> dict[tuple[str, str, str, str, str], dict]:
    if not path.exists():
        return {}
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row.get("chunk_id", ""),
                row.get("head", ""),
                row.get("relation", ""),
                row.get("tail", ""),
                row.get("evidence", ""),
            )
            labels[key] = {field: row.get(field, "") for field in MANUAL_FIELDS}
    return labels


def triple_key(tri: dict) -> tuple[str, str, str, str, str]:
    return (
        str(tri.get("chunk_id", "")),
        str(tri.get("head", "")),
        str(tri.get("relation", "")),
        str(tri.get("tail", "")),
        str(tri.get("evidence", "")),
    )


def parse_binary(value: object) -> int | None:
    text = str(value).strip()
    if text in {"1", "true", "True", "yes", "Y", "y"}:
        return 1
    if text in {"0", "false", "False", "no", "N", "n"}:
        return 0
    return None


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
