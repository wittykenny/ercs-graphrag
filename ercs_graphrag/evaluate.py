from __future__ import annotations

import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def is_relevant(chunk: dict, item: dict) -> bool:
    return bool(evidence_hit_indices(chunk, item))


def evidence_hit_indices(chunk: dict, item: dict) -> set[int]:
    title = normalize(chunk.get("title", ""))
    text = normalize(chunk.get("text", ""))
    hits: set[int] = set()
    for idx, ev in enumerate(item.get("evidence_list", [])):
        ev_title = normalize(ev.get("title", ""))
        ev_fact = normalize(ev.get("fact", ""))
        title_hit = bool(title and ev_title and title == ev_title)
        fact_hit = bool(ev_fact and (ev_fact in text or text in ev_fact))
        if title_hit or fact_hit:
            hits.add(idx)
    return hits


def retrieval_metrics(results: list[dict], item: dict, k: int = 5) -> dict[str, float]:
    top = results[:k]
    hits = [is_relevant(chunk, item) for chunk in top]
    evidence_total = max(1, len(item.get("evidence_list", [])))
    covered = set()
    for chunk in top:
        covered.update(evidence_hit_indices(chunk, item))
    first_rank = next((i + 1 for i, hit in enumerate(hits) if hit), 0)
    first_two_total = min(2, evidence_total)
    first_two_covered = all(i in covered for i in range(first_two_total))
    return {
        f"hit@{k}": 1.0 if any(hits) else 0.0,
        f"recall@{k}": len(covered) / evidence_total,
        f"precision@{k}": sum(hits) / k,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        f"evidence_coverage@{k}": len(covered) / evidence_total,
        f"all_evidence_hit@{k}": 1.0 if len(covered) >= evidence_total else 0.0,
        f"first_two_evidence_hit@{k}": 1.0 if first_two_covered else 0.0,
        "evidence_hit_count": float(len(covered)),
        "evidence_total": float(evidence_total),
    }


def covered_evidence_ids(results: list[dict], item: dict, k: int = 5) -> list[int]:
    covered = set()
    for chunk in results[:k]:
        covered.update(evidence_hit_indices(chunk, item))
    return sorted(covered)


def answer_accuracy(answer: str, generated: str) -> float:
    gold = normalize(answer)
    pred = normalize(generated)
    return 1.0 if gold and gold in pred else 0.0


def average_metrics(rows: list[dict]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
