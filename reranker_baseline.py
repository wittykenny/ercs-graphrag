# -*- coding: utf-8 -*-
"""Cross-Encoder reranker baseline for Project 3.

Designed for RTX 4070 Laptop GPU:
- default model: BAAI/bge-reranker-base
- batch_size: 16
- max_length: 512
- optional fp16 on CUDA

If sentence-transformers is not installed, the script falls back to a TF-IDF
candidate ranking baseline and clearly records that in summary.json.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from scripts.ercs_exp_utils import ensure_dir, normalize_text, pick_first, read_config, read_json, setup_logger, set_seed, write_csv, write_json


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    obj = read_json(path, [])
    if isinstance(obj, dict):
        iterable = obj.items()
    else:
        iterable = enumerate(obj or [])
    docs: List[Dict[str, Any]] = []
    for key, item in iterable:
        if isinstance(item, dict):
            doc_id = str(pick_first(item, ["id", "doc_id", "title", "_id"], key))
            title = normalize_text(pick_first(item, ["title", "name"], ""))
            text = normalize_text(pick_first(item, ["text", "content", "body", "passage", "abstract"], item))
        else:
            doc_id, title, text = str(key), "", normalize_text(item)
        full = f"{title}. {text}".strip(". ")
        if full:
            docs.append({"doc_id": doc_id, "title": title, "text": full})
    return docs


def load_questions(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    obj = read_json(path, [])
    if isinstance(obj, dict):
        # common patterns: {data: [...]} or {qid: {...}}
        if isinstance(obj.get("data"), list):
            items = obj["data"]
        elif isinstance(obj.get("questions"), list):
            items = obj["questions"]
        else:
            items = list(obj.values())
    else:
        items = obj or []
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        q = normalize_text(pick_first(item, ["question", "query", "q"], ""))
        if not q:
            continue
        qid = str(pick_first(item, ["id", "qid", "_id"], i))
        gold = pick_first(
            item,
            ["evidence_list", "evidence", "gold_evidence", "supporting_facts", "supporting_docs", "answers", "context"],
            [],
        )
        out.append({"qid": qid, "question": q, "gold": gold, "raw": item})
        if limit and len(out) >= limit:
            break
    return out


def build_tfidf(docs: Sequence[Dict[str, Any]]):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        vectorizer = TfidfVectorizer(max_features=80000, ngram_range=(1, 2), stop_words="english")
        mat = vectorizer.fit_transform([d["text"] for d in docs])
        return vectorizer, mat, cosine_similarity
    except Exception as e:
        raise RuntimeError("scikit-learn is required for candidate retrieval. Install scikit-learn.") from e


def extract_gold_strings(gold: Any) -> List[str]:
    vals: List[str] = []
    if isinstance(gold, str):
        vals.append(gold)
    elif isinstance(gold, dict):
        for v in gold.values():
            vals.extend(extract_gold_strings(v))
    elif isinstance(gold, list):
        for x in gold:
            vals.extend(extract_gold_strings(x))
    return [normalize_text(v) for v in vals if normalize_text(v)]


def is_hit(doc: Dict[str, Any], gold: Any) -> bool:
    golds = extract_gold_strings(gold)
    if not golds:
        return False
    hay = f"{doc.get('doc_id','')} {doc.get('title','')} {doc.get('text','')}".lower()
    for g in golds:
        gl = g.lower()
        if len(gl) >= 3 and gl in hay:
            return True
    return False


def precision_recall_at_k(ranked: Sequence[Dict[str, Any]], gold: Any, k: int) -> Tuple[float, float, int]:
    top = ranked[:k]
    hits = sum(1 for d in top if is_hit(d, gold))
    gold_count = max(1, len(set(extract_gold_strings(gold))))
    return hits / max(1, k), min(1.0, hits / gold_count), hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ercs_hotpotqa_500.yaml")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--candidate-k", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = read_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    rcfg = cfg.get("reranker", {})
    logger = setup_logger("reranker_baseline")

    corpus_path = Path(rcfg.get("corpus_path", "data/hotpotqa_raw_500/corpus.json"))
    questions_path = Path(rcfg.get("questions_path", "data/hotpotqa_raw_500/MultiHopRAG.json"))
    model_name = args.model_name or rcfg.get("model_name", "BAAI/bge-reranker-base")
    sample_size = int(args.sample_size or rcfg.get("sample_size", 500))
    candidate_k = int(args.candidate_k or rcfg.get("candidate_k", 80))
    top_k = int(args.top_k or rcfg.get("top_k", 5))
    batch_size = int(args.batch_size or rcfg.get("batch_size", 16))
    out_dir = ensure_dir(args.output_dir or rcfg.get("output_dir", "results_cross_encoder_hotpotqa_500_new"))

    docs = load_corpus(corpus_path)
    questions = load_questions(questions_path, limit=sample_size)
    logger.info("Loaded docs=%d questions=%d", len(docs), len(questions))

    vectorizer, doc_mat, cosine_similarity = build_tfidf(docs)

    cross_encoder = None
    backend = "tfidf_only_fallback"
    try:
        import torch  # type: ignore
        from sentence_transformers import CrossEncoder  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cross_encoder = CrossEncoder(model_name, device=device, max_length=int(rcfg.get("max_length", 512)))
        if bool(rcfg.get("use_fp16", True)) and device == "cuda":
            cross_encoder.model.half()
        backend = f"cross_encoder:{model_name}:{device}"
        logger.info("Using CrossEncoder backend: %s", backend)
    except Exception as e:
        logger.warning("CrossEncoder unavailable; using TF-IDF fallback. Reason: %s", e)

    detail_rows: List[Dict[str, Any]] = []
    p_sum = r_sum = hit_sum = 0.0
    query_time_sum = 0.0
    for idx, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        q_vec = vectorizer.transform([q["question"]])
        sims = cosine_similarity(q_vec, doc_mat).ravel()
        cand_idx = sims.argsort()[-candidate_k:][::-1]
        candidates = [dict(docs[int(i)], candidate_score=float(sims[int(i)])) for i in cand_idx]

        if cross_encoder is not None:
            pairs = [[q["question"], c["text"][:4000]] for c in candidates]
            scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["reranker_score"] = float(s)
            candidates.sort(key=lambda x: x.get("reranker_score", x.get("candidate_score", 0.0)), reverse=True)
        else:
            for c in candidates:
                c["reranker_score"] = c["candidate_score"]

        prec, rec, hits = precision_recall_at_k(candidates, q["gold"], top_k)
        query_time_ms = (time.perf_counter() - t0) * 1000.0
        query_time_sum += query_time_ms
        p_sum += prec
        r_sum += rec
        hit_sum += 1 if hits > 0 else 0
        for rank, c in enumerate(candidates[:top_k], 1):
            detail_rows.append({
                "qid": q["qid"],
                "question": q["question"],
                "rank": rank,
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "candidate_score": c.get("candidate_score"),
                "reranker_score": c.get("reranker_score"),
                "hit": int(is_hit(c, q["gold"])),
                "query_time_ms": query_time_ms,
                "text_preview": c.get("text", "")[:500],
            })
        if idx % 50 == 0:
            logger.info("Processed %d/%d", idx, len(questions))

    n = max(1, len(questions))
    summary = {
        "backend": backend,
        "model_name": model_name,
        "sample_size": len(questions),
        "candidate_k": candidate_k,
        "top_k": top_k,
        f"precision_at_{top_k}": p_sum / n,
        f"evidence_coverage_at_{top_k}": r_sum / n,
        f"hit_rate_at_{top_k}": hit_sum / n,
        "avg_query_time_ms": query_time_sum / n,
        "relative_cost_note": "Includes TF-IDF candidate scoring plus CrossEncoder reranking when available.",
    }
    write_csv(out_dir / "details.csv", detail_rows)
    write_json(out_dir / "summary.json", summary)
    logger.info("Saved reranker results to %s", out_dir)


if __name__ == "__main__":
    main()
