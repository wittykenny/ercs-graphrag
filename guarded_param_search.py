from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ercs_graphrag.data import build_chunks, load_dataset, split_qa_dataset
from ercs_graphrag.evaluate import answer_accuracy, average_metrics, retrieval_metrics
from ercs_graphrag.extract import extract_triples
from ercs_graphrag.generation import faithfulness_score, generate_answer
from ercs_graphrag.graph import build_kg, compute_community_score, compute_entity_importance, detect_communities
from ercs_graphrag.retrieval import build_retrieval_index, ercs_graphrag_retrieve, hybrid_rag_retrieve
from ercs_graphrag.summarize import build_community_texts, summarize_communities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search guarded Hybrid-UCP parameters on a dev split.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed_guarded_dev")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="dev")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--dev-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-backend", choices=["auto", "sbert", "tfidf"], default="sbert")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--params-file", default="results/best_params.json")
    parser.add_argument("--hybrid-vector-weight", type=float, default=0.60)
    parser.add_argument("--base-score-weights", default="0.5,0.6,0.7,0.75,0.8,0.9")
    parser.add_argument("--preserve-base-top-n-values", default="0,1,2,3,4")
    parser.add_argument("--output-dir", default="results_guarded_search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params_file)
    qa_all, corpus = load_dataset(args.data_dir)
    qa = split_qa_dataset(
        qa_all,
        split=args.split,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )[: args.max_queries]
    chunks = build_chunks(corpus, qa, max_docs=args.max_docs)

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    triples_path = processed_dir / "triples.json"
    if args.reuse_cache and triples_path.exists():
        triples = json.loads(triples_path.read_text(encoding="utf-8"))
    else:
        triples = extract_triples(chunks)
        triples_path.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")

    graph = build_kg(triples)
    entity_scores = compute_entity_importance(graph)
    partition, _ = detect_communities(graph)
    community_scores = compute_community_score(partition, entity_scores)
    summary_path = processed_dir / "community_summary.json"
    if args.reuse_cache and summary_path.exists():
        community_summaries = {int(k): v for k, v in json.loads(summary_path.read_text(encoding="utf-8")).items()}
    else:
        community_summaries = summarize_communities(build_community_texts(chunks, partition, triples))
        summary_path.write_text(json.dumps(community_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    index = build_retrieval_index(
        chunks,
        partition,
        triples=triples,
        community_summaries=community_summaries,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )

    rows = []
    hybrid_metrics = evaluate_hybrid(qa, index, args.top_k, args.hybrid_vector_weight)
    rows.append({"method": "hybrid_rag", "base_score_weight": "", "preserve_base_top_n": "", **hybrid_metrics})
    for base_score_weight in floats(args.base_score_weights):
        for preserve_n in ints(args.preserve_base_top_n_values):
            metrics = evaluate_guarded(
                qa=qa,
                index=index,
                graph=graph,
                entity_scores=entity_scores,
                community_scores=community_scores,
                partition=partition,
                top_k=args.top_k,
                hybrid_vector_weight=args.hybrid_vector_weight,
                base_score_weight=base_score_weight,
                preserve_base_top_n=preserve_n,
                params=params,
            )
            rows.append(
                {
                    "method": "guarded_hybrid_ucp_ercs_graphrag",
                    "base_score_weight": base_score_weight,
                    "preserve_base_top_n": preserve_n,
                    **metrics,
                }
            )

    objective = f"evidence_coverage@{args.top_k}"
    guarded_rows = [row for row in rows if row["method"] != "hybrid_rag"]
    guarded_rows.sort(key=lambda row: (row[objective], row["mrr"], row[f"first_two_evidence_hit@{args.top_k}"]), reverse=True)
    best = guarded_rows[0]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "guarded_param_search.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best_payload = {
        "objective_metric": objective,
        "base_score_weight": best["base_score_weight"],
        "preserve_base_top_n": best["preserve_base_top_n"],
        "hybrid_vector_weight": args.hybrid_vector_weight,
        "metrics": {k: v for k, v in best.items() if k not in {"method", "base_score_weight", "preserve_base_top_n"}},
        "source_params_file": args.params_file,
    }
    (out_dir / "best_guarded_params.json").write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best:", best_payload)
    print(f"saved {out_dir / 'guarded_param_search.csv'}")
    print(f"saved {out_dir / 'best_guarded_params.json'}")


def evaluate_hybrid(qa: list[dict], index, top_k: int, hybrid_vector_weight: float) -> dict:
    rows = []
    for item in qa:
        results = hybrid_rag_retrieve(item["query"], index, top_k, vector_weight=hybrid_vector_weight)
        rows.append(metrics_for(item, results, top_k))
    return average_metrics(rows)


def evaluate_guarded(
    qa: list[dict],
    index,
    graph,
    entity_scores: dict[str, float],
    community_scores: dict[int, float],
    partition: dict[str, int],
    top_k: int,
    hybrid_vector_weight: float,
    base_score_weight: float,
    preserve_base_top_n: int,
    params: dict,
) -> dict:
    rows = []
    for item in qa:
        results = ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=params.get("alpha", 0.4),
            beta=params.get("beta", 0.0),
            gamma=params.get("gamma", 0.6),
            delta=params.get("delta", 0.0),
            epsilon=params.get("epsilon", 0.05),
            candidate_pool=params.get("candidate_pool", 80),
            query_entity_lambda=params.get("query_entity_lambda", 0.6),
            adaptive_eta=params.get("adaptive_eta", 0.2),
            redundancy_weight=0.0,
            path_cutoff=params.get("path_cutoff", 4),
            base_retrieval="hybrid",
            hybrid_vector_weight=hybrid_vector_weight,
            base_score_weight=base_score_weight,
            preserve_base_top_n=preserve_base_top_n,
        )
        rows.append(metrics_for(item, results, top_k))
    return average_metrics(rows)


def metrics_for(item: dict, results: list[dict], top_k: int) -> dict:
    metrics = retrieval_metrics(results, item, top_k)
    generated = generate_answer(item["query"], results)
    metrics["answer_accuracy"] = answer_accuracy(item.get("answer", ""), generated)
    metrics["faithfulness"] = faithfulness_score(generated, results)
    return metrics


def load_params(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("params", payload)


def floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
