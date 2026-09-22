from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from ercs_graphrag.data import build_chunks, load_dataset, split_qa_dataset
from ercs_graphrag.evaluate import answer_accuracy, average_metrics, retrieval_metrics
from ercs_graphrag.extract import extract_triples
from ercs_graphrag.generation import faithfulness_score, generate_answer
from ercs_graphrag.graph import build_kg, compute_community_score, compute_entity_importance, detect_communities
from ercs_graphrag.retrieval import build_retrieval_index, ercs_graphrag_retrieve
from ercs_graphrag.summarize import build_community_texts, summarize_communities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search alpha/beta/gamma/delta for ERCS-GraphRAG.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", choices=["all", "train", "dev", "test"], default="dev")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--dev-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-evidence-chunks", action="store_true")
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--query-entity-lambda", type=float, default=0.60)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--adaptive-eta", type=float, default=0.20)
    parser.add_argument("--redundancy-weight", type=float, default=0.05)
    parser.add_argument("--path-cutoff", type=int, default=4)
    parser.add_argument("--embedding-backend", choices=["auto", "sbert", "tfidf"], default="auto")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--base-retrieval", choices=["vector", "bm25", "hybrid"], default="vector")
    parser.add_argument("--hybrid-vector-weight", type=float, default=0.60)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument(
        "--objective",
        choices=["mrr", "recall", "evidence_coverage", "all_evidence_hit", "first_two_evidence_hit", "answer_accuracy"],
        default="evidence_coverage",
    )
    parser.add_argument("--epsilon-values", default=None)
    parser.add_argument("--adaptive-eta-values", default=None)
    parser.add_argument("--redundancy-weight-values", default=None)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_all, corpus = load_dataset(args.data_dir)
    qa = split_qa_dataset(
        qa_all,
        split=args.split,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    qa = qa[: args.max_queries]
    chunks = build_chunks(
        corpus,
        qa,
        args.max_docs,
        include_evidence_chunks=args.include_evidence_chunks,
    )

    processed_dir = Path(args.processed_dir)
    triples_path = processed_dir / "triples.json"
    if args.reuse_cache and triples_path.exists():
        triples = json.loads(triples_path.read_text(encoding="utf-8"))
    else:
        triples = extract_triples(chunks)

    graph = build_kg(triples)
    entity_scores = compute_entity_importance(graph)
    partition, _ = detect_communities(graph)
    community_scores = compute_community_score(partition, entity_scores)
    summary_path = processed_dir / "community_summary.json"
    if args.reuse_cache and summary_path.exists():
        community_summaries = {int(k): v for k, v in json.loads(summary_path.read_text(encoding="utf-8")).items()}
    else:
        community_summaries = summarize_communities(build_community_texts(chunks, partition, triples))
    index = build_retrieval_index(
        chunks,
        partition,
        triples=triples,
        community_summaries=community_summaries,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )

    rows = []
    epsilon_values = parse_float_list(args.epsilon_values, [args.epsilon])
    adaptive_eta_values = parse_float_list(args.adaptive_eta_values, [args.adaptive_eta])
    redundancy_values = parse_float_list(args.redundancy_weight_values, [args.redundancy_weight])
    for alpha, beta, gamma, delta in weight_grid(args.step):
        for epsilon, adaptive_eta, redundancy_weight in itertools.product(
            epsilon_values, adaptive_eta_values, redundancy_values
        ):
            rows.append(evaluate_params(
                qa=qa,
                index=index,
                graph=graph,
                entity_scores=entity_scores,
                community_scores=community_scores,
                partition=partition,
                top_k=args.top_k,
                candidate_pool=args.candidate_pool,
                query_entity_lambda=args.query_entity_lambda,
                path_cutoff=args.path_cutoff,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                epsilon=epsilon,
                adaptive_eta=adaptive_eta,
                redundancy_weight=redundancy_weight,
                base_retrieval=args.base_retrieval,
                hybrid_vector_weight=args.hybrid_vector_weight,
            ))

    objective_key = objective_metric_name(args.objective, args.top_k)
    rows.sort(key=lambda x: (x[objective_key], x["mrr"], x[f"recall@{args.top_k}"]), reverse=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "param_search.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    best_params = {
        "objective": args.objective,
        "objective_metric": objective_key,
        "max_queries": args.max_queries,
        "max_docs": args.max_docs,
        "split": args.split,
        "include_evidence_chunks": args.include_evidence_chunks,
        "top_k": args.top_k,
        "embedding_backend": index.embedding_backend,
        "base_retrieval": args.base_retrieval,
        "hybrid_vector_weight": args.hybrid_vector_weight,
        "params": {
            "alpha": best["alpha"],
            "beta": best["beta"],
            "gamma": best["gamma"],
            "delta": best["delta"],
            "epsilon": best["epsilon"],
            "adaptive_eta": best["adaptive_eta"],
            "redundancy_weight": best["redundancy_weight"],
            "query_entity_lambda": best["query_entity_lambda"],
            "candidate_pool": best["candidate_pool"],
            "path_cutoff": best["path_cutoff"],
            "base_retrieval": args.base_retrieval,
            "hybrid_vector_weight": args.hybrid_vector_weight,
        },
        "metrics": {k: v for k, v in best.items() if k not in PARAM_KEYS},
    }
    (out_dir / "best_params.json").write_text(json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best:", best)
    print(f"saved {out_dir / 'param_search.csv'}")
    print(f"saved {out_dir / 'best_params.json'}")


def evaluate_params(
    qa: list[dict],
    index,
    graph,
    entity_scores: dict[str, float],
    community_scores: dict[int, float],
    partition: dict[str, int],
    top_k: int,
    candidate_pool: int,
    query_entity_lambda: float,
    path_cutoff: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    epsilon: float,
    adaptive_eta: float,
    redundancy_weight: float,
    base_retrieval: str,
    hybrid_vector_weight: float,
) -> dict:
    metric_rows = []
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
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
            candidate_pool=candidate_pool,
            query_entity_lambda=query_entity_lambda,
            adaptive_eta=adaptive_eta,
            redundancy_weight=redundancy_weight,
            path_cutoff=path_cutoff,
            base_retrieval=base_retrieval,
            hybrid_vector_weight=hybrid_vector_weight,
        )
        generated = generate_answer(item["query"], results)
        metrics = retrieval_metrics(results, item, top_k)
        metrics["answer_accuracy"] = answer_accuracy(item.get("answer", ""), generated)
        metrics["faithfulness"] = faithfulness_score(generated, results)
        metric_rows.append(metrics)
    avg = average_metrics(metric_rows)
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "epsilon": epsilon,
        "adaptive_eta": adaptive_eta,
        "redundancy_weight": redundancy_weight,
        "query_entity_lambda": query_entity_lambda,
        "candidate_pool": candidate_pool,
        "path_cutoff": path_cutoff,
        "base_retrieval": base_retrieval,
        "hybrid_vector_weight": hybrid_vector_weight,
        **avg,
    }


PARAM_KEYS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "adaptive_eta",
    "redundancy_weight",
    "query_entity_lambda",
    "candidate_pool",
    "path_cutoff",
    "base_retrieval",
    "hybrid_vector_weight",
}


def objective_metric_name(objective: str, top_k: int) -> str:
    if objective in {"recall", "evidence_coverage", "all_evidence_hit", "first_two_evidence_hit"}:
        return f"{objective}@{top_k}"
    return objective


def parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return default
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def weight_grid(step: float) -> list[tuple[float, float, float, float]]:
    units = round(1 / step)
    values = []
    for a, b, c in itertools.product(range(units + 1), repeat=3):
        d = units - a - b - c
        if d < 0:
            continue
        values.append((a * step, b * step, c * step, d * step))
    return values


if __name__ == "__main__":
    main()
