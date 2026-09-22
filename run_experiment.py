from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from ercs_graphrag.data import build_chunks, download_multihoprag, load_dataset, split_qa_dataset
from ercs_graphrag.evaluate import answer_accuracy, average_metrics, retrieval_metrics, covered_evidence_ids
from ercs_graphrag.extract import extract_triples
from ercs_graphrag.generation import faithfulness_score, generate_answer
from ercs_graphrag.graph import (
    build_kg,
    compute_community_score,
    compute_entity_importance,
    detect_communities,
    save_graphml,
)
from ercs_graphrag.llm import LLMClient
from ercs_graphrag.retrieval import (
    bm25_rag_retrieve,
    build_retrieval_index,
    cross_encoder_rag_retrieve,
    ercs_graphrag_retrieve,
    graph_rag_retrieve,
    hybrid_rag_retrieve,
    vector_rag_retrieve,
)
from ercs_graphrag.summarize import build_community_texts, summarize_communities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact ERCS-GraphRAG experiment.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", choices=["all", "train", "dev", "test"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--dev-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-evidence-chunks", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    parser.add_argument("--llm-extract-max-chunks", type=int, default=0)
    parser.add_argument(
        "--llm-extract-allow-rule-fallback",
        action="store_true",
        help="Allow rule extraction when --extractor llm is requested but no LLM credentials are configured.",
    )
    parser.add_argument("--answer-generator", choices=["extractive", "llm"], default="extractive")
    parser.add_argument("--community-summarizer", choices=["extractive", "llm"], default="extractive")
    parser.add_argument("--llm-summary-max-communities", type=int, default=0)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--llm-timeout", type=int, default=90)
    parser.add_argument("--validate-llm", action="store_true")
    parser.add_argument("--embedding-backend", choices=["auto", "sbert", "tfidf"], default="auto")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--params-file", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods to run, or all. Available: bm25_rag, hybrid_rag, vector_rag, graph_rag, ablations, ercs_graphrag, ucp_ercs_graphrag, hybrid_ucp_ercs_graphrag, guarded_hybrid_ucp_ercs_graphrag, safe_hybrid_ucp_ercs_graphrag, cross_encoder_rag.",
    )
    parser.add_argument("--hybrid-vector-weight", type=float, default=0.60)
    parser.add_argument("--base-score-weight", type=float, default=0.0)
    parser.add_argument("--preserve-base-top-n", type=int, default=0)
    parser.add_argument(
        "--safe-preserve-top-n",
        type=int,
        default=0,
        help="Top-N Hybrid candidates to preserve for safe_hybrid_ucp_ercs_graphrag. 0 means top_k.",
    )
    parser.add_argument(
        "--safe-base-score-weight",
        type=float,
        default=0.95,
        help="Semantic/base score weight used by safe_hybrid_ucp_ercs_graphrag when preserve N is smaller than top_k.",
    )
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--query-entity-lambda", type=float, default=0.60)
    parser.add_argument("--adaptive-eta", type=float, default=0.20)
    parser.add_argument("--redundancy-weight", type=float, default=0.05)
    parser.add_argument("--path-cutoff", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_env_file:
        load_env_file(Path(args.llm_env_file))
    apply_params_file(args)
    if args.download:
        download_multihoprag(args.data_dir)

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=os.getenv(args.llm_api_key_env),
        timeout=args.llm_timeout,
    )
    if args.extractor == "llm" and not llm.enabled and not args.llm_extract_allow_rule_fallback:
        raise SystemExit(
            f"LLM triple extraction requested, but {args.llm_api_key_env} and LLM_MODEL are not both set. "
            "Set credentials, pass --llm-model, or add --llm-extract-allow-rule-fallback for a non-LLM smoke run."
        )
    if args.validate_llm and llm.enabled:
        llm.validate_json_array()

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
        max_docs=args.max_docs,
        include_evidence_chunks=args.include_evidence_chunks,
    )
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    _save_json(processed_dir / "chunks.json", chunks)
    _save_json(
        processed_dir / "experiment_config.json",
        {
            "split": args.split,
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "seed": args.seed,
            "max_queries": args.max_queries,
            "max_docs": args.max_docs,
            "max_chunks": args.max_chunks,
            "include_evidence_chunks": args.include_evidence_chunks,
            "retrieval_corpus": "corpus_plus_gold_evidence" if args.include_evidence_chunks else "corpus_only",
        },
    )

    triples_path = processed_dir / "triples.json"
    if args.reuse_cache and triples_path.exists():
        triples = json.loads(triples_path.read_text(encoding="utf-8"))
    else:
        triples = extract_triples_controlled(
            chunks,
            args.extractor,
            llm,
            args.llm_extract_max_chunks,
            allow_rule_fallback=args.llm_extract_allow_rule_fallback,
        )
        _save_json(triples_path, triples)

    graph = build_kg(triples)
    entity_scores = compute_entity_importance(graph)
    partition, communities = detect_communities(graph)
    community_scores = compute_community_score(partition, entity_scores)
    save_graphml(graph, str(processed_dir / "graph.graphml"))
    _save_json(processed_dir / "entity_scores.json", entity_scores)
    _save_json(processed_dir / "partition.json", partition)
    _save_json(processed_dir / "communities.json", communities)

    summary_path = processed_dir / "community_summary.json"
    if args.reuse_cache and summary_path.exists():
        community_summaries = {int(k): v for k, v in json.loads(summary_path.read_text(encoding="utf-8")).items()}
    else:
        community_texts = build_community_texts(chunks, partition, triples)
        summary_llm = llm if args.community_summarizer == "llm" else None
        summary_limit = args.llm_summary_max_communities if args.llm_summary_max_communities > 0 else None
        community_summaries = summarize_communities(community_texts, summary_llm, summary_limit)
        _save_json(summary_path, community_summaries)

    index = build_retrieval_index(
        chunks,
        partition,
        triples=triples,
        community_summaries=community_summaries,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )

    methods = {
        "bm25_rag": lambda item: bm25_rag_retrieve(item["query"], index, args.top_k),
        "hybrid_rag": lambda item: hybrid_rag_retrieve(
            item["query"], index, args.top_k, vector_weight=args.hybrid_vector_weight
        ),
        "cross_encoder_rag": lambda item: cross_encoder_rag_retrieve(
            item["query"],
            index,
            args.top_k,
            candidate_pool=args.candidate_pool,
            model_name=args.cross_encoder_model,
            base_retrieval="hybrid",
            hybrid_vector_weight=args.hybrid_vector_weight,
        ),
        "vector_rag": lambda item: vector_rag_retrieve(item["query"], index, args.top_k),
        "graph_rag": lambda item: graph_rag_retrieve(item["query"], index, partition, args.top_k),
        "ablation_sim_only": lambda item: ercs_graphrag_retrieve(
            item["query"], index, entity_scores, community_scores, partition, args.top_k,
            question_type=item.get("question_type"),
            alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
        ),
        "ablation_sim_er": lambda item: ercs_graphrag_retrieve(
            item["query"], index, entity_scores, community_scores, partition, args.top_k,
            question_type=item.get("question_type"),
            alpha=0.80, beta=0.20, gamma=0.0, delta=0.0, candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
        ),
        "ablation_sim_er_rw": lambda item: ercs_graphrag_retrieve(
            item["query"], index, entity_scores, community_scores, partition, args.top_k,
            question_type=item.get("question_type"),
            alpha=0.70, beta=0.20, gamma=0.10, delta=0.0, candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
        ),
        "ablation_sim_er_rw_c": lambda item: ercs_graphrag_retrieve(
            item["query"], index, entity_scores, community_scores, partition, args.top_k,
            question_type=item.get("question_type"),
            alpha=0.55, beta=0.20, gamma=0.15, delta=0.10, epsilon=0.0,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
        ),
        "ablation_sim_er_rw_c_pcs": lambda item: ercs_graphrag_retrieve(
            item["query"], index, entity_scores, community_scores, partition, args.top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=0.52, beta=0.19, gamma=0.14, delta=0.10, epsilon=0.05,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=0.0,
            redundancy_weight=0.0,
            path_cutoff=args.path_cutoff,
        ),
        "ablation_hybrid_only": lambda item: hybrid_rag_retrieve(
            item["query"], index, args.top_k, vector_weight=args.hybrid_vector_weight
        ),
        "ablation_hybrid_ucp": lambda item: ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            args.top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=args.epsilon,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=args.adaptive_eta,
            redundancy_weight=args.redundancy_weight,
            path_cutoff=args.path_cutoff,
            base_retrieval="hybrid",
            hybrid_vector_weight=args.hybrid_vector_weight,
            base_score_weight=args.base_score_weight,
            preserve_base_top_n=args.preserve_base_top_n,
        ),
        "ablation_guarded_hybrid_ucp": lambda item: ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            args.top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=args.epsilon,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=args.adaptive_eta,
            redundancy_weight=0.0 if args.redundancy_weight > 0.03 else args.redundancy_weight,
            path_cutoff=args.path_cutoff,
            base_retrieval="hybrid",
            hybrid_vector_weight=args.hybrid_vector_weight,
            base_score_weight=args.base_score_weight if args.base_score_weight else 0.75,
            preserve_base_top_n=args.preserve_base_top_n if args.preserve_base_top_n else 3,
        ),
        "ercs_graphrag": lambda item: ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            args.top_k,
            question_type=item.get("question_type"),
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=0.0,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=0.0,
            redundancy_weight=0.0,
        ),
        "ucp_ercs_graphrag": lambda item: ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            args.top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=args.epsilon,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=args.adaptive_eta,
            redundancy_weight=args.redundancy_weight,
            path_cutoff=args.path_cutoff,
            base_retrieval="vector",
            hybrid_vector_weight=args.hybrid_vector_weight,
        ),
        "hybrid_ucp_ercs_graphrag": lambda item: ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            args.top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=args.epsilon,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=args.adaptive_eta,
            redundancy_weight=args.redundancy_weight,
            path_cutoff=args.path_cutoff,
            base_retrieval="hybrid",
            hybrid_vector_weight=args.hybrid_vector_weight,
            base_score_weight=args.base_score_weight,
            preserve_base_top_n=args.preserve_base_top_n,
        ),
        "guarded_hybrid_ucp_ercs_graphrag": lambda item: ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            args.top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=args.epsilon,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=args.adaptive_eta,
            redundancy_weight=0.0 if args.redundancy_weight > 0.03 else args.redundancy_weight,
            path_cutoff=args.path_cutoff,
            base_retrieval="hybrid",
            hybrid_vector_weight=args.hybrid_vector_weight,
            base_score_weight=args.base_score_weight if args.base_score_weight else 0.75,
            preserve_base_top_n=args.preserve_base_top_n if args.preserve_base_top_n else 3,
        ),
        "safe_hybrid_ucp_ercs_graphrag": lambda item: ercs_graphrag_retrieve(
            item["query"],
            index,
            entity_scores,
            community_scores,
            partition,
            args.top_k,
            question_type=item.get("question_type"),
            graph=graph,
            alpha=0.90,
            beta=0.10,
            gamma=0.0,
            delta=0.0,
            epsilon=0.0,
            candidate_pool=args.candidate_pool,
            query_entity_lambda=args.query_entity_lambda,
            adaptive_eta=0.0,
            redundancy_weight=0.0,
            path_cutoff=args.path_cutoff,
            base_retrieval="hybrid",
            hybrid_vector_weight=args.hybrid_vector_weight,
            base_score_weight=args.safe_base_score_weight,
            preserve_base_top_n=args.safe_preserve_top_n if args.safe_preserve_top_n else args.top_k,
        ),
    }

    methods = select_methods(methods, args.methods)
    detailed_rows, summary, grouped = [], {}, {}
    for name, retrieve in methods.items():
        metric_rows = []
        for qid, item in enumerate(qa):
            t0 = time.perf_counter()
            results = retrieve(item)
            query_time_ms = (time.perf_counter() - t0) * 1000.0
            answer_llm = llm if args.answer_generator == "llm" else None
            generated = generate_answer(item["query"], results, answer_llm)
            metrics = retrieval_metrics(results, item, args.top_k)
            metrics["answer_accuracy"] = answer_accuracy(item.get("answer", ""), generated)
            metrics["faithfulness"] = faithfulness_score(generated, results, answer_llm)
            metrics["query_time_ms"] = query_time_ms
            metric_rows.append(metrics)
            qtype = item.get("question_type", "unknown")
            grouped.setdefault(name, {}).setdefault(qtype, []).append(metrics)
            detailed_rows.append(
                {
                    "method": name,
                    "query_id": qid,
                    "original_query_id": item.get("original_query_id", qid),
                    "question_type": item.get("question_type", ""),
                    "query": item["query"],
                    "answer": item.get("answer", ""),
                    "generated_answer": generated,
                    "top_titles": " | ".join(r.get("title", "") for r in results),
                    "top_nodes": " | ".join(str(r.get("node", "")) for r in results),
                    "top_relations": " | ".join(str(r.get("relation", "")) for r in results),
                    "covered_evidence_ids": ",".join(map(str, covered_evidence_ids(results, item, args.top_k))),
                    "semantic_uncertainty": results[0].get("semantic_uncertainty", "") if results else "",
                    "adaptive_alpha": results[0].get("adaptive_alpha", "") if results else "",
                    "adaptive_beta": results[0].get("adaptive_beta", "") if results else "",
                    "adaptive_gamma": results[0].get("adaptive_gamma", "") if results else "",
                    "adaptive_delta": results[0].get("adaptive_delta", "") if results else "",
                    "adaptive_epsilon": results[0].get("adaptive_epsilon", "") if results else "",
                    "query_time_ms": query_time_ms,
                    **metrics,
                }
            )
        summary[name] = average_metrics(metric_rows)
    grouped_summary = {
        method: {qtype: average_metrics(rows) for qtype, rows in type_rows.items()}
        for method, type_rows in grouped.items()
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary_by_question_type.json").write_text(
        json.dumps(grouped_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if detailed_rows:
        with (out_dir / "details.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detailed_rows)

    print(f"chunks={len(chunks)} triples={len(triples)} nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(f"communities={len(communities)} queries={len(qa)} top_k={args.top_k}")
    for method, metrics in summary.items():
        nice = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{method}: {nice}")
    print(f"embedding_backend={index.embedding_backend}")


def _save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def extract_triples_controlled(
    chunks: list[dict],
    extractor: str,
    llm: LLMClient,
    llm_max_chunks: int,
    allow_rule_fallback: bool = False,
) -> list[dict]:
    if extractor != "llm" or llm_max_chunks <= 0:
        return extract_triples(chunks, extractor=extractor, llm=llm)
    if not llm.enabled and not allow_rule_fallback:
        raise RuntimeError("LLM extractor is unavailable. Set OPENAI_API_KEY and LLM_MODEL.")
    llm_chunks = chunks[:llm_max_chunks]
    rule_chunks = chunks[llm_max_chunks:]
    triples = extract_triples(llm_chunks, extractor="llm", llm=llm)
    triples.extend(extract_triples(rule_chunks, extractor="rule", llm=None))
    return triples


def apply_params_file(args: argparse.Namespace) -> None:
    if not args.params_file:
        return
    payload = json.loads(Path(args.params_file).read_text(encoding="utf-8"))
    params = payload.get("params", payload)
    mapping = {
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "delta": "delta",
        "epsilon": "epsilon",
        "adaptive_eta": "adaptive_eta",
        "redundancy_weight": "redundancy_weight",
        "query_entity_lambda": "query_entity_lambda",
        "candidate_pool": "candidate_pool",
        "path_cutoff": "path_cutoff",
        "hybrid_vector_weight": "hybrid_vector_weight",
        "base_score_weight": "base_score_weight",
        "preserve_base_top_n": "preserve_base_top_n",
        "safe_preserve_top_n": "safe_preserve_top_n",
        "safe_base_score_weight": "safe_base_score_weight",
    }
    for key, attr in mapping.items():
        if key in params:
            setattr(args, attr, params[key])


def select_methods(methods: dict, spec: str) -> dict:
    if spec == "all":
        return methods
    aliases = {
        "ablations": [
            "ablation_sim_only",
            "ablation_sim_er",
            "ablation_sim_er_rw",
            "ablation_sim_er_rw_c",
            "ablation_sim_er_rw_c_pcs",
            "ablation_hybrid_only",
            "ablation_hybrid_ucp",
            "ablation_guarded_hybrid_ucp",
        ],
        "safe": [
            "safe_hybrid_ucp_ercs_graphrag",
        ],
    }
    requested: list[str] = []
    for raw in spec.split(","):
        name = raw.strip()
        if not name:
            continue
        requested.extend(aliases.get(name, [name]))
    unknown = [name for name in requested if name not in methods]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    return {name: methods[name] for name in requested}


if __name__ == "__main__":
    main()
