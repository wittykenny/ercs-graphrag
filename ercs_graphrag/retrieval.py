from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .extract import extract_entities, get_relation_weight, infer_relation


_CROSS_ENCODER_CACHE: dict[str, Any] = {}


@dataclass
class RetrievalIndex:
    chunks: list[dict]
    encoder: Any
    embedding_backend: str
    chunk_matrix: Any
    bm25: "BM25Index"
    chunk_nodes: dict[str, str]
    chunk_relations: dict[str, str]
    entity_specificity: dict[str, float]
    community_relations: dict[int, dict[str, int]]
    community_summaries: dict[int, str]
    community_matrix: Any | None
    score_cache: dict[str, np.ndarray]


def build_retrieval_index(
    chunks: list[dict],
    partition: dict[str, int],
    triples: list[dict] | None = None,
    community_summaries: dict[int, str] | None = None,
    embedding_backend: str = "sbert",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> RetrievalIndex:
    texts = [c["text"] for c in chunks]
    encoder, backend, chunk_matrix = _fit_embeddings(texts, embedding_backend, embedding_model)
    bm25 = BM25Index.from_texts(texts)

    chunk_nodes, chunk_relations = {}, {}
    community_texts: dict[int, list[str]] = {}
    community_relations: dict[int, dict[str, int]] = {}
    triple_meta = _chunk_meta_from_triples(triples or [])
    for chunk in chunks:
        node, relation = triple_meta.get(chunk["chunk_id"], ("", ""))
        if not node:
            entities = extract_entities(chunk["text"], limit=3)
            node = entities[0] if entities else chunk["title"]
        if not relation:
            relation = infer_relation(chunk["text"])
        chunk_nodes[chunk["chunk_id"]] = node
        chunk_relations[chunk["chunk_id"]] = relation
        cid = partition.get(node)
        if cid is not None:
            community_texts.setdefault(cid, []).append(chunk["text"])
            rels = community_relations.setdefault(cid, {})
            rels[relation] = rels.get(relation, 0) + 1
    entity_specificity = _compute_entity_specificity(chunk_nodes, len(chunks))

    community_summaries = community_summaries or {
        cid: " ".join(items[:8])[:2500] for cid, items in community_texts.items()
    }
    community_matrix = None
    if community_summaries:
        community_matrix = _encode(list(community_summaries.values()), encoder, backend)
    return RetrievalIndex(
        chunks=chunks,
        encoder=encoder,
        embedding_backend=backend,
        chunk_matrix=chunk_matrix,
        bm25=bm25,
        chunk_nodes=chunk_nodes,
        chunk_relations=chunk_relations,
        entity_specificity=entity_specificity,
        community_relations=community_relations,
        community_summaries=community_summaries,
        community_matrix=community_matrix,
        score_cache={},
    )


def vector_rag_retrieve(query: str, index: RetrievalIndex, top_k: int = 5) -> list[dict]:
    sims = _chunk_sims(query, index)
    return _rank(index.chunks, sims, "vector_score", top_k)


def bm25_rag_retrieve(query: str, index: RetrievalIndex, top_k: int = 5) -> list[dict]:
    scores = index.bm25.score(query)
    return _rank(index.chunks, scores, "bm25_score", top_k)


def hybrid_rag_retrieve(
    query: str,
    index: RetrievalIndex,
    top_k: int = 5,
    vector_weight: float = 0.60,
) -> list[dict]:
    vector_scores = minmax(_chunk_sims(query, index))
    bm25_scores = minmax(index.bm25.score(query))
    scores = vector_weight * vector_scores + (1.0 - vector_weight) * bm25_scores
    ranked = _rank(index.chunks, scores, "hybrid_score", top_k)
    for item in ranked:
        chunk_id = item.get("chunk_id")
        idx = next((i for i, c in enumerate(index.chunks) if c.get("chunk_id") == chunk_id), None)
        if idx is not None:
            item["vector_component"] = float(vector_scores[idx])
            item["bm25_component"] = float(bm25_scores[idx])
    return ranked


def cross_encoder_rag_retrieve(
    query: str,
    index: RetrievalIndex,
    top_k: int = 5,
    candidate_pool: int = 80,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    base_retrieval: str = "hybrid",
    hybrid_vector_weight: float = 0.60,
) -> list[dict]:
    base_scores = candidate_scores(query, index, base_retrieval, hybrid_vector_weight)
    candidate_pool = min(candidate_pool, len(index.chunks))
    candidate_ids = np.argsort(base_scores)[::-1][:candidate_pool]
    try:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        from sentence_transformers import CrossEncoder

        model = _CROSS_ENCODER_CACHE.get(model_name)
        if model is None:
            model = CrossEncoder(model_name)
            _CROSS_ENCODER_CACHE[model_name] = model
        pairs = [(query, index.chunks[int(i)]["text"]) for i in candidate_ids]
        rerank_scores = np.asarray(model.predict(pairs), dtype=float)
        scorer = "cross_encoder"
    except Exception:
        rerank_scores = base_scores[candidate_ids]
        scorer = "base_fallback"
    ranked_ids = candidate_ids[np.argsort(rerank_scores)[::-1][:top_k]]
    ranked = []
    score_map = {int(i): float(s) for i, s in zip(candidate_ids, rerank_scores)}
    for idx in ranked_ids:
        item = dict(index.chunks[int(idx)])
        item["score"] = score_map[int(idx)]
        item["cross_encoder_score"] = score_map[int(idx)]
        item["base_retrieval"] = base_retrieval
        item["cross_encoder_scorer"] = scorer
        ranked.append(item)
    return ranked


def graph_rag_retrieve(
    query: str,
    index: RetrievalIndex,
    partition: dict[str, int],
    top_k: int = 5,
) -> list[dict]:
    chunk_sims = _chunk_sims(query, index)
    community_sims = _community_sims(query, index)
    scores = []
    for i, chunk in enumerate(index.chunks):
        node = index.chunk_nodes.get(chunk["chunk_id"], "")
        cid = partition.get(node)
        graph_sim = community_sims.get(cid, 0.0)
        scores.append(0.65 * chunk_sims[i] + 0.35 * graph_sim)
    return _rank(index.chunks, np.array(scores), "graph_score", top_k)


def ercs_graphrag_retrieve(
    query: str,
    index: RetrievalIndex,
    entity_scores: dict[str, float],
    community_scores: dict[int, float],
    partition: dict[str, int],
    top_k: int = 5,
    question_type: str | None = None,
    graph: nx.Graph | None = None,
    alpha: float = 0.55,
    beta: float = 0.20,
    gamma: float = 0.15,
    delta: float = 0.10,
    epsilon: float = 0.0,
    candidate_pool: int = 80,
    query_entity_lambda: float = 0.60,
    adaptive_eta: float = 0.0,
    redundancy_weight: float = 0.0,
    path_cutoff: int = 4,
    base_retrieval: str = "vector",
    hybrid_vector_weight: float = 0.60,
    base_score_weight: float = 0.0,
    preserve_base_top_n: int = 0,
) -> list[dict]:
    vector_sims = _chunk_sims(query, index)
    retrieval_scores = candidate_scores(query, index, base_retrieval, hybrid_vector_weight, vector_sims)
    candidate_pool = min(candidate_pool, len(index.chunks))
    candidate_ids = np.argsort(retrieval_scores)[::-1][:candidate_pool]
    pool_sims = retrieval_scores[candidate_ids]
    sim_min, sim_max = float(pool_sims.min()), float(pool_sims.max())
    sim_den = sim_max - sim_min or 1.0
    uncertainty = semantic_uncertainty(pool_sims)
    weights = adaptive_feature_weights(alpha, beta, gamma, delta, epsilon, uncertainty, adaptive_eta)
    community_sims = _community_sims(query, index)
    query_nodes = infer_query_nodes(query, graph) if graph is not None and epsilon > 0 else []
    query_path_lengths = query_graph_distances(query_nodes, graph, path_cutoff) if graph is not None and epsilon > 0 else {}
    results = []
    for i in candidate_ids:
        chunk = index.chunks[int(i)]
        node = index.chunk_nodes.get(chunk["chunk_id"], "")
        relation = index.chunk_relations.get(chunk["chunk_id"], "related_to")
        cid = partition.get(node)
        global_er = entity_scores.get(node, 0.0)
        q_match = query_entity_match(query, node, chunk["text"])
        specificity = index.entity_specificity.get(node, 0.0)
        er = query_aware_entity_importance(global_er, q_match, specificity, query_entity_lambda)
        rw = get_relation_weight(relation, question_type)
        cs = query_aware_community_score(
            cid, community_scores, community_sims, index.community_relations, question_type
        )
        pcs = path_consistency_score(node, query_path_lengths, path_cutoff)
        semantic_sim = (float(retrieval_scores[i]) - sim_min) / sim_den
        graph_score = (
            weights["alpha"] * semantic_sim
            + weights["beta"] * er
            + weights["gamma"] * rw
            + weights["delta"] * cs
            + weights["epsilon"] * pcs
        )
        score = (1.0 - base_score_weight) * graph_score + base_score_weight * semantic_sim
        base_rank = int(np.where(candidate_ids == i)[0][0]) + 1
        item = dict(chunk)
        item.update(
            {
                "score": float(score),
                "base_score": float(score),
                "graph_rerank_score": float(graph_score),
                "semantic_sim": semantic_sim,
                "raw_semantic_sim": float(vector_sims[i]),
                "base_retrieval_score": float(retrieval_scores[i]),
                "base_retrieval": base_retrieval,
                "base_rank": base_rank,
                "base_score_weight": float(base_score_weight),
                "preserve_base_top_n": float(preserve_base_top_n),
                "safe_no_regret_policy": float(base_retrieval == "hybrid" and preserve_base_top_n >= top_k),
                "global_entity_importance": float(global_er),
                "query_entity_match": float(q_match),
                "entity_specificity": float(specificity),
                "query_aware_entity_importance": float(er),
                "entity_importance": float(er),
                "relation_weight": float(rw),
                "community_score": float(cs),
                "community_id": cid,
                "path_consistency": float(pcs),
                "semantic_uncertainty": float(uncertainty),
                "adaptive_alpha": float(weights["alpha"]),
                "adaptive_beta": float(weights["beta"]),
                "adaptive_gamma": float(weights["gamma"]),
                "adaptive_delta": float(weights["delta"]),
                "adaptive_epsilon": float(weights["epsilon"]),
                "candidate_pool": float(candidate_pool),
                "rerank_stage": f"ucp_{base_retrieval}_candidate_rerank",
                "node": node,
                "relation": relation,
                "question_type": question_type or "",
            }
        )
        results.append(item)
    return coverage_aware_select(results, top_k, redundancy_weight, preserve_base_top_n)


def candidate_scores(
    query: str,
    index: RetrievalIndex,
    base_retrieval: str = "vector",
    hybrid_vector_weight: float = 0.60,
    vector_sims: np.ndarray | None = None,
) -> np.ndarray:
    cache_key = f"{base_retrieval}::{hybrid_vector_weight:.6f}::{query}"
    if vector_sims is None and cache_key in index.score_cache:
        return index.score_cache[cache_key]
    vector_sims = _chunk_sims(query, index) if vector_sims is None else vector_sims
    if base_retrieval == "vector":
        index.score_cache[cache_key] = vector_sims
        return vector_sims
    bm25_scores = index.bm25.score(query)
    if base_retrieval == "bm25":
        index.score_cache[cache_key] = bm25_scores
        return bm25_scores
    if base_retrieval == "hybrid":
        scores = hybrid_vector_weight * minmax(vector_sims) + (1.0 - hybrid_vector_weight) * minmax(bm25_scores)
        index.score_cache[cache_key] = scores
        return scores
    raise ValueError(f"Unknown base_retrieval: {base_retrieval}")


def generate_answer(query: str, retrieved: list[dict]) -> str:
    query_terms = {t for t in re.findall(r"[a-zA-Z0-9-]+", query.lower()) if len(t) > 3}
    best_sentence, best_overlap = "", -1
    for item in retrieved:
        for sent in re.split(r"(?<=[.!?])\s+", item["text"]):
            terms = set(re.findall(r"[a-zA-Z0-9-]+", sent.lower()))
            overlap = len(query_terms & terms)
            if overlap > best_overlap:
                best_sentence, best_overlap = sent, overlap
    return best_sentence or (retrieved[0]["text"] if retrieved else "")


def _chunk_sims(query: str, index: RetrievalIndex) -> np.ndarray:
    cache_key = f"vector::1.000000::{query}"
    if cache_key in index.score_cache:
        return index.score_cache[cache_key]
    query_vec = _encode([query], index.encoder, index.embedding_backend)
    sims = cosine_similarity(query_vec, index.chunk_matrix).ravel()
    index.score_cache[cache_key] = sims
    return sims


def query_entity_match(query: str, node: str, chunk_text: str = "") -> float:
    query_norm = _normalize(query)
    node_norm = _normalize(node)
    if not node_norm:
        return 0.0
    if node_norm in query_norm:
        return 1.0
    node_tokens = _content_tokens(node_norm)
    query_tokens = _content_tokens(query_norm)
    if not node_tokens:
        return 0.0
    overlap = len(node_tokens & query_tokens) / len(node_tokens)
    chunk_bonus = 0.15 if node_norm and node_norm in _normalize(chunk_text) else 0.0
    return min(1.0, overlap + chunk_bonus)


def query_aware_entity_importance(
    global_er: float,
    query_match: float,
    specificity: float,
    query_entity_lambda: float,
) -> float:
    base = query_entity_lambda * global_er + (1.0 - query_entity_lambda) * query_match
    return base * (0.50 + 0.50 * specificity)


def semantic_uncertainty(pool_sims: np.ndarray) -> float:
    if len(pool_sims) <= 1:
        return 0.0
    ordered = np.sort(pool_sims)[::-1]
    margin = float(ordered[0] - ordered[-1])
    scale = max(abs(float(ordered[0])), 1e-6)
    confidence = min(1.0, max(0.0, margin / scale))
    return 1.0 - confidence


def adaptive_feature_weights(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    epsilon: float,
    uncertainty: float,
    eta: float,
) -> dict[str, float]:
    sem = max(0.0, alpha * (1.0 - eta * uncertainty))
    graph_boost = 1.0 + eta * uncertainty
    weights = {
        "alpha": sem,
        "beta": max(0.0, beta * graph_boost),
        "gamma": max(0.0, gamma * graph_boost),
        "delta": max(0.0, delta * graph_boost),
        "epsilon": max(0.0, epsilon * graph_boost),
    }
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def query_aware_community_score(
    cid: int | None,
    community_scores: dict[int, float],
    community_sims: dict[int, float],
    community_relations: dict[int, dict[str, int]],
    question_type: str | None,
) -> float:
    if cid is None:
        return 0.0
    base = community_scores.get(cid, 0.0)
    sim = max(0.0, community_sims.get(cid, 0.0))
    rel_profile = community_relations.get(cid, {})
    if rel_profile:
        total = sum(rel_profile.values()) or 1
        type_weight = sum(get_relation_weight(rel, question_type) * count for rel, count in rel_profile.items()) / total
    else:
        type_weight = 0.5
    return base * (0.35 + 0.45 * sim + 0.20 * type_weight)


def infer_query_nodes(query: str, graph: nx.Graph | None, limit: int = 8) -> list[str]:
    if graph is None:
        return []
    query_norm = _normalize(query)
    candidates = []
    seen = set()
    for ent in extract_entities(query, limit=limit):
        if graph.has_node(ent) and ent not in seen:
            candidates.append(ent)
            seen.add(ent)
    for node in graph.nodes:
        if len(candidates) >= limit:
            break
        node_norm = _normalize(str(node))
        if len(node_norm) < 3 or node in seen:
            continue
        if node_norm in query_norm:
            candidates.append(str(node))
            seen.add(node)
    return candidates


def query_graph_distances(query_nodes: list[str], graph: nx.Graph | None, cutoff: int) -> dict[str, int]:
    if graph is None:
        return {}
    distances: dict[str, int] = {}
    for node in query_nodes:
        if not graph.has_node(node):
            continue
        for target, length in nx.single_source_shortest_path_length(graph, node, cutoff=cutoff).items():
            if target not in distances or length < distances[target]:
                distances[target] = length
    return distances


def path_consistency_score(node: str, query_path_lengths: dict[str, int], cutoff: int) -> float:
    if not node or node not in query_path_lengths:
        return 0.0
    length = query_path_lengths[node]
    if length == 0:
        return 1.0
    return max(0.0, (cutoff - length + 1) / cutoff)


def coverage_aware_select(
    candidates: list[dict],
    top_k: int,
    redundancy_weight: float,
    preserve_base_top_n: int = 0,
) -> list[dict]:
    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    selected: list[dict] = []
    if preserve_base_top_n > 0:
        base_ranked = sorted(candidates, key=lambda x: x.get("base_rank", 10**9))
        for item in base_ranked[:preserve_base_top_n]:
            chosen = dict(item)
            chosen["selection_score"] = chosen["score"]
            chosen["diversity_penalty"] = 0.0
            chosen["preserved_base_candidate"] = 1.0
            selected.append(chosen)
        ranked = [item for item in ranked if item.get("chunk_id") not in {x.get("chunk_id") for x in selected}]
        if len(selected) >= top_k:
            return selected[:top_k]
    if redundancy_weight <= 0:
        for item in ranked[:top_k]:
            item["selection_score"] = item["score"]
            item["diversity_penalty"] = 0.0
            item["preserved_base_candidate"] = 0.0
        return selected + ranked[: max(0, top_k - len(selected))]

    remaining = ranked[:]
    while remaining and len(selected) < top_k:
        best_idx, best_selection_score, best_penalty = 0, -1e9, 0.0
        for idx, item in enumerate(remaining):
            penalty = redundancy_penalty(item, selected)
            selection_score = item["score"] - redundancy_weight * penalty
            if selection_score > best_selection_score:
                best_idx = idx
                best_selection_score, best_penalty = selection_score, penalty
        chosen = remaining.pop(best_idx)
        chosen["selection_score"] = float(best_selection_score)
        chosen["diversity_penalty"] = float(best_penalty)
        chosen["preserved_base_candidate"] = 0.0
        selected.append(chosen)
    return selected


def redundancy_penalty(item: dict, selected: list[dict]) -> float:
    if not selected:
        return 0.0
    penalty = 0.0
    for other in selected:
        if item.get("title") and item.get("title") == other.get("title"):
            penalty = max(penalty, 0.45)
        if item.get("node") and item.get("node") == other.get("node"):
            penalty = max(penalty, 0.35)
        if item.get("community_id") is not None and item.get("community_id") == other.get("community_id"):
            penalty = max(penalty, 0.20)
    return penalty


def _community_sims(query: str, index: RetrievalIndex) -> dict[int, float]:
    if not index.community_summaries or index.community_matrix is None:
        return {}
    query_vec = _encode([query], index.encoder, index.embedding_backend)
    sims = cosine_similarity(query_vec, index.community_matrix).ravel()
    return dict(zip(index.community_summaries.keys(), map(float, sims)))


def _rank(chunks: list[dict], scores: np.ndarray, score_name: str, top_k: int) -> list[dict]:
    order = np.argsort(scores)[::-1][:top_k]
    ranked = []
    for idx in order:
        item = dict(chunks[int(idx)])
        item["score"] = float(scores[int(idx)])
        item[score_name] = float(scores[int(idx)])
        ranked.append(item)
    return ranked


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    low, high = float(scores.min()), float(scores.max())
    den = high - low
    if den <= 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - low) / den


@dataclass
class BM25Index:
    tokenized_docs: list[list[str]]
    doc_freqs: dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def from_texts(cls, texts: list[str]) -> "BM25Index":
        tokenized = [bm25_tokens(text) for text in texts]
        doc_freqs: dict[str, int] = {}
        for doc in tokenized:
            for token in set(doc):
                doc_freqs[token] = doc_freqs.get(token, 0) + 1
        avgdl = sum(len(doc) for doc in tokenized) / max(1, len(tokenized))
        return cls(tokenized, doc_freqs, avgdl)

    def score(self, query: str) -> np.ndarray:
        query_terms = bm25_tokens(query)
        scores = np.zeros(len(self.tokenized_docs), dtype=float)
        if not query_terms or not self.tokenized_docs:
            return scores
        n_docs = len(self.tokenized_docs)
        for term in query_terms:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, doc in enumerate(self.tokenized_docs):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = len(doc) or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


def bm25_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]{1,}", str(text).lower())
        if token not in _BM25_STOPWORDS and len(token) > 1
    ]


def _fit_embeddings(texts: list[str], backend: str, model_name: str) -> tuple[Any, str, Any]:
    if backend in {"sbert", "auto"}:
        try:
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            matrix = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            return model, "sbert", np.asarray(matrix)
        except Exception:
            if backend == "sbert":
                raise

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, "tfidf", matrix


def _encode(texts: list[str], encoder: Any, backend: str) -> Any:
    if backend == "sbert":
        return np.asarray(
            encoder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        )
    return encoder.transform(texts)


def _chunk_meta_from_triples(triples: list[dict]) -> dict[str, tuple[str, str]]:
    meta: dict[str, tuple[str, str]] = {}
    for tri in triples:
        chunk_id = tri.get("chunk_id")
        if chunk_id and chunk_id not in meta:
            meta[chunk_id] = (tri.get("head", ""), tri.get("relation", "related_to"))
    return meta


def _compute_entity_specificity(chunk_nodes: dict[str, str], chunk_count: int) -> dict[str, float]:
    df: dict[str, int] = {}
    for node in chunk_nodes.values():
        if node:
            df[node] = df.get(node, 0) + 1
    raw = {node: np.log((chunk_count + 1) / (count + 1)) for node, count in df.items()}
    max_score = max(raw.values(), default=1.0) or 1.0
    return {node: float(score / max_score) for node, score in raw.items()}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "who", "what", "when",
        "where", "which", "whose", "was", "were", "are", "has", "have", "had", "its",
        "into", "about", "according", "reported", "both",
    }
    return {t for t in text.split() if len(t) > 2 and t not in stop}


_BM25_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "who", "what", "when",
    "where", "which", "whose", "was", "were", "are", "has", "have", "had", "its",
    "into", "about", "according", "reported", "both", "source", "category", "will",
    "would", "could", "should", "there", "their", "they", "them", "than", "then",
    "also", "after", "before", "over", "under", "between", "among",
}
