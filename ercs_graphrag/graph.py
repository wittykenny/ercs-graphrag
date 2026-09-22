from __future__ import annotations

import networkx as nx


def build_kg(triples: list[dict]) -> nx.Graph:
    graph = nx.Graph()
    for tri in triples:
        head, tail = tri["head"], tri["tail"]
        graph.add_node(head)
        graph.add_node(tail)
        if graph.has_edge(head, tail):
            graph[head][tail]["weight"] += 1
            graph[head][tail]["relations"].add(tri["relation"])
        else:
            graph.add_edge(
                head,
                tail,
                weight=1,
                relations={tri["relation"]},
                evidence=tri.get("evidence", ""),
            )
    return graph


def compute_entity_importance(graph: nx.Graph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    pagerank = nx.pagerank(graph, weight="weight")
    degree = nx.degree_centrality(graph)
    raw = {node: 0.7 * pagerank[node] + 0.3 * degree[node] for node in graph.nodes}
    max_score = max(raw.values()) or 1.0
    return {node: score / max_score for node, score in raw.items()}


def detect_communities(graph: nx.Graph) -> tuple[dict[str, int], dict[int, list[str]]]:
    if graph.number_of_nodes() == 0:
        return {}, {}
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    partition: dict[str, int] = {}
    grouped: dict[int, list[str]] = {}
    for cid, nodes in enumerate(communities):
        grouped[cid] = sorted(nodes)
        for node in nodes:
            partition[node] = cid
    return partition, grouped


def compute_community_score(partition: dict[str, int], entity_scores: dict[str, float]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for node, cid in partition.items():
        scores[cid] = scores.get(cid, 0.0) + entity_scores.get(node, 0.0)
    max_score = max(scores.values(), default=1.0) or 1.0
    return {cid: score / max_score for cid, score in scores.items()}


def save_graphml(graph: nx.Graph, path: str) -> None:
    export_graph = nx.Graph()
    for node, data in graph.nodes(data=True):
        export_graph.add_node(str(node), **{k: str(v) for k, v in data.items()})
    for head, tail, data in graph.edges(data=True):
        attrs = {}
        for key, value in data.items():
            if isinstance(value, set):
                attrs[key] = ",".join(sorted(value))
            else:
                attrs[key] = value
        export_graph.add_edge(str(head), str(tail), **attrs)
    nx.write_graphml(export_graph, path)
