from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a compact knowledge graph subgraph for thesis figures.")
    parser.add_argument("--graph", default="data/processed/graph.graphml")
    parser.add_argument("--output", default="results/kg_subgraph.png")
    parser.add_argument("--nodes-csv", default="results/kg_central_nodes.csv")
    parser.add_argument("--top-n", type=int, default=45)
    parser.add_argument("--label-n", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = nx.read_graphml(args.graph)
    ranked = rank_nodes(graph)
    selected = [node for node, _ in ranked[: args.top_n]]
    subgraph = graph.subgraph(selected).copy()
    if subgraph.number_of_edges() == 0 and selected:
        subgraph = graph.subgraph(expand_with_neighbors(graph, selected[: min(10, len(selected))])).copy()

    draw_subgraph(subgraph, ranked, Path(args.output), args.label_n, args.seed)
    write_nodes_csv(Path(args.nodes_csv), graph, ranked[: args.top_n])
    print(f"saved {args.output}")
    print(f"saved {args.nodes_csv}")


def rank_nodes(graph: nx.Graph) -> list[tuple[str, float]]:
    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except Exception:
        pagerank = {node: 1.0 for node in graph.nodes}
    degree = nx.degree_centrality(graph)
    scores = {
        node: 0.7 * pagerank.get(node, 0.0) + 0.3 * degree.get(node, 0.0)
        for node in graph.nodes
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def expand_with_neighbors(graph: nx.Graph, seeds: list[str]) -> list[str]:
    nodes = set(seeds)
    for seed in seeds:
        nodes.update(list(graph.neighbors(seed))[:5])
    return list(nodes)


def draw_subgraph(
    graph: nx.Graph,
    ranked: list[tuple[str, float]],
    output: Path,
    label_n: int,
    seed: int,
) -> None:
    output.parent.mkdir(exist_ok=True)
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    rank_map = dict(ranked)
    node_scores = [rank_map.get(node, 0.0) for node in graph.nodes]
    max_score = max(node_scores) or 1.0
    node_sizes = [250 + 1800 * score / max_score for score in node_scores]
    pos = nx.spring_layout(graph, seed=seed, k=0.8)

    plt.figure(figsize=(14, 10), dpi=180)
    nx.draw_networkx_edges(graph, pos, alpha=0.25, width=0.8, edge_color="#7a8798")
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_scores,
        cmap=plt.cm.viridis,
        linewidths=0.6,
        edgecolors="#263238",
        alpha=0.92,
    )
    label_nodes = {node for node, _ in ranked[:label_n] if node in graph.nodes}
    labels = {node: shorten(node, 28) for node in label_nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_color="#111827")
    plt.title("Knowledge Graph Core Subgraph", fontsize=16, pad=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def write_nodes_csv(path: Path, graph: nx.Graph, ranked: list[tuple[str, float]]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "node", "score", "degree"])
        writer.writeheader()
        for idx, (node, score) in enumerate(ranked, 1):
            writer.writerow(
                {
                    "rank": idx,
                    "node": node,
                    "score": score,
                    "degree": graph.degree(node),
                }
            )


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    main()

