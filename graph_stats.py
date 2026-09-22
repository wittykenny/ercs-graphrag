from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate knowledge graph statistics for the thesis.")
    parser.add_argument("--graph", default="data/processed/graph.graphml")
    parser.add_argument("--triples", default="data/processed/triples.json")
    parser.add_argument("--communities", default="data/processed/communities.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    graph = nx.read_graphml(args.graph)
    triples = json.loads(Path(args.triples).read_text(encoding="utf-8"))
    communities = json.loads(Path(args.communities).read_text(encoding="utf-8"))

    stats = graph_summary(graph, triples, communities)
    (out_dir / "graph_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counter_csv(out_dir / "relation_distribution.csv", relation_counter(triples), "relation", args.top_n)
    write_counter_csv(out_dir / "entity_frequency.csv", entity_counter(triples), "entity", args.top_n)
    write_community_csv(out_dir / "community_stats.csv", communities)
    print(f"saved {out_dir / 'graph_stats.json'}")
    print(f"saved {out_dir / 'relation_distribution.csv'}")
    print(f"saved {out_dir / 'entity_frequency.csv'}")
    print(f"saved {out_dir / 'community_stats.csv'}")


def graph_summary(graph: nx.Graph, triples: list[dict], communities: dict) -> dict:
    components = list(nx.connected_components(graph.to_undirected())) if graph.number_of_nodes() else []
    largest = max((len(c) for c in components), default=0)
    degrees = [degree for _, degree in graph.degree()]
    community_sizes = [len(nodes) for nodes in communities.values()]
    return {
        "triple_count": len(triples),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "relation_type_count": len(relation_counter(triples)),
        "avg_degree": sum(degrees) / len(degrees) if degrees else 0.0,
        "max_degree": max(degrees, default=0),
        "connected_components": len(components),
        "largest_component_ratio": largest / graph.number_of_nodes() if graph.number_of_nodes() else 0.0,
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
        "community_count": len(communities),
        "avg_community_size": sum(community_sizes) / len(community_sizes) if community_sizes else 0.0,
        "max_community_size": max(community_sizes, default=0),
    }


def relation_counter(triples: list[dict]) -> Counter:
    return Counter(tri.get("relation", "unknown") for tri in triples)


def entity_counter(triples: list[dict]) -> Counter:
    counter = Counter()
    for tri in triples:
        counter[tri.get("head", "")] += 1
        counter[tri.get("tail", "")] += 1
    counter.pop("", None)
    return counter


def write_counter_csv(path: Path, counter: Counter, key_name: str, top_n: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key_name, "count"])
        writer.writeheader()
        for key, count in counter.most_common(top_n):
            writer.writerow({key_name: key, "count": count})


def write_community_csv(path: Path, communities: dict) -> None:
    rows = sorted(
        [{"community_id": cid, "size": len(nodes), "sample_nodes": " | ".join(nodes[:8])} for cid, nodes in communities.items()],
        key=lambda x: x["size"],
        reverse=True,
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["community_id", "size", "sample_nodes"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

