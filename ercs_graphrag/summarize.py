from __future__ import annotations

from .extract import extract_entities
from .llm import LLMClient


def build_community_texts(chunks: list[dict], partition: dict[str, int], triples: list[dict]) -> dict[int, list[str]]:
    chunk_nodes: dict[str, str] = {}
    for tri in triples:
        chunk_nodes.setdefault(tri.get("chunk_id", ""), tri.get("head", ""))

    community_texts: dict[int, list[str]] = {}
    for chunk in chunks:
        node = chunk_nodes.get(chunk["chunk_id"], "")
        if not node:
            entities = extract_entities(chunk["text"], limit=1)
            node = entities[0] if entities else chunk.get("title", "")
        cid = partition.get(node)
        if cid is not None:
            community_texts.setdefault(cid, []).append(chunk["text"])
    return community_texts


def summarize_communities(
    community_texts: dict[int, list[str]],
    llm: LLMClient | None = None,
    max_communities: int | None = None,
) -> dict[int, str]:
    summaries: dict[int, str] = {}
    for n, (cid, texts) in enumerate(community_texts.items()):
        if max_communities is not None and n >= max_communities:
            summaries[cid] = extractive_summary(texts)
            continue
        if llm and llm.enabled:
            summaries[cid] = summarize_with_llm(cid, texts, llm)
        else:
            summaries[cid] = extractive_summary(texts)
    return summaries


def summarize_with_llm(cid: int, texts: list[str], llm: LLMClient) -> str:
    joined = "\n".join(f"- {text[:500]}" for text in texts[:12])
    system = "You summarize graph communities for GraphRAG retrieval. Be factual and compact."
    user = f"""
Community ID: {cid}

Texts:
{joined}

Write a concise community summary covering:
- main entities
- important relations
- events or claims useful for question answering
"""
    try:
        return llm.complete(system, user, max_tokens=350).strip()
    except Exception:
        return extractive_summary(texts)


def extractive_summary(texts: list[str], max_chars: int = 2500) -> str:
    return " ".join(texts[:8])[:max_chars]

