from __future__ import annotations

import re

from .evaluate import normalize
from .llm import LLMClient
from .retrieval import generate_answer as extractive_answer


def generate_answer(query: str, retrieved: list[dict], llm: LLMClient | None = None) -> str:
    if not llm or not llm.enabled:
        return extractive_answer(query, retrieved)
    context = "\n\n".join(
        f"[{i + 1}] {item.get('title', '')}: {item.get('text', '')}" for i, item in enumerate(retrieved)
    )
    system = "Answer questions using only the supplied context. Be concise."
    user = f"""
Question:
{query}

Context:
{context}

Return only the final answer. If the context is insufficient, say "Insufficient evidence".
"""
    try:
        return llm.complete(system, user, max_tokens=300).strip()
    except Exception:
        return extractive_answer(query, retrieved)


def faithfulness_score(answer: str, retrieved: list[dict], llm: LLMClient | None = None) -> float:
    context = "\n\n".join(item.get("text", "") for item in retrieved)
    if not answer.strip():
        return 0.0
    if llm and llm.enabled:
        system = "You judge whether an answer is fully supported by context. Return JSON only."
        user = f"""
Context:
{context}

Answer:
{answer}

Return JSON: {{"faithful": true}} if every factual claim in the answer is supported by the context,
otherwise {{"faithful": false}}.
"""
        try:
            verdict = llm.complete(system, user, max_tokens=80).lower()
            return 1.0 if "true" in verdict and "false" not in verdict else 0.0
        except Exception:
            pass
    return lexical_faithfulness(answer, context)


def lexical_faithfulness(answer: str, context: str) -> float:
    ans = normalize(answer)
    ctx = normalize(context)
    if ans and ans in ctx:
        return 1.0
    tokens = [t for t in re.findall(r"[a-z0-9-]+", ans) if len(t) > 3]
    if not tokens:
        return 0.0
    supported = sum(1 for token in tokens if token in ctx)
    return supported / len(tokens)

