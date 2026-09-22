from __future__ import annotations

import re
from itertools import combinations

from .llm import LLMClient, parse_json_array


RELATION_KEYWORDS = {
    "acquired": "acquired",
    "acquires": "acquired",
    "acquisition": "acquired",
    "bought": "acquired",
    "invested": "invested",
    "funding": "invested",
    "founded": "founded",
    "co-founded": "founded",
    "ceo": "led_by",
    "chief executive": "led_by",
    "trial": "accused_of",
    "fraud": "accused_of",
    "lawsuit": "accused_of",
    "partnership": "partnered_with",
    "partnered": "partnered_with",
    "launched": "launched",
    "released": "launched",
    "announced": "announced",
    "located": "located_in",
    "based": "located_in",
}

RELATION_WEIGHTS = {
    "accused_of": 1.0,
    "acquired": 0.9,
    "invested": 0.85,
    "founded": 0.85,
    "led_by": 0.8,
    "partnered_with": 0.75,
    "launched": 0.7,
    "announced": 0.6,
    "located_in": 0.35,
    "related_to": 0.5,
}

QUESTION_TYPE_RELATION_BONUS = {
    "inference_query": {
        "accused_of": 0.20,
        "acquired": 0.15,
        "invested": 0.12,
        "founded": 0.12,
        "led_by": 0.12,
        "partnered_with": 0.10,
    },
    "comparison_query": {
        "related_to": 0.12,
        "announced": 0.10,
        "launched": 0.10,
        "located_in": 0.08,
        "founded": 0.06,
    },
    "temporal_query": {
        "announced": 0.20,
        "launched": 0.18,
        "released": 0.18,
        "acquired": 0.10,
        "invested": 0.10,
    },
    "null_query": {},
}


ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.'-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9&.'-]+|[A-Z]{2,}))*"
)


ENTITY_STOPWORDS = {
    "A", "An", "The", "This", "That", "These", "Those",
    "Source", "Category", "Table", "Contents", "Update", "Updates",
    "News", "Live", "Everything", "How", "What", "Why", "When", "Where",
    "Who", "Whose", "Which", "Is", "Are", "Was", "Were", "Did", "Does",
    "Do", "Can", "Could", "Should", "Will", "Would", "There", "Here",
    "It", "He", "She", "They", "We", "I", "You", "From", "Meet",
}

GENERIC_ENTITY_TERMS = {
    "TV", "NFL", "NBA", "MLB", "Apps", "App", "Moneyline", "Find", "Live",
    "Source", "Category", "Health", "Business", "Technology", "Entertainment",
    "Sports", "Science", "Life", "Style", "Music", "News", "Watch",
}

SOURCE_NAMES = {
    "The Verge", "TechCrunch", "Engadget", "Polygon", "The Guardian",
    "Sporting News", "CBSSports.com", "FOX News", "Fortune", "Business Line",
    "The Independent", "Yardbarker", "TalkSport", "The Age", "BBC News",
    "Sky Sports", "Hacker News", "Essentially Sports", "Globes English",
    "Live Science", "Eos", "Music Business Worldwide",
}


def clean_text_for_extraction(text: str) -> str:
    """Remove retrieval metadata and common boilerplate before KG extraction."""
    text = str(text)
    text = re.sub(r"\bSource:\s*[^.]{0,120}\.\s*Category:\s*[^.]{0,80}\.", " ", text)
    text = re.sub(r"\bSource:\s*[^.]{0,120}\.", " ", text)
    text = re.sub(r"\bCategory:\s*[^.]{0,80}\.", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(text: str, limit: int = 8) -> list[str]:
    text = clean_text_for_extraction(text)
    entities: list[str] = []
    for match in ENTITY_RE.finditer(text):
        ent = match.group(0).strip(" .,:;()[]")
        if not is_valid_entity(ent):
            continue
        if ent not in entities:
            entities.append(ent)
        if len(entities) >= limit:
            break
    return entities


def is_valid_entity(entity: str) -> bool:
    ent = entity.strip(" .,:;()[]'\"")
    low = ent.lower()
    if len(ent) < 2 or low.startswith("http"):
        return False
    if ent in ENTITY_STOPWORDS or ent in SOURCE_NAMES:
        return False
    if "source" in low or "category" in low:
        return False
    if ". category" in low or low.endswith(".com. category"):
        return False
    tokens = ent.split()
    if tokens and (tokens[0] in ENTITY_STOPWORDS or tokens[-1] in ENTITY_STOPWORDS):
        return False
    if len(tokens) == 1 and ent in GENERIC_ENTITY_TERMS:
        return False
    if len(tokens) == 1 and len(ent) <= 2 and not ent.isupper():
        return False
    if re.search(r"\b(Source|Category)\b", ent):
        return False
    if re.fullmatch(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*", ent):
        return False
    return True


def infer_relation(text: str) -> str:
    low = clean_text_for_extraction(text).lower()
    for key, relation in RELATION_KEYWORDS.items():
        if key in low:
            return relation
    return "related_to"


def extract_triples(chunks: list[dict], extractor: str = "rule", llm: LLMClient | None = None) -> list[dict]:
    triples: list[dict] = []
    for chunk in chunks:
        if extractor == "llm" and llm and llm.enabled:
            triples.extend(extract_triples_with_llm(chunk, llm))
            continue
        cleaned_text = clean_text_for_extraction(chunk["text"])
        entities = extract_entities(cleaned_text)
        relation = infer_relation(cleaned_text)
        for head, tail in combinations(entities[:4], 2):
            triples.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "evidence": cleaned_text[:280],
                    "chunk_id": chunk["chunk_id"],
                }
            )
    return triples


def extract_triples_with_llm(
    chunk: dict,
    llm: LLMClient,
    max_triples: int = 8,
    allow_rule_fallback: bool = True,
) -> list[dict]:
    system = "You extract concise knowledge graph triples from text. Return JSON only."
    user = f"""
Extract up to {max_triples} knowledge graph triples from the text.
Rules:
1. Only extract relationships that are explicitly stated or directly supported.
2. Return a JSON array. Each object must contain: head, relation, tail, evidence.
3. relation should be a short snake_case phrase in English.
4. evidence must be a short quote or paraphrase from the given text.

Text:
{clean_text_for_extraction(chunk["text"])}
"""
    try:
        rows = parse_json_array(llm.complete(system, user, max_tokens=1200))
    except Exception:
        if allow_rule_fallback:
            return extract_triples([chunk], extractor="rule")
        raise

    triples: list[dict] = []
    for row in rows[:max_triples]:
        head = str(row.get("head", "")).strip()
        relation = str(row.get("relation", "")).strip() or "related_to"
        tail = str(row.get("tail", "")).strip()
        evidence = str(row.get("evidence", "")).strip() or clean_text_for_extraction(chunk["text"])[:280]
        if is_valid_entity(head) and is_valid_entity(tail) and head != tail:
            triples.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "evidence": evidence[:400],
                    "chunk_id": chunk["chunk_id"],
                }
            )
    if triples:
        return triples
    if allow_rule_fallback:
        return extract_triples([chunk], extractor="rule")
    return []


def get_relation_weight(relation: str, question_type: str | None = None) -> float:
    base = RELATION_WEIGHTS.get(relation, 0.5)
    bonus = QUESTION_TYPE_RELATION_BONUS.get(question_type or "", {}).get(relation, 0.0)
    return min(1.0, base + bonus)
