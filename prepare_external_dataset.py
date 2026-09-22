from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert external multi-hop QA datasets to the local MultiHop-RAG format.")
    parser.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa"], default="hotpotqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir or f"data/{args.dataset}_raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "hotpotqa":
        qa, corpus = convert_hotpotqa(args.split, args.max_queries)
    else:
        qa, corpus = convert_2wiki(args.split, args.max_queries)
    (out_dir / "MultiHopRAG.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(qa)} queries and {len(corpus)} docs to {out_dir}")


def convert_hotpotqa(split: str, max_queries: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    qa, docs, seen_titles = [], [], set()
    for original_id, item in enumerate(ds):
        if len(qa) >= max_queries:
            break
        context = item.get("context", {})
        titles = list(context.get("title", []))
        sentences = list(context.get("sentences", []))
        evidence_titles = list(item.get("supporting_facts", {}).get("title", []))
        evidence_sent_ids = list(item.get("supporting_facts", {}).get("sent_id", []))

        title_to_doc = {}
        for title, sent_list in zip(titles, sentences):
            body = " ".join(sent_list) if isinstance(sent_list, list) else str(sent_list)
            if not body.strip():
                continue
            title_to_doc[title] = body
            if title not in seen_titles:
                docs.append(
                    {
                        "title": title,
                        "source": "HotpotQA",
                        "category": "wikipedia",
                        "published_at": "",
                        "body": body,
                    }
                )
                seen_titles.add(title)

        evidence_list = []
        for title, sent_id in zip(evidence_titles, evidence_sent_ids):
            sent = ""
            idx = titles.index(title) if title in titles else -1
            if idx >= 0 and sent_id < len(sentences[idx]):
                sent = sentences[idx][sent_id]
            evidence_list.append({"title": title, "source": "HotpotQA", "fact": sent or title})

        qa.append(
            {
                "query": item.get("question", ""),
                "answer": item.get("answer", ""),
                "question_type": item.get("type", "hotpotqa"),
                "evidence_list": evidence_list,
                "original_query_id": original_id,
            }
        )
    return qa, docs


def convert_2wiki(split: str, max_queries: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("xanhho/2WikiMultihopQA", split=split)
    qa, docs, seen_titles = [], [], set()
    for original_id, item in enumerate(ds):
        if len(qa) >= max_queries:
            break
        contexts = item.get("context") or item.get("contexts") or []
        evidence_list = []
        for ctx in normalize_contexts(contexts):
            title, body = ctx["title"], ctx["body"]
            if title and body and title not in seen_titles:
                docs.append(
                    {
                        "title": title,
                        "source": "2WikiMultiHopQA",
                        "category": "wikipedia",
                        "published_at": "",
                        "body": body,
                    }
                )
                seen_titles.add(title)
        supports = item.get("supporting_facts") or item.get("supports") or []
        for support in normalize_supports(supports):
            evidence_list.append({"title": support["title"], "source": "2WikiMultiHopQA", "fact": support["fact"]})
        if not evidence_list:
            for ctx in normalize_contexts(contexts)[:2]:
                evidence_list.append({"title": ctx["title"], "source": "2WikiMultiHopQA", "fact": ctx["body"][:240]})
        qa.append(
            {
                "query": item.get("question", ""),
                "answer": item.get("answer", ""),
                "question_type": item.get("type", "2wikimultihopqa"),
                "evidence_list": evidence_list,
                "original_query_id": original_id,
            }
        )
    return qa, docs


def normalize_contexts(contexts: object) -> list[dict]:
    rows = []
    if isinstance(contexts, dict):
        titles = contexts.get("title") or contexts.get("titles") or []
        texts = contexts.get("sentences") or contexts.get("text") or contexts.get("texts") or []
        for title, text in zip(titles, texts):
            body = " ".join(text) if isinstance(text, list) else str(text)
            rows.append({"title": str(title), "body": body})
    elif isinstance(contexts, list):
        for ctx in contexts:
            if isinstance(ctx, dict):
                title = ctx.get("title", "")
                text = ctx.get("text") or ctx.get("sentences") or ctx.get("paragraph", "")
            elif isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                title, text = ctx[0], ctx[1]
            else:
                continue
            body = " ".join(text) if isinstance(text, list) else str(text)
            rows.append({"title": str(title), "body": body})
    return rows


def normalize_supports(supports: object) -> list[dict]:
    rows = []
    if isinstance(supports, dict):
        titles = supports.get("title") or supports.get("titles") or []
        facts = supports.get("fact") or supports.get("facts") or supports.get("sentences") or []
        for title, fact in zip(titles, facts):
            rows.append({"title": str(title), "fact": " ".join(fact) if isinstance(fact, list) else str(fact)})
    elif isinstance(supports, list):
        for support in supports:
            if isinstance(support, dict):
                rows.append(
                    {
                        "title": str(support.get("title", "")),
                        "fact": str(support.get("fact") or support.get("text") or support.get("sentence") or ""),
                    }
                )
            elif isinstance(support, (list, tuple)) and support:
                rows.append({"title": str(support[0]), "fact": str(support[-1])})
    return rows


if __name__ == "__main__":
    main()
