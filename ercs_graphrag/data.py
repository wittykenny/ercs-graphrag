from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from html import unescape
from pathlib import Path


DATA_URLS = {
    "MultiHopRAG.json": [
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json",
        "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json",
    ],
    "corpus.json": [
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json",
        "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/corpus.json",
    ],
}


TOY_QA = [
    {
        "query": "Which company was founded by the person who created OpenAI?",
        "answer": "OpenAI",
        "question_type": "toy",
        "evidence_list": [
            {"title": "Sam Altman", "source": "Toy", "fact": "Sam Altman co-founded OpenAI."},
            {"title": "OpenAI", "source": "Toy", "fact": "OpenAI is an artificial intelligence company."},
        ],
    }
]


TOY_CORPUS = [
    {
        "title": "Sam Altman",
        "source": "Toy",
        "category": "technology",
        "published_at": "2023-01-01T00:00:00+00:00",
        "body": "Sam Altman co-founded OpenAI. OpenAI is known for generative AI systems.",
    },
    {
        "title": "OpenAI",
        "source": "Toy",
        "category": "technology",
        "published_at": "2023-01-01T00:00:00+00:00",
        "body": "OpenAI is an artificial intelligence company.",
    },
]


def download_multihoprag(data_dir: str | Path = "data/raw") -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, urls in DATA_URLS.items():
        target = data_dir / filename
        if target.exists() and target.stat().st_size > 1000:
            continue
        ok = False
        for url in urls:
            ok = _download_file(url, target)
            if ok:
                break
        if not ok:
            crawled = crawl_dataset_url(filename)
            ok = bool(crawled and _download_file(crawled, target))
        if not ok:
            _write_toy_dataset(data_dir)
            return


def _download_file(url: str, target: Path) -> bool:
    tmp = target.with_suffix(target.suffix + ".download")
    tmp.unlink(missing_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1024 * 128)
                if not chunk:
                    break
                f.write(chunk)
        if tmp.stat().st_size > 1000:
            target.unlink(missing_ok=True)
            tmp.replace(target)
            return True
    except Exception:
        time.sleep(1)
    finally:
        tmp.unlink(missing_ok=True)
    return False


def crawl_dataset_url(filename: str) -> str | None:
    pages = [
        "https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/main",
        "https://github.com/yixuantt/MultiHop-RAG/tree/main/dataset",
    ]
    for page in pages:
        try:
            req = urllib.request.Request(page, headers={"User-Agent": "Mozilla/5.0"})
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
        except Exception:
            continue
        for href in re.findall(r'href=["\']([^"\']+)["\']', html):
            href = unescape(href)
            if filename not in href:
                continue
            if href.startswith("/datasets/yixuantt/MultiHopRAG/blob/main/"):
                return f"https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/{filename}"
            if href.startswith("/yixuantt/MultiHop-RAG/blob/main/dataset/"):
                return f"https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/{filename}"
            if href.startswith("http"):
                return href
    return None


def _write_toy_dataset(data_dir: Path) -> None:
    (data_dir / "MultiHopRAG.json").write_text(
        json.dumps(TOY_QA, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data_dir / "corpus.json").write_text(
        json.dumps(TOY_CORPUS, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_dataset(data_dir: str | Path = "data/raw") -> tuple[list[dict], list[dict]]:
    data_dir = Path(data_dir)
    qa_path = data_dir / "MultiHopRAG.json"
    corpus_path = data_dir / "corpus.json"
    if not qa_path.exists() or not corpus_path.exists():
        download_multihoprag(data_dir)
    return (
        json.loads(qa_path.read_text(encoding="utf-8")),
        json.loads(corpus_path.read_text(encoding="utf-8")),
    )


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def split_qa_dataset(
    qa: list[dict],
    split: str = "all",
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
    seed: int = 42,
) -> list[dict]:
    indexed = [dict(item, original_query_id=i) for i, item in enumerate(qa)]
    if split == "all":
        return indexed
    if not 0 < train_ratio < 1 or not 0 <= dev_ratio < 1:
        raise ValueError("train_ratio and dev_ratio must be in [0, 1], and train_ratio must be > 0.")
    if train_ratio + dev_ratio >= 1:
        raise ValueError("train_ratio + dev_ratio must be less than 1.")
    rng = random.Random(seed)
    rng.shuffle(indexed)
    train_end = int(len(indexed) * train_ratio)
    dev_end = train_end + int(len(indexed) * dev_ratio)
    if split == "train":
        return indexed[:train_end]
    if split == "dev":
        return indexed[train_end:dev_end]
    if split == "test":
        return indexed[dev_end:]
    raise ValueError(f"Unknown split: {split}")


def build_chunks(
    corpus: list[dict],
    qa: list[dict],
    max_docs: int = 300,
    include_evidence_chunks: bool = False,
) -> list[dict]:
    chunks: list[dict] = []
    for doc_id, doc in enumerate(corpus[:max_docs]):
        title = doc.get("title") or f"doc_{doc_id}"
        body = doc.get("body") or doc.get("text") or ""
        prefix = f"{title}. Source: {doc.get('source', '')}. Category: {doc.get('category', '')}."
        for idx, chunk in enumerate(split_text(body)):
            chunks.append(
                {
                    "chunk_id": f"corpus_{doc_id}_{idx}",
                    "doc_id": f"corpus_{doc_id}",
                    "title": title,
                    "text": f"{prefix} {chunk}",
                    "source": doc.get("source", ""),
                    "category": doc.get("category", ""),
                }
            )

    if include_evidence_chunks:
        seen = {c["text"] for c in chunks}
        for qid, item in enumerate(qa):
            for eid, ev in enumerate(item.get("evidence_list", [])):
                fact = ev.get("fact", "")
                if fact and fact not in seen:
                    chunks.append(
                        {
                            "chunk_id": f"evidence_{qid}_{eid}",
                            "doc_id": f"evidence_{qid}",
                            "title": ev.get("title", ""),
                            "text": f"{ev.get('title', '')}. Source: {ev.get('source', '')}. {fact}",
                            "source": ev.get("source", ""),
                            "category": ev.get("category", ""),
                        }
                    )
                    seen.add(fact)
    return chunks
