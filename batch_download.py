import os
from typing import List, Dict
from simple_arxiv_fetcher import SimpleArxivFetcher


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_query() -> str:
    # Combine ML/AI/NLP keywords
    keywords = [
        "machine learning",
        "artificial intelligence",
        "AI",
        "natural language processing",
        "NLP",
        "deep learning"
    ]
    # Join with OR for arXiv search
    return " OR ".join(f"{kw}" for kw in keywords)


def download_top_papers(max_results: int = 10, out_dir: str = "downloads/ai_ml_nlp") -> List[Dict]:
    ensure_dir(out_dir)
    fetcher = SimpleArxivFetcher()

    query = build_query()
    papers = fetcher.search_papers(query, max_results=max_results)

    downloaded: List[Dict] = []
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            continue
        # Sanitize filename
        filename = f"{arxiv_id}.pdf"
        save_path = os.path.join(out_dir, filename)
        ok = fetcher.download_paper(arxiv_id, save_path)
        if ok:
            downloaded.append({
                "title": paper.get("title"),
                "arxiv_id": arxiv_id,
                "path": save_path
            })
    return downloaded


def main():
    out_dir = os.path.join("downloads", "ai_ml_nlp")
    items = download_top_papers(max_results=10, out_dir=out_dir)
    print(f"Downloaded {len(items)} papers to {out_dir}")
    for i, it in enumerate(items, 1):
        print(f"{i}. {it['title']} ({it['arxiv_id']}) -> {it['path']}")


if __name__ == "__main__":
    main()
