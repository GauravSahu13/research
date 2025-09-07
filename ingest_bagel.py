import os
import sys
import uuid
import gc
from typing import List, Dict, Tuple, Iterator

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# Use Google's Gemini embedding API directly
import google.generativeai as genai


def iter_pdf_text(pdf_path: str, max_pages: int = 50) -> Iterator[str]:
    reader = PdfReader(pdf_path)
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    for page in pages:
        try:
            text = page.extract_text() or ""
            if text.strip():
                yield text
        except Exception:
            continue


def iter_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> Iterator[str]:
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= n:
            break
        start = max(0, end - chunk_overlap)


def embed_texts(texts: List[str], api_key: str) -> List[List[float]]:
    genai.configure(api_key=api_key)
    emb_model = "models/embedding-001"

    vectors: List[List[float]] = []
    for t in texts:
        try:
            r = genai.embed_content(model=emb_model, content=t)
            if hasattr(r, 'embedding') and hasattr(r.embedding, 'values'):
                vectors.append(r.embedding.values)
            else:
                d = getattr(r, 'to_dict', lambda: {})()
                vectors.append(d.get('embedding', {}).get('values', []))
        except Exception:
            vectors.append([])
    return vectors


def collect_pdf_paths(root: str) -> List[str]:
    pdfs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith('.pdf'):
                pdfs.append(os.path.join(dirpath, name))
    return sorted(pdfs)


def ingest_folder_to_chroma(
    pdf_folder: str,
    db_dir: str,
    collection_name: str,
    api_key: str,
    max_pages_per_pdf: int = 50,
    max_chunks_per_pdf: int = 500,
    embed_batch_size: int = 8,
) -> Tuple[int, int]:
    client = chromadb.PersistentClient(path=db_dir, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(name=collection_name)

    pdf_paths = collect_pdf_paths(pdf_folder)
    total_chunks = 0
    added = 0

    for pdf_path in pdf_paths:
        fname = os.path.basename(pdf_path)
        chunk_buffer: List[str] = []
        chunk_ids: List[str] = []
        metas: List[Dict] = []
        chunks_for_pdf = 0

        for page_text in iter_pdf_text(pdf_path, max_pages=max_pages_per_pdf):
            for ch in iter_chunks(page_text):
                if chunks_for_pdf >= max_chunks_per_pdf:
                    break
                chunk_buffer.append(ch)
                chunk_ids.append(str(uuid.uuid4()))
                metas.append({"source": fname, "path": pdf_path})
                total_chunks += 1
                chunks_for_pdf += 1

                # When buffer reaches batch size, embed and add
                if len(chunk_buffer) >= embed_batch_size:
                    vecs = embed_texts(chunk_buffer, api_key)
                    docs: List[str] = []
                    ids: List[str] = []
                    mds: List[Dict] = []
                    for cid, doc, md, vec in zip(chunk_ids, chunk_buffer, metas, vecs):
                        if vec:
                            ids.append(cid)
                            docs.append(doc)
                            mds.append(md)
                    if ids:
                        collection.add(ids=ids, documents=docs, metadatas=mds, embeddings=vecs[:len(ids)])
                        added += len(ids)
                    # clear buffers
                    chunk_buffer.clear()
                    chunk_ids.clear()
                    metas.clear()
                    gc.collect()
            if chunks_for_pdf >= max_chunks_per_pdf:
                break

        # Flush remaining
        if chunk_buffer:
            vecs = embed_texts(chunk_buffer, api_key)
            docs = []
            ids = []
            mds = []
            for cid, doc, md, vec in zip(chunk_ids, chunk_buffer, metas, vecs):
                if vec:
                    ids.append(cid)
                    docs.append(doc)
                    mds.append(md)
            if ids:
                collection.add(ids=ids, documents=docs, metadatas=mds, embeddings=vecs[:len(ids)])
                added += len(ids)
            chunk_buffer.clear()
            chunk_ids.clear()
            metas.clear()
            gc.collect()

        print(f"Indexed ~{chunks_for_pdf} chunks from {fname}")

    print(f"Total chunks prepared: {total_chunks}")
    print(f"Total chunks indexed: {added}")
    return total_chunks, added


def main():
    pdf_folder = os.path.join('downloads', 'ai_ml_nlp')
    db_dir = 'bagelDB'
    collection_name = 'ai_mi-nlp'

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set. Please set your Gemini API key.")
        sys.exit(1)

    os.makedirs(db_dir, exist_ok=True)
    total, added = ingest_folder_to_chroma(pdf_folder, db_dir, collection_name, api_key)

    # Verify count
    client = chromadb.PersistentClient(path=db_dir)
    col = client.get_collection(collection_name)
    count = col.count()
    print(f"Collection '{collection_name}' now has {count} vectors.")


if __name__ == '__main__':
    main()
