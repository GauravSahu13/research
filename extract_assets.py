import os
import sys
import json
from typing import List, Optional

import fitz  # PyMuPDF
from pypdf import PdfReader

# LlamaParse (optional)
try:
    from llama_parse import LlamaParse
except Exception:
    LlamaParse = None

# Camelot for tables (may require Ghostscript for lattice). We'll use stream flavor.
import camelot
import pandas as pd
import re


def list_first_two_pdfs(root: str) -> List[str]:
    pdfs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith('.pdf'):
                pdfs.append(os.path.join(dirpath, name))
    pdfs.sort()
    return pdfs[:2]


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_text_llamaparse(pdf_path: str) -> Optional[str]:
    if LlamaParse is None:
        return None
    try:
        parser = LlamaParse(result_type="markdown")
        extra_info = {"file_name": os.path.basename(pdf_path)}
        with open(pdf_path, "rb") as f:
            docs = parser.load_data(f, extra_info=extra_info)
        # docs may be a list of Document objects with .text
        parts: List[str] = []
        for d in docs:
            text = getattr(d, 'text', None)
            if not text and isinstance(d, dict):
                text = d.get('text')
            if text:
                parts.append(text)
        return "\n\n".join(parts) if parts else None
    except Exception:
        return None


def extract_text_pypdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
        except Exception:
            continue
    return "\n".join(texts)


def extract_images_pymupdf(pdf_path: str, out_dir: str) -> int:
    ensure_dir(out_dir)
    doc = fitz.open(pdf_path)
    saved = 0
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            img_filename = os.path.join(out_dir, f"page{page_index+1}_img{img_index+1}.{ext}")
            with open(img_filename, "wb") as f:
                f.write(image_bytes)
            saved += 1
    return saved


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace and normalize spaces
    df = df.applymap(lambda x: re.sub(r"\s+", " ", str(x).strip()) if pd.notna(x) else x)
    # Drop fully empty rows/cols
    df = df.replace("", pd.NA)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


def extract_tables_camelot(pdf_path: str, out_dir: str) -> int:
    ensure_dir(out_dir)
    tables_all = []
    # Try lattice first (better for ruled tables)
    try:
        t_lat = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if t_lat and t_lat.n > 0:
            tables_all.extend(list(t_lat))
    except Exception:
        pass
    # Fallback to stream
    try:
        t_str = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        if t_str and t_str.n > 0:
            tables_all.extend(list(t_str))
    except Exception:
        pass

    if not tables_all:
        return 0

    count = 0
    for i, table in enumerate(tables_all):
        try:
            df = table.df if hasattr(table, "df") else None
            if df is None:
                # try reading via CSV then back to DataFrame as fallback
                tmp_csv = os.path.join(out_dir, "_tmp.csv")
                table.to_csv(tmp_csv)
                df = pd.read_csv(tmp_csv, header=None)
                os.remove(tmp_csv)
            df = _clean_dataframe(df)
            if df.empty:
                continue
            base = os.path.join(out_dir, f"table_{i+1}")
            df.to_csv(base + ".csv", index=False)
            # Also write Excel for readability
            df.to_excel(base + ".xlsx", index=False)
            count += 1
        except Exception:
            continue
    return count


def process_pdf(pdf_path: str, base_out_dir: str) -> dict:
    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
    paper_dir = os.path.join(base_out_dir, paper_id)
    text_dir = os.path.join(paper_dir, "text")
    images_dir = os.path.join(paper_dir, "images")
    tables_dir = os.path.join(paper_dir, "tables")
    ensure_dir(text_dir)
    ensure_dir(images_dir)
    ensure_dir(tables_dir)

    # Text extraction
    md_text = extract_text_llamaparse(pdf_path)
    if not md_text:
        md_text = extract_text_pypdf(pdf_path)
    text_path = os.path.join(text_dir, f"{paper_id}.md")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(md_text or "")

    # Images extraction
    num_images = extract_images_pymupdf(pdf_path, images_dir)

    # Tables extraction
    num_tables = extract_tables_camelot(pdf_path, tables_dir)

    summary = {
        "paper_id": paper_id,
        "pdf_path": pdf_path,
        "text_path": text_path,
        "images_dir": images_dir,
        "tables_dir": tables_dir,
        "num_images": num_images,
        "num_tables": num_tables,
    }
    with open(os.path.join(paper_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    src_dir = os.path.join("downloads", "ai_ml_nlp")
    out_dir = os.path.join("downloads", "ai_ml_nlp_extracted")
    ensure_dir(out_dir)

    pdfs = list_first_two_pdfs(src_dir)
    if not pdfs:
        print("No PDFs found to extract.")
        sys.exit(1)

    results = []
    for pdf in pdfs:
        print(f"Processing: {pdf}")
        res = process_pdf(pdf, out_dir)
        print(f"  -> text: {res['text_path']}")
        print(f"  -> images: {res['num_images']} saved in {res['images_dir']}")
        print(f"  -> tables: {res['num_tables']} saved in {res['tables_dir']}")
        results.append(res)

    print("\nExtraction complete. Outputs under:")
    for r in results:
        print(f"- {os.path.dirname(r['text_path'])}")


if __name__ == "__main__":
    main()
