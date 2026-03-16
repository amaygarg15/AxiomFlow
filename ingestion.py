import csv
import io

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import Chunk

def read_pdf_pages(file):
    doc = fitz.open(stream = file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append((text, i + 1))
    return pages

def read_txt_pages(file):
    text = file.read().decode("utf-8")
    return[(text, 1)]

def read_csv_pages(file):
    raw = file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))

    lines = []
    for row in reader:
        line = " | ".join(f"{col}: {val}" for col, val in row.items())
        lines.append(line)

    text = "\n".join(lines)
    return [(text, 1)]

def chunk_pages(pages, source, content_type="text", summary=""):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        separators = ["\n\n", "\n", ".", " "]
    )
    chunks = []
    for text, page_number in pages:
        split_texts = splitter.split_text(text)
        for part in split_texts:
            chunks.append(
                Chunk(
                    content = part,
                    source = source,
                    page_number = page_number,
                    content_type = "text",
                    summary = summary
                )
            )
    return chunks
