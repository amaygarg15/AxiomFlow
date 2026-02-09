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

def chunk_pages(pages, source, summary=""):
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












   