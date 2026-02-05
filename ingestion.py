import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_pdf(file):
    doc = fitz.open(stream = file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)