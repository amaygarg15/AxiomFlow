import streamlit as st
from ingestion import read_pdf_pages, chunk_pages

st.set_page_config(page_title="AxiomFlow")
st.title("AxiomFlow")
st.write("Learning RAG step by step")

st.divider()
st.write("Step 1: Upload a document")

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type = ["pdf"]
)

if uploaded_file is not None:
    st.success("PDF uploaded successfully")

    pages = read_pdf_pages(uploaded_file)
    st.write("Step 2: Extracted Page Preview")
    st.write(pages[0][0][:500])

    chunks = chunk_pages(pages, source=uploaded_file.name)
    st.write("Step 3: Chunking Result")
    st.write(f"Total Chunks Created: {len(chunks)}")

    st.write("Preview first 2 chunks with metadata:")
    for c in chunks[:2]:
        st.write({
            "content": c.content[:200],
            "source": c.source,
            "page_number": c.page_number,
            "content_type": c.content_type,
            "summary": c.summary
        })











