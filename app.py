import streamlit as st
from ingestion import read_pdf, chunk_text

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

    text = read_pdf(uploaded_file) #Read PDF
    st.write("Step 2: Extracted Text Preview")
    st.write(text[:500])

    chunks = chunk_text(text) #Chunking
    st.write("Step 3: Chunking Result")
    st.write(f"Total Chunks Created: {len(chunks)}")

    st.write("Preview first 2 chunks:")
    st.write(chunks[:2])




