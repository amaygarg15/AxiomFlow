import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingestion import read_pdf_pages, chunk_pages
from retrieval import HybridRetriever
from grader import grade_relevance
from generator import generate_answer

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
    st.subheader("Step 2: Extracted Page Preview")
    st.write(pages[0][0][:500])

    chunks = chunk_pages(pages, source=uploaded_file.name)
    st.subheader("Step 3: Chunking Result")
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

    #query
    st.divider()
    st.subheader("Step 4: Ask a question")

    question = st.text_input("Enter your question about the document")

    if question:
        #retrieve
        retriever = HybridRetriever(chunks)
        results = retriever.search(question, top_k=5)

        st.subheader("Step 5: Retrieved chunks (before grading)")
        for i, r in enumerate(results, start = 1):
            with st.expander(f"Chunk {i} - score {r.score:.4f} - {r.chunk.source} p.{r.chunk.page_number}"):
                st.write(r.chunk.content[:300])
        
        #grade
        with st.spinner("Grading relevance of retrieved chunks"):
            graded = grade_relevance(question, results)

        st.subheader(f"Step 6: After Grading - {len(graded)} of {len(results)} chunks kept")
        if not graded:
            st.warning("No chunks passed relevance grading.")

        for i, r in enumerate(graded, start=1):
            with st.expander(f"Relevant Chunk {i} - {r.chunk.source} p.{r.chunk.page_number}"):
                st.write(r.chunk.content[:300])

        #generate
        with st.spinner("Generating grounded answer"):
            answer = generate_answer(question, graded)

        st.subheader("Step 7: Answer")
        st.markdown(answer)











