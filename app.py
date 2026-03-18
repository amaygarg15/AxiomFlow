import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingestion import read_pdf_pages, read_txt_pages, read_csv_pages, chunk_pages
from retrieval import HybridRetriever
from grader import grade_relevance
from generator import generate_answer
from query_writer import rewrite_query

#page config
st.set_page_config(page_title="AxiomFlow")
st.title("AxiomFlow")
st.write("A Self-Correcting RAG for Multi-Format Documents")

#Step 1: upload
st.divider()
st.write("Step 1: Upload a document")

uploaded_file = st.file_uploader(
    "Upload your PDF, TXT or CSV file",
    type = ["pdf", "txt", "csv"]
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_ext = file_name.rsplit(".", 1)[-1].lower()

    st.success(f"Uploaded: {file_name}")

    #Step 2: route by file type
    if file_ext == "pdf":
        pages = read_pdf_pages(uploaded_file)
        content_type = "text"
    elif file_ext == "txt":
        pages = read_txt_pages(uploaded_file)
        content_type = "text"
    elif file_ext == "csv":
        pages = read_csv_pages(uploaded_file)
        content_type = "csv"
    else:
        st.error(f"Unsupported file type: {file_ext}")
        st.stop()

    st.subheader("Step 2: Extracted Content Preview")
    st.write(pages[0][0][:500])

    #Step 3: Chunk
    chunks = chunk_pages(pages, source=file_name, content_type=content_type)
    st.subheader("Step 3: Chunking Result")
    st.write(f"Total Chunks Created: {len(chunks)}")

    st.write("Preview first 2 chunks with metadata:")
    for c in chunks[:2]:
        st.write({
            "content": c.content[:200],
            "source": c.source,
            "page_number": c.page_number,
            "content_type": c.content_type,
        })

    #Step 4: query
    st.divider()
    st.subheader("Step 4: Ask a question")

    question = st.text_input("Enter your question about the document")

    if question:
        #retrieve
        retriever = HybridRetriever(chunks)

        current_query = question
        max_attempts = 2
        attempt = 0
        graded = []

        #corrective RAG loop
        while attempt < max_attempts:
            attempt += 1

            st.subheader(f"Attempt {attempt}: Retrieving with query")
            st.info(f"Query: {current_query}")

            results = retriever.search(current_query, top_k=5)

            with st.expander(f"Retrieved {len(results)} chunks (before grading)"):
                for i, r in enumerate(results, start=1):
                    st.write(f"**Chunk {i}** - score {r.score:.4f} - {r.chunk.source} p.{r.chunk.page_number}")
                    st.write(r.chunk.content[:200])
                    st.write("---")
            
            #grade
            with st.spinner("Grading relevance of retrieved chunks"):
                graded = grade_relevance(current_query, results)

            st.write(f"**Grading result:** {len(graded)} of {len(results)} chunks relevant")

            #check if we have relevant chunks
            if graded:
                st.success("Found relevant chunks!")
                break
            else:
                st.warning("No relevant chunks found")
                if attempt < max_attempts:
                    st.write("Rewriting query and trying again")
                    with st.spinner("Rewriting query"):
                        current_query = rewrite_query(question, results)
                    st.info(f"Rewritten query: {current_query}")

        #final graded chunks
        st.divider()
        st.subheader("Step 5: Final Relevant Chunks")
        if graded:
            for i, r in enumerate(graded, start = 1):
                with st.expander(f"Relevant Chunk {i} - {r.chunk.source} p.{r.chunk.page_number}"):
                    st.write(r.chunk.content[:300])
        else:
            st.warning("No relevant chunks found after all attempts.")
        
        #generate
        st.subheader("Step 6: Answer")

        with st.spinner("Generating grounded answer"):
            answer = generate_answer(question, graded)

        st.markdown(answer)











