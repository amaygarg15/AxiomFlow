import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingestion import read_pdf_pages, read_txt_pages, read_csv_pages, chunk_pages
from rag_graph import run_rag_pipeline
from evaluation import evaluate_response

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
        #run LangGraph pipeline
        with st.spinner("Running Agentic RAG pipeline..."):
            result = run_rag_pipeline(question, chunks, max_attempts=2)

        #results
        st.subheader("Step 5: Pipeline Results")

        st.write(f"**Attempts made:** {result['attempt']}")
        st.write(f"**Final query:** {result['current_query']}")

        if result["current_query"] != question:
            st.info(f"Query was rewritten from: '{question}'")

        #final graded chunks
        st.divider()
        st.subheader("Step 6: Final Relevant Chunks")
        graded = result["graded"]
    
        if graded:
            st.success(f"Found {len(graded)} relevant chunks")
            for i, r in enumerate(graded, start = 1):
                with st.expander(f"Relevant Chunk {i} - {r.chunk.source} p.{r.chunk.page_number}"):
                    st.write(r.chunk.content[:300])
        else:
            st.warning("No relevant chunks found after all attempts.")
        
        #generate
        st.subheader("Step 7: Answer")
        answer = result["answer"]
        st.markdown(answer)

        #evaluation
        st.divider()
        st.subheader("Step 8: RAGAS Evaluation")

        if st.button("Evaluate Response Quality"):
            if graded:
                with st.spinner("Running RAGAS evaluation..."):
                    scores = evaluate_response(question, answer, graded)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Faithfulness", scores["faithfulness"])
                with col2:
                    st.metric("Answer Relevancy", scores["answer_relevancy"])
                with col3:
                    st.metric("Context Precision", scores["context_precision"])

                avg_score = sum(scores.values()) / 3
                if avg_score >= 0.8:
                    st.success(f"Excellent! Average score: {avg_score:.2f}")
                elif avg_score >= 0.6:
                    st.info(f"Good. Average score: {avg_score:.2f}")
                else:
                    st.warning(f"Needs improvement. Average score: {avg_score:.2f}")
            else:
                st.warning("No graded chunks available for evaluation.")








