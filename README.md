# AxiomFlow

A self-correcting, agentic RAG system for multi-format documents (PDF, TXT, CSV), built with Streamlit, LangGraph, hybrid retrieval (BM25 + vector), and Gemini.

## What This Project Does

AxiomFlow lets a user upload a document, ask a question, and get a grounded answer supported by retrieved chunks.

The pipeline is corrective and agentic:
- It retrieves chunks for the query.
- It grades relevance chunk-by-chunk.
- If relevant context is insufficient, it rewrites the query and retries.
- It then generates a grounded answer and can evaluate response quality with RAGAS metrics.

## Key Features

- Multi-format ingestion:
	- PDF (PyMuPDF)
	- TXT
	- CSV
- Recursive text chunking with overlap for context retention.
- Hybrid retrieval:
	- BM25 lexical scoring
	- Chroma vector search
	- Reciprocal Rank Fusion (RRF)
- Agentic control flow using LangGraph (retrieve -> grade -> rewrite -> retrieve -> generate).
- Relevance grading and query rewriting with Gemini.
- Answer generation with grounding/citation-focused prompt.
- Optional quality evaluation via RAGAS:
	- Faithfulness
	- Answer Relevancy
	- Context Precision

## Architecture

High-level module responsibilities:

- `app.py`
	- Streamlit UI and end-to-end user flow.
- `ingestion.py`
	- File parsing and chunk creation.
- `models.py`
	- Core data model (`Chunk`).
- `retrieval.py`
	- HybridRetriever (BM25 + Chroma + RRF) and `SearchResult`.
- `rag_graph.py`
	- LangGraph state machine and pipeline execution.
- `grader.py`
	- LLM-based relevance filtering.
- `query_writer.py`
	- Query rewriting when initial retrieval is not useful.
- `generator.py`
	- Final grounded answer generation.
- `evaluation.py`
	- RAGAS-based evaluation metrics.

## Data Flow

1. User uploads file.
2. Content is extracted and chunked.
3. User asks question.
4. Graph runs:
	 - Retrieve top chunks
	 - Grade relevance
	 - If none relevant and attempts remain: rewrite query and retry
	 - Generate answer from final selected chunks
5. UI displays:
	 - Attempts used
	 - Final query
	 - Relevant chunks
	 - Final answer
6. Optional: evaluate with RAGAS.

## Requirements

From `requirements.txt`:

- streamlit
- langchain
- langchain-community
- chromadb
- pymupdf
- google-genai
- openai (optional; used by some evaluation setups)
- ragas
- datasets
- python-dotenv
- langgraph
- tiktoken

## Setup

### 1) Clone and enter project

```powershell
git clone <your-repo-url>
cd AxiomFlow
```

### 2) Create and activate virtual environment (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in project root:

```env
GOOGLE_API_KEY=your_google_or_gemini_api_key
GEMINI_MODEL=models/gemini-2.5-flash
```

Notes:
- The code accepts `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
- `GEMINI_MODEL` is optional but recommended.

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## UI Workflow

- Step 1: Upload document
- Step 2: Extracted content preview
- Step 3: Chunking result + metadata preview
- Step 4: Ask question
- Step 5: Pipeline results (attempt count, final query)
- Step 6: Final relevant chunks
- Step 7: Answer
- Step 8: Optional RAGAS evaluation

## Corrective RAG Logic

The graph state keeps:
- `question`
- `current_query`
- `chunks`
- `results`
- `graded`
- `answer`
- `attempt`
- `max_attempts`

Decision policy:
- If `graded` is non-empty -> generate.
- Else if attempts remain -> rewrite query and retrieve again.
- Else -> generate using available fallback context.

## Troubleshooting

### 1) Model not found (404)

Symptoms:
- errors like `models/<name> is not found for API version v1beta`.

Fix:
- Set `GEMINI_MODEL` to a model your key supports, for example:
	- `models/gemini-2.5-flash`
	- `models/gemini-2.0-flash`
- Ensure Gemini API is enabled for your project.

### 2) Quota/billing errors (429)

Symptoms:
- `ResourceExhausted` or quota limit exceeded.

Fix:
- Enable billing or use a project/key with available quota.
- Generate a new API key in the correctly configured project.

### 3) Missing package errors

```powershell
pip install -r requirements.txt
```

### 4) Streamlit app not opening

- Confirm app started without Python traceback.
- Check local URL printed in terminal.
- Try restarting terminal and rerunning `streamlit run app.py`.

## Known Limitations

- Current retriever index is rebuilt during pipeline execution, which can be expensive on larger corpora.
- Collection naming in Chroma is static (`chunks`), which may require better session/document isolation for multi-user scale.
- README examples assume Windows PowerShell commands.

## Suggested Improvements

- Persist and cache vector index per uploaded file/session.
- Add automated tests for retrieval, grading, rewriting, and graph transitions.
- Add richer metadata filtering and improved citation formatting.
- Expand README with diagrams and benchmark/evaluation tables.

## License

This project is licensed under the MIT License.

Copyright (c) 2026 Amay Garg

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the Software), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
