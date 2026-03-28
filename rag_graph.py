from typing import List, TypedDict
from langgraph.graph import StateGraph, END

from retrieval import HybridRetriever, SearchResult
from grader import grade_relevance
from generator import generate_answer
from query_writer import rewrite_query

#state defination
class RAGState(TypedDict):
    """State that flows through the RAG graph"""
    question: str
    current_query: str
    chunks: list
    results: List[SearchResult]
    graded: List[SearchResult]
    answer: str
    attempt: int
    max_attempts: int

#node functions
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks using hybrid search."""
    retriever = HybridRetriever(state["chunks"])
    results = retriever.search(state["current_query"], top_k = 5)
    return{"results": results}

def grade_node(state: RAGState) -> RAGState:
    """Grade retrieved chunks for relevance."""
    graded = grade_relevance(state["current_query"], state["results"])
    attempt = state["attempt"] + 1
    return {"graded": graded, "attempt": attempt}

def rewrite_node(state: RAGState) -> RAGState:
    """Rewrite the query when no relevant chunks found."""
    new_query = rewrite_query(state["question"], state["results"])
    return {"current_query": new_query}

def generate_node(state: RAGState) -> RAGState:
    """Generate final answer from graded chunks."""
    answer = generate_answer(state["question"], state["graded"])
    return {"answer": answer}

#conditional edge
def should_continue(state: RAGState) -> str:
    """generate, rewrite or end"""
    if state["graded"]:
        return "generate"
    elif state["attempt"] < state["max_attempts"]:
        return "rewrite"
    else:
        return "generate"
    
def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        should_continue,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()

def run_rag_pipeline(question: str, chunks: list, max_attempts: int = 2) -> dict:
    """
    Run the full RAG pipeline and return results.

    Args:
        question: User's question
        chunks: Document chunks from ingestion
        max_attempts: Maximum retrieval attempts

    Returns:
        Final state with answer and metadata
    """

    #build graph 
    app = build_rag_graph()

    #initial state
    initial_state = {
        "question": question,
        "current_query": question,
        "chunks": chunks,
        "results": [],
        "graded": [],
        "answer": "",
        "attempt": 0,
        "max_attempts": max_attempts,
    }

    final_state = app.invoke(initial_state)  #run graph

    return final_state
