from typing import List
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

from retrieval import SearchResult

def evaluate_response(
    question: str,
    answer: str,
    contexts: List[SearchResult],
) -> dict:
    """
    Evaluate a RAG response using RAGAS metrics.

    Args:
        question: The user's original question
        answer: The generated answer
        contexts: The graded chunks used for generation

    Returns:
        Dictionary with faithfulness, answer_relevancy, and context_precision scores
    """
    #extract text
    context_texts = [r.chunk.content for r in contexts]

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [context_texts],
    }
    dataset = Dataset.from_dict(data)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    return {
        "faithfulness": round(results["faithfulness"], 3),
        "answer_relevancy": round(results["answer_relevancy"], 3),
        "context_precision": round(results["context_precision"], 3),
    }