from typing import List
from openai import OpenAI

from retrieval import SearchResult

client = OpenAI()

def grade_relevance(question: str, results: List[SearchResult], model: str = "gpt-4o-mini") -> List[SearchResult]:
    """
    keep only relevant chunks
    return a filtered list of SearchResult objects
    """ 
    if not results:
        return []
    
    kept = []
    for result in results:
        chunk = result.chunk

        prompt = (
            "You are a strict relevance grader.\n"
            "Given a user question and one retrieved chunk, answer only 'yes' or 'no'.\n"
            "Answer yes only if the chunk is clearly useful to asnwer the question.\n\n"
            f"Question: {question}\n\n"
            f"Source: {chunk.source}\n"
            f"Page: {chunk.page_number}\n"
            f"Content Type: {chunk.content_type}\n"
            f"Text: {chunk.content}\n"
        )

        response = client.chat.completions.create(
            model = model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        verdict = response.choices[0].message.content.strip().lower()
        if verdict.startswith("yes"):
            kept.append(result)

    return kept