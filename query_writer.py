from typing import List
from openai import OpenAI

from retrieval import SearchResult

client = OpenAI()

def rewrite_query(
        original_query: str,
        rejected_results: List[SearchResult],
        model: str = "gpt-4o-mini",
) -> str:
    """
    Rewrite a query that failed to retrieve relevant chunks.

        Uses the rejected chunks to understand document vocabulary
        and generates a better query more likely to find relevant info.

        Args:
            original_query: The user's original question
            rejected_results: Chunks that were retrieved but rejected by grader
            model: OpenAI model to use

        Returns:
            A rewritten query string
    """
    if not rejected_results:
        #no chunks to learn from , just rephrase the query
        context_hint = "No relevant content was found"
    else:
        #build context from rejected chunks so LLM can see document vocabulary
        snippets = []
        for i, result in enumerate(rejected_results[:3], start=1):
            snippet = result.chunk.content[:300]
            snippets.append(f"[Chunk {i}]: {snippet}")
        context_hint = "\n".join(snippets)
    
    prompt = f"""You are a query to rewriting assisstant for a document search system.

The user asked a question, but the retrieved content was not relevant enough.
Your job is to rewrite the query to be more likely to find relevant information.

ORIGINAL QUERY:
{original_query}

RETRIEVED CONTENT (not relevant enough):
{context_hint}

INSTRUCTIONS:
1. Look at the vocabulary and topics in the retrieved content.
2. Rewrite the query using similar terminology.
3. Make the query more specific if it was too vague.
4. Make the query more general if it was too narrow.
5. Return ONLY the rewritten query, nothing else.

REWRITTEN QUERY: """
    
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()