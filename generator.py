from typing import List
from openai import OpenAI

from retrieval import SearchResult

client = OpenAI()

def build_context(results: List[SearchResult], max_chunk_chars: int = 1200) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        chunk_text = (chunk.content or "")[:max_chunk_chars]

        blocks.append(
            f"[Chunk{i}]\n"
            f"Source: {chunk.source}\n"
            f"Page: {chunk.page_number}\n"
            f"Content Type: {chunk.content_type}\n"
            f"Text: {chunk.content}\n"
        )
    return "\n".join(blocks)

def generate_answer(question: str, results: List[SearchResult], model: str = "gpt-4o-mini") -> str:
    if not results:
        return "I don't know based on the uploaded document."
    
    context = build_context(results)

    system_prompt = (
        "You are a strict retrieval-augmented assistant.\n"
        "Rules:\n"
        "1) Answer ONLY from the provided context.\n"
        "2) If the context is insufficient, say exactly: 'I don't know.'\n"
        "3) Add citations at the end of each key sentence in this format: "
        "[Source: <filename>, Page <number>].\n"
        "4) Do not invent facts.\n"
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Now provide the best possible grounded answer."
    )

    response = client.chat.completions.create(
        model = model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()
