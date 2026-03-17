import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List

import chromadb

from models import Chunk

_WORD_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)

        #BM25 index
        self._doc_tokens = [_tokenize(c.content) for c in self.chunks]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._term_df = self._build_document_frequencies()
        self._bm25_idf = {
            term: math.log(
                (len(self._doc_tokens) - freq + 0.5) / (freq + 0.5) + 1
            )
            for term, freq in self._term_df.items()
        }

        #ChromaDB vector index
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=[str(i) for i in range(len(self.chunks))],
            documents=[c.content for c in self.chunks],
        )

    def _build_document_frequencies(self) -> dict:
        frequencies = defaultdict(int)
        for tokens in self._doc_tokens:
            for term in set(tokens):
                frequencies[term] += 1
        return frequencies

    def _bm25_scores(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        scores = []
        query_terms = Counter(query_tokens)
        for tokens, doc_len in zip(self._doc_tokens, self._doc_len):
            tf = Counter(tokens)
            score = 0.0
            for term, qf in query_terms.items():
                if term not in tf:
                    continue
                idf = self._bm25_idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + k1 * (1 - b + b * doc_len / self._avgdl)
                score += idf * (freq * (k1 + 1) / denom) * qf
            scores.append(score)
        return scores

    def _vector_ranked(self, query: str, top_k: int) -> List[int]:
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.chunks)),
        )
        return [int(doc_id) for doc_id in results["ids"][0]]

    def search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        bm25_top: int = 10,
        vector_top: int = 10,
    ) -> List[SearchResult]:
        query_tokens = _tokenize(query)

        #BM25 ranking
        bm25_scores = self._bm25_scores(query_tokens)
        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:bm25_top]

        #Vector ranking
        vector_ranked = self._vector_ranked(query, top_k=vector_top)

        #RRF fusion
        rrf_scores = defaultdict(float)
        for rank, (index, _) in enumerate(bm25_ranked, start=1):
            rrf_scores[index] += 1 / (rrf_k + rank)
        for rank, index in enumerate(vector_ranked, start=1):
            rrf_scores[index] += 1 / (rrf_k + rank)

        combined = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(chunk=self.chunks[index], score=score)
            for index, score in combined
        ]
