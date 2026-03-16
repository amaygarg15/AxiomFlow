import math  
import re                                    
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from models import Chunk

_WORD_RE = re.compile(r"\b\w+\b")

def _tokenize(text:str) -> List[str]:
    return _WORD_RE.findall(text.lower())

@dataclass
class SearchResult:
    chunk: Chunk
    score: float

class HybridRetriever:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks = list(chunks)
        self._doc_tokens = [_tokenize(chunk.content) for chunk in self.chunks]
        self._doc_len = [len(tokens) for tokens in self._doc_tokens]
        self._avgdl = sum(self._doc_len)/max(len(self._doc_len), 1)
        self._term_df = self._build_document_frequencies()
        self._bm25_idf = {
            term: math.log((len(self._doc_tokens) - freq + 0.5) / (freq + 0.5) + 1)  #total docs/no of docs containing the word
            for term, freq in self._term_df.items()
        }
        self._tfidf_idf = {
            term: math.log((len(self._doc_tokens) + 1) / (freq + 1)) + 1
            for term, freq in self._term_df.items()
        }
        self._tfidf_norms = self._build_tfidf_norms()

    def _build_document_frequencies(self) -> dict:
        frequencies = defaultdict(int)
        for tokens in self._doc_tokens:
            for term in set(tokens):
                frequencies[term] += 1
        return frequencies
    
    def _build_tfidf_norms(self) -> List[float]:
        norms = []
        for tokens in self._doc_tokens:
            tf = Counter(tokens)
            total = 0.0
            for term, count in tf.items():
                weight = (count / len(tokens)) * self._tfidf_idf.get(term,0.0)
                total += weight * weight
            norms.append(math.sqrt(total))
        return norms
    
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
    
    def _tfidf_scores(self, query_tokens: List[str]) -> List[float]:
        if not query_tokens:
            return[0.0 for _ in self._doc_tokens]
        query_tf = Counter(query_tokens)
        query_vec = {
            term: (count / len(query_tokens)) * self._tfidf_idf.get(term, 0.0)
            for term, count in  query_tf.items()
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_vec.values()))
        scores = []
        for tokens, doc_norm in zip(self._doc_tokens, self._tfidf_norms):
            if doc_norm == 0 or query_norm == 0:
                scores.append(0.0)
                continue
            tf = Counter(tokens)
            dot = 0.0
            for term, query_weight in query_vec.items():
                doc_weight = (tf.get(term,0) / len(tokens)) * self._tfidf_idf.get(term, 0.0)
                dot += query_weight * doc_weight
            scores.append(dot / (doc_norm * query_norm))
        return scores
    
    def search(
            self,
            query: str,
            top_k: int = 5,
            rrf_k: int = 60,
            bm25_k: int = 10,
            tfidf_k: int = 10,
    ) -> List[SearchResult]:
        query_tokens = _tokenize(query)
        bm25_scores = self._bm25_scores(query_tokens)
        tfidf_scores = self._tfidf_scores(query_tokens)

        bm25_ranked = sorted(enumerate(bm25_scores), key = lambda item: item[1], reverse = True)
        tfidf_ranked = sorted(enumerate(tfidf_scores), key = lambda item: item[1], reverse = True)

        rrf_scores = defaultdict(float)
        for rank, (index, _) in enumerate(bm25_ranked, start = 1):
            rrf_scores[index] += 1 / (rrf_k + rank)
        for rank , (index, _) in enumerate(tfidf_ranked, start = 1):
            rrf_scores[index] += 1 / (rrf_k + rank)
        
        combined = sorted(rrf_scores.items(), key = lambda item: item[1], reverse = True)[:top_k]
        return [SearchResult(chunk=self.chunks[index], score=score) for index, score in combined]

    

    

    