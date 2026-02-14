import math  
import re                                    #search
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
        self._avgdl = sum(self._doc_len)/max(len(self._doc_len), 1)
        self._term_df = self._build_document_frequencies()
        self._bm25_idf = {
            term: math.log((len(self._doc_tokens) - freq + 0.5) / (freq + 0.5) + 1)
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
    
    
    

    

    