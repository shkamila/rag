from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss
import numpy as np

from .indexing import Chunk, Index


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, with its score and 1-based rank."""
    chunk: Chunk
    score: float
    rank: int

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def domain(self) -> str:
        return self.chunk.domain

def keyword_search(index: Index, query: str, k: int = 5) -> List[RetrievedChunk]:
    """BM25 lexical search; exact term matches, fast, no semantic understanding."""
    tokens = query.lower().split()
    scores = index.bm25.get_scores(tokens)
    top_k = np.argsort(scores)[::-1][:k]
    return [
        RetrievedChunk(
            chunk=index.chunks[int(i)],
            score=float(scores[int(i)]),
            rank=r + 1,
        )
        for r, i in enumerate(top_k)
    ]


def semantic_search(index: Index, query: str, k: int = 5) -> List[RetrievedChunk]:
    """FAISS cosine-similarity search over sentence-transformer embeddings."""
    q_vec = index.embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.faiss.search(q_vec, k)
    return [
        RetrievedChunk(
            chunk=index.chunks[int(indices[0][i])],
            score=float(scores[0][i]),
            rank=i + 1,
        )
        for i in range(k)
    ]


def hybrid_search(
    index: Index,
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> List[RetrievedChunk]:
    """Reciprocal Rank Fusion of BM25 (keyword) and FAISS (semantic).

    Args:
        index: prebuilt Index.
        query: natural-language query.
        k: number of chunks to return.
        alpha: weight of semantic vs. keyword. 0.0 = BM25 only, 1.0 = semantic only.
        rrf_k: RRF smoothing constant (Cormack et al., 2009). 60 is a robust default.

    Returns:
        Top-k chunks ordered by the fused RRF score.
    """
    n = index.n_chunks

    # BM25 ranking
    bm25_scores = index.bm25.get_scores(query.lower().split())
    bm25_order = np.argsort(bm25_scores)[::-1]
    bm25_rank = {int(i): r for r, i in enumerate(bm25_order)}

    # semantic ranking
    q_vec = index.embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    _, sem_indices = index.faiss.search(q_vec, n)
    sem_rank = {int(i): r for r, i in enumerate(sem_indices[0])}

    # fuse
    rrf_scores = {}
    for i in range(n):
        kw  = (1.0 - alpha) / (bm25_rank.get(i, n) + rrf_k)
        sem =        alpha  / (sem_rank.get(i, n)  + rrf_k)
        rrf_scores[i] = kw + sem

    top_k = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [
        RetrievedChunk(
            chunk=index.chunks[i],
            score=rrf_scores[i],
            rank=r + 1,
        )
        for r, i in enumerate(top_k)
    ]
