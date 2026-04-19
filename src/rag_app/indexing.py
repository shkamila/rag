"""Chunking and index construction.

Given a list of Documents, produce a bundled Index object that contains:
- the chunks (with their metadata)
- a BM25 index for keyword search
- a FAISS index for semantic search
- the sentence-transformers embedder (reused at query time)

Keeping all four in one object makes the retrieval functions in
`retrieval.py` trivially simple: they just take an `Index` and a query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .loaders import Document


@dataclass
class Chunk:
    """A text chunk with its metadata inherited from the parent Document."""
    text: str
    source: str
    format: str
    domain: str


class Index:
    """Bundled BM25 + FAISS index over a list of Chunks.

    Attributes:
        chunks: the underlying chunks in insertion order (index-aligned with BM25/FAISS).
        embedder: the SentenceTransformer used for both indexing and query encoding.
        faiss: the FAISS IndexFlatIP of normalized embeddings (inner product = cosine).
        bm25: the BM25Okapi index built on whitespace-tokenized lowercased chunks.
    """

    def __init__(
        self,
        chunks: List[Chunk],
        embedder: SentenceTransformer,
        faiss_index: faiss.Index,
        bm25: BM25Okapi,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.faiss = faiss_index
        self.bm25 = bm25

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Index(n_chunks={self.n_chunks}, dim={self.faiss.d})"


# --- Builders ----------------------------------------------------------------

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> List[Chunk]:
    """Split each Document into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Chunk] = []
    for doc in docs:
        for piece in splitter.split_text(doc.text):
            chunks.append(Chunk(
                text=piece,
                source=doc.source,
                format=doc.format,
                domain=doc.domain,
            ))
    return chunks


def build_bm25(chunks: List[Chunk]) -> BM25Okapi:
    """Build a BM25 index over whitespace-tokenized, lowercased chunks."""
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def build_faiss(chunks: List[Chunk], embedder: SentenceTransformer) -> faiss.Index:
    """Embed all chunks and build a FAISS inner-product index over L2-normalized vectors."""
    texts = [c.text for c in chunks]
    vectors = embedder.encode(texts, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx


def build_index(
    docs: List[Document],
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> Index:
    """End-to-end: chunk documents, then build BM25 + FAISS."""
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks produced — is your documents list empty?")
    embedder = SentenceTransformer(embedding_model)
    bm25 = build_bm25(chunks)
    faiss_idx = build_faiss(chunks, embedder)
    return Index(chunks=chunks, embedder=embedder, faiss_index=faiss_idx, bm25=bm25)
