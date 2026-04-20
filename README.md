# Hybrid RAG System 

> A production-oriented Retrieval-Augmented Generation pipeline that combines lexical and semantic search over multi-format documents, with a clean modular architecture and an interactive Streamlit interface.

---

## Overview

This project implements a Hybrid RAG system designed to answer natural-language questions over PDF, DOCX, PPTX, and XLSX documentation with citations.

The retrieval layer fuses BM25 (keyword/lexical) and FAISS (dense/semantic) rankings through Reciprocal Rank Fusion (RRF), a rank-based fusion strategy that is more robust than raw score averaging because it does not require BM25 and cosine similarity scores to be on the same scale.

The system supports both local LLMs via Ollama and OpenAI models, switchable through a single environment variable, making it easy to run fully offline or integrate with cloud APIs.

---

## Key Features

- **Multi-format document ingestion** - PDF, DOCX (paragraphs + tables), PPTX (slide-aware), XLSX (sheet-aware)
- **Hybrid retrieval** - BM25 + sentence-transformers / FAISS, fused via Reciprocal Rank Fusion
- **Grounded generation** - the LLM receives only retrieved, numbered context and is instructed to cite `[1]`, `[2]`, … or to explicitly state when information is unavailable
- **Swappable LLM backend** - `LLM_PROVIDER=ollama` (default, fully local) or `LLM_PROVIDER=openai`
- **Streamlit UI** - upload documents, tune retrieval parameters (`k`, `α`), and query interactively
- **Clean modular architecture** - each concern (config, ingestion, indexing, retrieval, generation, UI) is isolated in its own module and reusable independently of the UI layer

---

## Architecture

```
┌─────────────────┐
│  streamlit_app  │  Upload docs · Tune k/α · Ask questions
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  src/rag_app/                                               │
│                                                             │
│  config.py     → Settings  (loads from .env)               │
│  loaders.py    → load_pdf / docx / pptx / xlsx → Document  │
│  indexing.py   → chunk + BM25 + FAISS          → Index     │
│  retrieval.py  → keyword / semantic / hybrid   → Chunks    │
│  generation.py → build context + call LLM      → Response  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Ollama (local)  /  OpenAI API
```

---

## Retrieval Strategy 

For each candidate chunk `i`, the hybrid score is computed as:

```
score(i) = α / (rank_semantic(i) + rrf_k)  +  (1 - α) / (rank_bm25(i) + rrf_k)
```

| α value | Behaviour |
|---------|-----------|
| `0.0`  | Pure keyword (BM25 only) |
| `1.0`  | Pure semantic (FAISS only) |
| `0.5`  | Equal blend (default) |

RRF is preferred over weighted score fusion because BM25 and cosine similarity live on incomparable scales; rank comparison is distribution-agnostic and requires no per-query calibration.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Semantic embeddings | sentence-transformers (all-MiniLM-L6-v2, runs locally) |
| Vector index | FAISS (IndexFlatIP + L2 normalisation) |
| Keyword index | rank_bm25 |
| LLM (local) | Ollama — gemma3:4b (default) |
| LLM (cloud) | OpenAI — gpt-4o-mini (optional) |
| UI | Streamlit |
| Config | pydantic-settings + .env |

---

## Project Structure

```
rag/
├── streamlit_app.py       ← Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── src/
    └── rag_app/
        ├── __init__.py
        ├── config.py      ← Settings dataclass (env-loaded)
        ├── loaders.py     ← Per-format document loaders
        ├── indexing.py    ← Chunking + BM25 + FAISS index builders
        ├── retrieval.py   ← Keyword / semantic / hybrid search
        └── generation.py  ← LLM factory + RAG answer generation
```

---

## Possible Extensions

- Persist the FAISS index to disk (faiss.write_index) to skip the embedding step on restart
- Add a cross-encoder reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) as a third retrieval stage
- Evaluation pipeline with RAGAS metrics (answer relevance, context precision, faithfulness)
- Streaming responses in the UI via st.write_stream

