# Hybrid RAG System — BM25 + FAISS + Reciprocal Rank Fusion

> A production-oriented Retrieval-Augmented Generation pipeline that combines lexical and semantic search over multi-format documents, with a clean modular architecture and an interactive Streamlit interface.

---

## Overview

This project implements a **Hybrid RAG (Retrieval-Augmented Generation)** system designed to answer natural-language questions over private document collections — PDF, DOCX, PPTX, and XLSX — with full inline source citations.

The retrieval layer fuses **BM25** (keyword/lexical) and **FAISS** (dense/semantic) rankings through **Reciprocal Rank Fusion (RRF)**, a rank-based fusion strategy that is more robust than raw score averaging because it does not require BM25 and cosine similarity scores to be on the same scale.

The system supports both **local LLMs via Ollama** and **OpenAI models**, switchable through a single environment variable — making it easy to run fully offline or integrate with cloud APIs.

---

## Key Features

- **Multi-format document ingestion** — PDF, DOCX (paragraphs + tables), PPTX (slide-aware), XLSX (sheet-aware)
- **Hybrid retrieval** — BM25 + sentence-transformers / FAISS, fused via Reciprocal Rank Fusion
- **Grounded generation** — the LLM receives only retrieved, numbered context and is instructed to cite `[1]`, `[2]`, … or to explicitly state when information is unavailable
- **Swappable LLM backend** — `LLM_PROVIDER=ollama` (default, fully local) or `LLM_PROVIDER=openai`
- **Streamlit UI** — upload documents, tune retrieval parameters (`k`, `α`), and query interactively
- **Clean modular architecture** — each concern (config, ingestion, indexing, retrieval, generation, UI) is isolated in its own module and reusable independently of the UI layer

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

## Retrieval Strategy — Reciprocal Rank Fusion

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

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env — at minimum set DATA_DIR to your document folder
```

### 3. LLM backend

**Option A — Ollama (local, no API key required)**
```bash
# Install from https://ollama.com, then:
ollama pull gemma3:4b
ollama serve
```

**Option B — OpenAI**
```bash
pip install langchain-openai
# In .env:
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
```

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

---

## Programmatic Usage

The RAG pipeline can be used directly as a Python library, independent of the Streamlit UI:

```python
from dotenv import load_dotenv
load_dotenv()

from rag_app import Settings, load_directory, build_index, rag_answer

settings = Settings.from_env()
docs     = load_directory(settings.data_dir, settings.domain_map)
index    = build_index(docs, settings.embedding_model, settings.chunk_size, settings.chunk_overlap)

response = rag_answer(
    "What are the diagnostic criteria for type 2 diabetes?",
    index, settings,
    k=4, alpha=0.5,
)

print(response.answer)
for source in response.sources:
    print(f"  [{source.source}] ({source.domain}) — score {source.score:.3f}")
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| LLM_PROVIDER | ollama | ollama or openai |
| LLM_MODEL | gemma3:4b | Ollama model name |
| OLLAMA_ENDPOINT | http://localhost:11434 | Ollama server URL |
| OPENAI_MODEL | gpt-4o-mini | Used when LLM_PROVIDER=openai |
| OPENAI_API_KEY | — | Required if provider = openai |
| LLM_TEMPERATURE | 0.3 | Sampling temperature |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Sentence-transformers model (local) |
| CHUNK_SIZE | 400 | Characters per chunk |
| CHUNK_OVERLAP | 60 | Overlap between consecutive chunks |
| DEFAULT_K | 4 | Top-K chunks retrieved |
| HYBRID_ALPHA | 0.5 | Semantic weight in RRF (0 = BM25, 1 = semantic) |
| RRF_K | 60 | RRF smoothing constant |
| DATA_DIR | ~/Desktop/rag | Root folder with document subfolders |

---

## Design Decisions

**FAISS IndexFlatIP with L2 normalisation** — For the corpus sizes this system targets, an exact brute-force index with cosine similarity is both simpler and more accurate than approximate-nearest-neighbour indexes (IVF, HNSW). At >10⁵ chunks, switching to IndexIVFFlat would be the natural next step.

**RRF over weighted score fusion** — BM25 and cosine similarity scores have different distributions; fusing by rank is statistically robust and eliminates the need for per-query score calibration.

**Lazy OpenAI import** — The OpenAI dependency is imported only when LLM_PROVIDER=openai, so the package installs and runs cleanly in offline environments without langchain-openai.

**@st.cache_resource on the index** — The FAISS index is rebuilt only when the document corpus changes (detected via a lightweight hash of source metadata), avoiding redundant embedding passes on every interaction.

---

## Possible Extensions

- Persist the FAISS index to disk (faiss.write_index) to skip the embedding step on restart
- Add a cross-encoder reranker (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) as a third retrieval stage
- Evaluation pipeline with RAGAS metrics (answer relevance, context precision, faithfulness)
- Streaming responses in the UI via st.write_stream

---

## Background

Developed during the **ELIS Innovation Hub Master in AI, Data Science & Generative AI**, then refactored from a single Jupyter notebook into a structured, maintainable Python package — demonstrating the full journey from prototype to production-ready code.
