# RAG Demo — Hybrid Search + Ollama

A Retrieval-Augmented Generation demo combining **BM25 keyword search**, **FAISS semantic search** and **Reciprocal Rank Fusion** to answer questions over local documents (PDF, DOCX, PPTX, XLSX) with inline citations.

Built during the ELIS Innovation Hub Master in AI, Data Science & Generative AI, then refactored from a single notebook into a small, modular Python package with a Streamlit UI.

---

## What this project demonstrates

- **Multi-format ingestion** — PDF, DOCX (paragraphs + tables), PPTX (slide-aware), XLSX (sheet-aware).
- **Hybrid retrieval** — BM25 (lexical) + sentence-transformers / FAISS (semantic), fused via Reciprocal Rank Fusion instead of raw score averaging.
- **Grounded generation** — the LLM receives only the retrieved, numbered context and is instructed to cite `[1]`, `[2]`, … and to say "I don't have enough information" when appropriate.
- **Swappable LLM provider** — `LLM_PROVIDER=ollama` (local, default) or `LLM_PROVIDER=openai`, chosen from `.env`.
- **Clean architecture** — each concern (config, ingestion, indexing, retrieval, generation, UI) lives in its own module and can be reused independently of Streamlit.

---

## Architecture

```
┌─────────────────┐
│  streamlit_app  │   UI: upload docs, tune k/α, ask question
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    src/rag_app/                             │
│                                                             │
│  config.py    → Settings (loads from .env)                  │
│  loaders.py   → load_pdf / docx / pptx / xlsx → Document    │
│  indexing.py  → chunk + BM25 + FAISS → Index                │
│  retrieval.py → keyword / semantic / hybrid → RetrievedChunk│
│  generation.py→ build context + LLM → RAGResponse           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Ollama / OpenAI
```

### Hybrid retrieval (Reciprocal Rank Fusion)

For each chunk `i`, the fused score is:

```
score(i) = α / (rank_semantic(i) + rrf_k) + (1 - α) / (rank_bm25(i) + rrf_k)
```

- `α = 0.0` → pure keyword
- `α = 1.0` → pure semantic
- `α = 0.5` (default) → equal weight

RRF is preferred over raw score fusion because BM25 and cosine similarity live on incomparable scales; comparing *ranks* is robust to that.

---

## Project structure

```
elis-rag-demo/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── streamlit_app.py          ← Streamlit entry point
└── src/
    └── rag_app/
        ├── __init__.py
        ├── config.py         ← Settings dataclass (env-loaded)
        ├── loaders.py        ← Per-format loaders + Document
        ├── indexing.py       ← Chunk + Index + builders
        ├── retrieval.py      ← keyword / semantic / hybrid search
        └── generation.py     ← LLM factory + rag_answer
```

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate              # Windows PowerShell
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# then edit .env, at minimum set DATA_DIR to your document folder
```

Default folder layout expected by `DATA_DIR`:

```
<DATA_DIR>/
├── argomenti_sportivi/     →  domain: "sport"
├── argomenti_social/       →  domain: "social"
├── argomenti_medicina/     →  domain: "medicina"
└── argomenti_legal/        →  domain: "legale"
```

(You can also drop-and-upload files from the Streamlit sidebar, no folder required.)

### 3. LLM backend

**Option A — Ollama (local, default).**

```bash
# Install from https://ollama.com, then:
ollama pull gemma3:4b
ollama serve
```

**Option B — OpenAI.**

```bash
pip install langchain-openai
# in .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

---

## Run

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 and:

1. Click **Carica cartella predefinita** (or upload files from the sidebar).
2. Adjust **top-K** and **α** if you want to see how retrieval changes.
3. Type a question and hit **Chiedi all'LLM** (or **Solo retrieval** to inspect the chunks the LLM would see).

---

## Using the library directly

```python
from dotenv import load_dotenv
load_dotenv()

from rag_app import Settings, load_directory, build_index, rag_answer

settings = Settings.from_env()
docs = load_directory(settings.data_dir, settings.domain_map)
index = build_index(docs, settings.embedding_model,
                    settings.chunk_size, settings.chunk_overlap)

response = rag_answer(
    "Quali sono i criteri diagnostici del diabete tipo 2?",
    index, settings, k=4, alpha=0.5,
)
print(response.answer)
for s in response.sources:
    print(f"  - {s.source} ({s.domain}) — score {s.score:.3f}")
```

---

## Configuration reference

| Variable         | Default                         | Description                                         |
| ---------------- | ------------------------------- | --------------------------------------------------- |
| `LLM_PROVIDER`   | `ollama`                        | `ollama` or `openai`                                |
| `LLM_MODEL`      | `gemma3:4b`                     | Ollama model name                                   |
| `OLLAMA_ENDPOINT`| `http://localhost:11434`        | Ollama server URL                                   |
| `OPENAI_MODEL`   | `gpt-4o-mini`                   | Used when `LLM_PROVIDER=openai`                     |
| `OPENAI_API_KEY` | — (required if provider=openai) | OpenAI key                                          |
| `LLM_TEMPERATURE`| `0.3`                           | Sampling temperature                                |
| `EMBEDDING_MODEL`| `all-MiniLM-L6-v2`              | sentence-transformers model (runs locally)          |
| `CHUNK_SIZE`     | `400`                           | Characters per chunk                                |
| `CHUNK_OVERLAP`  | `60`                            | Chunk overlap in characters                         |
| `DEFAULT_K`      | `4`                             | Top-K chunks retrieved                              |
| `HYBRID_ALPHA`   | `0.5`                           | Semantic weight in RRF (0=BM25, 1=semantic)         |
| `RRF_K`          | `60`                            | RRF smoothing constant                              |
| `DATA_DIR`       | `~/Desktop/rag`                 | Root folder with domain subfolders                  |

---

## Design choices & trade-offs

- **FAISS `IndexFlatIP` + L2 normalization.** Small corpus, so an exact brute-force index with cosine similarity is simpler and more accurate than ANN indexes like IVF or HNSW. For >10⁵ chunks I'd switch to `IndexIVFFlat`.
- **RRF over weighted score fusion.** Scores from BM25 and cosine are on different scales and distributions; fusing by rank is more robust and needs no per-query calibration.
- **Lazy OpenAI import.** The OpenAI path is imported only when `LLM_PROVIDER=openai`, so the package installs cleanly even if `langchain-openai` isn't available.
- **`@st.cache_resource` on the index.** Rebuilding the FAISS index for every interaction would make the app unusable. Cache key is a short hash of `(source, format, domain, len)` tuples — rebuilds only when the corpus actually changes.

---

## Possible extensions

- Persist the FAISS index to disk (`faiss.write_index`) to skip the embedding step across restarts.
- Add a reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a third stage after hybrid retrieval.
- Evaluate with a small RAGAS-style set (answer relevance, context precision, faithfulness).
- Add streaming responses to the UI (`st.write_stream`).
