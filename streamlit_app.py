from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_app.config import Settings  # noqa: E402
from rag_app.generation import rag_answer  # noqa: E402
from rag_app.indexing import Index, build_index  # noqa: E402
from rag_app.loaders import Document, load_directory, load_document  # noqa: E402
from rag_app.retrieval import hybrid_search  # noqa: E402

load_dotenv()

st.set_page_config(
    page_title="Knowledge Base",
    layout="wide",
)



@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings.from_env()


def docs_signature(docs: List[Document]) -> str:
    """Short stable hash of the current document list — used as cache key."""
    parts = [f"{d.source}|{d.format}|{d.domain}|{len(d.text)}" for d in docs]
    return hashlib.md5("\n".join(parts).encode()).hexdigest()[:12]


@st.cache_resource(show_spinner="Costruisco l'indice...")
def build_cached_index(signature: str, _docs: List[Document]) -> Index:
    """Build and cache an Index keyed by a content signature.

    Note: the leading underscore on `_docs` tells Streamlit not to hash it
    (Document isn't directly hashable). We control the cache key via
    `signature` instead.
    """
    settings = get_settings()
    return build_index(
        _docs,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )



if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []  # type: List[Document]
if "last_response" not in st.session_state:
    st.session_state.last_response = None

settings = get_settings()


# Sidebar 


with st.sidebar:
    st.title("📁 Documenti")

    #  Documents
    st.subheader("Documenti")

    uploaded = st.file_uploader(
        "Upload a file",
        type=["pdf", "docx", "pptx", "xlsx"],
        accept_multiple_files=True,
    )
    if uploaded:
        upload_domain = st.text_input("Dominio", value="custom", help="Etichetta usata nelle citazioni")
        if st.button(f"Aggiungi {len(uploaded)} file", use_container_width=True):
            added = 0
            for f in uploaded:
                suffix = Path(f.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    doc = load_document(tmp_path, domain=upload_domain)

                    doc = Document(
                        text=doc.text,
                        source=f.name,
                        format=doc.format,
                        domain=upload_domain,
                    )
                    st.session_state.loaded_docs.append(doc)
                    added += 1
                except Exception as e:  # noqa: BLE001
                    st.error(f"{f.name}: {e}")
                finally:
                    os.unlink(tmp_path)
            if added:
                st.success(f"Aggiunti {added} file. L'indice verrà ricostruito.")

    if st.session_state.loaded_docs and st.button(
        "Svuota documenti", use_container_width=True
    ):
        st.session_state.loaded_docs = []
        st.session_state.last_response = None
        build_cached_index.clear()
        st.rerun()

    st.divider()

    #  Retrieval settings 
    st.subheader("Retrieval")
    k = st.slider("Top-K chunk", 1, 10, settings.default_k)
    alpha = st.slider(
        "α (0 = keyword, 1 = semantic)",
        0.0, 1.0, settings.hybrid_alpha, step=0.1,
        help="Peso del semantic rispetto al keyword nella RRF",
    )

    st.divider()

    # LLM info 
    st.subheader("LLM")
    st.caption(f"**Provider:** `{settings.llm_provider}`")
    if settings.llm_provider == "ollama":
        st.caption(f"**Modello:** `{settings.llm_model}`")
        st.caption(f"**Endpoint:** `{settings.ollama_endpoint}`")
    else:
        st.caption(f"**Modello:** `{settings.openai_model}`")
    st.caption(f"**Embedder:** `{settings.embedding_model}`")
    st.caption(f"**Temperature:** `{settings.llm_temperature}`")

    st.divider()
    st.subheader("Stato")
    st.metric("Documenti", len(st.session_state.loaded_docs))


st.title("Knowledge Base")
if not st.session_state.loaded_docs:
    st.info("Per iniziare, carica la cartella predefinita o i tuoi documenti dalla barra laterale.")
    with st.expander("Come funziona"):
        st.markdown(
            """
            1. **Ingestion** — `loaders.py` estrae testo da PDF/DOCX/PPTX/XLSX
               mantenendo il nome file e il dominio come metadati.
            2. **Chunking** — `RecursiveCharacterTextSplitter` (400 char, overlap 60).
            3. **Indexing** — BM25 su token lowercased + FAISS `IndexFlatIP` su
               embedding `all-MiniLM-L6-v2` L2-normalizzati (cosine similarity).
            4. **Retrieval** — hybrid search con RRF: fonde i ranking dei due
               motori con peso α configurabile.
            5. **Generation** — prompt con contesto citato, LLM via Ollama locale
               (o OpenAI API, configurabile da `.env`).
            """
        )
    st.stop()

# build the index
sig = docs_signature(st.session_state.loaded_docs)
index = build_cached_index(sig, st.session_state.loaded_docs)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Documenti", len(st.session_state.loaded_docs))
col_b.metric("Chunk", index.n_chunks)
col_c.metric("Dim. embedding", index.faiss.d)

st.divider()

question = st.text_input(
    "La tua domanda",
    placeholder="Es. Quali sono i criteri diagnostici del diabete tipo 2?",
)

c1, c2, _ = st.columns([1, 1, 4])
ask_llm = c1.button("Chiedi all'LLM", type="primary", disabled=not question)
only_retrieve = c2.button("Solo retrieval", disabled=not question)

# ---
if ask_llm and question:
    t0 = time.time()
    try:
        with st.spinner("Cerco chunk rilevanti e genero la risposta..."):
            response = rag_answer(
                question=question,
                index=index,
                settings=settings,
                k=k,
                alpha=alpha,
            )
        elapsed = time.time() - t0
        st.session_state.last_response = (response, elapsed)
    except Exception as e:  # noqa: BLE001
        st.error(f"Errore LLM: {e}")
        st.caption(
            "Controlla che Ollama sia in esecuzione (`ollama serve`) e "
            "che il modello sia installato (`ollama pull gemma3:4b`)."
        )

# retrieval 
if only_retrieve and question:
    t0 = time.time()
    with st.spinner("Cerco chunk rilevanti..."):
        retrieved = hybrid_search(index, question, k=k, alpha=alpha)
    elapsed = time.time() - t0
    st.subheader(f"🔎 Chunk recuperati — {elapsed:.2f}s")
    for r in retrieved:
        label = (
            f"[{r.rank}] {r.source} — {r.domain} "
            f"(score {r.score:.4f})"
        )
        with st.expander(label):
            st.write(r.text)

if st.session_state.last_response is not None and not only_retrieve:
    response, elapsed = st.session_state.last_response

    st.subheader(f"Risposta — {elapsed:.2f}s")
    st.write(response.answer)

    st.subheader("Fonti")
    for i, src in enumerate(response.sources, 1):
        label = (
            f"[{i}] {src.source} — {src.domain} "
            f"(rank #{src.rank}, score {src.score:.4f})"
        )
        with st.expander(label):
            st.write(src.text)
