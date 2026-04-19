"""LLM wiring and end-to-end RAG answering.

`rag_answer` is the one-line entry point: given a question and an Index,
it runs hybrid retrieval, builds a cited context, and asks the LLM to
answer using only that context. The provider is chosen via `Settings`,
so the same code runs against a local Ollama model or the OpenAI API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .config import Settings
from .indexing import Index
from .retrieval import RetrievedChunk, hybrid_search


SYSTEM_PROMPT = (
    "Sei un assistente esperto. Rispondi alla domanda dell'utente "
    "usando solo le informazioni presenti nel contesto fornito.\n"
    "Se la risposta non è nel contesto, dì onestamente: "
    '"Non ho informazioni sufficienti per rispondere."\n'
    "Cita sempre i frammenti che hai usato indicandoli con [1], [2], [3], ecc."
)


@dataclass
class RAGResponse:
    """End-to-end response: the question, the generated answer, and cited chunks."""
    question: str
    answer: str
    sources: List[RetrievedChunk]


def get_llm(settings: Settings) -> Any:
    """Factory for the LLM client. Supports 'ollama' (local) and 'openai' (API)."""
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_endpoint,
        )

    if provider == "openai":
        # Imported lazily so the package works without the openai LangChain extra.
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from e
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is empty — set it in .env or your shell.")
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unknown LLM provider: {settings.llm_provider!r}. "
        "Valid values: 'ollama', 'openai'."
    )


def build_context(retrieved: List[RetrievedChunk]) -> str:
    """Format retrieved chunks as a numbered, source-tagged context block."""
    return "\n\n".join(
        f"[{i + 1}] (fonte: {r.source})\n{r.text}"
        for i, r in enumerate(retrieved)
    )


def rag_answer(
    question: str,
    index: Index,
    settings: Settings,
    k: int = 4,
    alpha: float = 0.5,
    system_prompt: str = SYSTEM_PROMPT,
) -> RAGResponse:
    """Hybrid retrieval + grounded LLM answer with inline citations.

    Args:
        question: user question in natural language.
        index: prebuilt Index.
        settings: Settings for LLM provider selection.
        k: number of chunks to retrieve.
        alpha: hybrid weight (0 = keyword only, 1 = semantic only).
        system_prompt: can be overridden to change tone / language.
    """
    retrieved = hybrid_search(index, question, k=k, alpha=alpha)
    context = build_context(retrieved)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Contesto:\n{context}\n\nDomanda: {question}\n\nRisposta:"),
    ])
    llm = get_llm(settings)
    chain = prompt | llm
    resp = chain.invoke({"context": context, "question": question})

    answer_text = getattr(resp, "content", str(resp))
    return RAGResponse(question=question, answer=answer_text, sources=retrieved)
