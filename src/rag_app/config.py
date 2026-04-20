

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class Settings:

    #  LLM provider 
    llm_provider: str = "ollama"         
    llm_temperature: float = 0.3

    #  ollama
    ollama_endpoint: str = "http://localhost:11434"
    ollama_api_key: str = "ollama"
    llm_model: str = "gemma3:4b"

    #  OpenAI 
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # embedding model 
    embedding_model: str = "all-MiniLM-L6-v2"

    # chunking 
    chunk_size: int = 400
    chunk_overlap: int = 60

    # retrieval 
    default_k: int = 4
    hybrid_alpha: float = 0.5  
    rrf_k: int = 60             

    # data 
    data_dir: Path = field(default_factory=lambda: Path.home() / "Desktop" / "rag")
    domain_map: Dict[str, str] = field(default_factory=lambda: {
        "argomenti_sportivi": "sport",
        "argomenti_social":   "social",
        "argomenti_medicina": "medicina",
        "argomenti_legal":    "legale",
    })

    @classmethod
    def from_env(cls) -> "Settings":
        """Build Settings from environment variables, falling back to defaults."""
        defaults = cls()
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", defaults.llm_temperature)),
            ollama_endpoint=os.getenv("OLLAMA_ENDPOINT", defaults.ollama_endpoint),
            ollama_api_key=os.getenv("OLLAMA_API_KEY", defaults.ollama_api_key),
            llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
            openai_api_key=os.getenv("OPENAI_API_KEY", defaults.openai_api_key),
            openai_model=os.getenv("OPENAI_MODEL", defaults.openai_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", defaults.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", defaults.chunk_overlap)),
            default_k=int(os.getenv("DEFAULT_K", defaults.default_k)),
            hybrid_alpha=float(os.getenv("HYBRID_ALPHA", defaults.hybrid_alpha)),
            rrf_k=int(os.getenv("RRF_K", defaults.rrf_k)),
            data_dir=Path(os.getenv("DATA_DIR", str(defaults.data_dir))),
        )
