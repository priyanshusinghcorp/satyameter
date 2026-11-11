"""Central configuration. Paths and model names live here, nowhere else."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # Paths
    root: Path = ROOT
    raw_dir: Path = ROOT / "data" / "raw"
    index_dir: Path = ROOT / "data" / "index"
    cache_path: Path = ROOT / "data" / "cache.sqlite"
    history_path: Path = ROOT / "data" / "history.sqlite"
    eval_dir: Path = ROOT / "data" / "eval"
    seed_titles: Path = ROOT / "data" / "seed_titles.txt"

    # Retrieval
    embed_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    chunk_words: int = 150
    chunk_overlap: int = 25
    top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 5))

    # Verification
    nli_model: str = field(
        default_factory=lambda: os.getenv(
            "NLI_MODEL", "cross-encoder/nli-deberta-v3-base"
        )
    )
    claim_k: int = field(default_factory=lambda: _env_int("CLAIM_TOP_K", 3))
    max_evidence: int = field(default_factory=lambda: _env_int("MAX_EVIDENCE", 8))
    nli_max_length: int = field(default_factory=lambda: _env_int("NLI_MAX_LENGTH", 384))
    cpu_threads: int = field(default_factory=lambda: _env_int("CPU_THREADS", 2))

    # LLM backend. Groq is the practical default for this CPU-only 8 GB laptop;
    # Ollama is provided as an optional local alternative.
    llm_backend: str = field(
        default_factory=lambda: os.getenv("LLM_BACKEND", "groq").strip().lower()
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    )
    groq_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("GROQ_REASONING_EFFORT", "low")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_timeout_s: int = field(default_factory=lambda: _env_int("OLLAMA_TIMEOUT_S", 180))


def get_settings() -> Settings:
    return Settings()
