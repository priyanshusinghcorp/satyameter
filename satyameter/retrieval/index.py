"""FAISS vector index over chunks, with an injectable embedding function."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from satyameter.ingestion.chunker import Chunk

EmbedFn = Callable[[list[str]], np.ndarray]


class SentenceTransformerEmbedder:
    """Lazy wrapper so importing this module never loads the model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # Keep CPU inference predictable on low-core laptops.
            torch.set_num_threads(2)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return np.asarray(embeddings, dtype="float32")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


class VectorIndex:
    def __init__(self, faiss_index, chunks: list[Chunk], embed_fn: EmbedFn):
        self._index = faiss_index
        self.chunks = chunks
        self._embed = embed_fn

    @classmethod
    def build(cls, chunks: list[Chunk], embed_fn: EmbedFn) -> "VectorIndex":
        import faiss

        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        vectors = np.ascontiguousarray(
            embed_fn([c.embed_text for c in chunks]), dtype="float32"
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
            raise ValueError(
                "Embedder must return a 2D array with one non-empty vector per chunk."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or infinite values.")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine here
        index.add(vectors)
        return cls(index, chunks, embed_fn)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        import faiss

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        vector = np.ascontiguousarray(self._embed([query]), dtype="float32")
        if vector.ndim != 2 or vector.shape != (1, self._index.d):
            raise ValueError(
                f"Query embedding has shape {vector.shape}; expected (1, {self._index.d}). "
                "Check that the same embedding model is used to build and query the index."
            )
        faiss.normalize_L2(vector)
        scores, ids = self._index.search(vector, min(k, len(self.chunks)))
        hits = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx == -1:
                continue
            hits.append(Hit(chunk=self.chunks[idx], score=float(score), rank=rank))
        return hits

    def save(self, directory: Path, embed_model: str = "unknown") -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "faiss.index"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        meta = {
            "embed_model": embed_model,
            "num_chunks": len(self.chunks),
            "dimension": int(self._index.d),
            "index_type": "IndexFlatIP",
            "normalized_vectors": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embed_fn: EmbedFn) -> "VectorIndex":
        import faiss

        index_file = directory / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError(
                f"No index at {directory}. Run: python -m satyameter.cli build-index"
            )
        chunks_file = directory / "chunks.jsonl"
        meta_file = directory / "meta.json"
        if not chunks_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"The vector index in {directory} is incomplete. Run: "
                "python -m satyameter.cli build-index"
            )
        index = faiss.read_index(str(index_file))
        chunks = []
        with chunks_file.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    chunks.append(Chunk(**json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid chunk metadata on line {line_number} of {chunks_file}."
                    ) from exc
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid index metadata file: {meta_file}") from exc
        expected_count = meta.get("num_chunks")
        if expected_count is not None and int(expected_count) != len(chunks):
            raise ValueError(
                f"Index metadata declares {expected_count} chunks, but {len(chunks)} were loaded. "
                "Rebuild the index."
            )
        expected_dimension = meta.get("dimension")
        if expected_dimension is not None and int(expected_dimension) != index.d:
            raise ValueError(
                f"Index metadata dimension {expected_dimension} does not match FAISS dimension {index.d}."
            )
        expected_model = meta.get("embed_model")
        actual_model = getattr(embed_fn, "model_name", None)
        if expected_model and actual_model and expected_model != actual_model:
            raise ValueError(
                "Embedding model mismatch: index was built with "
                f"'{expected_model}', but configuration requests '{actual_model}'. "
                "Restore the original model or rebuild the index."
            )
        if len(chunks) != index.ntotal:
            raise ValueError(
                f"Index integrity error: FAISS contains {index.ntotal} vectors but "
                f"chunks.jsonl contains {len(chunks)} records. Rebuild the index."
            )
        return cls(index, chunks, embed_fn)
