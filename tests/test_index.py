import zlib

import numpy as np
import pytest

from satyameter.ingestion.chunker import Chunk

faiss = pytest.importorskip("faiss")

from satyameter.retrieval.index import VectorIndex  # noqa: E402


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedder, so tests need no model download."""
    out = np.zeros((len(texts), 128), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().replace("(", " ").replace(")", " ").replace(":", " ").split():
            out[row, zlib.crc32(word.encode()) % 128] += 1.0
    return out


def make_chunks():
    data = [
        ("Python", "Python is a programming language created by Guido van Rossum."),
        ("Everest", "Mount Everest is the highest mountain above sea level in the Himalayas."),
        ("Penicillin", "Penicillin was discovered by Alexander Fleming in 1928."),
    ]
    return [Chunk(f"c{i}", t, f"https://x/{i}", "Introduction", i, txt) for i, (t, txt) in enumerate(data)]


def test_search_returns_most_relevant_chunk_first():
    index = VectorIndex.build(make_chunks(), fake_embed)
    hits = index.search("highest mountain Everest", k=2)
    assert hits[0].chunk.doc_title == "Everest"
    assert hits[0].rank == 1 and hits[0].score >= hits[1].score


def test_save_and_load_roundtrip(tmp_path):
    built = VectorIndex.build(make_chunks(), fake_embed)
    built.save(tmp_path, embed_model="fake")
    loaded = VectorIndex.load(tmp_path, fake_embed)
    assert len(loaded.chunks) == 3
    assert loaded.search("who discovered penicillin", k=1)[0].chunk.doc_title == "Penicillin"


def test_k_larger_than_corpus_is_safe():
    index = VectorIndex.build(make_chunks(), fake_embed)
    assert len(index.search("anything", k=50)) == 3
