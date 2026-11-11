"""Turn a Wikipedia article into sentence-aware, overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
SKIP_SECTIONS = {
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "citations",
    "footnotes",
}
MIN_CHUNK_WORDS = 15


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_title: str
    url: str
    section: str
    chunk_index: int
    text: str

    @property
    def embed_text(self) -> str:
        """Text that is embedded. The title helps retrieval a lot."""
        return f"{self.doc_title} ({self.section}): {self.text}"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split article text into (section_title, body) pairs, dropping junk sections."""
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    skip_level: int | None = None
    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            level, title = len(match.group(1)), match.group(2).strip()
            if title.lower() in SKIP_SECTIONS:
                skip_level = level
            elif skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None:
                sections.append((title, []))
            continue
        if skip_level is None:
            sections[-1][1].append(line)
    result = []
    for title, lines in sections:
        body = " ".join(part.strip() for part in lines if part.strip())
        if body:
            result.append((title, body))
    return result


def _split_sentences(body: str, max_words: int) -> list[str]:
    sentences = []
    for sentence in SENTENCE_SPLIT_RE.split(body):
        words = sentence.split()
        if len(words) <= max_words:
            sentences.append(sentence.strip())
        else:  # extremely long sentence: hard-split by words
            for i in range(0, len(words), max_words):
                sentences.append(" ".join(words[i : i + max_words]))
    return [s for s in sentences if s]


def chunk_body(body: str, max_words: int, overlap_words: int) -> list[str]:
    """Pack whole sentences into chunks of <= max_words with sentence overlap."""
    sentences = _split_sentences(body, max_words)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and current_words + n > max_words:
            chunks.append(" ".join(current))
            # carry over trailing sentences as overlap
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                pn = len(prev.split())
                if carried + pn > overlap_words:
                    break
                carry.insert(0, prev)
                carried += pn
            if carried + n > max_words:  # never let overlap push a chunk over the limit
                carry, carried = [], 0
            current, current_words = carry, carried
        current.append(sentence)
        current_words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_article(
    title: str,
    url: str,
    text: str,
    slug: str,
    max_words: int = 150,
    overlap_words: int = 25,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for section, body in split_sections(text):
        for piece in chunk_body(body, max_words, overlap_words):
            if len(piece.split()) < MIN_CHUNK_WORDS:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{slug}-{index:04d}",
                    doc_title=title,
                    url=url,
                    section=section,
                    chunk_index=index,
                    text=piece,
                )
            )
            index += 1
    return chunks
