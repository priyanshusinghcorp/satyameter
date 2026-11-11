from satyameter.ingestion.chunker import chunk_article, chunk_body, split_sections

ARTICLE = """Marie Curie was a physicist and chemist. She won two Nobel Prizes. She was born in Warsaw in 1867 and later moved to Paris to study.

== Early life ==
She grew up in Warsaw. Her parents were teachers. She studied secretly at the Flying University because women could not attend regular university.

=== Family ===
She had four siblings. Her mother died when she was young.

== References ==
Some citation text that must never appear in any chunk at all.

=== Sub reference ===
More citation junk.

== Legacy ==
The element curium is named after her and her husband. Many institutes carry her name. She remains one of the most celebrated scientists in history.
"""


def test_split_sections_skips_reference_blocks():
    sections = dict(split_sections(ARTICLE))
    assert "Introduction" in sections
    assert "Early life" in sections
    assert "Family" in sections
    assert "Legacy" in sections
    assert "References" not in sections
    assert "Sub reference" not in sections
    assert all("citation" not in body for body in sections.values())


def test_chunks_respect_max_words_and_have_metadata():
    long_body = " ".join(f"Sentence number {i} is here with some filler words." for i in range(60))
    pieces = chunk_body(long_body, max_words=40, overlap_words=10)
    assert len(pieces) > 5
    assert all(len(p.split()) <= 40 for p in pieces)


def test_chunk_overlap_repeats_trailing_sentence():
    body = "Alpha one two three. Beta four five six. Gamma seven eight nine. Delta ten eleven twelve."
    pieces = chunk_body(body, max_words=8, overlap_words=4)
    assert len(pieces) >= 2
    # last sentence of a chunk reappears at the start of the next
    assert pieces[0].split(". ")[-1].rstrip(".") in pieces[1]


def test_chunk_article_ids_unique_and_titles_kept():
    chunks = chunk_article("Marie Curie", "https://example.org/mc", ARTICLE, slug="marie-curie", max_words=30, overlap_words=5, )
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.doc_title == "Marie Curie" for c in chunks)
    assert all("citation" not in c.text for c in chunks)
    assert "Marie Curie" in chunks[0].embed_text
