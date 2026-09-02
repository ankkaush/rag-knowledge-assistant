from app.ingestion.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("", chunk_size_chars=100, chunk_overlap_chars=20) == []


def test_short_text_produces_one_chunk():
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size_chars=1000, chunk_overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].char_start == 0


def test_long_text_splits_into_multiple_overlapping_chunks():
    # 500 words, each 5 chars + space = enough to force several chunks at size 200.
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size_chars=200, chunk_overlap_chars=40)

    assert len(chunks) > 1
    # chunk_index is sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # every chunk respects the size bound (snapping to whitespace can only shrink it)
    assert all(len(c.content) <= 200 for c in chunks)
    # consecutive chunks overlap: the tail of one chunk's source range and the
    # head of the next chunk's source range share characters.
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end


def test_no_word_is_split_across_a_chunk_boundary():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size_chars=150, chunk_overlap_chars=30)
    words = set(text.split())
    for c in chunks:
        for token in c.content.split():
            assert token in words, f"chunk boundary split a word: {token!r}"


def test_page_number_and_section_are_attached_to_every_chunk():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size_chars=100, chunk_overlap_chars=20, page_number=3)
    assert all(c.page_number == 3 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size_chars=100, chunk_overlap_chars=100)
