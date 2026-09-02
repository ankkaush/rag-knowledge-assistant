from app.ingestion.cleaning import clean_text, is_meaningful


def test_joins_mid_sentence_line_breaks():
    raw = "This sentence was\nwrapped by the PDF\nlayout engine."
    cleaned = clean_text(raw)
    assert "\n" not in cleaned
    assert cleaned == "This sentence was wrapped by the PDF layout engine."


def test_preserves_paragraph_breaks():
    raw = "First paragraph.\n\nSecond paragraph."
    cleaned = clean_text(raw)
    assert cleaned == "First paragraph.\n\nSecond paragraph."


def test_collapses_excess_blank_lines():
    raw = "Para one.\n\n\n\n\nPara two."
    cleaned = clean_text(raw)
    assert cleaned == "Para one.\n\nPara two."


def test_collapses_repeated_spaces():
    raw = "Too    many     spaces."
    assert clean_text(raw) == "Too many spaces."


def test_unicode_ligature_normalized():
    # "ﬁ" (U+FB01, single ligature glyph) should normalize toward "fi".
    raw = "The ﬁrst test."
    cleaned = clean_text(raw)
    assert "ﬁ" not in cleaned


def test_is_meaningful_rejects_near_empty_text():
    assert not is_meaningful("   ")
    assert not is_meaningful("ok")


def test_is_meaningful_accepts_real_content():
    assert is_meaningful("This is a real sentence with enough content.")
