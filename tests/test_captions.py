from cerebro.ingest._captions import clean_caption_text


def test_strips_bracketed_non_speech_tags():
    assert clean_caption_text("[Music]") == ""
    assert clean_caption_text("(Applause)") == ""
    assert clean_caption_text("[laughter]") == ""
    assert clean_caption_text("[MUSIC]") == ""  # case-insensitive


def test_strips_a_tag_embedded_mid_sentence():
    assert clean_caption_text("before the [Music] chorus starts") == "before the chorus starts"


def test_preserves_normal_speech_about_music():
    text = "I really love music and I think music theory is fascinating"
    assert clean_caption_text(text) == text


def test_leaves_ordinary_text_untouched():
    assert clean_caption_text("hello world") == "hello world"


def test_strips_multiple_known_tags():
    for tag in ("music", "applause", "laughter", "silence", "inaudible", "crosstalk", "background noise"):
        assert clean_caption_text(f"[{tag}]") == ""
        assert clean_caption_text(f"({tag})") == ""
