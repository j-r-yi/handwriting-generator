"""Keyboard symbols beyond the standard set: validation, placement and rendering."""

from __future__ import annotations

import pytest

from handwriting.baseline import Anchor, rule_for
from handwriting.charset import COMMON_SYMBOLS, DEFAULT_CHARSET, describe_char, validate_symbol_char
from handwriting.errors import InvalidCharacterError
from handwriting.renderer import find_missing_characters, render_text
from handwriting.sample_store import ProfileStore
from handwriting.settings import NO_VARIATION, PageSettings, PaperStyle, RenderSettings
from tests.helpers import make_new_samples


def test_common_symbols_are_new_and_valid() -> None:
    assert len(set(COMMON_SYMBOLS)) == len(COMMON_SYMBOLS)
    assert not set(COMMON_SYMBOLS) & set(DEFAULT_CHARSET)
    assert all(validate_symbol_char(c) == c for c in COMMON_SYMBOLS)


@pytest.mark.parametrize("text", ["😀", "❤️", "⭐", "👍🏽", "🇺🇸", "ab", " ", "́", "\x07"])
def test_emoji_and_non_characters_are_rejected(text: str) -> None:
    with pytest.raises(InvalidCharacterError):
        validate_symbol_char(text)


def test_symbols_are_normalised_and_described() -> None:
    assert validate_symbol_char("é") == "é"  # decomposed input shares samples with é
    assert describe_char("→") == "→  (rightwards arrow)"
    assert describe_char("a") == "a"


def test_symbols_have_sensible_size_and_position() -> None:
    assert rule_for("•").anchor is Anchor.CENTER and rule_for("•").height_xh < 0.5
    assert rule_for("°").anchor is Anchor.TOP
    assert rule_for("é").anchor is Anchor.BASELINE and rule_for("é").height_xh > rule_for("e").height_xh
    assert rule_for("Ü").height_xh > rule_for("U").height_xh
    assert rule_for("ç").anchor is Anchor.DESCENDER


def test_added_symbols_are_written_in_place(store: ProfileStore) -> None:
    meta = store.create_profile("Symbols")
    store.add_samples(meta.profile_id, make_new_samples("ao→•°é", variants=1))
    glyphs = store.load_glyph_set(meta.profile_id)
    text = "a → o • 20° é"
    assert set(find_missing_characters(text, glyphs)) == {"2", "0"}

    def placed(char: str):
        glyph = glyphs.variants(char)[0]
        return glyph.height / glyph.x_height_ref, glyph.descent / glyph.x_height_ref  # in x-heights

    assert placed("•")[0] < 0.5 and placed("•")[1] < 0  # small, floating at mid height
    assert placed("°")[1] < -1.0  # raised like a superscript
    assert placed("é")[1] == 0 and placed("é")[0] > placed("o")[0]

    settings = RenderSettings(page=PageSettings(dpi=100, paper_style=PaperStyle.BLANK), variation=NO_VARIATION,
                              seed=1)
    result = render_text(text, glyphs, settings)
    assert "".join(p.char for p in result.placements) == text.replace(" ", "")
