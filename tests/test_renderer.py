"""Rendering: determinism, content integrity, variants, missing glyphs, pages."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from handwriting.errors import EmptyTextError, MissingGlyphError
from handwriting.layout import normalize_text
from handwriting.models import GlyphSet, LoadedGlyph
from handwriting.renderer import VariantSelector, find_missing_characters, render_text
from handwriting.settings import (
    MISSING_GLYPH_COLOR,
    NO_VARIATION,
    MissingGlyphPolicy,
    PageFormat,
    PageSettings,
    PaperStyle,
    RenderSettings,
    VariantMode,
)
from tests.helpers import make_glyph_set

LOWER = "abcdefghijklmnopqrstuvwxyz"
FAST = PageSettings(dpi=100)  # low resolution keeps tests quick
STILL = NO_VARIATION


def _settings(**kwargs) -> RenderSettings:
    kwargs.setdefault("page", FAST)
    kwargs.setdefault("seed", 1234)
    return RenderSettings(**kwargs)


def test_same_seed_gives_identical_pages() -> None:
    glyphs = make_glyph_set(LOWER)
    text = "machine learning is interesting " * 20
    first = render_text(text, glyphs, _settings(seed=7))
    second = render_text(text, glyphs, _settings(seed=7))
    assert first.seed == second.seed == 7
    assert [p.tobytes() for p in first.pages] == [p.tobytes() for p in second.pages]
    assert [p.sample_id for p in first.placements] == [p.sample_id for p in second.placements]


def test_different_seed_changes_appearance_but_not_content() -> None:
    glyphs = make_glyph_set(LOWER)
    text = "eeeeee aaaa banana"
    a = render_text(text, glyphs, _settings(seed=1))
    b = render_text(text, glyphs, _settings(seed=2))
    assert a.pages[0].tobytes() != b.pages[0].tobytes()
    assert [p.char for p in a.placements] == [p.char for p in b.placements]


def test_random_seed_is_reported_when_not_given() -> None:
    result = render_text("abc", make_glyph_set("abc"), _settings(seed=None))
    assert isinstance(result.seed, int)
    again = render_text("abc", make_glyph_set("abc"), _settings(seed=result.seed))
    assert again.pages[0].tobytes() == result.pages[0].tobytes()


def test_rendered_content_matches_input_exactly() -> None:
    glyphs = make_glyph_set(LOWER + ".")
    text = "Machine learning is interesting.\n\n  new paragraph\twith tab é"
    result = render_text(text, glyphs, _settings())
    normalized = normalize_text(text)
    placed = "".join(p.char for p in result.placements)
    assert placed == "".join(c for c in normalized if not c.isspace())
    assert [p.index for p in result.placements] == sorted(p.index for p in result.placements)


def test_missing_characters_listed_and_rendered_as_placeholder() -> None:
    glyphs = make_glyph_set("ab")
    text = "abZ?Z"
    assert find_missing_characters(text, glyphs) == {"Z": 2, "?": 1}
    result = render_text(text, glyphs, _settings())
    assert result.missing == {"Z": 2, "?": 1}
    # Missing characters are still placed (never silently dropped)...
    assert "".join(p.char for p in result.placements) == "abZ?Z"
    assert [p.sample_id is None for p in result.placements] == [False, False, True, True, True]
    # ...and drawn in the warning colour.
    pixels = np.asarray(result.pages[0])
    assert np.all(pixels == MISSING_GLYPH_COLOR, axis=-1).any()


def test_missing_policy_error_raises() -> None:
    with pytest.raises(MissingGlyphError) as info:
        render_text("abc!", make_glyph_set("abc"), _settings(missing_policy=MissingGlyphPolicy.ERROR))
    assert info.value.missing == {"!": 1}


def test_missing_policy_typed_draws_in_ink_colour() -> None:
    ink = (10, 120, 30)
    result = render_text("X", make_glyph_set("a"),
                         _settings(missing_policy=MissingGlyphPolicy.TYPED, ink_color=ink,
                                   page=PageSettings(dpi=100, paper_style=PaperStyle.BLANK)))
    pixels = np.asarray(result.pages[0])
    assert np.all(pixels == ink, axis=-1).any()


def test_typographic_quotes_use_ascii_samples() -> None:
    glyphs = make_glyph_set("it's")
    assert find_missing_characters("it’s", glyphs) == {}


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_empty_text_is_rejected(text: str) -> None:
    with pytest.raises(EmptyTextError):
        render_text(text, make_glyph_set("a"), _settings())


@pytest.mark.parametrize("page_format, dpi, size", [
    (PageFormat.LETTER, 300, (2550, 3300)),
    (PageFormat.A4, 300, (2480, 3508)),
    (PageFormat.LETTER, 100, (850, 1100)),
])
def test_output_page_dimensions(page_format: PageFormat, dpi: int, size: tuple[int, int]) -> None:
    result = render_text("abc", make_glyph_set("abc"),
                         _settings(page=PageSettings(page_format=page_format, dpi=dpi)))
    assert result.pages[0].size == size
    assert result.pages[0].mode == "RGB"


def test_long_text_overflows_onto_multiple_pages() -> None:
    glyphs = make_glyph_set(LOWER)
    text = "\n".join(["line"] * 70)  # more lines than fit on one ruled page
    result = render_text(text, glyphs, _settings())
    assert len(result.pages) >= 2
    pages_used = Counter(p.page for p in result.placements)
    assert set(pages_used) == set(range(len(result.pages)))
    assert sum(pages_used.values()) == 70 * 4


def test_variant_selection_cycles_through_all_variants() -> None:
    selector = VariantSelector(random.Random(0), VariantMode.SHUFFLE)
    picks = [selector.choose("e", 3) for _ in range(30)]
    for start in range(0, 30, 3):
        assert sorted(picks[start:start + 3]) == [0, 1, 2]  # each "deck" uses every variant
    assert all(a != b for a, b in zip(picks, picks[1:]))  # never the same twice in a row


def test_random_mode_never_repeats_immediately() -> None:
    selector = VariantSelector(random.Random(5), VariantMode.RANDOM)
    picks = [selector.choose("e", 2) for _ in range(50)]
    assert all(a != b for a, b in zip(picks, picks[1:]))
    assert VariantSelector(random.Random(5), VariantMode.FIRST).choose("e", 4) == 0


def test_single_variant_always_used() -> None:
    selector = VariantSelector(random.Random(0), VariantMode.SHUFFLE)
    assert {selector.choose("x", 1) for _ in range(5)} == {0}


def test_repeated_letters_use_different_variants() -> None:
    result = render_text("eeeeee", make_glyph_set("e", variants=3), _settings())
    ids = [p.sample_id for p in result.placements]
    assert len(set(ids)) == 3
    assert all(a != b for a, b in zip(ids, ids[1:]))


def _ink_columns(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    return np.flatnonzero((gray < 128).any(axis=0))


def test_character_widths_are_proportional() -> None:
    narrow = LoadedGlyph("i", "i0", Image.new("RGBA", (8, 40), (0, 0, 0, 255)), 40, 0)
    wide = LoadedGlyph("m", "m0", Image.new("RGBA", (40, 40), (0, 0, 0, 255)), 40, 0)
    glyphs = GlyphSet({"i": [narrow], "m": [wide]})
    plain = PageSettings(dpi=100, paper_style=PaperStyle.BLANK)
    settings = _settings(page=plain, variation=STILL)
    width_i = _ink_columns(render_text("iiii", glyphs, settings).pages[0])
    width_m = _ink_columns(render_text("mmmm", glyphs, settings).pages[0])
    assert width_m.max() - width_m.min() > 2.5 * (width_i.max() - width_i.min())


def test_descenders_hang_below_the_baseline() -> None:
    glyphs = make_glyph_set("ag", variants=1)
    plain = PageSettings(dpi=300, paper_style=PaperStyle.BLANK)
    settings = _settings(page=plain, variation=STILL)
    a = np.asarray(render_text("a", glyphs, settings).pages[0].convert("L"))
    g = np.asarray(render_text("g", glyphs, settings).pages[0].convert("L"))
    bottom_a = np.flatnonzero((a < 128).any(axis=1)).max()
    bottom_g = np.flatnonzero((g < 128).any(axis=1)).max()
    x_height_px = settings.x_height_mm / 25.4 * 300
    assert bottom_g - bottom_a > 0.25 * x_height_px


def test_ink_colour_is_applied_with_alpha_preserved() -> None:
    soft = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    soft.paste((0, 0, 0, 255), (5, 5, 15, 15))
    soft.paste((0, 0, 0, 128), (0, 0, 5, 20))  # half-transparent edge
    glyphs = GlyphSet({"o": [LoadedGlyph("o", "o0", soft, 20, 0)]})
    blue = (20, 42, 116)
    page = PageSettings(dpi=100, paper_style=PaperStyle.BLANK)
    pixels = np.asarray(render_text("o", glyphs, _settings(ink_color=blue, page=page, variation=STILL)).pages[0]).reshape(-1, 3)
    colours = {tuple(p) for p in pixels}
    assert blue in colours
    # Anti-aliased pixels lie between the ink colour and white paper.
    assert any(c != blue and c != (255, 255, 255) and c[2] > c[0] for c in colours)
