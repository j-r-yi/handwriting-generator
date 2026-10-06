"""Natural variation: presets, word/line effects, letter shape and pen pressure."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from handwriting.layout import compute_page_geometry
from handwriting.renderer import _layout_blocks, _LineShape, _place_glyph, _Planner, build_blocks, find_missing_characters, render_text
from handwriting.settings import NO_VARIATION, VARIATION_PRESETS, PageSettings, PaperStyle, RenderSettings
from tests.helpers import make_glyph_set

LOWER = "abcdefghijklmnopqrstuvwxyz"
BLANK = PageSettings(dpi=100, paper_style=PaperStyle.BLANK)
TEXT = "the quick brown fox jumps over the lazy dog\n" * 6


def _render(variation, text: str = TEXT, seed: int = 5, **kwargs):
    settings = RenderSettings(page=kwargs.pop("page", BLANK), variation=variation, seed=seed, **kwargs)
    return render_text(text, make_glyph_set(LOWER), settings)


def _ink(page) -> np.ndarray:
    return np.asarray(page.convert("L")) < 160


@pytest.mark.parametrize("preset", list(VARIATION_PRESETS))
def test_every_preset_writes_the_text_exactly_and_reproducibly(preset: str) -> None:
    variation = VARIATION_PRESETS[preset]
    first, second = _render(variation), _render(variation)
    assert "".join(p.char for p in first.placements) == TEXT.replace(" ", "").replace("\n", "")
    assert [p.tobytes() for p in first.pages] == [p.tobytes() for p in second.pages]


def test_presets_get_progressively_messier() -> None:
    def spread(preset: str) -> float:
        """How unevenly the line starts and baselines sit: std of ink row centres per line."""
        ink = _ink(_render(VARIATION_PRESETS[preset]).pages[0])
        rows = np.flatnonzero(ink.any(axis=1))
        lefts = [np.flatnonzero(ink[r]).min() for r in rows]
        return float(np.std(lefts))

    spreads = [spread(name) for name in ("Very Consistent", "Natural", "Messy", "Rushed notes")]
    assert spreads == sorted(spreads)
    assert spreads[-1] > spreads[0] * 2


def test_no_variation_lines_start_at_the_same_x() -> None:
    page = _render(NO_VARIATION, "abc\nabc\nabc").pages[0]
    ink = _ink(page)
    rows = np.flatnonzero(ink.any(axis=1))
    lines = np.split(rows, np.flatnonzero(np.diff(rows) > 5) + 1)
    lefts = {int(np.flatnonzero(ink[line].any(axis=0)).min()) for line in lines}
    assert len(lefts) == 1


def test_line_shape_follows_slope_and_reports_matching_tilt() -> None:
    uphill = _LineShape(origin=100.0, slope=-math.tan(math.radians(2.0)))
    assert uphill.dy(100.0) == 0.0
    assert uphill.dy(600.0) < 0  # further right is higher on the page
    assert uphill.tilt(300.0) == pytest.approx(2.0)
    wavy = _LineShape(waves=((3.0, 0.05, 0.0),))
    assert max(abs(wavy.dy(x)) for x in range(200)) == pytest.approx(3.0, abs=0.05)


def test_line_slope_makes_lines_drift_from_their_baseline() -> None:
    sloped = replace(NO_VARIATION, line_slope_deg=2.0)
    text = "a" * 60

    def line_drift(variation) -> float:
        ink = _ink(_render(variation, text).pages[0])
        cols = np.flatnonzero(ink.any(axis=0))
        left, right = ink[:, cols[:15]], ink[:, cols[-15:]]
        return abs(np.flatnonzero(left.any(axis=1)).mean() - np.flatnonzero(right.any(axis=1)).mean())

    assert line_drift(NO_VARIATION) < 1
    assert line_drift(sloped) > 3


def test_word_settings_move_whole_words_together() -> None:
    bouncy = replace(NO_VARIATION, word_baseline=0.3)
    result = _render(bouncy, "aaaa aaaa aaaa aaaa aaaa aaaa", seed=2)
    ink = _ink(result.pages[0])
    cols = np.flatnonzero(ink.any(axis=0))
    words = np.split(cols, np.flatnonzero(np.diff(cols) > 6) + 1)
    bottoms = [np.flatnonzero(ink[:, w].any(axis=1)).max() for w in words]
    assert len(words) == 6
    assert len(set(bottoms)) > 1  # words sit at different heights...
    for w in words:  # ...but the letters inside one word share a baseline
        letter_bottoms = {np.flatnonzero(ink[:, c]).max() for c in w if ink[:, c].sum() > 2}
        assert max(letter_bottoms) - min(letter_bottoms) <= 2


def test_margin_jitter_keeps_text_inside_the_page_margin() -> None:
    ragged = replace(VARIATION_PRESETS["Rushed notes"], margin_jitter=1.5)
    text = "word " * 400
    result = _render(ragged, text)
    geometry = compute_page_geometry(RenderSettings(page=BLANK))
    for page in result.pages:
        cols = np.flatnonzero(_ink(page).any(axis=0))
        assert cols.max() <= geometry.text_right + 0.5 * geometry.x_height


def _ring() -> np.ndarray:
    from PIL import Image, ImageDraw

    glyph = Image.new("L", (40, 50), 0)
    ImageDraw.Draw(glyph).ellipse((8, 10, 32, 42), outline=255, width=4)
    return np.asarray(glyph, dtype=np.float32)


def test_place_glyph_without_effects_is_untouched() -> None:
    ring = _ring()
    out, left, top = _place_glyph(ring, 10.4, 20.6, slant=0.0, rotation=0.0, warp=0.0, seed=1)
    assert out is ring and (left, top) == (10, 21)


def test_place_glyph_warp_changes_shape_but_not_amount_of_ink() -> None:
    ring = _ring()
    warped, left, top = _place_glyph(ring, 100.0, 100.0, 0.0, 0.0, warp=3.0, seed=1)
    again, *_ = _place_glyph(ring, 100.0, 100.0, 0.0, 0.0, warp=3.0, seed=1)
    other, *_ = _place_glyph(ring, 100.0, 100.0, 0.0, 0.0, warp=3.0, seed=2)
    assert np.array_equal(again, warped) and not np.array_equal(other, warped)
    assert left < 100 and top < 100  # padded for the displacement
    assert warped.sum() == pytest.approx(ring.sum(), rel=0.15)


def test_place_glyph_slant_and_rotation_keep_the_glyph_in_place() -> None:
    ring = _ring()
    for slant, rotation in ((0.3, 0.0), (-0.3, 0.0), (0.0, 8.0), (0.2, -5.0)):
        out, left, top = _place_glyph(ring, 50.0, 60.0, slant, rotation, warp=0.0, seed=0)
        assert out.sum() == pytest.approx(ring.sum(), rel=0.05)
        rows, cols = np.nonzero(out > 128)
        # The ink's centre moves by at most the shear of half the glyph height.
        assert abs(cols.mean() + left - (50 + 20 + slant * 25)) < 3
        assert abs(rows.mean() + top - (60 + 26)) < 3

    leaning, left, top = _place_glyph(ring, 0.0, 0.0, 0.4, 0.0, warp=0.0, seed=0)
    ink = leaning > 128
    top_rows, bottom_rows = ink[: ink.shape[0] // 2], ink[ink.shape[0] // 2:]
    assert np.nonzero(top_rows)[1].mean() > np.nonzero(bottom_rows)[1].mean()  # top leans right


def test_ink_fade_lightens_some_words() -> None:
    faded = replace(NO_VARIATION, ink_fade=0.5)
    darkest = lambda r: int(np.asarray(r.pages[0].convert("L")).astype(int).sum())  # noqa: E731
    assert darkest(_render(faded)) > darkest(_render(NO_VARIATION))  # lighter page = higher sum


def test_fatigue_makes_the_end_messier_than_the_start() -> None:
    tired = replace(NO_VARIATION, baseline_jitter=0.1, fatigue=3.0)
    text = "\n".join(["aaaaaaaaaaaaaaaaaaaa"] * 12)
    ink = _ink(_render(tired, text).pages[0])
    rows = np.flatnonzero(ink.any(axis=1))
    lines = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1)

    def unevenness(line_rows) -> float:
        band = ink[line_rows.min():line_rows.max() + 1]
        bottoms = [np.flatnonzero(band[:, c]).max() for c in range(band.shape[1]) if band[:, c].any()]
        return float(np.std(bottoms))

    assert unevenness(lines[-1]) > unevenness(lines[0])


def _layout(text: str, variation, chars: str = LOWER + "-", seed: int = 3):
    settings = RenderSettings(page=BLANK, variation=variation, markdown=True, seed=seed)
    geometry = compute_page_geometry(settings)
    planner = _Planner(make_glyph_set(chars), settings, geometry.x_height, seed)
    lines, _ = _layout_blocks(build_blocks(text, True), planner, settings.markdown_style, geometry)
    return lines


def test_list_indents_wander_but_nesting_stays_clear() -> None:
    text = "- parent item\n  - child item\n" * 12
    still = [line for line in _layout(text, NO_VARIATION) if line.first]
    assert len({round(line.marker_x, 3) for line in still if line.block.indent == 0}) == 1

    rushed = [line for line in _layout(text, VARIATION_PRESETS["Rushed notes"]) if line.first]
    parents = [line.marker_x for line in rushed if line.block.indent == 0]
    children = [line.marker_x for line in rushed if line.block.indent == 1]
    assert len({round(x, 3) for x in parents}) > 4  # items don't sit in a perfect column...
    assert min(children) > max(parents)  # ...but a nested item is always clearly nested


def test_wrapped_lines_drift_but_never_start_left_of_their_bullet() -> None:
    text = "\n".join(["- " + "lorem ipsum dolor sit amet " * 4] * 8)
    lines = _layout(text, VARIATION_PRESETS["Messy"])
    starts: dict[int, float] = {}
    continued = []
    for line in lines:
        if line.first:
            starts[id(line.block)] = line.text_x
        else:
            assert line.text_x >= line.marker_x
            continued.append(line.text_x - starts[id(line.block)])
    assert continued and len({round(x, 3) for x in continued}) > 3


def test_uneven_line_ends_wrap_earlier_but_keep_the_text() -> None:
    text = "lorem ipsum dolor sit amet " * 30
    even = _layout(text, NO_VARIATION)
    ragged = _layout(text, replace(NO_VARIATION, ragged_right=8.0))
    assert len(ragged) > len(even)
    widths = [line.line.width for line in ragged[:-1]]
    assert max(widths) - min(widths) > 2 * (max(w.line.width for w in even[:-1]) - min(w.line.width for w in even[:-1]))


def test_bullets_are_written_as_typed() -> None:
    lines = _layout("- dash item\n* star item", NO_VARIATION)
    dash, star = (line for line in lines if line.first)
    assert [p.char for p in dash.marker_plans] == ["-"]  # the writer's own dash
    assert star.marker_plans == ()  # drawn dot (no sample for •)
    without_dash = make_glyph_set(LOWER)
    assert find_missing_characters("- dash item", without_dash, markdown=True) == {}  # drawn instead
    assert _layout("- dash item", NO_VARIATION, chars=LOWER)[0].marker_plans == ()


def test_layout_variation_keeps_every_character() -> None:
    text = "# Title\n- one two three four five six seven\n  - nested words here and there\n1. numbered item\n" * 6
    result = _render(VARIATION_PRESETS["Rushed notes"], text, markdown=True)
    expected = text.replace("# ", "").replace("- ", "").replace("1. ", "1.")
    assert "".join(p.char for p in result.placements) == "".join(c for c in expected if not c.isspace())


def test_letters_may_touch_only_as_far_as_the_overlap_allows() -> None:
    def gaps(variation) -> list[float]:
        settings = RenderSettings(page=BLANK, variation=variation, seed=4, letter_spacing=0.0)
        planner = _Planner(make_glyph_set(LOWER), settings, 20.0, 4)
        plans = [planner.plan(c) for c in "abcdefghij" * 30]
        return [(p.advance - p.ink_width) / 20.0 for p in plans]

    jittery = replace(NO_VARIATION, letter_spacing_jitter=0.2)
    assert min(gaps(jittery)) >= 0  # no overlap allowed: letters never collide
    touching = gaps(replace(jittery, letter_overlap=0.08))
    assert min(touching) >= -0.08 - 1e-9
    assert sum(g < 0 for g in touching) > 20  # but quite a few letters run into the next one


def test_headings_are_not_underlined_unless_asked() -> None:
    from handwriting.settings import MarkdownStyle

    def page(**kwargs) -> bytes:
        return _render(NO_VARIATION, "# abc def", markdown=True, **kwargs).pages[0].tobytes()

    assert page() == page(markdown_style=MarkdownStyle(underline_levels=0))
    assert page() != page(markdown_style=MarkdownStyle(underline_levels=2))
