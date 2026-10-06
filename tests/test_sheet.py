"""Sample-sheet template geometry and photo/scan import."""

from __future__ import annotations

import numpy as np
import pytest

from handwriting.charset import DEFAULT_CHARSET
from handwriting.errors import SheetDetectionError
from handwriting.settings import PageFormat
from handwriting.sheet_import import ExtractedSample, assign_x_height, import_sheet
from handwriting.template import (
    SheetIdentity,
    build_sheet_layouts,
    decode_identity,
    encode_identity,
    render_sample_sheet,
)
from tests.helpers import filled_sheet, simulate_photo


@pytest.mark.parametrize("page_format", list(PageFormat))
@pytest.mark.parametrize("variants", [1, 3, 5])
def test_layout_covers_every_character_exactly(page_format: PageFormat, variants: int) -> None:
    layouts = build_sheet_layouts(page_format, variants)
    cells = [(c.char, c.variant) for layout in layouts for c in layout.cells]
    assert sorted(cells) == sorted((ch, v) for ch in DEFAULT_CHARSET for v in range(variants))
    for layout in layouts:
        width, height = page_format.size_px(300)
        rects = [c.rect for c in layout.cells] + [r for _, r in layout.labels]
        for r in rects:
            assert 0 <= r.x and r.x + r.w <= width and 0 <= r.y and r.y + r.h <= height
        # Labels never overlap writing cells.
        for _, label in layout.labels:
            for cell in layout.cells:
                r = cell.rect
                assert label.x + label.w <= r.x or r.x + r.w <= label.x or \
                    label.y + label.h <= r.y or r.y + r.h <= label.y


def test_default_sheet_is_three_pages_with_three_samples() -> None:
    pages = render_sample_sheet(PageFormat.LETTER)
    assert len(pages) == 3
    assert pages[0].size == (2550, 3300)


@pytest.mark.parametrize("identity", [
    SheetIdentity(PageFormat.LETTER, 0, 3),
    SheetIdentity(PageFormat.A4, 7, 1),
    SheetIdentity(PageFormat.A4, 15, 8),
])
def test_page_code_round_trip(identity: SheetIdentity) -> None:
    bits = encode_identity(identity)
    assert decode_identity(bits) == identity
    flipped = list(bits)
    flipped[5] ^= 1
    assert decode_identity(flipped) is None  # parity catches single-bit errors
    assert decode_identity(bits[::-1]) is None


@pytest.mark.parametrize("page_format, page_index, rotate_180", [
    (PageFormat.LETTER, 0, False),
    (PageFormat.A4, 1, True),
])
def test_imports_photographed_sheet(page_format: PageFormat, page_index: int, rotate_180: bool) -> None:
    skipped = {(c.char, 2) for c in build_sheet_layouts(page_format)[page_index].cells[:9]}
    layout, page = filled_sheet(page_format, page_index, skip=skipped)
    photo = simulate_photo(page, angle=-4 if rotate_180 else 6, rotate_180=rotate_180)

    result = import_sheet(photo)

    assert result.identity == SheetIdentity(page_format, page_index, 3)
    assert result.identity_detected and result.method == "markers"
    found = {(s.char, s.variant) for s in result.samples}
    expected = {(c.char, c.variant) for c in layout.cells} - skipped
    assert found == expected
    assert set(result.empty_cells) == skipped
    for sample in result.samples:
        alpha = sample.rgba[:, :, 3]
        assert alpha.max() == 255 and alpha[0, 0] == 0
        assert sample.x_height_ref > 0


def test_extracted_descender_baseline_comes_from_guide() -> None:
    _, page = filled_sheet(PageFormat.LETTER, 0)
    result = import_sheet(simulate_photo(page, angle=2))
    g = next(s for s in result.samples if s.char == "g")
    a = next(s for s in result.samples if s.char == "a")
    height_g = g.rgba.shape[0]
    assert 0.15 * height_g < height_g - g.baseline_from_top < 0.6 * height_g
    assert abs(a.rgba.shape[0] - a.baseline_from_top) < 0.15 * a.rgba.shape[0]


def test_manual_identity_override() -> None:
    _, page = filled_sheet(PageFormat.LETTER, 2)
    override = SheetIdentity(PageFormat.LETTER, 2, 3)
    result = import_sheet(simulate_photo(page), identity_override=override)
    assert result.identity == override and not result.identity_detected
    assert {s.char for s in result.samples} <= set(DEFAULT_CHARSET)
    assert len(result.samples) > 50


def test_image_without_sheet_raises_detection_error() -> None:
    noise = np.random.default_rng(3).integers(0, 255, (900, 700, 3), dtype=np.uint8)
    with pytest.raises(SheetDetectionError):
        import_sheet(noise)
    with pytest.raises(SheetDetectionError):
        import_sheet(np.full((50, 50, 3), 255, dtype=np.uint8))


def _sample(char: str, height: float) -> ExtractedSample:
    return ExtractedSample(key=char, char=char, variant=0, page_index=0, rgba=np.zeros((1, 1, 4), np.uint8),
                           ink_width=height, ink_height=height, baseline_from_top=None)


def test_assign_x_height_uses_lowercase_then_known_reference() -> None:
    batch = [_sample("a", 50), _sample("e", 52), _sample("o", 54), _sample("H", 90)]
    assert {s.x_height_ref for s in assign_x_height(batch)} == {52}
    capitals = [_sample("H", 80), _sample("K", 80)]
    assert {s.x_height_ref for s in assign_x_height(capitals, known_reference=47.0)} == {47.0}
    assert {s.x_height_ref for s in assign_x_height(capitals)} == {50.0}  # 80 / 1.6
