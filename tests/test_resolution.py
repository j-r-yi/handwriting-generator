"""Resolution: high-resolution capture, 600 DPI output and lossless-looking export."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest

from handwriting.errors import HandwritingError
from handwriting.export import pdf_bytes
from handwriting.renderer import MAX_PAGES, max_pages, render_text
from handwriting.sample_store import NewSample, ProfileStore
from handwriting.settings import INK_COLORS, NO_VARIATION, PageFormat, PageSettings, PaperStyle, RenderSettings
from handwriting.sheet_import import MAX_CAPTURE_DPI, ExtractedSample, assign_x_height, import_sheet
from handwriting.utils import decode_pdf_pages
from tests.helpers import filled_sheet, make_glyph_set
from tests.test_pdf_import import scanned_pdf


@pytest.fixture(scope="module")
def sheet_page() -> np.ndarray:
    return filled_sheet(PageFormat.LETTER, 0)[1]


def _sample_for(result, char: str):
    return next(s for s in result.samples if s.char == char)


def test_sheets_are_read_at_the_scan_resolution(sheet_page: np.ndarray) -> None:
    normal = import_sheet(sheet_page)
    sharp = import_sheet(cv2.resize(sheet_page, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    assert len(sharp.samples) == len(normal.samples)
    e300, e600 = _sample_for(normal, "e"), _sample_for(sharp, "e")
    assert e300.capture_dpi == 300 and e600.capture_dpi == MAX_CAPTURE_DPI
    assert e600.rgba.shape[0] == pytest.approx(2 * e300.rgba.shape[0], rel=0.1)
    # Same physical size, measured in the samples' own pixels.
    assert e600.x_height_ref == pytest.approx(2 * e300.x_height_ref, rel=0.05)


def test_capture_resolution_is_capped(sheet_page: np.ndarray) -> None:
    huge = cv2.resize(sheet_page, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_LINEAR)
    assert _sample_for(import_sheet(huge), "a").capture_dpi == MAX_CAPTURE_DPI


def test_x_height_is_shared_across_pages_read_at_different_resolutions() -> None:
    def sample(char: str, height: float, dpi: float) -> ExtractedSample:
        return ExtractedSample(key=f"{char}{dpi}", char=char, variant=0, page_index=0,
                               rgba=np.zeros((1, 1, 4), np.uint8), ink_width=height, ink_height=height,
                               baseline_from_top=None, capture_dpi=dpi)

    batch = [sample("a", 40, 300), sample("e", 42, 300), sample("o", 84, 600), sample("H", 130, 600)]
    refs = {(s.char, s.x_height_ref) for s in assign_x_height(batch)}
    assert refs == {("a", 42.0), ("e", 42.0), ("o", 84.0), ("H", 84.0)}
    capitals = [sample("H", 160, 600)]
    assert assign_x_height(capitals, known_reference=40.0)[0].x_height_ref == 80.0


def test_store_keeps_capture_resolution_and_normalises_reference(store: ProfileStore) -> None:
    meta = store.create_profile("Sharp")
    rgba = np.zeros((20, 10, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    store.add_samples(meta.profile_id, [NewSample("a", rgba, 40.0, source="template"),
                                        NewSample("b", rgba, 84.0, source="template", capture_dpi=600),
                                        NewSample("c", rgba, 42.0, source="template", capture_dpi=300)])
    reloaded = ProfileStore(store.data_dir).load_profile(meta.profile_id)
    assert reloaded.characters["b"].samples[0].capture_dpi == 600
    assert reloaded.characters["a"].samples[0].capture_dpi is None  # older samples: 300 DPI
    assert store.template_x_height_reference(meta.profile_id) == 42.0


def test_pdf_scans_are_rasterised_at_their_own_resolution() -> None:
    page = np.full((3300, 2550, 3), 255, dtype=np.uint8)
    big = cv2.resize(page, (5100, 6600))
    assert abs(decode_pdf_pages(scanned_pdf([big], dpi=600))[0].shape[1] - 5100) <= 1
    assert abs(decode_pdf_pages(scanned_pdf([page], dpi=300))[0].shape[1] - 2550) <= 1
    assert abs(decode_pdf_pages(scanned_pdf([big], dpi=600), dpi=300)[0].shape[1] - 2550) <= 1


def test_page_limit_scales_with_resolution() -> None:
    assert max_pages(150) == max_pages(300) == MAX_PAGES
    assert max_pages(600) == MAX_PAGES // 4
    settings = RenderSettings(page=PageSettings(dpi=600), variation=NO_VARIATION, seed=1)
    with pytest.raises(HandwritingError, match="lower output resolution"):
        render_text("a\n" * 40 * (MAX_PAGES // 4 + 1), make_glyph_set("a"), settings)


def test_600_dpi_output_has_twice_the_pixels_and_same_layout() -> None:
    glyphs = make_glyph_set("abc ")
    text = "abc cab bca " * 30

    def render(dpi: int):
        page = PageSettings(dpi=dpi, paper_style=PaperStyle.BLANK)
        return render_text(text, glyphs, RenderSettings(page=page, variation=NO_VARIATION, seed=4))

    low, high = render(300), render(600)
    assert high.pages[0].size == (2 * low.pages[0].size[0], 2 * low.pages[0].size[1])
    assert [(p.line, p.char) for p in high.placements] == [(p.line, p.char) for p in low.placements]


def test_pdf_export_is_not_visibly_compressed() -> None:
    page = render_text("minimum " * 40, make_glyph_set("minu "),
                       RenderSettings(page=PageSettings(dpi=200), ink_color=INK_COLORS["Blue"], seed=2)).pages[0]
    document = pdfium.PdfDocument(pdf_bytes([page], 200))
    rendered = document[0].render(scale=200 / 72).to_pil().convert("RGB").resize(page.size)
    error = np.abs(np.asarray(rendered, dtype=int) - np.asarray(page, dtype=int)).mean()
    document.close()
    assert error < 0.8  # Pillow's default JPEG settings give about 1.1 here


def test_no_variation_letters_are_not_resampled_beyond_resizing() -> None:
    """Without slant/rotation/warp a glyph is only resized, so a crisp edge stays crisp."""
    glyphs = make_glyph_set("l")
    settings = RenderSettings(page=PageSettings(dpi=300, paper_style=PaperStyle.BLANK),
                              variation=replace(NO_VARIATION), seed=1)
    gray = np.asarray(render_text("l", glyphs, settings).pages[0].convert("L"))
    row = gray[np.flatnonzero((gray < 128).any(axis=1))[5]]
    partial = ((row > 40) & (row < 215)).sum()
    assert partial <= 4  # at most a couple of anti-aliased pixels per edge
