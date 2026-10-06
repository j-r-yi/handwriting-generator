"""Importing completed sample sheets from (multi-page) PDF scans."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from handwriting.errors import ImageDecodeError
from handwriting.settings import PageFormat
from handwriting.sheet_import import import_sheet
from handwriting.template import SheetIdentity
from handwriting.utils import decode_pages, decode_pdf_pages, is_pdf
from tests.helpers import draw_char_image, filled_sheet, png_bytes_of


def scanned_pdf(pages: list[np.ndarray], dpi: int = 300) -> bytes:
    """Bundle page images into a PDF, like a scanner's 'save as PDF'."""
    images = [Image.fromarray(page).convert("RGB") for page in pages]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:], resolution=float(dpi))
    return buffer.getvalue()


def test_pdf_pages_are_rendered_in_order_at_300_dpi() -> None:
    red = np.full((220, 170, 3), (255, 0, 0), dtype=np.uint8)
    blue = np.full((220, 170, 3), (0, 0, 255), dtype=np.uint8)
    data = scanned_pdf([red, blue], dpi=20)  # 170x220 px at 20 DPI = 8.5x11 in
    assert is_pdf(data)
    pages = decode_pdf_pages(data)
    assert len(pages) == 2
    height, width, channels = pages[0].shape
    # US Letter at 300 DPI is 2550x3300; PDFium may round up by one pixel.
    assert abs(width - 2550) <= 1 and abs(height - 3300) <= 1 and channels == 3
    assert tuple(pages[0][1650, 1275]) == pytest.approx((255, 0, 0), abs=8)
    assert tuple(pages[1][1650, 1275]) == pytest.approx((0, 0, 255), abs=8)


def test_decode_pages_handles_images_and_pdfs() -> None:
    image = png_bytes_of(draw_char_image("a"))
    assert not is_pdf(image)
    assert len(decode_pages(image)) == 1
    assert len(decode_pages(scanned_pdf([draw_char_image("a")] * 3, dpi=72))) == 3


@pytest.mark.parametrize("data", [b"%PDF-1.7\nthis is not really a pdf", b"%PDF-"])
def test_corrupted_pdf_raises_friendly_error(data: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        decode_pages(data)


def test_too_many_pdf_pages_are_rejected() -> None:
    data = scanned_pdf([np.zeros((10, 10, 3), np.uint8)] * 3, dpi=72)
    with pytest.raises(ImageDecodeError, match="at most 2"):
        decode_pdf_pages(data, max_pages=2)


def test_multi_page_scanned_pdf_imports_every_page() -> None:
    sheets = [filled_sheet(PageFormat.LETTER, index)[1] for index in (0, 2)]
    pages = decode_pages(scanned_pdf(sheets))
    results = [import_sheet(page) for page in pages]
    assert [r.identity for r in results] == [SheetIdentity(PageFormat.LETTER, 0, 3),
                                             SheetIdentity(PageFormat.LETTER, 2, 3)]
    assert all(r.method == "markers" for r in results)
    assert {s.char for s in results[0].samples} >= set("abcxyz")
    assert {s.char for s in results[1].samples} >= set("()[]{}")
