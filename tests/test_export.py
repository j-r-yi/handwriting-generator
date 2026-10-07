"""PNG / PDF / ZIP export."""

from __future__ import annotations

import io
import re
import zipfile

import pytest
from PIL import Image

from handwriting.export import (
    MAX_FILE_STEM,
    pdf_bytes,
    png_bytes,
    safe_file_stem,
    zip_of_png_bytes,
    zip_of_pngs,
)


def _pages(count: int) -> list[Image.Image]:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (90, 90, 90)]
    return [Image.new("RGB", (170, 220), colors[i % len(colors)]) for i in range(count)]


def test_png_keeps_size_and_dpi() -> None:
    data = png_bytes(Image.new("RGBA", (120, 80), (0, 0, 0, 0)), dpi=300)
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (120, 80) and image.mode == "RGB"
        assert round(image.info["dpi"][0]) == 300


def test_pdf_contains_all_pages_in_order() -> None:
    pages = _pages(3)
    data = pdf_bytes(pages, dpi=100)
    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page\b", data)) == 3
    # 170 px at 100 DPI = 1.7 in = 122.4 pt
    assert b"122.4" in data


def test_zip_has_numbered_pngs() -> None:
    archive = zipfile.ZipFile(io.BytesIO(zip_of_pngs(_pages(2), dpi=150, stem="page")))
    assert archive.namelist() == ["page_01.png", "page_02.png"]
    with Image.open(io.BytesIO(archive.read("page_02.png"))) as second:
        assert second.getpixel((0, 0)) == (0, 255, 0)


@pytest.mark.parametrize(("typed", "stem"), [
    ("Republic IV - Reason, Appetite, and Justice", "Republic IV - Reason, Appetite, and Justice"),
    ("notes.PDF", "notes"),
    ("week 3/4: ethics?", "week 34 ethics"),
    ('a\\b*c"d<e>f|g', "abcdefg"),
    ("  two\nlines\t ", "two lines"),
    ("Éthique — notes", "Éthique — notes"),
    ("...", "handwriting"),
    ("", "handwriting"),
])
def test_typed_file_names_are_made_safe(typed: str, stem: str) -> None:
    assert safe_file_stem(typed) == stem


def test_long_file_names_are_shortened() -> None:
    assert len(safe_file_stem("x" * 500)) == MAX_FILE_STEM


def test_zip_of_encoded_pngs_uses_the_given_name() -> None:
    pngs = [png_bytes(page, 100) for page in _pages(2)]
    with zipfile.ZipFile(io.BytesIO(zip_of_png_bytes(pngs, "Ethics_page"))) as archive:
        assert archive.namelist() == ["Ethics_page_01.png", "Ethics_page_02.png"]
        assert archive.read("Ethics_page_02.png") == pngs[1]
