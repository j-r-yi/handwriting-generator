"""PNG / PDF / ZIP export."""

from __future__ import annotations

import io
import re
import zipfile

from PIL import Image

from handwriting.export import pdf_bytes, png_bytes, zip_of_pngs


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
