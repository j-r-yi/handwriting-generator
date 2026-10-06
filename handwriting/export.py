"""Exporting rendered pages as PNG, multi-page PDF, or a ZIP of PNGs.

Uses only Pillow and the standard library.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence

from PIL import Image


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def png_bytes(image: Image.Image, dpi: int) -> bytes:
    """Encode one page as PNG with physical resolution metadata."""
    buffer = io.BytesIO()
    _as_rgb(image).save(buffer, format="PNG", dpi=(dpi, dpi))
    return buffer.getvalue()


# Pillow stores PDF pages as JPEG. Its defaults (quality 75, half-resolution
# colour) leave visible blocks around pen strokes, so use near-lossless settings.
PDF_JPEG_QUALITY = 95


def pdf_bytes(pages: Sequence[Image.Image], dpi: int) -> bytes:
    """Combine pages, in order, into one PDF that prints at the correct size."""
    if not pages:
        raise ValueError("There are no pages to export.")
    rgb_pages = [_as_rgb(page) for page in pages]
    buffer = io.BytesIO()
    rgb_pages[0].save(
        buffer,
        format="PDF",
        save_all=True,
        append_images=rgb_pages[1:],
        resolution=float(dpi),
        quality=PDF_JPEG_QUALITY,
        subsampling=0,  # keep full colour resolution (4:4:4)
    )
    return buffer.getvalue()


def zip_of_pngs(pages: Sequence[Image.Image], dpi: int, stem: str = "page") -> bytes:
    """Bundle every page as ``<stem>_01.png``, ``<stem>_02.png``, ... in a ZIP."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for number, page in enumerate(pages, start=1):
            archive.writestr(f"{stem}_{number:02d}.png", png_bytes(page, dpi))
    return buffer.getvalue()
