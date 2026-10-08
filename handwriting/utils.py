"""Small shared helpers: image decoding, conversions, colours, fonts."""

from __future__ import annotations

import io
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont, ImageOps, UnidentifiedImageError

from .errors import ImageDecodeError

# Refuse absurdly large images (protects memory; ~60 megapixels is plenty).
MAX_IMAGE_PIXELS = 60_000_000

_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "Arial.ttf",
    "arial.ttf",
    "Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into an RGB or RGBA ``uint8`` array.

    EXIF orientation from phone cameras is applied so photos are upright.

    Raises:
        ImageDecodeError: if the data is empty, corrupted or not an image.
    """
    if not data:
        raise ImageDecodeError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ImageDecodeError(
                    f"Image is too large ({img.width}x{img.height}). Please use a smaller image."
                )
            img.load()
            img = ImageOps.exif_transpose(img)
            mode = "RGBA" if _has_transparency(img) else "RGB"
            return np.array(img.convert(mode))
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageDecodeError(
            "Unsupported or unrecognised image format. Please upload PNG, JPEG, BMP, TIFF or WebP."
        ) from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ImageDecodeError("The image file appears to be corrupted or truncated.") from exc


# PDF pages are rasterised at the resolution of the scan inside them (between
# these limits), so a 600 DPI scan keeps its detail and a vector PDF uses 300.
PDF_RENDER_DPI = 300
MAX_PDF_RENDER_DPI = 600
MAX_PDF_PAGES = 30
_PDF_POINTS_PER_INCH = 72


def is_pdf(data: bytes) -> bool:
    """True if *data* looks like a PDF file (checks the ``%PDF`` signature)."""
    return data[:1024].lstrip().startswith(b"%PDF")


def decode_pdf_pages(data: bytes, dpi: int | None = None,
                     max_pages: int = MAX_PDF_PAGES) -> list[np.ndarray]:
    """Render every page of a PDF to an RGB ``uint8`` array, locally.

    With *dpi* ``None`` each page is rendered at the resolution of the largest
    scanned image on it (clamped to ``PDF_RENDER_DPI``..``MAX_PDF_RENDER_DPI``).
    Large pages are rendered at a lower resolution so no page exceeds
    :data:`MAX_IMAGE_PIXELS`.

    Raises:
        ImageDecodeError: if the PDF is corrupted, encrypted, empty or too long.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - only without the dependency
        raise ImageDecodeError(
            "PDF support needs the 'pypdfium2' package. Run: pip install -r requirements.txt"
        ) from exc

    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise ImageDecodeError(
            "The PDF could not be opened. It may be corrupted or password-protected."
        ) from exc

    try:
        page_count = len(document)
        if page_count == 0:
            raise ImageDecodeError("The PDF has no pages.")
        if page_count > max_pages:
            raise ImageDecodeError(f"The PDF has {page_count} pages; at most {max_pages} are supported.")
        pages: list[np.ndarray] = []
        for index in range(page_count):
            page = document[index]
            try:
                width_pt, height_pt = page.get_size()
                page_dpi = dpi if dpi is not None else _scan_dpi(page)
                scale = page_dpi / _PDF_POINTS_PER_INCH
                pixels = width_pt * height_pt * scale * scale
                if pixels > MAX_IMAGE_PIXELS:
                    scale *= (MAX_IMAGE_PIXELS / pixels) ** 0.5
                bitmap = page.render(scale=scale)
                pages.append(np.array(bitmap.to_pil().convert("RGB")))
            finally:
                page.close()
        return pages
    except pdfium.PdfiumError as exc:
        raise ImageDecodeError(f"A page of the PDF could not be rendered: {exc}") from exc
    finally:
        document.close()


def _scan_dpi(page) -> float:
    """Resolution of the largest embedded image on a PDF page, clamped to the supported range."""
    import pypdfium2.raw as pdfium_raw

    best_pixels, best_dpi = 0, float(PDF_RENDER_DPI)
    try:
        for obj in page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE], max_depth=2):
            meta = obj.get_metadata()
            if meta.width * meta.height > best_pixels and meta.horizontal_dpi > 0:
                best_pixels = meta.width * meta.height
                best_dpi = min(meta.horizontal_dpi, meta.vertical_dpi or meta.horizontal_dpi)
    except Exception:  # metadata is only a hint; fall back to the default resolution
        return float(PDF_RENDER_DPI)
    return float(min(MAX_PDF_RENDER_DPI, max(PDF_RENDER_DPI, best_dpi)))


def decode_pages(data: bytes) -> list[np.ndarray]:
    """Decode an upload into one image per page: PDFs give every page, images give one."""
    if is_pdf(data):
        return decode_pdf_pages(data)
    return [decode_image(data)]


def _has_transparency(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def rgba_array_to_image(rgba: np.ndarray) -> Image.Image:
    """Wrap an (H, W, 4) uint8 array as a PIL RGBA image."""
    array = np.ascontiguousarray(rgba, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"Expected an (H, W, 4) array, got shape {array.shape}.")
    return Image.fromarray(array)


def recolor(glyph: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    """Return *glyph* drawn in *color*, keeping its alpha (anti-aliasing) intact.

    Works for any RGB colour: the glyph's ink coverage lives entirely in the
    alpha channel, so recolouring just replaces the RGB planes.
    """
    alpha = glyph.getchannel("A") if glyph.mode == "RGBA" else glyph.convert("L")
    colored = Image.new("RGBA", glyph.size, (*color, 0))
    colored.putalpha(alpha)
    return colored


_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into an RGB tuple."""
    match = _HEX_COLOR.match(value.strip())
    if not match:
        raise ValueError(f"Not a #RRGGBB colour: {value!r}")
    digits = match.group(1)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def to_hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@lru_cache(maxsize=32)
def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a readable sans-serif font, falling back to Pillow's bundled font."""
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


# Folder names Windows reserves for devices (in any letter case, with or without an extension).
_WINDOWS_RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)),
                               *(f"lpt{i}" for i in range(10))})


def slugify(name: str, max_length: int = 48) -> str:
    """Make a filesystem-safe, lowercase identifier from a display name (also safe on Windows)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    slug = slug[:max_length].strip("-") or "profile"
    return f"{slug}-profile" if slug in _WINDOWS_RESERVED else slug


def odd(value: float, minimum: int = 3) -> int:
    """Round to the nearest odd integer >= *minimum* (OpenCV kernel sizes)."""
    n = max(minimum, int(round(value)))
    return n if n % 2 == 1 else n + 1


def ensure_writable_dir(path: Path) -> None:
    """Create *path* if needed and verify that files can be written into it."""
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    probe.write_bytes(b"ok")
    probe.unlink()
