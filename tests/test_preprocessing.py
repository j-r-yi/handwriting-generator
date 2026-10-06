"""Glyph extraction: transparency, cropping, noise/line removal, decoding errors."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from handwriting.errors import ImageDecodeError, NoInkFoundError
from handwriting.preprocessing import extract_glyph, to_grayscale
from handwriting.utils import decode_image, recolor
from tests.helpers import PAPER, draw_char_image, png_bytes_of


def _ink_bbox(array: np.ndarray, threshold: int = 128) -> tuple[int, int, int, int]:
    gray = to_grayscale(array)
    ys, xs = np.nonzero(gray < threshold)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_extracts_transparent_tightly_cropped_glyph() -> None:
    source = draw_char_image("k")
    glyph = extract_glyph(source)
    rgba = glyph.rgba
    assert rgba.shape[2] == 4
    alpha = rgba[:, :, 3]
    # Background is fully transparent, ink is opaque.
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0
    assert alpha.max() == 255
    # Crop hugs the ink (only the small configured padding remains).
    x0, y0, x1, y1 = _ink_bbox(source)
    assert abs(glyph.width - (x1 - x0 + 1)) <= 8
    assert abs(glyph.height - (y1 - y0 + 1)) <= 8
    assert glyph.width < source.shape[1] / 2


def test_soft_alpha_preserves_antialiasing() -> None:
    glyph = extract_glyph(draw_char_image("o", size=140))
    alpha = glyph.rgba[:, :, 3]
    partial = np.count_nonzero((alpha > 0) & (alpha < 255))
    assert partial > 20  # stroke edges keep intermediate values


def test_uneven_lighting_and_shadow_are_ignored() -> None:
    image = draw_char_image("e", size=120).astype(np.float32)
    h, w = image.shape[:2]
    gradient = np.linspace(0.45, 1.0, w, dtype=np.float32)[None, :, None]
    shaded = np.clip(image * gradient, 0, 255).astype(np.uint8)
    glyph = extract_glyph(shaded)
    # The dark (shadowed) left side must not be mistaken for ink.
    assert glyph.bbox[0] > w * 0.15
    assert glyph.width < w * 0.7


def test_small_specks_are_removed() -> None:
    image = draw_char_image("x", size=120)
    rng = np.random.default_rng(1)
    for _ in range(25):
        y, x = rng.integers(0, image.shape[0]), rng.integers(0, 15)
        image[y, x] = (20, 20, 20)  # isolated dark pixels near the left edge
    glyph = extract_glyph(image)
    clean = extract_glyph(draw_char_image("x", size=120))
    assert abs(glyph.width - clean.width) <= 2


def test_ruled_lines_touching_the_edges_are_removed() -> None:
    image = draw_char_image("n", size=110)
    h, w = image.shape[:2]
    cv2.line(image, (0, int(h * 0.85)), (w - 1, int(h * 0.85)), (90, 120, 200), 2)  # notebook rule
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), (40, 40, 40), 3)  # box border
    glyph = extract_glyph(image)
    clean = extract_glyph(draw_char_image("n", size=110))
    assert abs(glyph.width - clean.width) <= 4
    assert glyph.height < h * 0.6


def test_blank_image_raises_no_ink() -> None:
    blank = np.full((120, 120, 3), PAPER, dtype=np.uint8)
    with pytest.raises(NoInkFoundError):
        extract_glyph(blank)


def test_transparent_png_input_is_supported() -> None:
    rgba = np.zeros((100, 80, 4), dtype=np.uint8)
    rgba[20:80, 30:40, 3] = 255  # a vertical stroke on a transparent background
    decoded = decode_image(png_bytes_of(rgba))
    assert decoded.shape[2] == 4
    glyph = extract_glyph(decoded)
    assert 8 <= glyph.width <= 16 and 58 <= glyph.height <= 66


def test_large_images_are_downscaled() -> None:
    big = draw_char_image("W", size=900)
    glyph = extract_glyph(big)
    assert max(glyph.width, glyph.height) <= 480
    assert glyph.scale < 1.0


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40])
def test_corrupted_or_unsupported_files_raise_friendly_error(data: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(data)


def test_truncated_jpeg_raises_friendly_error() -> None:
    buffer = io.BytesIO()
    Image.fromarray(draw_char_image("a")).save(buffer, format="JPEG")
    with pytest.raises(ImageDecodeError):
        decode_image(buffer.getvalue()[:200])


def test_exif_orientation_is_applied() -> None:
    image = Image.fromarray(np.zeros((40, 100, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise when displayed
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    assert decode_image(buffer.getvalue()).shape[:2] == (100, 40)


def test_recolor_keeps_alpha() -> None:
    glyph = Image.new("RGBA", (4, 1))
    glyph.putdata([(0, 0, 0, 0), (0, 0, 0, 64), (0, 0, 0, 200), (0, 0, 0, 255)])
    colored = recolor(glyph, (20, 42, 116))
    assert list(colored.getdata()) == [(20, 42, 116, a) for a in (0, 64, 200, 255)]
