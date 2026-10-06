"""Shared test helpers.

All handwriting images are generated programmatically (text drawn with a
regular font, then run through the real extraction pipeline), so the tests do
not depend on anyone's personal handwriting files.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from handwriting.baseline import descent_px, estimate_x_height_ref
from handwriting.models import GlyphSet, LoadedGlyph
from handwriting.preprocessing import extract_glyph
from handwriting.sample_store import NewSample
from handwriting.settings import PageFormat
from handwriting.template import SheetPageLayout, build_sheet_layouts, render_sheet_page
from handwriting.utils import get_font

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAPER = (238, 234, 226)
INK = (30, 32, 70)


def draw_char_image(char: str, size: int = 110, offset: int = 0, paper=PAPER, ink=INK) -> np.ndarray:
    """A fake 'photo' of one handwritten character: text on slightly tinted paper."""
    image = Image.new("RGB", (int(size * 1.6), int(size * 1.8)), paper)
    ImageDraw.Draw(image).text((image.width // 2 + offset, int(image.height * 0.68)), char,
                               font=get_font(size), fill=ink, anchor="ms")
    return np.array(image)


def png_bytes_of(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def make_new_samples(chars: Iterable[str], variants: int = 2) -> list[NewSample]:
    samples = []
    for char in chars:
        for v in range(variants):
            glyph = extract_glyph(draw_char_image(char, offset=3 * v))
            samples.append(NewSample(char=char, rgba=glyph.rgba,
                                     x_height_ref=estimate_x_height_ref(char, glyph.width, glyph.height)))
    return samples


def make_glyph_set(chars: Iterable[str], variants: int = 3, width: int = 30, height: int = 40) -> GlyphSet:
    """Synthetic glyph set: solid rectangles whose width encodes the variant index."""
    glyphs: dict[str, list[LoadedGlyph]] = {}
    for char in chars:
        for v in range(variants):
            w = width + 2 * v
            image = Image.new("RGBA", (w, height), (0, 0, 0, 255))
            glyphs.setdefault(char, []).append(LoadedGlyph(
                char=char, sample_id=f"{char}_{v}", image=image, x_height_ref=height,
                descent=descent_px(char, height, height),
            ))
    return GlyphSet(glyphs, name="synthetic")


def filled_sheet(page_format: PageFormat, page_index: int, variants: int = 3,
                 skip: Iterable[tuple[str, int]] = ()) -> tuple[SheetPageLayout, np.ndarray]:
    """A printed sample-sheet page with every cell 'written' using a font."""
    layout = build_sheet_layouts(page_format, variants)[page_index]
    image = render_sheet_page(layout)
    draw = ImageDraw.Draw(image)
    font = get_font(95)
    skipped = set(skip)
    for cell in layout.cells:
        if (cell.char, cell.variant) in skipped:
            continue
        x = cell.rect.x + cell.rect.w // 2 + (cell.variant - 1) * 5
        draw.text((x, cell.baseline_y), cell.char, font=font, fill=INK, anchor="ms")
    return layout, np.array(image)


def simulate_photo(page: np.ndarray, angle: float = 5.0, perspective: float = 0.05,
                   rotate_180: bool = False, scale: float = 0.7) -> np.ndarray:
    """Place a page on a dark table with rotation, perspective, uneven light and noise."""
    height, width = page.shape[:2]
    canvas_w, canvas_h = int(width * 1.3), int(height * 1.3)
    theta = np.deg2rad(angle)
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    stretch = [(-perspective, 0), (perspective, perspective), (0, 0), (0, -perspective)]
    target = []
    for (x, y), (sx, sy) in zip(source, stretch):
        dx, dy = (x - width / 2) * (1 + sx), (y - height / 2) * (1 + sy)
        target.append([canvas_w / 2 + dx * np.cos(theta) - dy * np.sin(theta),
                       canvas_h / 2 + dx * np.sin(theta) + dy * np.cos(theta)])
    matrix = cv2.getPerspectiveTransform(source, np.float32(target))
    table = (60, 50, 40)
    photo = cv2.warpPerspective(page, matrix, (canvas_w, canvas_h), borderValue=table).astype(np.float32)
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    light = 0.65 + 0.35 * (xx / canvas_w) * (1 - 0.3 * yy / canvas_h)
    photo = photo * light[..., None] + np.random.default_rng(0).normal(0, 5, photo.shape)
    photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (5, 5), 0)
    photo = cv2.resize(photo, (int(canvas_w * scale), int(canvas_h * scale)), interpolation=cv2.INTER_AREA)
    if rotate_180:
        photo = cv2.rotate(photo, cv2.ROTATE_180)
    return photo

