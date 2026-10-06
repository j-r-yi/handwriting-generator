"""Turning photos/scans of handwriting into clean, transparent glyph images.

Pipeline (see :func:`extract_glyph`):

1. grayscale conversion (transparent uploads are composited on white)
2. illumination normalisation: divide by an estimate of the paper brightness so
   shadows and uneven lighting do not look like ink
3. thresholding (Otsu, clamped to sane limits) to find ink pixels
4. removal of ruled/border lines that touch the crop edge, and of tiny specks
5. bounding box of the remaining ink
6. a *soft* alpha channel computed from the normalised darkness, so stroke
   edges keep their natural anti-aliasing instead of becoming jagged
7. crop to the ink (plus a small margin) and return an RGBA array

Strokes are never smoothed, thinned or vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .errors import NoInkFoundError
from .utils import odd

# Background estimation runs on a downscaled copy for speed; the background is
# low-frequency, so nothing important is lost.
_BACKGROUND_WORK_SIZE = 480


@dataclass(frozen=True)
class ExtractionOptions:
    """Tunable parameters for glyph extraction."""

    # Minimum darkness (0 = paper, 1 = black) that can count as ink.
    min_threshold: float = 0.16
    # Upper clamp for the Otsu threshold, so faint strokes are kept.
    max_threshold: float = 0.45
    # Speck removal: components smaller than the largest of these are dropped.
    speck_min_area: int = 4
    speck_fraction_of_largest: float = 0.012
    speck_fraction_of_image: float = 0.00004
    # Remove straight lines (ruled paper, box borders) touching the crop edge.
    remove_edge_lines: bool = True
    edge_line_min_length: float = 0.35  # fraction of the crop width/height
    # Pixels of transparent margin kept around the ink.
    padding: int = 2
    # Fewer ink pixels than this means "nothing written here".
    min_ink_pixels: int = 12
    # Size of the background-estimation kernel relative to the image (when no
    # explicit kernel size is given). Must be larger than the stroke width.
    background_kernel_fraction: float = 0.3
    # Large glyphs are downscaled so stored samples stay reasonably small.
    max_glyph_dimension: int = 480


@dataclass(frozen=True)
class ExtractedGlyph:
    """Result of extracting one handwritten character.

    Attributes:
        rgba: (H, W, 4) uint8 array; RGB is black, alpha holds ink coverage.
        bbox: (x, y, w, h) of the cropped region in the *input* image coordinates.
        scale: factor applied to the crop to produce ``rgba`` (1.0 = unchanged).
        ink_pixels: number of pixels classified as ink.
    """

    rgba: np.ndarray
    bbox: tuple[int, int, int, int]
    scale: float
    ink_pixels: int

    @property
    def width(self) -> int:
        return int(self.rgba.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgba.shape[0])


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a gray / RGB / RGBA uint8 array to grayscale.

    Transparent pixels are treated as white paper, so already-cut-out PNGs work.
    Input channel order is RGB(A) (as produced by :func:`utils.decode_image`).
    """
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape {image.shape}.")
    rgb = image[:, :, :3].astype(np.float32)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
    gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray


def estimate_background(gray: np.ndarray, kernel_px: int) -> np.ndarray:
    """Estimate local paper brightness by morphologically closing away dark ink.

    *kernel_px* must exceed the stroke width (in full-resolution pixels).
    """
    height, width = gray.shape
    factor = min(1.0, _BACKGROUND_WORK_SIZE / max(height, width))
    small = gray
    if factor < 1.0:
        small = cv2.resize(gray, (max(1, round(width * factor)), max(1, round(height * factor))),
                           interpolation=cv2.INTER_AREA)
    k = odd(kernel_px * factor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    closed = cv2.GaussianBlur(closed, (odd(k / 2), odd(k / 2)), 0)
    background = cv2.resize(closed, (width, height), interpolation=cv2.INTER_LINEAR)
    background = background.astype(np.float32)
    # Never let the background be darker than the pixel itself, and never far
    # below the overall paper level (protects large solid blobs of ink).
    paper_level = float(np.percentile(gray, 95))
    background = np.maximum(background, gray.astype(np.float32))
    return np.maximum(background, np.float32(0.5 * paper_level))


def darkness_map(gray: np.ndarray, kernel_px: int | None = None,
                 kernel_fraction: float = 0.3) -> np.ndarray:
    """Illumination-normalised darkness in [0, 1] (0 = paper, 1 = black ink)."""
    if kernel_px is None:
        kernel_px = max(15, round(min(gray.shape) * kernel_fraction))
    background = estimate_background(gray, kernel_px)
    ratio = gray.astype(np.float32) / np.maximum(background, 1.0)
    return np.clip(1.0 - ratio, 0.0, 1.0)


def ink_threshold(darkness: np.ndarray, options: ExtractionOptions) -> float:
    """Otsu threshold on the darkness map, clamped to sensible limits."""
    as_uint8 = np.round(darkness * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(as_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.clip(otsu / 255.0, options.min_threshold, options.max_threshold))


def remove_edge_lines(mask: np.ndarray, options: ExtractionOptions) -> np.ndarray:
    """Remove straight lines that touch the crop border (box borders, rules).

    Two complementary checks:
    * long horizontal/vertical runs found with a morphological opening, kept
      only if the line component touches the border band;
    * short, thin border fragments lying parallel to the edge they touch.
    Handwriting in the middle of the crop is never affected.
    """
    height, width = mask.shape
    band = max(2, round(0.05 * min(height, width)))
    remove = np.zeros_like(mask)

    horiz_len = max(3, round(width * options.edge_line_min_length))
    vert_len = max(3, round(height * options.edge_line_min_length))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len)))
    lines = cv2.bitwise_or(horiz, vert)
    if lines.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        for i in range(1, count):
            x, y, w, h, _ = stats[i]
            if x <= band or y <= band or x + w >= width - band or y + h >= height - band:
                remove[labels == i] = 255

    thickness_limit = max(3, round(0.05 * min(height, width)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        thin_horizontal = h <= thickness_limit and w >= 4 * h
        thin_vertical = w <= thickness_limit and h >= 4 * w
        touches_top_bottom = y <= band or y + h >= height - band
        touches_left_right = x <= band or x + w >= width - band
        if (thin_horizontal and touches_top_bottom) or (thin_vertical and touches_left_right):
            remove[labels == i] = 255

    if not remove.any():
        return mask
    remove = cv2.dilate(remove, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(mask, cv2.bitwise_not(remove))


def remove_specks(mask: np.ndarray, options: ExtractionOptions) -> np.ndarray:
    """Drop connected components too small to be part of a character."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(
        options.speck_min_area,
        options.speck_fraction_of_largest * float(areas.max()),
        options.speck_fraction_of_image * mask.size,
    )
    keep_labels = np.flatnonzero(areas >= min_area) + 1
    return np.where(np.isin(labels, keep_labels), 255, 0).astype(np.uint8)


def extract_glyph_from_darkness(darkness: np.ndarray,
                                options: ExtractionOptions | None = None) -> ExtractedGlyph:
    """Extract a glyph from an already illumination-normalised darkness map.

    Raises:
        NoInkFoundError: if no handwriting is visible.
    """
    options = options or ExtractionOptions()
    if darkness.size == 0:
        raise NoInkFoundError("The image is empty.")

    threshold = ink_threshold(darkness, options)
    mask = np.where(darkness >= threshold, 255, 0).astype(np.uint8)
    if options.remove_edge_lines:
        mask = remove_edge_lines(mask, options)
    mask = remove_specks(mask, options)

    ink_pixels = int(np.count_nonzero(mask))
    if ink_pixels < options.min_ink_pixels:
        raise NoInkFoundError("No visible handwriting was found in the image.")

    # Soft alpha preserves anti-aliasing and pen-pressure variation.
    low = threshold * 0.5
    high = max(float(np.percentile(darkness[mask > 0], 70)), threshold + 0.05)
    alpha = np.clip((darkness - low) / (high - low), 0.0, 1.0)
    support = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = np.where(support > 0, alpha, 0.0)

    ys, xs = np.nonzero(mask)
    pad = options.padding
    height, width = mask.shape
    x0, x1 = max(0, xs.min() - pad), min(width, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(height, ys.max() + 1 + pad)

    crop_alpha = np.round(alpha[y0:y1, x0:x1] * 255).astype(np.uint8)
    rgba = np.zeros((*crop_alpha.shape, 4), dtype=np.uint8)
    rgba[:, :, 3] = crop_alpha

    rgba, scale = _limit_size(rgba, options.max_glyph_dimension)
    return ExtractedGlyph(rgba=rgba, bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                          scale=scale, ink_pixels=ink_pixels)


def extract_glyph(image: np.ndarray, options: ExtractionOptions | None = None) -> ExtractedGlyph:
    """Full extraction pipeline for a photo/scan/crop containing one character.

    Raises:
        NoInkFoundError: if no handwriting is visible.
    """
    options = options or ExtractionOptions()
    gray = to_grayscale(image)
    if min(gray.shape) < 3:
        raise NoInkFoundError("The image is too small to contain handwriting.")
    darkness = darkness_map(gray, kernel_fraction=options.background_kernel_fraction)
    return extract_glyph_from_darkness(darkness, options)


def _limit_size(rgba: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    height, width = rgba.shape[:2]
    largest = max(height, width)
    if largest <= max_dim:
        return rgba, 1.0
    scale = max_dim / largest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # RGB is constant (black), so resizing the RGBA array directly is safe.
    return cv2.resize(rgba, size, interpolation=cv2.INTER_AREA), scale
