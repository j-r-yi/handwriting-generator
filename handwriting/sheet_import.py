"""Reading completed sample sheets from scans or phone photos.

Steps:

1. find the four black corner markers (fallback: find the paper's outline);
2. read the page code to learn the paper format, page number and number of
   samples per character, and to determine which way up the page is;
3. warp the photo into exact template coordinates (corrects perspective and
   rotation), at the photo's own resolution up to 600 DPI so no detail of the
   handwriting is thrown away;
4. normalise illumination once for the whole page;
5. for every cell: snap to the actual printed box borders (absorbs small
   alignment errors and paper curl), crop inside the border, and extract the
   handwriting with :mod:`handwriting.preprocessing`.

No OCR is involved: the template geometry says which character each cell holds.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Sequence
import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .baseline import X_HEIGHT_CHARS, estimate_x_height_ref
from .charset import char_to_id
from .errors import NoInkFoundError, SheetDetectionError
from .preprocessing import ExtractionOptions, darkness_map, extract_glyph_from_darkness, to_grayscale
from .settings import PageFormat
from .template import (
    TEMPLATE_DPI,
    CellSpec,
    Rect,
    SheetIdentity,
    SheetPageLayout,
    build_sheet_layouts,
    code_bit_rects,
    decode_identity,
    marker_rects,
)

_DETECT_MAX_DIM = 1800
_MARKER_DARKNESS = 0.35
_BIT_DARKNESS = 0.4
_MAX_MARKER_CANDIDATES = 14
_PREVIEW_WIDTH = 1000
# Page normalisation kernel: wider than any pen stroke, narrower than a shadow.
_PAGE_BACKGROUND_KERNEL = round(0.25 * TEMPLATE_DPI)
_BORDER_DARKNESS = 0.3
_CELL_INSET_FRACTION = 0.09
_CELL_SEARCH_FRACTION = 0.14
_MIN_XHEIGHT_SAMPLES = 3
# Highest resolution handwriting is captured at; sharper scans are reduced to this.
MAX_CAPTURE_DPI = 600


@dataclass(frozen=True)
class ExtractedSample:
    """A glyph cut out of a sample sheet, awaiting user review."""

    key: str
    char: str
    variant: int
    page_index: int
    rgba: np.ndarray
    ink_width: float
    ink_height: float
    baseline_from_top: float | None
    x_height_ref: float = 0.0
    capture_dpi: float = TEMPLATE_DPI

    @property
    def scale(self) -> float:
        """Capture pixels per 300 DPI template pixel."""
        return self.capture_dpi / TEMPLATE_DPI


@dataclass
class SheetImportResult:
    identity: SheetIdentity
    identity_detected: bool
    method: str
    samples: list[ExtractedSample]
    empty_cells: list[tuple[str, int]]
    preview: Image.Image
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Alignment:
    warped: np.ndarray  # grayscale page in template coordinates times ``scale``
    identity: SheetIdentity
    identity_detected: bool
    method: str
    scale: float = 1.0


# --------------------------------------------------------------------------- API

def import_sheet(image: np.ndarray, identity_override: SheetIdentity | None = None,
                 options: ExtractionOptions | None = None) -> SheetImportResult:
    """Detect, align and extract every handwritten cell of one sheet page.

    Args:
        image: RGB/RGBA/gray array of the photo or scan.
        identity_override: use this format/page/variant count instead of the
            printed page code (for when the code cannot be read).

    Raises:
        SheetDetectionError: if the sheet cannot be located reliably.
    """
    options = options or ExtractionOptions()
    gray = to_grayscale(image)
    if min(gray.shape) < 200:
        raise SheetDetectionError("The image is too small. Please use a higher-resolution scan or photo.")

    alignment = _align_sheet(gray, identity_override)
    layouts = build_sheet_layouts(alignment.identity.page_format, alignment.identity.variants)
    if alignment.identity.page_index >= len(layouts):
        raise SheetDetectionError(
            f"Page {alignment.identity.page_index + 1} does not exist for a sheet with "
            f"{alignment.identity.variants} samples per character."
        )
    layout = layouts[alignment.identity.page_index]
    scale = alignment.scale
    darkness = darkness_map(alignment.warped, kernel_px=round(_PAGE_BACKGROUND_KERNEL * scale))
    border_mask = (darkness > _BORDER_DARKNESS).astype(np.uint8)
    cell_options = _scaled_options(options, scale)

    samples: list[ExtractedSample] = []
    empty: list[tuple[str, int]] = []
    drawn: list[tuple[Rect, bool]] = []
    snapped_cells = 0
    for cell in layout.cells:
        rect, edges_found = _snap_cell(border_mask, _scale_rect(cell.rect, scale))
        snapped_cells += edges_found >= 3
        sample = _extract_cell(darkness, cell, rect, layout, cell_options, TEMPLATE_DPI * scale)
        drawn.append((rect, sample is not None))
        if sample is None:
            empty.append((cell.char, cell.variant))
        else:
            samples.append(sample)

    warnings: list[str] = []
    if layout.cells and snapped_cells < 0.4 * len(layout.cells):
        warnings.append("The printed boxes could not be matched precisely; check the alignment preview "
                        "and exclude any samples that look cut off.")
    if not samples:
        warnings.append("No handwriting was found on this page.")
    if alignment.method == "page outline":
        warnings.append("Corner markers were not found; the page outline was used instead, which is less precise.")

    return SheetImportResult(
        identity=alignment.identity,
        identity_detected=alignment.identity_detected,
        method=alignment.method,
        samples=assign_x_height(samples),
        empty_cells=empty,
        preview=_preview(alignment.warped, drawn, scale),
        warnings=warnings,
    )


def assign_x_height(samples: Sequence[ExtractedSample],
                    known_reference: float | None = None) -> list[ExtractedSample]:
    """Give every sample the same physical x-height reference.

    All sheet pages are warped to the same physical scale, so one x-height
    describes the whole batch and preserves the writer's natural proportions
    (capitals vs. lowercase, small punctuation...). It is worked out in 300 DPI
    template pixels and stored in each sample's own pixels (pages may be
    captured at different resolutions). Priority:

    1. median ink height of x-height letters (a, c, e, m, n, o, ...) in the batch;
    2. *known_reference* (template pixels), e.g. from samples already in the profile;
    3. median of per-character heuristic estimates.
    """
    if not samples:
        return []
    x_letters = [s.ink_height / s.scale for s in samples if s.char in X_HEIGHT_CHARS]
    if len(x_letters) >= _MIN_XHEIGHT_SAMPLES:
        reference = statistics.median(x_letters)
    elif known_reference:
        reference = known_reference
    else:
        reference = statistics.median(
            estimate_x_height_ref(s.char, s.ink_width, s.ink_height) / s.scale for s in samples
        )
    reference = max(reference, 1.0)
    return [replace(s, x_height_ref=float(reference * s.scale)) for s in samples]


# --------------------------------------------------------------------- alignment

def _align_sheet(gray: np.ndarray, override: SheetIdentity | None) -> _Alignment:
    factor = min(1.0, _DETECT_MAX_DIM / max(gray.shape))
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else gray
    small_dark = darkness_map(small, kernel_px=round(max(small.shape) * 0.08))

    quad = _find_marker_quad(small_dark)
    if quad is not None:
        found = _identify(small_dark, quad, override, use_markers=True)
        if found is not None:
            identity, ordered, detected = found
            points = ordered / factor
            scale = _capture_scale(points, identity.page_format, use_markers=True)
            return _Alignment(_warp(gray, points, identity.page_format, use_markers=True, scale=scale),
                              identity, detected, "markers", scale)

    page_quad = _find_page_quad(small)
    if page_quad is not None:
        # Flatten the page first, then look for markers in the clean image.
        for page_format in _formats_to_try(override):
            for ordered in _orientations(page_quad):
                points = ordered / factor
                scale = _capture_scale(points, page_format, use_markers=False)
                rough = _warp(gray, points, page_format, use_markers=False, scale=scale)
                inner = _try_markers_only(rough, override, scale)
                if inner is not None:
                    return inner
        found = _identify(small_dark, page_quad, override, use_markers=False)
        if found is not None:
            identity, ordered, detected = found
            points = ordered / factor
            scale = _capture_scale(points, identity.page_format, use_markers=False)
            return _Alignment(_warp(gray, points, identity.page_format, use_markers=False, scale=scale),
                              identity, detected, "page outline", scale)

    if quad is not None or page_quad is not None:
        raise SheetDetectionError(
            "The sheet was found, but its page code could not be read. "
            "Choose the page number manually (\"Identify page manually\") and try again."
        )
    raise SheetDetectionError(
        "Could not find the sample sheet in this image. Make sure the whole page is visible, "
        "all four black corner squares are in the frame, the photo is in focus, and lighting is even."
    )


def _try_markers_only(rough: np.ndarray, override: SheetIdentity | None, scale: float) -> _Alignment | None:
    """Find the markers on a page already flattened at *scale* times template size."""
    factor = min(1.0, _DETECT_MAX_DIM / max(rough.shape))
    small = cv2.resize(rough, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else rough
    small_dark = darkness_map(small, kernel_px=round(max(small.shape) * 0.08))
    quad = _find_marker_quad(small_dark)
    if quad is None:
        return None
    found = _identify(small_dark, quad, override, use_markers=True)
    if found is None:
        return None
    identity, ordered, detected = found
    return _Alignment(_warp(rough, ordered / factor, identity.page_format, use_markers=True, scale=scale),
                      identity, detected, "page outline + markers", scale)


def _formats_to_try(override: SheetIdentity | None) -> list[PageFormat]:
    return [override.page_format] if override else [PageFormat.LETTER, PageFormat.A4]


def _template_points(page_format: PageFormat, use_markers: bool) -> np.ndarray:
    if use_markers:
        return np.array([m.center for m in marker_rects(page_format)], dtype=np.float32)
    width, height = page_format.size_px(TEMPLATE_DPI)
    return np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)


def _identify(darkness: np.ndarray, quad: np.ndarray, override: SheetIdentity | None,
              use_markers: bool) -> tuple[SheetIdentity, np.ndarray, bool] | None:
    """Find the orientation whose page code decodes; returns identity, ordered quad, detected flag."""
    for page_format in _formats_to_try(override):
        source = _template_points(page_format, use_markers)
        for ordered in _orientations(quad):
            transform = cv2.getPerspectiveTransform(source, ordered.astype(np.float32))
            decoded = decode_identity(_read_code_bits(darkness, transform))
            if decoded is None:
                continue
            if override is not None:
                return override, ordered, False
            if decoded.page_format != page_format:
                # Read with the wrong format's geometry; redo with the right one.
                source_ok = _template_points(decoded.page_format, use_markers)
                retry = cv2.getPerspectiveTransform(source_ok, ordered.astype(np.float32))
                if decode_identity(_read_code_bits(darkness, retry)) != decoded:
                    continue
            return decoded, ordered, True
    if override is not None:
        upright = _orientations(quad)[0]
        return override, upright, False
    return None


def _read_code_bits(darkness: np.ndarray, transform: np.ndarray) -> list[int]:
    rects = code_bit_rects()
    centers = np.array([[r.center] for r in rects], dtype=np.float32)
    mapped = cv2.perspectiveTransform(centers, transform).reshape(-1, 2)
    # Sampling radius: a third of a bit, measured in the image.
    corner = cv2.perspectiveTransform(np.array([[[rects[0].x, rects[0].y]]], dtype=np.float32), transform)
    radius = max(1, int(np.linalg.norm(corner.reshape(2) - mapped[0]) / 3))
    height, width = darkness.shape
    bits: list[int] = []
    for x, y in mapped:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < width and 0 <= yi < height):
            return []
        patch = darkness[max(0, yi - radius):yi + radius + 1, max(0, xi - radius):xi + radius + 1]
        bits.append(int(float(patch.mean()) > _BIT_DARKNESS))
    return bits


def _orientations(quad: np.ndarray) -> list[np.ndarray]:
    """The four cyclic orderings of a clockwise quad, upright guess first."""
    start = int(np.argmin(quad.sum(axis=1)))
    upright = np.roll(quad, -start, axis=0)
    return [np.roll(upright, -k, axis=0) for k in range(4)]


def _order_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]  # image y points down, so this is clockwise


def _find_marker_quad(darkness: np.ndarray) -> np.ndarray | None:
    """Locate the four corner markers; returns their centres (clockwise) or None."""
    mask = (darkness > _MARKER_DARKNESS).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # RETR_LIST: markers may be nested inside a dark surrounding (e.g. a table top).
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(mask.shape[0] * mask.shape[1])

    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 0.00003 * image_area <= area <= 0.02 * image_area:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(contour)
        if min(w, h) <= 0 or min(w, h) / max(w, h) < 0.7 or area / (w * h) < 0.8:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < 0.9:
            continue
        moments = cv2.moments(contour)
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        candidates.append((area, center))

    if len(candidates) < 4:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:_MAX_MARKER_CANDIDATES]

    expected_aspects: list[float] = []
    for page_format in PageFormat:
        pts = _template_points(page_format, use_markers=True)
        aspect = (pts[1][0] - pts[0][0]) / (pts[3][1] - pts[0][1])
        expected_aspects += [aspect, 1 / aspect]
    # Expected (marker quad area) / (single marker area); similar for both formats.
    letter = _template_points(PageFormat.LETTER, use_markers=True)
    marker_side = marker_rects(PageFormat.LETTER)[0].w
    ratio_to_marker = (letter[1][0] - letter[0][0]) * (letter[3][1] - letter[0][1]) / marker_side ** 2

    best: tuple[float, np.ndarray] | None = None
    for combo in itertools.combinations(candidates, 4):
        areas = [c[0] for c in combo]
        if max(areas) > 2.5 * min(areas):
            continue
        quad = _order_clockwise(np.array([c[1] for c in combo]))
        if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
            continue
        quad_area = cv2.contourArea(quad.astype(np.float32))
        mean_marker = float(np.mean(areas))
        if not 0.25 * ratio_to_marker <= quad_area / mean_marker <= 4 * ratio_to_marker:
            continue
        widths = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
        heights = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
        aspect = widths / max(heights, 1e-6)
        if not any(0.7 * e <= aspect <= 1.3 * e for e in expected_aspects):
            continue
        if best is None or quad_area > best[0]:
            best = (quad_area, quad)
    return None if best is None else best[1].astype(np.float32)


def _find_page_quad(gray: np.ndarray) -> np.ndarray | None:
    """Find the outline of a bright sheet of paper against a darker background."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    image_area = gray.shape[0] * gray.shape[1]
    if cv2.contourArea(contour) < 0.2 * image_area or cv2.contourArea(contour) > 0.98 * image_area:
        return None
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) != 4:
        return None
    return _order_clockwise(approx.reshape(4, 2).astype(np.float32))


def _capture_scale(image_points: np.ndarray, page_format: PageFormat, use_markers: bool) -> float:
    """Resolution to read the sheet at, as a multiple of the 300 DPI template.

    Matches the photo's own resolution (no point inventing pixels it does not
    have), between 1x and ``MAX_CAPTURE_DPI``.
    """
    template = _template_points(page_format, use_markers)

    def perimeter(points: np.ndarray) -> float:
        return sum(float(np.linalg.norm(points[i] - points[i - 1])) for i in range(4))

    ratio = perimeter(np.asarray(image_points, dtype=np.float64)) / perimeter(template.astype(np.float64))
    if not math.isfinite(ratio):
        return 1.0
    return round(min(MAX_CAPTURE_DPI / TEMPLATE_DPI, max(1.0, ratio)), 2)


def _scale_rect(rect: Rect, scale: float) -> Rect:
    return Rect(round(rect.x * scale), round(rect.y * scale), round(rect.w * scale), round(rect.h * scale))


def _scaled_options(options: ExtractionOptions, scale: float) -> ExtractionOptions:
    """Pixel-based extraction limits adjusted to the capture resolution."""
    if scale == 1.0:
        return options
    area = scale * scale
    return replace(options,
                   speck_min_area=max(1, round(options.speck_min_area * area)),
                   min_ink_pixels=max(1, round(options.min_ink_pixels * area)),
                   padding=max(1, round(options.padding * scale)),
                   max_glyph_dimension=round(options.max_glyph_dimension * scale))


def _warp(gray: np.ndarray, image_points: np.ndarray, page_format: PageFormat, use_markers: bool,
          scale: float = 1.0) -> np.ndarray:
    """Flatten the page into template coordinates multiplied by *scale*."""
    width, height = page_format.size_px(TEMPLATE_DPI)
    width, height = round(width * scale), round(height * scale)
    transform = cv2.getPerspectiveTransform(image_points.astype(np.float32),
                                            _template_points(page_format, use_markers) * np.float32(scale))
    return cv2.warpPerspective(gray, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


# ------------------------------------------------------------------------ cells

def _snap_cell(border_mask: np.ndarray, rect: Rect) -> tuple[Rect, int]:
    """Move each side of *rect* onto the printed border line, if one is found nearby."""
    search = max(6, round(_CELL_SEARCH_FRACTION * min(rect.w, rect.h)))
    height, width = border_mask.shape

    def find_edge(expected: int, horizontal: bool) -> int | None:
        if horizontal:
            lo, hi = max(0, expected - search), min(height, expected + search + 1)
            band = border_mask[lo:hi, rect.x + rect.w // 6: rect.x + rect.w * 5 // 6]
            profile = band.sum(axis=1)
            span = band.shape[1]
        else:
            lo, hi = max(0, expected - search), min(width, expected + search + 1)
            band = border_mask[rect.y + rect.h // 6: rect.y + rect.h * 5 // 6, lo:hi]
            profile = band.sum(axis=0)
            span = band.shape[0]
        if profile.size == 0 or span == 0 or profile.max() < 0.5 * span:
            return None
        peak = np.flatnonzero(profile >= 0.85 * profile.max())
        # Prefer the line closest to the expected position.
        best = peak[np.argmin(np.abs(peak + lo - expected))]
        return int(best + lo)

    top = find_edge(rect.y, True)
    bottom = find_edge(rect.y + rect.h - 1, True)
    left = find_edge(rect.x, False)
    right = find_edge(rect.x + rect.w - 1, False)
    found = sum(edge is not None for edge in (top, bottom, left, right))

    if top is None and bottom is not None:
        top = bottom - rect.h + 1
    if bottom is None and top is not None:
        bottom = top + rect.h - 1
    if left is None and right is not None:
        left = right - rect.w + 1
    if right is None and left is not None:
        right = left + rect.w - 1
    x0 = rect.x if left is None else left
    y0 = rect.y if top is None else top
    x1 = rect.x + rect.w - 1 if right is None else right
    y1 = rect.y + rect.h - 1 if bottom is None else bottom
    # Reject implausible snaps (e.g. onto handwriting) by size check.
    if abs((x1 - x0 + 1) - rect.w) > search or abs((y1 - y0 + 1) - rect.h) > search:
        return rect, 0
    return Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1), found


def _extract_cell(darkness: np.ndarray, cell: CellSpec, rect: Rect, layout: SheetPageLayout,
                  options: ExtractionOptions, capture_dpi: float = TEMPLATE_DPI) -> ExtractedSample | None:
    """Cut one character out of *rect* (in capture pixels); *cell* gives template geometry."""
    inset = max(8, round(_CELL_INSET_FRACTION * min(rect.w, rect.h)))
    x0, y0 = rect.x + inset, rect.y + inset
    x1, y1 = rect.x + rect.w - inset, rect.y + rect.h - inset
    crop = darkness[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if crop.size == 0:
        return None
    try:
        glyph = extract_glyph_from_darkness(crop, options)
    except NoInkFoundError:
        return None

    pad = options.padding
    ink_width = max(1.0, (glyph.bbox[2] - 2 * pad)) * glyph.scale
    ink_height = max(1.0, (glyph.bbox[3] - 2 * pad)) * glyph.scale
    baseline_in_crop = rect.y + (cell.baseline_y - cell.rect.y) * rect.h / cell.rect.h - y0
    baseline_from_top = (baseline_in_crop - glyph.bbox[1]) * glyph.scale
    page = layout.identity.page_index
    return ExtractedSample(
        key=f"p{page}-{char_to_id(cell.char)}-{cell.variant}",
        char=cell.char,
        variant=cell.variant,
        page_index=page,
        rgba=glyph.rgba,
        ink_width=ink_width,
        ink_height=ink_height,
        baseline_from_top=float(baseline_from_top),
        capture_dpi=float(capture_dpi),
    )


def _preview(warped: np.ndarray, cells: Sequence[tuple[Rect, bool]], scale: float = 1.0) -> Image.Image:
    image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(image)
    for rect, has_ink in cells:
        color = (30, 170, 60) if has_ink else (200, 200, 200)
        draw.rectangle((rect.x, rect.y, rect.x + rect.w - 1, rect.y + rect.h - 1), outline=color,
                       width=round(6 * scale))
    scale = _PREVIEW_WIDTH / image.width
    return image.resize((_PREVIEW_WIDTH, round(image.height * scale)), Image.Resampling.LANCZOS)
