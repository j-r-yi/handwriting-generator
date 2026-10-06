"""Printable handwriting sample sheets with fixed, known geometry.

Each sheet page has:

* four solid black **corner markers** used to find and un-distort the page in a
  photo or scan;
* a small **page code** (a row of black/white squares next to the top-left
  marker) encoding the paper format, page number and samples-per-character,
  so uploaded pages identify themselves;
* a grid of **cells**: a printed label to the left (outside the cells) and N
  empty boxes per character. Small tick marks on the box sides show the
  baseline.

All coordinates are in template pixels at :data:`TEMPLATE_DPI`; extraction
warps photos into exactly this coordinate system.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw

from .charset import DEFAULT_CHARSET, describe_char
from .settings import PageFormat
from .utils import get_font

TEMPLATE_DPI = 300
MIN_VARIANTS = 1
MAX_VARIANTS = 8
DEFAULT_VARIANTS = 3
MAX_PAGES = 15  # limited by the 4-bit page number in the page code


def _px(inches: float) -> int:
    return round(inches * TEMPLATE_DPI)


MARKER_SIZE = _px(0.35)
MARKER_INSET = _px(0.40)
CODE_BIT_SIZE = _px(0.11)
CODE_OFFSET_X = _px(0.25)  # gap between the top-left marker and the page code
HEADER_HEIGHT = _px(1.10)  # from the top marker edge to the first cell row
FOOTER_GAP = _px(0.10)
LABEL_WIDTH = _px(0.50)
CELL_WIDTH = _px(0.58)
CELL_HEIGHT = _px(0.66)
CELL_GAP = _px(0.06)
GROUP_GAP = _px(0.15)
ROW_GAP = _px(0.12)
BORDER_WIDTH = 3
BORDER_COLOR = (70, 70, 70)
BASELINE_RATIO = 0.68  # baseline tick position from the top of the cell
TICK_LENGTH = _px(0.04)
TICK_WIDTH = 3

# Page code: start pattern, 1 format bit, 4 page bits, 4 variant bits, parity, end pattern.
_CODE_START = (1, 1, 0)
_CODE_END = (0, 1)  # asymmetric with the start, so a reversed read never validates
_FORMAT_BITS = {PageFormat.LETTER: 0, PageFormat.A4: 1}
CODE_LENGTH = len(_CODE_START) + 1 + 4 + 4 + 1 + len(_CODE_END)

INSTRUCTIONS = (
    "Write the indicated character once inside each box. Avoid touching the box borders.",
    "Write at your normal size with a dark pen. The small marks on the box sides show the baseline;",
    "letters such as g, j, p, q, y should hang below it. Leave a box empty to skip it.",
    "Keep all four black corner squares visible when you scan or photograph the page.",
)


@dataclass(frozen=True)
class SheetIdentity:
    """Which sheet page an image shows."""

    page_format: PageFormat
    page_index: int  # zero-based
    variants: int


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True)
class CellSpec:
    """One handwriting box on the sheet."""

    char: str
    variant: int  # zero-based
    rect: Rect

    @property
    def baseline_y(self) -> float:
        return self.rect.y + BASELINE_RATIO * self.rect.h


@dataclass(frozen=True)
class SheetPageLayout:
    identity: SheetIdentity
    page_count: int
    width: int
    height: int
    cells: tuple[CellSpec, ...]
    labels: tuple[tuple[str, Rect], ...]
    markers: tuple[Rect, Rect, Rect, Rect]  # TL, TR, BR, BL
    code_bits: tuple[Rect, ...]

    @property
    def marker_centers(self) -> list[tuple[float, float]]:
        return [m.center for m in self.markers]


def marker_rects(page_format: PageFormat) -> tuple[Rect, Rect, Rect, Rect]:
    width, height = page_format.size_px(TEMPLATE_DPI)
    far_x = width - MARKER_INSET - MARKER_SIZE
    far_y = height - MARKER_INSET - MARKER_SIZE
    return (
        Rect(MARKER_INSET, MARKER_INSET, MARKER_SIZE, MARKER_SIZE),
        Rect(far_x, MARKER_INSET, MARKER_SIZE, MARKER_SIZE),
        Rect(far_x, far_y, MARKER_SIZE, MARKER_SIZE),
        Rect(MARKER_INSET, far_y, MARKER_SIZE, MARKER_SIZE),
    )


def code_bit_rects() -> tuple[Rect, ...]:
    """Positions of the page-code squares (identical for every format)."""
    x0 = MARKER_INSET + MARKER_SIZE + CODE_OFFSET_X
    y0 = MARKER_INSET + (MARKER_SIZE - CODE_BIT_SIZE) // 2
    return tuple(Rect(x0 + i * CODE_BIT_SIZE, y0, CODE_BIT_SIZE, CODE_BIT_SIZE) for i in range(CODE_LENGTH))


def encode_identity(identity: SheetIdentity) -> list[int]:
    """Encode a sheet identity as the list of page-code bits."""
    if not 0 <= identity.page_index < 16:
        raise ValueError("page_index must be between 0 and 15")
    if not MIN_VARIANTS <= identity.variants <= 16:
        raise ValueError("variants must be between 1 and 16")
    data = [_FORMAT_BITS[identity.page_format]]
    data += [(identity.page_index >> shift) & 1 for shift in (3, 2, 1, 0)]
    data += [((identity.variants - 1) >> shift) & 1 for shift in (3, 2, 1, 0)]
    parity = sum(data) % 2
    return [*_CODE_START, *data, parity, *_CODE_END]


def decode_identity(bits: Sequence[int]) -> SheetIdentity | None:
    """Decode page-code bits; returns ``None`` if they are not a valid code."""
    if len(bits) != CODE_LENGTH:
        return None
    start, end = len(_CODE_START), CODE_LENGTH - len(_CODE_END)
    if tuple(bits[:start]) != _CODE_START or tuple(bits[end:]) != _CODE_END:
        return None
    data, parity = list(bits[start:end - 1]), bits[end - 1]
    if sum(data) % 2 != parity:
        return None
    page_format = PageFormat.A4 if data[0] else PageFormat.LETTER
    page_index = int("".join(map(str, data[1:5])), 2)
    variants = int("".join(map(str, data[5:9])), 2) + 1
    return SheetIdentity(page_format, page_index, variants)


def build_sheet_layouts(page_format: PageFormat, variants: int = DEFAULT_VARIANTS,
                        charset: Sequence[str] = DEFAULT_CHARSET) -> list[SheetPageLayout]:
    """Compute the deterministic layout of every page of a sample sheet."""
    if not MIN_VARIANTS <= variants <= MAX_VARIANTS:
        raise ValueError(f"Samples per character must be between {MIN_VARIANTS} and {MAX_VARIANTS}.")
    width, height = page_format.size_px(TEMPLATE_DPI)
    markers = marker_rects(page_format)

    grid_left = MARKER_INSET
    grid_right = width - MARKER_INSET
    grid_top = MARKER_INSET + HEADER_HEIGHT
    grid_bottom = height - MARKER_INSET - MARKER_SIZE - FOOTER_GAP

    group_width = LABEL_WIDTH + variants * CELL_WIDTH + (variants - 1) * CELL_GAP
    available_width = grid_right - grid_left
    groups_per_row = max(1, (available_width + GROUP_GAP) // (group_width + GROUP_GAP))
    rows_per_page = (grid_bottom - grid_top + ROW_GAP) // (CELL_HEIGHT + ROW_GAP)
    per_page = groups_per_row * rows_per_page
    page_count = math.ceil(len(charset) / per_page)
    if page_count > MAX_PAGES:
        raise ValueError("Too many characters for one sample sheet.")

    used_width = groups_per_row * group_width + (groups_per_row - 1) * GROUP_GAP
    x_start = grid_left + (available_width - used_width) // 2

    layouts: list[SheetPageLayout] = []
    for page_index in range(page_count):
        chars = charset[page_index * per_page:(page_index + 1) * per_page]
        cells: list[CellSpec] = []
        labels: list[tuple[str, Rect]] = []
        for slot, char in enumerate(chars):
            row, column = divmod(slot, groups_per_row)
            gx = x_start + column * (group_width + GROUP_GAP)
            gy = grid_top + row * (CELL_HEIGHT + ROW_GAP)
            labels.append((char, Rect(gx, gy, LABEL_WIDTH, CELL_HEIGHT)))
            for variant in range(variants):
                cx = gx + LABEL_WIDTH + variant * (CELL_WIDTH + CELL_GAP)
                cells.append(CellSpec(char, variant, Rect(cx, gy, CELL_WIDTH, CELL_HEIGHT)))
        layouts.append(SheetPageLayout(
            identity=SheetIdentity(page_format, page_index, variants),
            page_count=page_count,
            width=width,
            height=height,
            cells=tuple(cells),
            labels=tuple(labels),
            markers=markers,
            code_bits=code_bit_rects(),
        ))
    return layouts


def render_sheet_page(layout: SheetPageLayout) -> Image.Image:
    """Draw one printable sheet page (RGB, 300 DPI)."""
    image = Image.new("RGB", (layout.width, layout.height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for marker in layout.markers:
        draw.rectangle(_box(marker), fill=(0, 0, 0))
    for bit, rect in zip(encode_identity(layout.identity), layout.code_bits):
        if bit:
            draw.rectangle(_box(rect), fill=(0, 0, 0))

    identity = layout.identity
    title_x = layout.code_bits[-1].x + CODE_BIT_SIZE + _px(0.25)
    draw.text((title_x, MARKER_INSET - _px(0.02)), "Handwriting Sample Sheet",
              font=get_font(_px(0.22)), fill=(0, 0, 0))
    subtitle = (f"Page {identity.page_index + 1} of {layout.page_count}  ·  "
                f"{identity.page_format.value}  ·  {identity.variants} per character")
    draw.text((title_x, MARKER_INSET + _px(0.24)), subtitle, font=get_font(_px(0.12)), fill=(60, 60, 60))

    instruction_font = get_font(_px(0.105))
    y = MARKER_INSET + MARKER_SIZE + _px(0.12)
    for line in INSTRUCTIONS:
        draw.text((MARKER_INSET, y), line, font=instruction_font, fill=(30, 30, 30))
        y += _px(0.15)

    label_font = get_font(_px(0.30))
    hint_font = get_font(_px(0.075))
    for char, rect in layout.labels:
        cx, cy = rect.center
        draw.text((cx, cy - _px(0.03)), char, font=label_font, fill=(0, 0, 0), anchor="mm")
        description = describe_char(char)
        if description != char:
            hint = description.split("(", 1)[1].rstrip(")")
            draw.text((cx, rect.y + rect.h - _px(0.02)), hint, font=hint_font, fill=(90, 90, 90), anchor="md")

    for cell in layout.cells:
        _draw_cell(draw, cell)
    return image


def _draw_cell(draw: ImageDraw.ImageDraw, cell: CellSpec) -> None:
    r = cell.rect
    draw.rectangle(_box(r), outline=BORDER_COLOR, width=BORDER_WIDTH)
    y = round(cell.baseline_y)
    half = TICK_WIDTH // 2
    draw.rectangle((r.x, y - half, r.x + TICK_LENGTH, y + half), fill=BORDER_COLOR)
    draw.rectangle((r.x + r.w - 1 - TICK_LENGTH, y - half, r.x + r.w - 1, y + half), fill=BORDER_COLOR)


def _box(rect: Rect) -> tuple[int, int, int, int]:
    return rect.x, rect.y, rect.x + rect.w - 1, rect.y + rect.h - 1


def render_sample_sheet(page_format: PageFormat, variants: int = DEFAULT_VARIANTS) -> list[Image.Image]:
    """Render every page of a blank sample sheet."""
    return [render_sheet_page(layout) for layout in build_sheet_layouts(page_format, variants)]
