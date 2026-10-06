"""Page geometry, word wrapping and pagination.

This module is pure arithmetic: it knows nothing about images. The renderer
measures every character first (advance widths) and this module decides where
each one goes. That keeps wrapping logic easy to test.

Guarantees:
* every non-whitespace character of the text is placed exactly once, in order;
* words are only split when a single word is wider than the line;
* ``\\n`` always starts a new line, and blank lines are preserved;
* pagination never drops a line.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .settings import (
    ASCENDER_ALLOWANCE,
    DESCENDER_ALLOWANCE,
    GRAPH_SQUARE_MM,
    RULE_SPACING_MM,
    RULED_TEXT_INDENT_MM,
    PaperStyle,
    RenderSettings,
    mm_to_px,
)

_EPSILON = 1e-6


def normalize_text(text: str) -> str:
    """Normalise line endings and Unicode composition.

    Only representation changes: ``\\r\\n``/``\\r`` become ``\\n`` and
    characters are NFC-composed (so ``e`` + combining accent is one character).
    No visible character is added, removed or replaced.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def is_breaking_space(char: str) -> bool:
    """Whitespace that separates words (everything except newline)."""
    return char != "\n" and char.isspace()


@dataclass(frozen=True)
class PlacedChar:
    """A visible character positioned on a line.

    ``index`` refers to the normalised text; ``x`` is the offset (px) of the
    character's left edge from the start of the line.
    """

    index: int
    x: float


@dataclass(frozen=True)
class Line:
    items: tuple[PlacedChar, ...]
    width: float
    starts_paragraph: bool

    @property
    def is_blank(self) -> bool:
        return not self.items


def wrap_text(text: str, advances: Sequence[float], max_width: float,
              stop_short: Callable[[], float] | None = None) -> list[Line]:
    """Greedy word wrapping.

    Args:
        text: normalised text (see :func:`normalize_text`).
        advances: horizontal advance (px) for every character of *text*;
            spaces carry the word-space width, newlines are ignored.
        max_width: usable line width in px.
        stop_short: called once per line; returns how many px before *max_width*
            that line should end, the way a writer moves to the next line before
            reaching the edge. A word is never split because of it.
    """
    if len(advances) != len(text):
        raise ValueError("advances must have one entry per character of text")
    if max_width <= 0:
        raise ValueError("max_width must be positive")

    lines: list[Line] = []
    start = 0
    for paragraph in text.split("\n"):
        end = start + len(paragraph)
        lines.extend(_wrap_paragraph(text, start, end, advances, max_width, stop_short))
        start = end + 1
    return lines


def _tokenize(text: str, start: int, end: int) -> list[tuple[bool, int, int]]:
    """Split ``text[start:end]`` into runs of (is_space, run_start, run_end)."""
    tokens: list[tuple[bool, int, int]] = []
    i = start
    while i < end:
        space = is_breaking_space(text[i])
        j = i + 1
        while j < end and is_breaking_space(text[j]) == space:
            j += 1
        tokens.append((space, i, j))
        i = j
    return tokens


def _wrap_paragraph(text: str, start: int, end: int, advances: Sequence[float],
                    max_width: float, stop_short: Callable[[], float] | None = None) -> list[Line]:
    lines: list[Line] = []
    items: list[PlacedChar] = []
    x = 0.0
    first_line = True
    pending_space = 0.0

    def line_limit() -> float:
        if stop_short is None:
            return max_width
        return max(0.5 * max_width, max_width - max(0.0, stop_short()))

    limit = line_limit()

    def emit() -> None:
        nonlocal items, x, first_line, limit
        lines.append(Line(tuple(items), x, first_line))
        items, x, first_line = [], 0.0, False
        limit = line_limit()

    for is_space, run_start, run_end in _tokenize(text, start, end):
        if is_space:
            pending_space += sum(advances[run_start:run_end])
            continue

        word_width = sum(advances[run_start:run_end])
        # Spaces before a word: kept between words and as paragraph indentation,
        # dropped at the start of a wrapped line.
        lead = pending_space if (items or first_line) else 0.0
        pending_space = 0.0

        if items and x + lead + word_width > limit + _EPSILON:
            emit()
            lead = 0.0
        # Stopping short never splits a word that fits the full width.
        room = limit if word_width <= limit else max_width
        if not items and lead + word_width > room + _EPSILON:
            lead = max(0.0, min(lead, room - word_width))
        x += lead

        if x + word_width <= room + _EPSILON:
            for i in range(run_start, run_end):
                items.append(PlacedChar(i, x))
                x += advances[i]
        else:
            # Single word wider than the line: break it between characters.
            for i in range(run_start, run_end):
                if items and x + advances[i] > max_width + _EPSILON:
                    emit()
                items.append(PlacedChar(i, x))
                x += advances[i]

    emit()
    return lines


@dataclass(frozen=True)
class SlotRequest:
    """Vertical needs of one line when assigning it to a baseline.

    Attributes:
        space_before: empty baselines to leave above the line (ignored at the
            top of a page), e.g. to give large headings room.
        keep_with_next: move the line to the next page rather than leaving it
            alone on the last baseline (used for headings).
    """

    space_before: int = 0
    keep_with_next: bool = False


def assign_slots(requests: Sequence[SlotRequest], slots_per_page: int) -> list[tuple[int, int]]:
    """Assign each line a ``(page, baseline index)``; lines are never dropped."""
    if slots_per_page < 1:
        raise ValueError("The page has no room for any lines; reduce margins or handwriting size.")
    positions: list[tuple[int, int]] = []
    page, slot = 0, 0
    last = len(requests) - 1
    for i, request in enumerate(requests):
        if slot > 0:
            slot += request.space_before
        if slot >= slots_per_page:
            page, slot = page + 1, 0
        if request.keep_with_next and i < last and 0 < slot == slots_per_page - 1:
            page, slot = page + 1, 0
        positions.append((page, slot))
        slot += 1
    return positions


@dataclass(frozen=True)
class PageGeometry:
    """Pixel geometry of one page, shared by the paper and the renderer.

    Attributes:
        rule_ys: y positions of horizontal ruled/grid lines (empty for blank paper).
        grid_xs: x positions of vertical grid lines (graph paper only).
        margin_rule_x: x of the red margin line, if drawn.
        baselines: y positions of text baselines, top to bottom.
    """

    width: int
    height: int
    dpi: int
    x_height: float
    text_left: float
    text_right: float
    baselines: tuple[float, ...]
    rule_ys: tuple[float, ...] = ()
    grid_xs: tuple[float, ...] = ()
    margin_rule_x: float | None = None

    @property
    def text_width(self) -> float:
        return self.text_right - self.text_left

    @property
    def line_pitch(self) -> float:
        """Distance between consecutive baselines."""
        if len(self.baselines) >= 2:
            return self.baselines[1] - self.baselines[0]
        return 2.5 * self.x_height


def _arithmetic_positions(first: float, step: float, limit: float) -> list[float]:
    positions: list[float] = []
    while True:
        y = first + len(positions) * step
        if y > limit + _EPSILON:
            return positions
        positions.append(y)


def compute_page_geometry(settings: RenderSettings) -> PageGeometry:
    """Compute rules, baselines and text area for the given settings."""
    page = settings.page
    dpi = page.dpi
    width, height = page.page_format.size_px(dpi)
    left = mm_to_px(page.margins.left, dpi)
    right = width - mm_to_px(page.margins.right, dpi)
    top = mm_to_px(page.margins.top, dpi)
    bottom = height - mm_to_px(page.margins.bottom, dpi)
    x_height = mm_to_px(settings.x_height_mm, dpi)
    lowest_baseline = bottom - DESCENDER_ALLOWANCE * x_height

    rule_ys: list[float] = []
    grid_xs: list[float] = []
    margin_rule_x: float | None = None
    text_left = left
    style = page.paper_style

    if style.is_ruled:
        spacing = mm_to_px(RULE_SPACING_MM[style], dpi)
        rule_ys = _arithmetic_positions(top, spacing, bottom)
        step = max(1, int(page.ruled_line_step))
        baselines = _arithmetic_positions(top + spacing, spacing * step, lowest_baseline)
        if page.show_margin_rule and style.has_margin_rule:
            margin_rule_x = left
            text_left = left + mm_to_px(RULED_TEXT_INDENT_MM, dpi)
    elif style is PaperStyle.GRAPH:
        square = mm_to_px(GRAPH_SQUARE_MM, dpi)
        rule_ys = _arithmetic_positions(top % square, square, height)
        grid_xs = _arithmetic_positions(left % square, square, width)
        squares_per_line = max(1, round(page.line_spacing_mm / GRAPH_SQUARE_MM))
        first = top + square * max(1, round(ASCENDER_ALLOWANCE * x_height / square))
        baselines = _arithmetic_positions(first, square * squares_per_line, lowest_baseline)
    else:
        pitch = mm_to_px(page.line_spacing_mm, dpi)
        baselines = _arithmetic_positions(top + ASCENDER_ALLOWANCE * x_height, pitch, lowest_baseline)

    if right - text_left < 4 * x_height:
        raise ValueError("Margins are too large for this page; there is no room for text.")

    return PageGeometry(
        width=width,
        height=height,
        dpi=dpi,
        x_height=x_height,
        text_left=text_left,
        text_right=right,
        baselines=tuple(baselines),
        rule_ys=tuple(rule_ys),
        grid_xs=tuple(grid_xs),
        margin_rule_x=margin_rule_x,
    )
