"""Compose text from handwritten glyph samples onto paper.

The renderer never changes the text: every visible character is drawn exactly
once, in order — either with one of its handwritten samples or, if none
exists, with a clearly visible fallback (see ``MissingGlyphPolicy``). Only the
*appearance* varies (variant choice; letter, word and line jitter; letter
shape and ink pressure — see ``VariationSettings``), and with a fixed seed
even that is fully reproducible.

Input is first turned into blocks (:mod:`handwriting.document`): plain text
becomes one paragraph per line, kept exactly; with ``RenderSettings.markdown``
the text is parsed by :mod:`handwriting.markdown` and the Markdown syntax
becomes formatting (larger headings, bullets, checkboxes, bold, ...).

Rendering then happens in three steps:

1. **plan** – choose a variant and jitter for every character, and measure its
   advance width (proportional to the real glyph image, times any heading size);
2. **layout** – wrap each block in its column and assign lines to baselines
   with :mod:`handwriting.layout`;
3. **compose** – draw paper, glyphs and hand-drawn marks on their baselines.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .decorations import InkPainter
from .document import LIST_KINDS, PLAIN, Block, BlockKind, Style, plain_blocks, visible_text
from .errors import EmptyTextError, HandwritingError, MissingGlyphError
from .layout import (
    Line,
    PageGeometry,
    SlotRequest,
    assign_slots,
    compute_page_geometry,
    is_breaking_space,
    normalize_text,
    wrap_text,
)
from .markdown import parse_markdown
from .models import GlyphSet, LoadedGlyph
from .paper import render_paper
from .settings import (
    DEFAULT_DPI,
    MISSING_GLYPH_COLOR,
    TAB_WIDTH_SPACES,
    MarkdownStyle,
    MissingGlyphPolicy,
    RenderSettings,
    VariantMode,
    VariationSettings,
)
from .utils import get_font

MAX_PAGES = 30  # at 300 DPI; fewer at higher resolutions (see max_pages)
_SEED_RANGE = 2**31 - 1
# Typical x-height of a sans-serif font as a fraction of its point size.
_FONT_X_HEIGHT_RATIO = 0.52
# Placeholder box for missing characters (fractions of the x-height).
_PLACEHOLDER_WIDTH = 0.7
_PLACEHOLDER_HEIGHT = 1.25
# A handwritten bullet sample is used when the profile has one; otherwise a dot is drawn.
BULLET_CHAR = "\u2022"
# Narrowest text column allowed for deeply nested content (x-heights).
_MIN_COLUMN = 4.0
# Shape warp: displacement grid per letter (cells across) and smallest warp worth doing (px).
_WARP_GRID = 4
_MIN_WARP_PX = 0.3
# Share of the line slope that is a document-wide habit rather than per line.
_SLOPE_HABIT = 0.6
# Largest indent wander, as a share of one nesting step.
_MAX_INDENT_DRIFT = 0.4


def max_pages(dpi: int) -> int:
    """Page limit for one generation: a fixed pixel budget, so memory use stays bounded."""
    return max(1, min(MAX_PAGES, int(MAX_PAGES * (DEFAULT_DPI / dpi) ** 2)))


def build_blocks(text: str, markdown: bool) -> list[Block]:
    """Turn input text into renderable blocks."""
    return parse_markdown(text) if markdown else plain_blocks(text)


def find_missing_characters(text: str, glyphs: GlyphSet, markdown: bool = False) -> dict[str, int]:
    """Characters that will be written but have no samples, with their counts.

    Ordered by first appearance. In Markdown mode, syntax characters that
    become formatting (``#``, ``**``, ``- ``...) are not counted.
    """
    counts: Counter[str] = Counter()
    for char in visible_text(build_blocks(text, markdown)):
        if not char.isspace() and not glyphs.has(char):
            counts[char] += 1
    return dict(counts)


class VariantSelector:
    """Chooses which sample to use for each occurrence of a character.

    In ``SHUFFLE`` mode each character has a shuffled "deck" of its variants;
    all variants are used before any repeats, and a new deck never starts with
    the variant that was just used. ``RANDOM`` picks freely but avoids using
    the same variant twice in a row.
    """

    def __init__(self, rng: random.Random, mode: VariantMode) -> None:
        self._rng = rng
        self._mode = mode
        self._decks: dict[str, list[int]] = {}
        self._last: dict[str, int] = {}

    def choose(self, char: str, count: int) -> int:
        if count <= 0:
            raise ValueError("count must be positive")
        if count == 1 or self._mode is VariantMode.FIRST:
            choice = 0
        elif self._mode is VariantMode.RANDOM:
            last = self._last.get(char)
            if last is None:
                choice = self._rng.randrange(count)
            else:
                choice = self._rng.randrange(count - 1)
                if choice >= last:
                    choice += 1  # uniform over every variant except the last one
        else:
            choice = self._draw_from_deck(char, count)
        self._last[char] = choice
        return choice

    def _draw_from_deck(self, char: str, count: int) -> int:
        deck = self._decks.get(char)
        if not deck:
            deck = list(range(count))
            self._rng.shuffle(deck)
            last = self._last.get(char)
            if last is not None and deck[-1] == last:
                deck[0], deck[-1] = deck[-1], deck[0]
            self._decks[char] = deck
        return deck.pop()


@dataclass(frozen=True)
class CharPlan:
    """How one character will be drawn.

    ``scale`` maps glyph pixels to page pixels (including heading ``size`` and
    jitter); ``ink_width`` is the drawn width before letter spacing. ``slant``
    is an extra shear (on top of italics), ``warp`` the shape distortion in
    page pixels (drawn with ``warp_seed``) and ``ink`` the opacity (pressure).
    """

    char: str
    advance: float
    glyph: LoadedGlyph | None = None
    scale: float = 1.0
    rotation: float = 0.0
    baseline_offset: float = 0.0
    size: float = 1.0
    style: Style = PLAIN
    ink_width: float = 0.0
    slant: float = 0.0
    warp: float = 0.0
    warp_seed: int = 0
    ink: float = 1.0


@dataclass(frozen=True)
class RenderedChar:
    """Record of where a written character ended up (for tests and diagnostics).

    ``index`` is the character's position in reading order.
    """

    index: int
    char: str
    page: int
    line: int
    sample_id: str | None


@dataclass
class RenderResult:
    pages: list[Image.Image]
    seed: int
    dpi: int
    missing: dict[str, int]
    placements: list[RenderedChar] = field(default_factory=list)
    line_count: int = 0


class _Jitter:
    """Clipped-normal random offsets; zero magnitude means no randomness."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def __call__(self, magnitude: float) -> float:
        if magnitude <= 0:
            return 0.0
        value = self._rng.gauss(0.0, magnitude / 2)
        return max(-magnitude, min(magnitude, value))


def resolve_seed(seed: int | None) -> int:
    """Return *seed*, or a fresh random seed when none was given."""
    if seed is not None:
        return int(seed)
    return random.SystemRandom().randrange(1, _SEED_RANGE)


def render_text(text: str, glyphs: GlyphSet, settings: RenderSettings) -> RenderResult:
    """Render *text* in handwriting.

    Raises:
        EmptyTextError: if the text has no visible characters.
        MissingGlyphError: if characters lack samples and the policy is ERROR.
        HandwritingError: if the page geometry is impossible or the text too long.
    """
    if not normalize_text(text).strip():
        raise EmptyTextError("Please enter some text to render.")
    blocks = build_blocks(text, settings.markdown)

    missing = find_missing_characters(text, glyphs, settings.markdown)
    if missing and settings.missing_policy is MissingGlyphPolicy.ERROR:
        raise MissingGlyphError(missing)

    try:
        geometry = compute_page_geometry(settings)
    except ValueError as exc:
        raise HandwritingError(str(exc)) from exc

    seed = resolve_seed(settings.seed)
    planner = _Planner(glyphs, settings, geometry.x_height, seed,
                       total_chars=sum(len(block.text) for block in blocks))
    lines, requests = _layout_blocks(blocks, planner, settings.markdown_style, geometry,
                                     reserve=_max_margin_shift(settings.variation) * geometry.x_height)
    try:
        positions = assign_slots(requests, len(geometry.baselines))
    except ValueError as exc:
        raise HandwritingError(str(exc)) from exc
    page_count = positions[-1][0] + 1 if positions else 1
    limit = max_pages(settings.page.dpi)
    if page_count > limit:
        lower = " or a lower output resolution (Settings / About)" if limit < MAX_PAGES else ""
        raise HandwritingError(
            f"The text needs {page_count} pages; the limit at {settings.page.dpi} DPI is {limit}. "
            f"Split it into smaller parts or use smaller handwriting{lower}."
        )

    composer = _PageComposer(settings, geometry, seed, line_count=len(lines))
    images = [render_paper(geometry, settings.page) for _ in range(page_count)]
    placements: list[RenderedChar] = []
    for placed, (page_number, slot) in zip(lines, positions):
        for plan in composer.draw_line(images[page_number], placed, geometry.baselines[slot]):
            placements.append(RenderedChar(
                index=len(placements),
                char=plan.char,
                page=page_number,
                line=slot,
                sample_id=plan.glyph.sample_id if plan.glyph else None,
            ))

    return RenderResult(pages=images, seed=seed, dpi=settings.page.dpi, missing=missing,
                        placements=placements, line_count=len(lines))


@dataclass
class _Word:
    """Variation shared by every letter of one word."""

    baseline: float = 0.0  # x-heights
    scale: float = 1.0
    tilt: float = 0.0  # degrees, counter-clockwise
    slant: float = 0.0
    ink: float = 1.0
    spacing: float = 0.0  # extra letter spacing for the whole word (x-heights)
    x: float = 0.0  # advance written so far in this word (px)


class _Planner:
    """Chooses variants and jitter for every character, deterministically.

    Separate random streams are used for variant choice and for each kind of
    jitter, so changing one variation slider does not reshuffle everything else.
    """

    def __init__(self, glyphs: GlyphSet, settings: RenderSettings, x_height: float, seed: int,
                 total_chars: int = 0) -> None:
        self.glyphs = glyphs
        self.settings = settings
        self.x_height = x_height
        variation = settings.variation
        self._variation = variation
        self._selector = VariantSelector(random.Random(f"{seed}:variants"), variation.variant_mode)
        self._size_jitter = _Jitter(random.Random(f"{seed}:size"))
        self._angle_jitter = _Jitter(random.Random(f"{seed}:angle"))
        self._baseline_jitter = _Jitter(random.Random(f"{seed}:baseline"))
        self._spacing_jitter = _Jitter(random.Random(f"{seed}:spacing"))
        self._word_jitter = _Jitter(random.Random(f"{seed}:words"))
        self._ink_jitter = _Jitter(random.Random(f"{seed}:ink"))
        self._looseness_jitter = _Jitter(random.Random(f"{seed}:looseness"))
        self.layout_rng = random.Random(f"{seed}:layout")
        self._drift_rng = random.Random(f"{seed}:drift")
        self._shape_rng = random.Random(f"{seed}:shape")
        self._typed = settings.missing_policy is MissingGlyphPolicy.TYPED
        self._total = max(1, total_chars)
        self._planned = 0
        self._drift = 0.0
        self._word: _Word | None = None

    def messiness(self) -> float:
        """Multiplier for jitter: grows with ``fatigue`` towards the end of the text."""
        return 1.0 + self._variation.fatigue * min(1.0, self._planned / self._total)

    def _new_word(self, m: float) -> _Word:
        v, jitter = self._variation, self._word_jitter
        return _Word(
            baseline=jitter(v.word_baseline * m),
            scale=1.0 + jitter(v.word_scale * m),
            tilt=jitter(v.word_tilt_deg * m),
            slant=jitter(v.slant_jitter * m),
            ink=1.0 - abs(self._ink_jitter(v.ink_fade * m)),
            # Some words are written more spread out, others more cramped.
            spacing=self._looseness_jitter(v.letter_spacing_jitter * m),
        )

    def end_word(self) -> None:
        """Start a new word with the next character (e.g. after a list marker)."""
        self._word = None

    def _next_drift(self, m: float) -> float:
        """Slowly wandering size factor (a smooth random walk in [-1, 1])."""
        if self._variation.size_drift <= 0:
            return 1.0
        self._drift = max(-1.0, min(1.0, 0.94 * self._drift + self._drift_rng.gauss(0.0, 0.3)))
        return 1.0 + self._variation.size_drift * m * self._drift

    def plan(self, char: str, size: float = 1.0, style: Style = PLAIN) -> CharPlan:
        m = self.messiness()
        self._planned += 1
        v = self._variation
        x_height = self.x_height * size
        if is_breaking_space(char) or char == "\n":
            self._word = None
            width = self.settings.word_spacing * x_height * (TAB_WIDTH_SPACES if char == "\t" else 1)
            width *= 1.0 + self._spacing_jitter(v.word_spacing_jitter * m)
            return CharPlan(char=char, advance=max(0.0, width), size=size, style=style)

        if self._word is None:
            self._word = self._new_word(m)
        word = self._word
        base = self.settings.letter_spacing
        gap = base + word.spacing + self._spacing_jitter(v.letter_spacing_jitter * m)
        # Jitter may pull a letter into its neighbour (touching, as in quick writing), but only so far.
        spacing = max(gap, min(base, 0.0) - v.letter_overlap * m) * x_height
        bold_extra = self.settings.markdown_style.bold_offset * x_height if style.bold else 0.0
        variants = self.glyphs.variants(char)
        if not variants:
            ink_width = self._fallback_width(char, x_height)
            plan = CharPlan(char=char, advance=ink_width + bold_extra + max(0.0, spacing),
                            size=size, style=style, ink_width=ink_width)
            word.x += plan.advance
            return plan

        glyph = variants[self._selector.choose(char, len(variants))]
        scale = (x_height / glyph.x_height_ref * word.scale * self._next_drift(m)
                 * (1.0 + self._size_jitter(v.scale_jitter * m)))
        ink_width = glyph.width * scale + bold_extra
        # A tilted word rises (or sinks) steadily from its first letter.
        rise = math.tan(math.radians(word.tilt)) * word.x
        plan = CharPlan(
            char=char,
            advance=max(ink_width * 0.5, ink_width + spacing),
            glyph=glyph,
            scale=scale,
            rotation=self._angle_jitter(v.rotation_deg * m) + word.tilt,
            baseline_offset=(self._baseline_jitter(v.baseline_jitter * m) + word.baseline) * x_height - rise,
            size=size,
            style=style,
            ink_width=ink_width,
            slant=word.slant,
            warp=v.shape_warp * m * x_height,
            warp_seed=self._shape_rng.randrange(_SEED_RANGE) if v.shape_warp > 0 else 0,
            ink=word.ink * (1.0 - abs(self._ink_jitter(v.ink_fade * m * 0.3))),
        )
        word.x += plan.advance
        return plan

    def _fallback_width(self, char: str, x_height: float) -> float:
        if self._typed:
            return float(get_font(max(6, round(x_height / _FONT_X_HEIGHT_RATIO))).getlength(char))
        return _PLACEHOLDER_WIDTH * x_height


@dataclass(frozen=True)
class _PlacedLine:
    """One wrapped line of a block, ready to draw.

    ``text_x`` and ``marker_x`` are offsets from the page's text column.
    """

    block: Block
    line: Line
    plans: tuple[CharPlan, ...]
    text_x: float
    marker_x: float
    marker_plans: tuple[CharPlan, ...]
    first: bool


def _max_margin_shift(variation: VariationSettings) -> float:
    """Largest rightward line shift from ``margin_jitter``, in x-heights."""
    return max(0.0, variation.margin_jitter) * (1.0 + max(0.0, variation.fatigue))


def _layout_blocks(blocks: Sequence[Block], planner: _Planner, style: MarkdownStyle,
                   geometry: PageGeometry, reserve: float = 0.0) -> tuple[list[_PlacedLine], list[SlotRequest]]:
    """Wrap blocks into lines; *reserve* px of each line are kept free for line shifts.

    The layout variation settings make the structure hand-made: each item
    starts at a slightly different indent, wrapped lines don't line up exactly
    under the text above, and lines end at different distances from the margin.
    """
    x_height = geometry.x_height
    variation = planner.settings.variation
    rng = planner.layout_rng
    jitter = _Jitter(rng)
    lines: list[_PlacedLine] = []
    requests: list[SlotRequest] = []
    blank_run = 0  # empty lines just above the current block
    drift: dict[tuple[int, int], float] = {}  # indent wander per nesting level
    cap = _MAX_INDENT_DRIFT * style.list_indent * x_height  # keeps nesting levels distinguishable
    for block in blocks:
        heading = block.kind is BlockKind.HEADING
        level = max(1, min(block.level, len(style.heading_scale))) if heading else 0
        size = style.heading_scale[level - 1] if heading else 1.0
        m = planner.messiness()
        indent = variation.indent_jitter * m * x_height
        ragged = variation.ragged_right * m * x_height

        marker_x = (block.quote_depth * style.quote_indent + block.indent * style.list_indent) * x_height
        if indent:
            # Each level's indent wanders slowly: items line up roughly with the one above.
            depth = (block.quote_depth, block.indent)
            drift[depth] = max(-cap, min(cap, 0.5 * drift.get(depth, 0.0) + jitter(indent)))
            marker_x = max(0.0, marker_x + drift[depth] + 0.5 * min(indent, cap))
        marker_plans: tuple[CharPlan, ...] = ()
        marker_width = 0.0
        if block.kind is BlockKind.NUMBERED:
            marker_plans = tuple(planner.plan(c) for c in block.marker)
            marker_width = max(sum(p.advance for p in marker_plans) + style.marker_gap * x_height,
                               style.bullet_column * x_height)
        elif block.kind in LIST_KINDS:
            marker_width = style.bullet_column * x_height
            bullet = _bullet_glyph_char(block, planner.glyphs)
            if bullet is not None:
                marker_plans = (planner.plan(bullet),)
        planner.end_word()  # the marker is not part of the first word
        if marker_width:  # the gap between a marker and its text varies too
            shortest = max(0.75 * marker_width, sum(p.advance for p in marker_plans) + 0.3 * x_height)
            marker_width = max(shortest, marker_width + jitter(0.4 * indent))
        text_x = marker_x + marker_width
        overflow = text_x - (geometry.text_width - _MIN_COLUMN * x_height)
        if overflow > 0:  # very deep nesting: keep a usable column
            text_x -= overflow
            marker_x = max(0.0, marker_x - overflow)

        plans = tuple(planner.plan(char, size, char_style)
                      for char, char_style in zip(block.text, block.styles()))
        # Wrapped lines may start up to half an indent step right of the text above.
        wrap_shift = 0.5 * indent
        if block.kind is BlockKind.RULE:
            wrapped = [Line((), 0.0, True)]
        else:
            width = max(_MIN_COLUMN * x_height, geometry.text_width - text_x - reserve - wrap_shift)
            stop_short = (lambda: rng.uniform(0.0, ragged)) if ragged > 0 else None
            wrapped = wrap_text(block.text, [p.advance for p in plans], width, stop_short)
        for number, line in enumerate(wrapped):
            line_x = text_x
            if number and indent:
                # Writers rarely line a wrapped line up exactly; it drifts back towards the margin more often.
                line_x = max(marker_x, text_x + rng.uniform(-1.0, 0.5) * indent)
            lines.append(_PlacedLine(block, line, plans, line_x, marker_x, marker_plans, first=number == 0))
            # A heading's extra space counts blank lines already above it.
            space = 0
            if heading and number == 0 and style.heading_gaps:
                space = style.heading_space_before[level - 1] - blank_run
            requests.append(SlotRequest(
                space_before=max(0, space),
                keep_with_next=heading and number == len(wrapped) - 1,
            ))
        is_blank = block.kind is BlockKind.PARAGRAPH and not block.text and not block.quote_depth
        blank_run = blank_run + 1 if is_blank else 0
    return lines, requests


def _bullet_glyph_char(block: Block, glyphs: GlyphSet) -> str | None:
    """The handwritten character to use as a bullet, if the profile has one.

    A ``-`` bullet is written with the writer's own dash, as in handwritten
    notes; ``*`` and ``+`` use their own ``•`` when there is a sample.
    """
    if block.kind is not BlockKind.BULLET:
        return None
    wanted = "-" if block.marker == "-" else BULLET_CHAR
    return wanted if glyphs.has(wanted) else None


@dataclass(frozen=True)
class _LineShape:
    """How a hand-written line departs from its ruled baseline.

    ``dy(x)`` is the vertical offset (px, down is positive) at page x and
    ``tilt(x)`` the local angle in degrees (counter-clockwise), so letters
    follow a line that runs uphill or wanders.
    """

    origin: float = 0.0
    shift: float = 0.0
    offset: float = 0.0
    slope: float = 0.0  # dy/dx
    waves: tuple[tuple[float, float, float], ...] = ()  # (amplitude px, radians per px, phase)

    def dy(self, x: float) -> float:
        t = x - self.origin
        return self.offset + self.slope * t + sum(a * math.sin(f * t + p) for a, f, p in self.waves)

    def tilt(self, x: float) -> float:
        t = x - self.origin
        gradient = self.slope + sum(a * f * math.cos(f * t + p) for a, f, p in self.waves)
        return -math.degrees(math.atan(gradient))


_STRAIGHT = _LineShape()


class _PageComposer:
    """Draws planned characters and Markdown marks onto page images."""

    def __init__(self, settings: RenderSettings, geometry: PageGeometry, seed: int, line_count: int = 1) -> None:
        self._settings = settings
        self._variation = settings.variation
        self._style = settings.markdown_style
        self._geometry = geometry
        self._x_height = geometry.x_height
        self._alpha: dict[int, np.ndarray] = {}
        self._painter = InkPainter(settings.ink_color, geometry.x_height,
                                   self._style.stroke_width * geometry.x_height, random.Random(f"{seed}:marks"))
        self._line_rng = random.Random(f"{seed}:lines")
        self._line_jitter = _Jitter(self._line_rng)
        self._lines_total = max(1, line_count)
        self._lines_drawn = 0
        # A writer's lines tend to run the same way: part of the slope is a habit for the whole text.
        self._slope_habit = _Jitter(random.Random(f"{seed}:slope"))(
            self._variation.line_slope_deg * _SLOPE_HABIT)

    # ------------------------------------------------------------------ lines

    def _next_line_shape(self) -> _LineShape:
        v, xh, rng = self._variation, self._x_height, self._line_rng
        m = 1.0 + v.fatigue * min(1.0, self._lines_drawn / self._lines_total)
        self._lines_drawn += 1
        slope_deg = self._slope_habit + self._line_jitter(v.line_slope_deg * (1 - _SLOPE_HABIT) * m)
        waves: list[tuple[float, float, float]] = []
        if v.line_wave > 0:
            # One long swell plus a smaller, quicker wobble.
            for share, (shortest, longest) in ((1.0, (12.0, 25.0)), (0.35, (5.0, 9.0))):
                amplitude = v.line_wave * m * share * rng.uniform(0.5, 1.0) * xh
                wavelength = rng.uniform(shortest, longest) * xh
                waves.append((amplitude, 2 * math.pi / wavelength, rng.uniform(0, 2 * math.pi)))
        shift = rng.uniform(-0.3, 1.0) * v.margin_jitter * m * xh if v.margin_jitter > 0 else 0.0
        return _LineShape(
            origin=self._geometry.text_left,
            shift=shift,
            offset=self._line_jitter(v.line_offset * m) * xh,
            slope=-math.tan(math.radians(slope_deg)),
            waves=tuple(waves),
        )

    def draw_line(self, page: Image.Image, placed: _PlacedLine, baseline: float) -> list[CharPlan]:
        """Draw one line and return the written characters in reading order."""
        block, xh = placed.block, self._x_height
        shape = self._next_line_shape()
        left = self._geometry.text_left + shape.shift
        written: list[CharPlan] = []
        self._draw_quote_bars(page, block, baseline)
        if block.kind is BlockKind.RULE:
            start, end = left + placed.marker_x, self._geometry.text_right
            y = baseline - 0.45 * xh
            self._painter.line(page, (start, y + shape.dy(start)), (end, y + shape.dy(end)))
            return written

        if placed.first:
            written.extend(self._draw_marker(page, placed, left + placed.marker_x, baseline, shape))

        positions: list[tuple[float, CharPlan]] = []
        for item in placed.line.items:
            plan = placed.plans[item.index]
            x = left + placed.text_x + item.x
            self.draw_char(page, plan, x, baseline, shape)
            positions.append((x, plan))
            written.append(plan)

        self._draw_strikes(page, positions, baseline, shape)
        if (block.kind is BlockKind.HEADING and block.level <= self._style.underline_levels and positions):
            start = positions[0][0] - 0.1 * xh
            end = positions[-1][0] + positions[-1][1].ink_width + 0.1 * xh
            y = baseline + 0.32 * xh
            self._painter.line(page, (start, y + shape.dy(start)), (end, y + shape.dy(end)))
        return written

    def _draw_quote_bars(self, page: Image.Image, block: Block, baseline: float) -> None:
        pitch = self._geometry.line_pitch
        for depth in range(block.quote_depth):
            x = self._geometry.text_left + (depth * self._style.quote_indent + 0.35) * self._x_height
            self._painter.line(page, (x, baseline - 0.8 * pitch), (x, baseline + 0.2 * pitch), wobble=0.02)

    def _draw_marker(self, page: Image.Image, placed: _PlacedLine, x: float, baseline: float,
                     shape: _LineShape = _STRAIGHT) -> list[CharPlan]:
        block, xh = placed.block, self._x_height
        if placed.marker_plans:
            for plan in placed.marker_plans:
                self.draw_char(page, plan, x, baseline, shape)
                x += plan.advance
            # A handwritten bullet glyph is decoration; written numbers are content.
            return [] if block.kind is BlockKind.BULLET else list(placed.marker_plans)
        baseline += shape.dy(x)
        if block.kind is BlockKind.TASK:
            self._painter.checkbox(page, x + 0.1 * xh, baseline, self._style.checkbox_size * xh, block.checked)
        elif block.kind is BlockKind.BULLET:
            middle = baseline - 0.5 * xh
            if block.marker == "-":  # written as typed: a short dash
                self._painter.line(page, (x + 0.2 * xh, middle), (x + 0.85 * xh, middle))
            else:
                self._painter.dot(page, (x + 0.5 * xh, middle), self._style.bullet_radius * xh)
        return []

    def _draw_strikes(self, page: Image.Image, positions: list[tuple[float, CharPlan]], baseline: float,
                      shape: _LineShape = _STRAIGHT) -> None:
        """Hand-drawn lines through struck-out runs and under underlined ones."""
        for marked, height in ((lambda st: st.strike, 0.5), (lambda st: st.underline, -0.32)):
            run: list[tuple[float, CharPlan]] = []
            for entry in [*positions, None]:
                if entry is not None and marked(entry[1].style):
                    run.append(entry)
                    continue
                if run:
                    size = run[0][1].size
                    y = baseline - height * self._x_height * size
                    start = run[0][0] - 0.1 * self._x_height
                    end = run[-1][0] + run[-1][1].ink_width + 0.1 * self._x_height
                    self._painter.line(page, (start, y + shape.dy(start)), (end, y + shape.dy(end)))
                    run = []

    # ------------------------------------------------------------- characters

    def draw_char(self, page: Image.Image, plan: CharPlan, x: float, baseline: float,
                  shape: _LineShape = _STRAIGHT) -> None:
        baseline += shape.dy(x)
        if plan.glyph is not None:
            self._draw_glyph(page, plan, x, baseline + plan.baseline_offset, shape.tilt(x))
        elif self._settings.missing_policy is MissingGlyphPolicy.TYPED:
            font = get_font(max(6, round(self._x_height * plan.size / _FONT_X_HEIGHT_RATIO)))
            ImageDraw.Draw(page).text((x, baseline), plan.char, font=font,
                                      fill=self._settings.ink_color, anchor="ls")
        else:
            self._draw_placeholder(page, plan.char, x, baseline, plan.size)

    def _glyph_alpha(self, glyph: LoadedGlyph) -> np.ndarray:
        """The sample's ink coverage as a float array (cached per sample)."""
        key = id(glyph.image)
        alpha = self._alpha.get(key)
        if alpha is None:
            alpha = np.asarray(glyph.image.getchannel("A"), dtype=np.float32)
            self._alpha[key] = alpha
        return alpha

    def _draw_glyph(self, page: Image.Image, plan: CharPlan, x: float, baseline: float, tilt: float = 0.0) -> None:
        glyph = plan.glyph
        assert glyph is not None
        alpha = self._glyph_alpha(glyph)
        width = max(1, round(glyph.width * plan.scale))
        height = max(1, round(glyph.height * plan.scale))
        # Resample as few times as possible, so strokes stay crisp: one area-averaged
        # resize to the final size, then slant, rotation and shape warp in one step.
        shrinking = width < alpha.shape[1]
        sized = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)
        top = baseline + glyph.descent * plan.scale - height
        slant = plan.slant + (self._style.italic_slant if plan.style.italic else 0.0)
        ink, left, top_px = _place_glyph(sized, x, top, slant, plan.rotation + tilt, plan.warp, plan.warp_seed)
        if plan.ink < 1.0:
            ink = ink * plan.ink
        image = Image.new("RGBA", (ink.shape[1], ink.shape[0]), (*self._settings.ink_color, 0))
        image.putalpha(Image.fromarray(np.clip(ink, 0, 255).astype(np.uint8)))
        page.paste(image, (left, top_px), image)
        if plan.style.bold:
            # A second, slightly offset pass reads as a heavier pen stroke.
            offset = max(1, round(self._style.bold_offset * self._x_height * plan.size))
            page.paste(image, (left + offset, top_px), image)

    def _draw_placeholder(self, page: Image.Image, char: str, x: float, baseline: float, size: float) -> None:
        draw = ImageDraw.Draw(page)
        x_height = self._x_height * size
        box_width = _PLACEHOLDER_WIDTH * x_height
        box_height = _PLACEHOLDER_HEIGHT * x_height
        stroke = max(1, round(x_height * 0.06))
        draw.rectangle((round(x), round(baseline - box_height), round(x + box_width), round(baseline)),
                       outline=MISSING_GLYPH_COLOR, width=stroke)
        draw.text((x + box_width / 2, baseline - box_height / 2), char,
                  font=get_font(max(6, round(x_height * 0.9))), fill=MISSING_GLYPH_COLOR, anchor="mm")


def _place_glyph(alpha: np.ndarray, x: float, top: float, slant: float, rotation: float,
                 warp: float, seed: int) -> tuple[np.ndarray, int, int]:
    """Slant, rotate and reshape a glyph in a single resampling step.

    *alpha* is the glyph's ink coverage, already at its final size, whose
    top-left corner belongs at page position (*x*, *top*). *slant* shears the
    top to the right (positive) or left with the bottom fixed; *rotation* is in
    degrees counter-clockwise about the glyph's centre; *warp* (px) pushes the
    ink around with a smooth random displacement field so no two copies of a
    sample are identical. Sub-pixel positions are kept.

    Returns the transformed coverage and the integer page position of its
    top-left corner.
    """
    height, width = alpha.shape
    warped = warp >= _MIN_WARP_PX
    if abs(slant) * height < 0.5 and abs(rotation) <= 0.01 and not warped:
        return alpha, round(x), round(top)

    # Forward map (glyph -> page, relative to x/top): shear, then rotate about the centre.
    shear = np.array([[1.0, -slant, slant * height], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cx, cy = (width + slant * height) / 2, height / 2
    theta = math.radians(rotation)
    cos, sin = math.cos(theta), math.sin(theta)
    rotate = np.array([[cos, sin, cx - cos * cx - sin * cy],
                       [-sin, cos, cy + sin * cx - cos * cy],
                       [0.0, 0.0, 1.0]])
    forward = rotate @ shear
    corners = forward @ np.array([[0, width, width, 0], [0, 0, height, height], [1, 1, 1, 1]], dtype=float)

    pad = math.ceil(warp) + 2  # room for the displacement and the interpolation kernel
    left = math.floor(x + corners[0].min()) - pad
    top_px = math.floor(top + corners[1].min()) - pad
    out_w = math.ceil(x + corners[0].max()) + pad - left
    out_h = math.ceil(top + corners[1].max()) + pad - top_px

    # Inverse map, sampled at output pixel centres.
    inverse = np.linalg.inv(forward)
    gx, gy = np.meshgrid(np.arange(out_w, dtype=np.float64) + (left + 0.5 - x),
                         np.arange(out_h, dtype=np.float64) + (top_px + 0.5 - top))
    map_x = (inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2] - 0.5).astype(np.float32)
    map_y = (inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2] - 0.5).astype(np.float32)
    if warped:
        rng = np.random.default_rng(seed)
        coarse = np.clip(rng.normal(0.0, warp / 2, (2, _WARP_GRID, _WARP_GRID)), -warp, warp).astype(np.float32)
        map_x += cv2.resize(coarse[0], (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        map_y += cv2.resize(coarse[1], (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    out = cv2.remap(alpha, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(out, 0, 255), left, top_px
