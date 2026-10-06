"""Centralised rendering settings, defaults and presets.

All tunable numbers used by the renderer live here so they can be adjusted in
one place. Sizes are expressed in millimetres (physical units) or as fractions
of the handwriting x-height (so variation scales with handwriting size).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum

MM_PER_INCH = 25.4
DEFAULT_DPI = 300


def mm_to_px(mm: float, dpi: int) -> float:
    """Convert millimetres to pixels at *dpi*."""
    return mm / MM_PER_INCH * dpi


class PageFormat(str, Enum):
    """Supported paper sizes (portrait)."""

    LETTER = "US Letter"
    A4 = "A4"

    @property
    def size_mm(self) -> tuple[float, float]:
        return _PAGE_SIZES_MM[self]

    def size_px(self, dpi: int) -> tuple[int, int]:
        width_mm, height_mm = self.size_mm
        return round(mm_to_px(width_mm, dpi)), round(mm_to_px(height_mm, dpi))


_PAGE_SIZES_MM: dict[PageFormat, tuple[float, float]] = {
    PageFormat.LETTER: (215.9, 279.4),
    PageFormat.A4: (210.0, 297.0),
}


class PaperStyle(str, Enum):
    """Programmatically generated paper backgrounds."""

    BLANK = "Blank white"
    COLLEGE_RULED = "College ruled"
    NARROW_RULED = "Narrow ruled"
    WIDE_RULED = "Wide ruled"
    GRAPH = "Graph paper"

    @property
    def is_ruled(self) -> bool:
        return self in RULE_SPACING_MM

    @property
    def has_margin_rule(self) -> bool:
        """Narrow-ruled sheets have no red margin line."""
        return self in (PaperStyle.COLLEGE_RULED, PaperStyle.WIDE_RULED)


# Standard US notebook rulings: narrow = 1/4", college = 9/32", wide = 11/32".
RULE_SPACING_MM: dict[PaperStyle, float] = {
    PaperStyle.NARROW_RULED: 1 / 4 * MM_PER_INCH,
    PaperStyle.COLLEGE_RULED: 9 / 32 * MM_PER_INCH,
    PaperStyle.WIDE_RULED: 11 / 32 * MM_PER_INCH,
}
GRAPH_SQUARE_MM = 5.0


@dataclass(frozen=True)
class PaperColors:
    """Colours used when drawing paper backgrounds."""

    background: tuple[int, int, int] = (255, 255, 255)
    rule: tuple[int, int, int] = (160, 190, 225)
    margin_rule: tuple[int, int, int] = (228, 120, 128)
    grid: tuple[int, int, int] = (190, 210, 232)


RULE_THICKNESS_MM = 0.25
GRAPH_LINE_THICKNESS_MM = 0.18


INK_COLORS: dict[str, tuple[int, int, int]] = {
    "Black": (24, 24, 30),
    "Dark blue": (20, 42, 116),
    "Blue": (32, 72, 168),
}
DEFAULT_INK = "Black"
MISSING_GLYPH_COLOR: tuple[int, int, int] = (205, 45, 45)


class VariantMode(str, Enum):
    """How a sample is picked when a character has several variants."""

    SHUFFLE = "Shuffle (use every variant before repeating)"
    RANDOM = "Random (never the same twice in a row)"
    FIRST = "Always the first sample"


class MissingGlyphPolicy(str, Enum):
    """What to do with characters that have no handwriting samples."""

    PLACEHOLDER = "Placeholder box (shows the missing character)"
    TYPED = "Typed fallback (render with a regular font)"
    ERROR = "Stop with an error"


@dataclass(frozen=True)
class VariationSettings:
    """Natural-variation parameters.

    Jitter values are *maximum* magnitudes; actual offsets are drawn from a
    clipped normal distribution so most characters vary only slightly.
    Fractions are relative to the rendered x-height.

    Independent per-letter jitter alone looks "neat but shaky". Real notes
    are messy in a *correlated* way, which the word, line and pen settings
    model: whole words bob, lean, change size and spread out together, lines
    drift uphill and wander off the rule, indents and line ends are uneven,
    every copy of a letter is shaped a little differently, ink pressure varies,
    and writing gets sloppier as it goes.
    """

    variant_mode: VariantMode = VariantMode.SHUFFLE
    # Letters
    rotation_deg: float = 1.5
    baseline_jitter: float = 0.04
    scale_jitter: float = 0.03
    letter_spacing_jitter: float = 0.05
    letter_overlap: float = 0.03  # neighbouring letters may touch or overlap by up to this much
    shape_warp: float = 0.02  # random distortion of each letter's shape
    # Words
    word_spacing_jitter: float = 0.15
    word_baseline: float = 0.03  # whole word sits higher or lower
    word_scale: float = 0.02  # whole word written larger or smaller
    word_tilt_deg: float = 0.6  # whole word rotated as a unit
    slant_jitter: float = 0.04  # whole word leans more or less (shear)
    size_drift: float = 0.02  # handwriting size wanders slowly
    # Lines
    line_slope_deg: float = 0.2  # line runs uphill or downhill
    line_wave: float = 0.025  # baseline wanders up and down along the line
    line_offset: float = 0.04  # line floats above or sinks below the rule
    margin_jitter: float = 0.15  # lines start at slightly different x
    # Layout (x-heights)
    indent_jitter: float = 0.25  # list items and wrapped lines don't line up exactly
    ragged_right: float = 1.5  # lines end up to this far before the right margin
    # Pen
    ink_fade: float = 0.06  # pressure: some words are lighter
    fatigue: float = 0.0  # extra messiness by the end of the text (0.5 = +50%)


# All variation switched off: every copy of a sample is placed exactly.
NO_VARIATION = VariationSettings(**{f.name: 0.0 for f in fields(VariationSettings) if f.name != "variant_mode"})

VARIATION_PRESETS: dict[str, VariationSettings] = {
    "Very Consistent": replace(
        NO_VARIATION,
        rotation_deg=0.4,
        baseline_jitter=0.012,
        scale_jitter=0.01,
        letter_spacing_jitter=0.015,
        word_spacing_jitter=0.04,
        margin_jitter=0.05,
        ink_fade=0.03,
    ),
    "Natural": VariationSettings(),
    "Messy": VariationSettings(
        variant_mode=VariantMode.RANDOM,
        rotation_deg=2.0,
        baseline_jitter=0.06,
        scale_jitter=0.05,
        letter_spacing_jitter=0.10,
        letter_overlap=0.08,
        shape_warp=0.05,
        word_spacing_jitter=0.30,
        word_baseline=0.10,
        word_scale=0.07,
        word_tilt_deg=1.2,
        slant_jitter=0.10,
        size_drift=0.07,
        line_slope_deg=0.7,
        line_wave=0.07,
        line_offset=0.10,
        margin_jitter=0.45,
        indent_jitter=0.6,
        ragged_right=4.0,
        ink_fade=0.18,
        fatigue=0.3,
    ),
    "Rushed notes": VariationSettings(
        variant_mode=VariantMode.RANDOM,
        rotation_deg=3.0,
        baseline_jitter=0.09,
        scale_jitter=0.08,
        letter_spacing_jitter=0.16,
        letter_overlap=0.14,
        shape_warp=0.08,
        word_spacing_jitter=0.45,
        word_baseline=0.16,
        word_scale=0.12,
        word_tilt_deg=2.0,
        slant_jitter=0.18,
        size_drift=0.12,
        line_slope_deg=1.1,
        line_wave=0.11,
        line_offset=0.16,
        margin_jitter=0.8,
        indent_jitter=1.0,
        ragged_right=7.0,
        ink_fade=0.28,
        fatigue=0.5,
    ),
}
DEFAULT_PRESET = "Natural"


@dataclass(frozen=True)
class Margins:
    """Page margins in millimetres."""

    left: float = 25.0
    right: float = 20.0
    top: float = 22.0
    bottom: float = 18.0


@dataclass(frozen=True)
class PageSettings:
    """Physical page configuration."""

    page_format: PageFormat = PageFormat.LETTER
    paper_style: PaperStyle = PaperStyle.COLLEGE_RULED
    margins: Margins = field(default_factory=Margins)
    dpi: int = DEFAULT_DPI
    # Line pitch for blank / graph paper.
    line_spacing_mm: float = 8.5
    # For ruled paper: write on every Nth ruled line.
    ruled_line_step: int = 1
    show_margin_rule: bool = True
    paper_colors: PaperColors = field(default_factory=PaperColors)


@dataclass(frozen=True)
class MarkdownStyle:
    """How Markdown formatting is drawn. Sizes are in x-heights unless noted."""

    # Size multiplier and extra ruled lines above, for heading levels 1-6.
    heading_scale: tuple[float, ...] = (1.7, 1.4, 1.2, 1.1, 1.0, 1.0)
    heading_space_before: tuple[int, ...] = (1, 1, 1, 0, 0, 0)
    # Headings of this level or higher-priority (1..n) are underlined by hand (0: none).
    underline_levels: int = 0
    list_indent: float = 3.0  # per nesting level
    bullet_column: float = 1.6  # room reserved for a bullet or checkbox
    bullet_radius: float = 0.22
    checkbox_size: float = 0.95
    marker_gap: float = 0.55  # gap after a written number such as "3."
    quote_indent: float = 1.3
    bold_offset: float = 0.06  # offset of the second pen pass for **bold**
    italic_slant: float = 0.22  # horizontal shear for *italic*
    stroke_width: float = 0.085  # drawn lines: bullets, boxes, rules, strikes


@dataclass(frozen=True)
class RenderSettings:
    """Everything the renderer needs besides the text and the glyphs."""

    page: PageSettings = field(default_factory=PageSettings)
    variation: VariationSettings = field(default_factory=VariationSettings)
    ink_color: tuple[int, int, int] = INK_COLORS[DEFAULT_INK]
    # Height of a lowercase "x" on the page.
    x_height_mm: float = 2.9
    # Extra space after each character, as a fraction of the x-height.
    letter_spacing: float = 0.04
    # Width of a space character, as a fraction of the x-height.
    word_spacing: float = 0.85
    missing_policy: MissingGlyphPolicy = MissingGlyphPolicy.PLACEHOLDER
    seed: int | None = None
    # Interpret the text as Markdown (headings, lists, bold, ...).
    markdown: bool = False
    markdown_style: MarkdownStyle = field(default_factory=MarkdownStyle)

    def with_seed(self, seed: int | None) -> RenderSettings:
        return replace(self, seed=seed)


# Layout constants (fractions of the x-height).
TAB_WIDTH_SPACES = 4
RULED_TEXT_INDENT_MM = 1.5  # gap between the red margin line and the text
DESCENDER_ALLOWANCE = 0.75  # room kept below the last baseline on a page
ASCENDER_ALLOWANCE = 1.7  # room kept above the first baseline on blank paper
