"""Heuristic vertical metrics: where each glyph sits relative to the baseline.

Glyph images are cropped tightly to their ink, so the image alone does not say
whether a ``g`` hangs below the line or a ``-`` floats at mid height. This
module encodes simple, typographic character classes for that.

It is intentionally isolated so it can be improved (or replaced by learned
metrics) without touching extraction or rendering. All public functions work
in *source pixels* of the stored glyph image.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from enum import Enum


class Anchor(str, Enum):
    """How a glyph is positioned vertically."""

    BASELINE = "baseline"  # ink bottom rests on the baseline
    DESCENDER = "descender"  # part of the ink hangs below the baseline
    CENTER = "center"  # vertically centred at a given height (e.g. "-")
    TOP = "top"  # top edge placed at a given height (e.g. quotes)


@dataclass(frozen=True)
class VerticalRule:
    """Typical geometry of a character, in units of the x-height.

    Attributes:
        anchor: positioning strategy.
        height_xh: typical ink height.
        position_xh: CENTER: height of the glyph centre above the baseline;
            TOP: height of the glyph top above the baseline (may be negative).
        descent_fraction: DESCENDER: default fraction of the glyph below the baseline.
        top_at_x_height: DESCENDER: the glyph's top normally sits at the x-height
            line (g, p, q, y), which allows a better estimate than a fixed fraction.
        width_xh: for flat glyphs (``-``, ``_``) the width is a more reliable size
            cue than the height, which is just the stroke thickness.
    """

    anchor: Anchor
    height_xh: float
    position_xh: float = 0.0
    descent_fraction: float = 0.0
    top_at_x_height: bool = False
    width_xh: float | None = None


def _rules(chars: str, rule: VerticalRule) -> dict[str, VerticalRule]:
    return {c: rule for c in chars}


_RULES: dict[str, VerticalRule] = {
    **_rules("acemnorsuvwxz", VerticalRule(Anchor.BASELINE, 1.0)),
    **_rules("i", VerticalRule(Anchor.BASELINE, 1.4)),
    **_rules("t", VerticalRule(Anchor.BASELINE, 1.35)),
    **_rules("bdfhkl", VerticalRule(Anchor.BASELINE, 1.65)),
    **_rules("gpqy", VerticalRule(Anchor.DESCENDER, 1.6, descent_fraction=0.38, top_at_x_height=True)),
    **_rules("j", VerticalRule(Anchor.DESCENDER, 1.95, descent_fraction=0.33)),
    **_rules("ABCDEFGHIJKLMNOPQRSTUVWXYZ", VerticalRule(Anchor.BASELINE, 1.6)),
    **_rules("0123456789", VerticalRule(Anchor.BASELINE, 1.55)),
    **_rules("&#%?!", VerticalRule(Anchor.BASELINE, 1.55)),
    **_rules("@", VerticalRule(Anchor.DESCENDER, 1.5, descent_fraction=0.08)),
    **_rules("$", VerticalRule(Anchor.DESCENDER, 1.9, descent_fraction=0.12)),
    **_rules("/\\", VerticalRule(Anchor.BASELINE, 1.75)),
    **_rules("()[]{}|", VerticalRule(Anchor.DESCENDER, 2.0, descent_fraction=0.17)),
    ".": VerticalRule(Anchor.BASELINE, 0.2),
    ":": VerticalRule(Anchor.BASELINE, 0.9),
    ",": VerticalRule(Anchor.DESCENDER, 0.45, descent_fraction=0.5),
    ";": VerticalRule(Anchor.DESCENDER, 1.05, descent_fraction=0.25),
    **_rules("'\"`", VerticalRule(Anchor.TOP, 0.5, position_xh=1.65)),
    "^": VerticalRule(Anchor.TOP, 0.55, position_xh=1.65),
    "*": VerticalRule(Anchor.CENTER, 0.7, position_xh=1.2),
    "-": VerticalRule(Anchor.CENTER, 0.1, position_xh=0.5, width_xh=0.6),
    "~": VerticalRule(Anchor.CENTER, 0.25, position_xh=0.55, width_xh=0.75),
    "=": VerticalRule(Anchor.CENTER, 0.45, position_xh=0.5, width_xh=0.7),
    "+": VerticalRule(Anchor.CENTER, 0.85, position_xh=0.55),
    **_rules("<>", VerticalRule(Anchor.CENTER, 0.9, position_xh=0.55)),
    "_": VerticalRule(Anchor.TOP, 0.1, position_xh=-0.2, width_xh=0.9),
    # Common keyboard symbols beyond ASCII (see charset.COMMON_SYMBOLS).
    "•": VerticalRule(Anchor.CENTER, 0.35, position_xh=0.5),
    "·": VerticalRule(Anchor.CENTER, 0.2, position_xh=0.5),
    **_rules("→←", VerticalRule(Anchor.CENTER, 0.55, position_xh=0.5, width_xh=1.3)),
    **_rules("↑↓", VerticalRule(Anchor.BASELINE, 1.6)),
    "°": VerticalRule(Anchor.TOP, 0.45, position_xh=1.65),
    "±": VerticalRule(Anchor.BASELINE, 1.1),
    "×": VerticalRule(Anchor.CENTER, 0.65, position_xh=0.5),
    "÷": VerticalRule(Anchor.CENTER, 0.8, position_xh=0.5),
    "≈": VerticalRule(Anchor.CENTER, 0.45, position_xh=0.5, width_xh=0.75),
    "≠": VerticalRule(Anchor.CENTER, 0.8, position_xh=0.5),
    **_rules("≤≥", VerticalRule(Anchor.BASELINE, 1.05)),
    "√": VerticalRule(Anchor.BASELINE, 1.7),
    "∞": VerticalRule(Anchor.CENTER, 0.6, position_xh=0.5, width_xh=1.3),
    "µ": VerticalRule(Anchor.DESCENDER, 1.5, descent_fraction=0.33),
    "π": VerticalRule(Anchor.BASELINE, 1.0),
    **_rules("€£¥©®", VerticalRule(Anchor.BASELINE, 1.6)),
    "¢": VerticalRule(Anchor.DESCENDER, 1.6, descent_fraction=0.15),
    "™": VerticalRule(Anchor.TOP, 0.6, position_xh=1.65),
    "§": VerticalRule(Anchor.DESCENDER, 1.9, descent_fraction=0.12),
    "¶": VerticalRule(Anchor.DESCENDER, 1.9, descent_fraction=0.17),
    "…": VerticalRule(Anchor.BASELINE, 0.2, width_xh=1.2),
    "–": VerticalRule(Anchor.CENTER, 0.1, position_xh=0.5, width_xh=0.9),
    "—": VerticalRule(Anchor.CENTER, 0.1, position_xh=0.5, width_xh=1.6),
    **_rules("“”‘’", VerticalRule(Anchor.TOP, 0.5, position_xh=1.65)),
    **_rules("«»", VerticalRule(Anchor.CENTER, 0.6, position_xh=0.5)),
    **_rules("¿¡", VerticalRule(Anchor.DESCENDER, 1.55, descent_fraction=0.3)),
    "ß": VerticalRule(Anchor.BASELINE, 1.65),
    "æ": VerticalRule(Anchor.BASELINE, 1.0, width_xh=1.5),
    "ø": VerticalRule(Anchor.BASELINE, 1.1),
}

# Combining marks drawn below the letter (cedilla, ogonek, dot below, …).
_MARKS_BELOW = frozenset("\u0323\u0324\u0325\u0326\u0327\u0328\u0331")
_ACCENT_XH = 0.4  # height an accent adds above or below its letter

# Lowercase letters whose height defines the x-height of a handwriting sample.
X_HEIGHT_CHARS = frozenset("acemnorsuvwxz")

# Plausible range of "fraction below baseline" when trusting a sheet guide line.
_GUIDE_DESCENT_RANGE = (0.08, 0.75)
# Maximum disagreement (in x-heights) between guide and heuristic for floating glyphs.
_GUIDE_TOLERANCE_XH = 0.6


def rule_for(char: str) -> VerticalRule:
    """Return the vertical rule for *char*, with a generic fallback."""
    rule = _RULES.get(char)
    if rule is not None:
        return rule
    accented = _accented_rule(char)
    if accented is not None:
        return accented
    if char.islower():
        return VerticalRule(Anchor.BASELINE, 1.35)
    if char.isupper() or char.isdigit():
        return VerticalRule(Anchor.BASELINE, 1.6)
    return VerticalRule(Anchor.BASELINE, 1.0)


def _accented_rule(char: str) -> VerticalRule | None:
    """Rule for an accented letter (é, ç, Ü, …): its base letter plus room for the accent."""
    base, *marks = unicodedata.normalize("NFD", char)
    rule = _RULES.get(base)
    if not marks or rule is None or rule.anchor is not Anchor.BASELINE:
        return None
    height = rule.height_xh + _ACCENT_XH
    if any(mark in _MARKS_BELOW for mark in marks):
        return VerticalRule(Anchor.DESCENDER, height, descent_fraction=_ACCENT_XH / height)
    if base in "ij":  # the accent replaces the dot
        height = 1.0 + _ACCENT_XH
    return replace(rule, height_xh=height, width_xh=None)


def estimate_x_height_ref(char: str, ink_width: float, ink_height: float) -> float:
    """Estimate the writer's x-height (source px) from a single glyph.

    Used when a sample has no better reference, e.g. a manually uploaded crop.
    """
    rule = rule_for(char)
    if rule.width_xh is not None and ink_width > 0:
        estimate = ink_width / rule.width_xh
    else:
        estimate = ink_height / rule.height_xh
    return max(estimate, 1.0)


def _heuristic_descent(rule: VerticalRule, height: float, x_height: float) -> float:
    if rule.anchor is Anchor.BASELINE:
        return 0.0
    if rule.anchor is Anchor.DESCENDER:
        if rule.top_at_x_height:
            descent = height - x_height
            low, high = _GUIDE_DESCENT_RANGE
            if low * height <= descent <= high * height:
                return descent
        return rule.descent_fraction * height
    if rule.anchor is Anchor.CENTER:
        return height / 2 - rule.position_xh * x_height
    # Anchor.TOP
    return height - rule.position_xh * x_height


def descent_px(
    char: str,
    glyph_height: float,
    x_height_ref: float,
    guide_baseline_from_top: float | None = None,
) -> float:
    """Distance from the baseline down to the glyph's bottom edge (source px).

    Positive values hang below the baseline (``g``), zero rests on it (``a``),
    negative values float above it (``-``, ``'``).

    Args:
        char: the character the glyph represents.
        glyph_height: height of the cropped glyph image.
        x_height_ref: the writer's x-height in the same pixel scale.
        guide_baseline_from_top: baseline position measured from the top of the
            glyph, when known from a sample-sheet guide. Only used when it is
            plausible, because people do not always write exactly on the guide.
    """
    rule = rule_for(char)
    heuristic = _heuristic_descent(rule, glyph_height, x_height_ref)
    if guide_baseline_from_top is None or rule.anchor is Anchor.BASELINE:
        return heuristic

    guide_descent = glyph_height - guide_baseline_from_top
    if rule.anchor is Anchor.DESCENDER:
        low, high = _GUIDE_DESCENT_RANGE
        if low * glyph_height <= guide_descent <= high * glyph_height:
            return guide_descent
        return heuristic
    if abs(guide_descent - heuristic) <= _GUIDE_TOLERANCE_XH * x_height_ref:
        return guide_descent
    return heuristic
