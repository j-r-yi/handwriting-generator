"""Programmatically drawn paper backgrounds (no external images)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from .layout import PageGeometry
from .settings import GRAPH_LINE_THICKNESS_MM, RULE_THICKNESS_MM, PageSettings, PaperStyle, mm_to_px


def render_paper(geometry: PageGeometry, page: PageSettings) -> Image.Image:
    """Draw a blank, ruled or graph page as an RGB image.

    Ruled lines come from *geometry*, the same object the renderer uses for
    baselines, so handwriting sits on the rules.
    """
    colors = page.paper_colors
    image = Image.new("RGB", (geometry.width, geometry.height), colors.background)
    if page.paper_style is PaperStyle.BLANK:
        return image

    draw = ImageDraw.Draw(image)
    if page.paper_style is PaperStyle.GRAPH:
        thickness = max(1, round(mm_to_px(GRAPH_LINE_THICKNESS_MM, geometry.dpi)))
        for y in geometry.rule_ys:
            _hline(draw, y, 0, geometry.width, thickness, colors.grid)
        for x in geometry.grid_xs:
            _vline(draw, x, 0, geometry.height, thickness, colors.grid)
        return image

    thickness = max(1, round(mm_to_px(RULE_THICKNESS_MM, geometry.dpi)))
    for y in geometry.rule_ys:
        _hline(draw, y, 0, geometry.width, thickness, colors.rule)
    if geometry.margin_rule_x is not None:
        _vline(draw, geometry.margin_rule_x, 0, geometry.height, thickness, colors.margin_rule)
    return image


def _hline(draw: ImageDraw.ImageDraw, y: float, x0: float, x1: float, thickness: int,
           color: tuple[int, int, int]) -> None:
    top = round(y - thickness / 2)
    draw.rectangle((round(x0), top, round(x1), top + thickness - 1), fill=color)


def _vline(draw: ImageDraw.ImageDraw, x: float, y0: float, y1: float, thickness: int,
           color: tuple[int, int, int]) -> None:
    left = round(x - thickness / 2)
    draw.rectangle((left, round(y0), left + thickness - 1, round(y1)), fill=color)
