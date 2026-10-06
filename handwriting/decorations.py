"""Hand-drawn marks used by Markdown formatting.

Bullets, checkboxes, rules, underlines, strike-throughs and quote bars are
drawn in the ink colour with a slight, seeded wobble so they look drawn by
hand rather than ruled. Shapes are rendered on a supersampled mask and then
downsampled, so their edges are anti-aliased like the handwriting glyphs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from PIL import Image, ImageDraw

Point = tuple[float, float]

_SUPERSAMPLE = 4
_SEGMENT_LENGTH = 0.9  # x-heights per segment of a wobbly line


class InkPainter:
    """Draws anti-aliased ink shapes onto a page."""

    def __init__(self, color: tuple[int, int, int], x_height: float, stroke: float, rng: random.Random) -> None:
        self.color = color
        self.x_height = x_height
        self.stroke = max(1.0, stroke)
        self.rng = rng

    # ------------------------------------------------------------ primitives

    def _paint(self, page: Image.Image, points: Sequence[Point],
               draw_fn: Callable[[ImageDraw.ImageDraw, Callable[[Point], Point]], None]) -> None:
        """Run *draw_fn* on a supersampled mask covering *points*, then ink it in."""
        margin = self.stroke * 2 + 2
        left = math.floor(min(p[0] for p in points) - margin)
        top = math.floor(min(p[1] for p in points) - margin)
        right = math.ceil(max(p[0] for p in points) + margin)
        bottom = math.ceil(max(p[1] for p in points) + margin)
        width, height = max(1, right - left), max(1, bottom - top)
        mask = Image.new("L", (width * _SUPERSAMPLE, height * _SUPERSAMPLE), 0)

        def to_mask(point: Point) -> Point:
            return (point[0] - left) * _SUPERSAMPLE, (point[1] - top) * _SUPERSAMPLE

        draw_fn(ImageDraw.Draw(mask), to_mask)
        mask = mask.resize((width, height), Image.Resampling.BOX)
        page.paste(Image.new("RGB", (width, height), self.color), (left, top), mask)

    def stroke_path(self, page: Image.Image, points: Sequence[Point], width: float | None = None) -> None:
        """A rounded polyline."""
        line_width = (width or self.stroke) * _SUPERSAMPLE

        def draw(d: ImageDraw.ImageDraw, at: Callable[[Point], Point]) -> None:
            mapped = [at(p) for p in points]
            d.line(mapped, fill=255, width=max(1, round(line_width)), joint="curve")
            r = line_width / 2
            for x, y in (mapped[0], mapped[-1]):
                d.ellipse((x - r, y - r, x + r, y + r), fill=255)

        self._paint(page, points, draw)

    def wobbly_line(self, start: Point, end: Point, wobble: float = 0.05) -> list[Point]:
        """Points along a gently uneven line from *start* to *end*."""
        length = math.dist(start, end)
        if length == 0:
            return [start, end]
        segments = max(2, round(length / (_SEGMENT_LENGTH * self.x_height)))
        dx, dy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        normal = (-dy, dx)
        amplitude = wobble * self.x_height
        offset, points = 0.0, []
        for i in range(segments + 1):
            t = i / segments
            if 0 < i < segments:
                offset = 0.6 * offset + self.rng.uniform(-amplitude, amplitude)
            else:
                offset = 0.0
            points.append((start[0] + (end[0] - start[0]) * t + normal[0] * offset,
                           start[1] + (end[1] - start[1]) * t + normal[1] * offset))
        return points

    # ---------------------------------------------------------------- shapes

    def line(self, page: Image.Image, start: Point, end: Point, wobble: float = 0.05) -> None:
        if math.dist(start, end) < 1:
            return
        tilt = self.rng.uniform(-0.04, 0.04) * self.x_height
        self.stroke_path(page, self.wobbly_line(start, (end[0], end[1] + tilt), wobble))

    def dot(self, page: Image.Image, center: Point, radius: float) -> None:
        cx, cy = center
        rx, ry = radius * self.rng.uniform(0.9, 1.1), radius * self.rng.uniform(0.85, 1.05)

        def draw(d: ImageDraw.ImageDraw, at: Callable[[Point], Point]) -> None:
            x0, y0 = at((cx - rx, cy - ry))
            x1, y1 = at((cx + rx, cy + ry))
            d.ellipse((x0, y0, x1, y1), fill=255)

        self._paint(page, [(cx - rx, cy - ry), (cx + rx, cy + ry)], draw)

    def checkbox(self, page: Image.Image, left: float, baseline: float, size: float, checked: bool) -> None:
        jitter = 0.04 * size
        top = baseline - size
        corners = [(left + self.rng.uniform(-jitter, jitter), top + self.rng.uniform(-jitter, jitter)),
                   (left + size + self.rng.uniform(-jitter, jitter), top + self.rng.uniform(-jitter, jitter)),
                   (left + size + self.rng.uniform(-jitter, jitter), baseline + self.rng.uniform(-jitter, jitter)),
                   (left + self.rng.uniform(-jitter, jitter), baseline + self.rng.uniform(-jitter, jitter))]
        self.stroke_path(page, [*corners, corners[0]])
        if checked:
            tick = [(left + 0.18 * size, baseline - 0.5 * size),
                    (left + 0.42 * size, baseline - 0.18 * size),
                    (left + 0.95 * size, baseline - 1.05 * size)]
            self.stroke_path(page, tick, width=self.stroke * 1.25)
