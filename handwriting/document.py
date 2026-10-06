"""Structured document model shared by plain-text and Markdown input.

The renderer works on a list of :class:`Block` objects. Plain text becomes one
paragraph block per line (preserving the text exactly); Markdown input is
parsed into headings, list items, quotes, rules and styled spans by
:mod:`handwriting.markdown`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from .layout import normalize_text


@dataclass(frozen=True)
class Style:
    """Inline formatting of a run of text."""

    bold: bool = False
    italic: bool = False
    strike: bool = False
    code: bool = False


PLAIN = Style()


@dataclass(frozen=True)
class Span:
    text: str
    style: Style = PLAIN


class BlockKind(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    BULLET = "bullet"
    NUMBERED = "numbered"
    TASK = "task"
    RULE = "rule"
    CODE = "code"


LIST_KINDS = frozenset({BlockKind.BULLET, BlockKind.NUMBERED, BlockKind.TASK})


@dataclass(frozen=True)
class Block:
    """One source line's worth of content.

    Attributes:
        level: heading level (1–6) for headings.
        indent: nesting depth for list items and indented paragraphs.
        marker: the written list marker for numbered items (e.g. ``"3."``), or
            the bullet character typed (``-``, ``*`` or ``+``).
        checked: state of a task-list checkbox.
        quote_depth: number of ``>`` quote levels around the block.
    """

    kind: BlockKind
    spans: tuple[Span, ...] = ()
    level: int = 0
    indent: int = 0
    marker: str = ""
    checked: bool = False
    quote_depth: int = 0

    @property
    def text(self) -> str:
        return "".join(span.text for span in self.spans)

    def styles(self) -> list[Style]:
        """The style of every character of :attr:`text`."""
        return [span.style for span in self.spans for _ in span.text]


def merge_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Join neighbouring spans with the same style and drop empty ones."""
    merged: list[Span] = []
    for span in spans:
        if not span.text:
            continue
        if merged and merged[-1].style == span.style:
            merged[-1] = Span(merged[-1].text + span.text, span.style)
        else:
            merged.append(span)
    return tuple(merged)


def plain_blocks(text: str) -> list[Block]:
    """Plain text: one unformatted paragraph per line, text kept exactly."""
    return [Block(BlockKind.PARAGRAPH, merge_spans([Span(line)]))
            for line in normalize_text(text).split("\n")]


def visible_text(blocks: Sequence[Block]) -> str:
    """Every character that will be written, in reading order (numbers of numbered items included).

    Bullets are decoration: drawn by hand when the profile has no sample for them.
    """
    return "".join((block.marker if block.kind is BlockKind.NUMBERED else "") + block.text for block in blocks)
