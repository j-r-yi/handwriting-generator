"""A small, predictable Markdown parser for handwritten notes.

Supported (a practical subset of CommonMark/GitHub Markdown):

* headings ``#`` … ``######``
* bullet lists ``-``/``*``/``+``, numbered lists ``1.``/``1)``, task lists
  ``- [ ]``/``- [x]``; nesting by indentation (2 spaces or a tab per level)
* block quotes ``>`` (may contain headings and lists)
* horizontal rules ``---``, ``***``, ``___``
* fenced code blocks (written literally)
* inline ``**bold**``/``__bold__``, ``*italic*``/``_italic_``,
  ``***both***``, ``~~strikethrough~~``, ```code```, ``[links](url)``
  (only the link text is written), ``![images](url)`` (alt text),
  ``<https://autolinks>`` and backslash escapes.

Unlike standard Markdown, every line break is kept (as in handwritten notes):
a single newline starts a new line. As in standard Markdown, a run of blank
lines is one gap, and blank lines at the start or end, or closing a quote,
take no space. Fenced code is kept exactly. Unrecognised syntax is simply
written as text, so nothing is lost.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable
from dataclasses import replace

from .document import PLAIN, Block, BlockKind, Span, Style, merge_spans
from .layout import normalize_text

MAX_INDENT = 6
_SPACES_PER_LEVEL = 2
_TAB_SPACES = 4

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_RULE = re.compile(r"^ {0,3}([-*_])(?: *\1){2,} *$")
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_TASK = re.compile(r"^([ \t]*)[-*+][ \t]+\[([ xX])\](?:[ \t]+(.*))?$")
_BULLET = re.compile(r"^([ \t]*)([-*+])(?:[ \t]+(.*))?$")
_NUMBERED = re.compile(r"^([ \t]*)(\d{1,9})([.)])(?:[ \t]+(.*))?$")
_AUTOLINK = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.-]{1,31}:[^\s<>]*|[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+)>")
_ESCAPABLE = set(string.punctuation)
StyleChange = Callable[[Style], Style]


def parse_markdown(text: str) -> list[Block]:
    """Parse Markdown into blocks, one per source line (fence lines excluded)."""
    blocks: list[Block] = []
    fence: str | None = None
    for line in normalize_text(text).split("\n"):
        if fence is not None:
            if line.strip().startswith(fence):
                fence = None
            else:
                blocks.append(Block(BlockKind.CODE, merge_spans([Span(line, Style(code=True))])))
            continue
        opening = _FENCE.match(line)
        if opening:
            fence = opening.group(1)[:3]
            continue
        blocks.append(_parse_line(line))
    return _tidy_blank_lines(blocks)


def _is_blank(block: Block) -> bool:
    return block.kind is BlockKind.PARAGRAPH and not block.spans


def _tidy_blank_lines(blocks: list[Block]) -> list[Block]:
    """Collapse runs of blank lines into one gap and drop blanks that add no meaning."""
    tidy: list[Block] = []
    for index, block in enumerate(blocks):
        if _is_blank(block) and block.quote_depth:
            following = blocks[index + 1] if index + 1 < len(blocks) else None
            if following is None or following.quote_depth < block.quote_depth or _is_blank(following):
                block = Block(BlockKind.PARAGRAPH)  # an empty "> " line that closes a quote
        if _is_blank(block) and (not tidy or _is_blank(tidy[-1])):
            continue  # leading blank line, or one more in a row
        tidy.append(block)
    while tidy and _is_blank(tidy[-1]):
        tidy.pop()
    return tidy


def _indent_level(whitespace: str) -> int:
    width = len(whitespace.replace("\t", " " * _TAB_SPACES))
    return min(MAX_INDENT, width // _SPACES_PER_LEVEL)


def _strip_quotes(line: str) -> tuple[str, int]:
    depth = 0
    while True:
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) > 3 or not stripped.startswith(">"):
            return line, depth
        line = stripped[1:]
        if line.startswith(" "):
            line = line[1:]
        depth += 1


def _parse_line(line: str) -> Block:
    line, quote_depth = _strip_quotes(line)
    block = _parse_content(line)
    return replace(block, quote_depth=quote_depth) if quote_depth else block


def _parse_content(line: str) -> Block:
    if not line.strip():
        return Block(BlockKind.PARAGRAPH)
    if _RULE.match(line):
        return Block(BlockKind.RULE)
    if match := _HEADING.match(line):
        return Block(BlockKind.HEADING, parse_inline(match.group(2) or ""), level=len(match.group(1)))
    if match := _TASK.match(line):
        return Block(BlockKind.TASK, parse_inline(match.group(3) or ""), indent=_indent_level(match.group(1)),
                     checked=match.group(2) in "xX")
    if match := _BULLET.match(line):
        return Block(BlockKind.BULLET, parse_inline(match.group(3) or ""), indent=_indent_level(match.group(1)),
                     marker=match.group(2))
    if match := _NUMBERED.match(line):
        return Block(BlockKind.NUMBERED, parse_inline(match.group(4) or ""),
                     indent=_indent_level(match.group(1)), marker=match.group(2) + match.group(3))
    stripped = line.lstrip(" \t")
    return Block(BlockKind.PARAGRAPH, parse_inline(stripped), indent=_indent_level(line[:len(line) - len(stripped)]))


# --------------------------------------------------------------------- inline

def parse_inline(text: str, style: Style = PLAIN) -> tuple[Span, ...]:
    """Parse inline formatting into styled spans."""
    return merge_spans(_InlineParser(text).parse(0, len(text), style))


class _InlineParser:
    def __init__(self, text: str) -> None:
        self.text = text

    def parse(self, start: int, end: int, style: Style) -> list[Span]:
        text = self.text
        spans: list[Span] = []
        literal: list[str] = []

        def flush() -> None:
            if literal:
                spans.append(Span("".join(literal), style))
                literal.clear()

        i = start
        while i < end:
            char = text[i]
            if char == "\\" and i + 1 < end and text[i + 1] in _ESCAPABLE:
                literal.append(text[i + 1])
                i += 2
                continue
            if char == "`":
                found = self._code_span(i, end)
                if found:
                    content, i = found
                    flush()
                    spans.append(Span(content, replace(style, code=True)))
                    continue
            if char == "<":
                found = self._autolink(i, end)
                if found:
                    content, i = found
                    flush()
                    spans.append(Span(content, style))
                    continue
            if char == "[" or (char == "!" and i + 1 < end and text[i + 1] == "["):
                found = self._link(i + (char == "!"), end)
                if found:
                    (label_start, label_end), i = found
                    flush()
                    spans.extend(self.parse(label_start, label_end, style))
                    continue
            if char in "*_~":
                found = self._emphasis(i, end)
                if found:
                    (inner_start, inner_end), new_style_fn, i = found
                    flush()
                    spans.extend(self.parse(inner_start, inner_end, new_style_fn(style)))
                    continue
                run = self._run_length(i, end)
                literal.append(text[i:i + run])
                i += run
                continue
            literal.append(char)
            i += 1
        flush()
        return spans

    def _run_length(self, i: int, end: int) -> int:
        j = i
        while j < end and self.text[j] == self.text[i]:
            j += 1
        return j - i

    def _code_span(self, i: int, end: int) -> tuple[str, int] | None:
        run = self._run_length(i, end)
        fence = "`" * run
        close = self.text.find(fence, i + run, end)
        while close != -1 and close + run < end and self.text[close + run] == "`":
            close = self.text.find(fence, close + run + 1, end)
        if close == -1:
            return None
        content = self.text[i + run:close]
        if len(content) > 2 and content[0] == content[-1] == " " and content.strip():
            content = content[1:-1]
        return content, close + run

    def _autolink(self, i: int, end: int) -> tuple[str, int] | None:
        match = _AUTOLINK.match(self.text, i, end)
        return (match.group(1), match.end()) if match else None

    def _link(self, i: int, end: int) -> tuple[tuple[int, int], int] | None:
        """``[label](target)`` starting at ``i`` (the ``[``)."""
        depth, j = 0, i
        while j < end:
            if self.text[j] == "\\":
                j += 2
                continue
            if self.text[j] == "[":
                depth += 1
            elif self.text[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= end or j + 1 >= end or self.text[j + 1] != "(":
            return None
        close = self.text.find(")", j + 2, end)
        if close == -1:
            return None
        return (i + 1, j), close + 1

    def _emphasis(self, i: int, end: int) -> tuple[tuple[int, int], StyleChange, int] | None:
        """Match an emphasis delimiter run at ``i`` with a closing run of equal length."""
        text = self.text
        char = text[i]
        run = self._run_length(i, end)
        if not self._can_open(i, run, end):
            return None
        sizes = [2] if char == "~" else [k for k in (3, 2, 1) if k <= run]
        if char == "~" and run != 2:
            return None
        for size in sizes:
            open_end = i + size
            j = open_end
            while j < end:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "`":
                    skipped = self._code_span(j, end)
                    if skipped:
                        j = skipped[1]
                        continue
                if text[j] == char:
                    closing_run = self._run_length(j, end)
                    if closing_run >= size and j > open_end and self._can_close(j, size, end):
                        return (open_end, j), _style_for(char, size), j + size
                    j += closing_run
                    continue
                j += 1
        return None

    def _can_open(self, i: int, run: int, end: int) -> bool:
        after = self.text[i + run] if i + run < end else " "
        before = self.text[i - 1] if i > 0 else " "
        if after.isspace():
            return False
        return not (self.text[i] == "_" and before.isalnum())

    def _can_close(self, j: int, size: int, end: int) -> bool:
        before = self.text[j - 1]
        after = self.text[j + size] if j + size < end else " "
        if before.isspace():
            return False
        return not (self.text[j] == "_" and after.isalnum())


def _style_for(char: str, size: int) -> StyleChange:
    if char == "~":
        return lambda s: replace(s, strike=True)
    if size == 3:
        return lambda s: replace(s, bold=True, italic=True)
    if size == 2:
        return lambda s: replace(s, bold=True)
    return lambda s: replace(s, italic=True)
