"""Markdown parsing and formatted rendering."""

from __future__ import annotations

import numpy as np
import pytest

from handwriting.document import BlockKind, Style, plain_blocks, visible_text
from handwriting.markdown import parse_inline, parse_markdown
from handwriting.renderer import find_missing_characters, render_text
from handwriting.settings import NO_VARIATION, MarkdownStyle, PageSettings, PaperStyle, RenderSettings
from tests.helpers import make_glyph_set

CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()'"
STILL = NO_VARIATION
BLANK = PageSettings(dpi=150, paper_style=PaperStyle.BLANK)


def _styles(text: str) -> list[tuple[str, set[str]]]:
    return [(span.text, {k for k, v in vars(span.style).items() if v}) for span in parse_inline(text)]


# ----------------------------------------------------------------- parsing

def test_block_types_are_recognised() -> None:
    blocks = parse_markdown(
        "# Title\n### Small\nplain\n- item\n  - nested\n1. first\n12) twelfth\n"
        "- [ ] todo\n- [x] done\n> quoted\n> - quoted item\n---\n\n#hashtag"
    )
    summary = [(b.kind, b.level, b.indent, b.marker, b.checked, b.quote_depth, b.text) for b in blocks]
    assert summary == [
        (BlockKind.HEADING, 1, 0, "", False, 0, "Title"),
        (BlockKind.HEADING, 3, 0, "", False, 0, "Small"),
        (BlockKind.PARAGRAPH, 0, 0, "", False, 0, "plain"),
        (BlockKind.BULLET, 0, 0, "-", False, 0, "item"),
        (BlockKind.BULLET, 0, 1, "-", False, 0, "nested"),
        (BlockKind.NUMBERED, 0, 0, "1.", False, 0, "first"),
        (BlockKind.NUMBERED, 0, 0, "12)", False, 0, "twelfth"),
        (BlockKind.TASK, 0, 0, "", False, 0, "todo"),
        (BlockKind.TASK, 0, 0, "", True, 0, "done"),
        (BlockKind.PARAGRAPH, 0, 0, "", False, 1, "quoted"),
        (BlockKind.BULLET, 0, 0, "-", False, 1, "quoted item"),
        (BlockKind.RULE, 0, 0, "", False, 0, ""),
        (BlockKind.PARAGRAPH, 0, 0, "", False, 0, ""),
        (BlockKind.PARAGRAPH, 0, 0, "", False, 0, "#hashtag"),
    ]


def test_inline_styles() -> None:
    assert _styles("**bold** *it* ***both*** ~~gone~~") == [
        ("bold", {"bold"}), (" ", set()), ("it", {"italic"}), (" ", set()),
        ("both", {"bold", "italic"}), (" ", set()), ("gone", {"strike"}),
    ]
    assert _styles("__b__ and _i_") == [("b", {"bold"}), (" and ", set()), ("i", {"italic"})]


@pytest.mark.parametrize("text", ["2 * 3 * 4", "snake_case_name", "**unclosed", "~single~", "a ** b", "5*"])
def test_ordinary_symbols_stay_literal(text: str) -> None:
    assert _styles(text) == [(text, set())]


def test_code_links_and_escapes() -> None:
    assert _styles("`*raw*` x") == [("*raw*", {"code"}), (" x", set())]
    assert _styles("see [the **docs**](http://x.y)") == [("see the ", set()), ("docs", {"bold"})]
    assert _styles("![a photo](img.png)") == [("a photo", set())]
    assert _styles("<https://example.com>") == [("https://example.com", set())]
    assert _styles(r"\*not italic\*") == [("*not italic*", set())]


def test_fenced_code_is_kept_literally() -> None:
    blocks = parse_markdown("```\n# not a heading\n**raw**\n```\nafter")
    assert [(b.kind, b.text) for b in blocks] == [
        (BlockKind.CODE, "# not a heading"), (BlockKind.CODE, "**raw**"), (BlockKind.PARAGRAPH, "after"),
    ]


def test_line_breaks_are_kept_and_blank_runs_collapse() -> None:
    blocks = parse_markdown("\n\none\ntwo\n\n\nthree\n\n")
    assert [b.text for b in blocks] == ["one", "two", "", "three"]


def test_empty_quote_line_closing_a_quote_is_a_plain_gap() -> None:
    blocks = parse_markdown("> one\n>\n> two\n> \n\n---")
    assert [(b.kind, b.text, b.quote_depth) for b in blocks] == [
        (BlockKind.PARAGRAPH, "one", 1), (BlockKind.PARAGRAPH, "", 1), (BlockKind.PARAGRAPH, "two", 1),
        (BlockKind.PARAGRAPH, "", 0), (BlockKind.RULE, "", 0),
    ]


def test_blank_lines_inside_code_fences_are_kept() -> None:
    blocks = parse_markdown("```\na\n\n\nb\n```")
    assert [b.text for b in blocks] == ["a", "", "", "b"]


def test_plain_mode_keeps_text_exactly() -> None:
    text = "# not a heading\n  **stars** stay\n- dash"
    assert visible_text(plain_blocks(text)) == text.replace("\n", "")
    assert all(b.kind is BlockKind.PARAGRAPH for b in plain_blocks(text))


# --------------------------------------------------------------- rendering

def _render(text: str, **kwargs):
    settings = RenderSettings(page=BLANK, variation=STILL, seed=3, markdown=True, **kwargs)
    return render_text(text, make_glyph_set(CHARS), settings)


def test_markdown_renders_text_without_syntax() -> None:
    text = "# Notes\n- **Key** idea\n1. first\n- [x] done\n> *quote*\n~~old~~ new"
    result = _render(text)
    written = "".join(p.char for p in result.placements)
    assert written == "NotesKeyidea1.firstdonequoteoldnew"
    assert written == visible_text(parse_markdown(text)).replace(" ", "")
    assert result.missing == {}


def test_missing_characters_ignore_markdown_syntax() -> None:
    glyphs = make_glyph_set("abc")
    assert find_missing_characters("# abc **a**", glyphs, markdown=True) == {}
    assert find_missing_characters("# abc **a**", glyphs, markdown=False) == {"#": 1, "*": 4}


def _ink_rows(page) -> np.ndarray:
    gray = np.asarray(page.convert("L"))
    return np.flatnonzero((gray < 128).any(axis=1))


def _ink_height(text: str) -> int:
    no_underline = MarkdownStyle(underline_levels=0)  # measure the letters only
    rows = _ink_rows(_render(text, markdown_style=no_underline).pages[0])
    return int(rows.max() - rows.min())


def test_headings_are_larger_by_level() -> None:
    h1, h2, h3, body = _ink_height("# abc"), _ink_height("## abc"), _ink_height("### abc"), _ink_height("abc")
    assert h1 > h2 > h3 > body
    assert h1 == pytest.approx(body * 1.7, rel=0.15)


def test_list_text_is_indented_and_marked() -> None:
    def ink_columns(text: str) -> np.ndarray:
        gray = np.asarray(_render(text).pages[0].convert("L"))
        return np.flatnonzero((gray < 128).any(axis=0))

    plain, bullet, nested = ink_columns("abc"), ink_columns("- abc"), ink_columns("  - abc")
    # The bullet dot sits near where plain text starts; the text itself moves right.
    x_height_px = 2.9 / 25.4 * 150
    assert bullet.min() <= plain.min() + 0.5 * x_height_px
    assert bullet.max() > plain.max() and nested.max() > bullet.max()
    # Something (the bullet) is drawn left of the list text.
    gap = np.diff(bullet)
    assert gap.max() > 3


def test_task_boxes_and_rules_draw_ink() -> None:
    for text in ("- [ ] ", "- [x] ", "---"):
        assert _ink_rows(_render(text).pages[0]).size > 0, text
    unchecked = np.asarray(_render("- [ ] ").pages[0].convert("L"))
    checked = np.asarray(_render("- [x] ").pages[0].convert("L"))
    assert (checked < 128).sum() > (unchecked < 128).sum()


def test_bold_and_strike_add_ink() -> None:
    def ink(text: str) -> int:
        return int((np.asarray(_render(text).pages[0].convert("L")) < 128).sum())

    assert ink("**abc**") > ink("abc")
    assert ink("~~abc~~") > ink("abc")
    assert ink("*abc*") == pytest.approx(ink("abc"), rel=0.25)  # slanted, not heavier


def test_markdown_rendering_is_deterministic() -> None:
    text = "# T\n- a\n- [x] b\n---\n> c ~~d~~"
    assert _render(text).pages[0].tobytes() == _render(text).pages[0].tobytes()


def test_big_heading_gets_a_free_line_above_on_ruled_paper() -> None:
    settings = RenderSettings(page=PageSettings(dpi=100), variation=STILL, seed=1, markdown=True)
    result = render_text("abc\n# Head\nabc", make_glyph_set(CHARS), settings)
    lines = sorted({p.line for p in result.placements})
    assert lines == [0, 2, 3]


def test_heading_space_does_not_stack_with_blank_lines() -> None:
    settings = RenderSettings(page=PageSettings(dpi=100), variation=STILL, seed=1, markdown=True)
    text = "# Title\n\n## Section\n\nabc\n\n\n\n---\n\n### Sub\n\nabc"
    result = render_text(text, make_glyph_set(CHARS), settings)
    # Title, gap, Section, gap, abc, gap, rule, gap, Sub, gap, abc
    assert sorted({p.line for p in result.placements}) == [0, 2, 4, 8, 10]


def test_style_dataclass_defaults_are_plain() -> None:
    assert Style() == Style(bold=False, italic=False, strike=False, code=False)
