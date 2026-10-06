"""Word wrapping, line breaks, pagination and page geometry."""

from __future__ import annotations

import pytest

from handwriting.layout import SlotRequest, assign_slots, compute_page_geometry, normalize_text, wrap_text
from handwriting.settings import (
    RULE_SPACING_MM,
    Margins,
    PageFormat,
    PageSettings,
    PaperStyle,
    RenderSettings,
    mm_to_px,
)


def _advances(text: str, char_width: float = 10.0, space: float = 10.0) -> list[float]:
    return [0.0 if c == "\n" else space if c.isspace() else char_width for c in text]


def _line_texts(text: str, lines) -> list[str]:
    return ["".join(text[item.index] for item in line.items) for line in lines]


def test_short_text_fits_on_one_line() -> None:
    text = "hello world"
    lines = wrap_text(text, _advances(text), max_width=500)
    assert _line_texts(text, lines) == ["helloworld"]
    assert [round(i.x) for i in lines[0].items][:6] == [0, 10, 20, 30, 40, 60]  # space advances x


def test_wraps_at_word_boundaries_without_splitting_words() -> None:
    text = "aaa bbb ccc ddd"
    # Each word is 30 wide, a space 10: "aaa bbb" = 70 fits in 75, "aaa bbb ccc" = 110 does not.
    lines = wrap_text(text, _advances(text), max_width=75)
    assert _line_texts(text, lines) == ["aaabbb", "cccddd"]
    # Wrapped lines start at x = 0 (the breaking space is dropped).
    assert lines[1].items[0].x == 0


def test_manual_line_breaks_and_blank_lines_are_preserved() -> None:
    text = "one\ntwo\n\n\nthree"
    lines = wrap_text(text, _advances(text), max_width=1000)
    assert _line_texts(text, lines) == ["one", "two", "", "", "three"]
    assert [line.is_blank for line in lines] == [False, False, True, True, False]


def test_trailing_newline_creates_final_blank_line() -> None:
    text = "end\n"
    lines = wrap_text(text, _advances(text), max_width=1000)
    assert _line_texts(text, lines) == ["end", ""]


def test_overlong_word_is_split_between_characters() -> None:
    text = "ab abcdefghij"
    lines = wrap_text(text, _advances(text), max_width=45)
    assert _line_texts(text, lines) == ["ab", "abcd", "efgh", "ij"]
    assert all(line.width <= 45 for line in lines)


def test_paragraph_indentation_is_kept() -> None:
    text = "   indented"
    lines = wrap_text(text, _advances(text), max_width=1000)
    assert lines[0].items[0].x == pytest.approx(30)


def test_every_visible_character_is_placed_once_in_order() -> None:
    text = "The quick brown fox\njumps over\n\n the lazy dog. " * 7
    lines = wrap_text(text, _advances(text, char_width=13, space=7), max_width=160)
    placed = [item.index for line in lines for item in line.items]
    expected = [i for i, c in enumerate(text) if not c.isspace()]
    assert placed == expected


def test_proportional_advances_are_respected() -> None:
    text = "im"
    lines = wrap_text(text, [5.0, 25.0], max_width=100)
    assert [i.x for i in lines[0].items] == [0.0, 5.0]
    assert lines[0].width == 30.0


def test_wrap_rejects_mismatched_advances() -> None:
    with pytest.raises(ValueError):
        wrap_text("abc", [1.0, 2.0], max_width=10)


def test_lines_flow_onto_new_pages() -> None:
    positions = assign_slots([SlotRequest()] * 25, slots_per_page=10)
    pages = [page for page, _ in positions]
    assert [pages.count(n) for n in range(3)] == [10, 10, 5]
    assert positions[10] == (1, 0)
    with pytest.raises(ValueError):
        assign_slots([SlotRequest()], 0)


def test_space_before_is_skipped_at_page_top() -> None:
    requests = [SlotRequest(space_before=1), SlotRequest(), SlotRequest(space_before=1), SlotRequest()]
    assert assign_slots(requests, slots_per_page=10) == [(0, 0), (0, 1), (0, 3), (0, 4)]


def test_heading_is_not_left_alone_at_page_bottom() -> None:
    requests = [SlotRequest()] * 4 + [SlotRequest(keep_with_next=True), SlotRequest()]
    positions = assign_slots(requests, slots_per_page=5)
    assert positions[4] == (1, 0) and positions[5] == (1, 1)


def test_normalize_text_only_changes_representation() -> None:
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"
    assert normalize_text("é") == "é"  # combining accent composed
    assert normalize_text("Tabs\tstay") == "Tabs\tstay"


@pytest.mark.parametrize("paper", [PaperStyle.COLLEGE_RULED, PaperStyle.NARROW_RULED, PaperStyle.WIDE_RULED])
def test_ruled_baselines_sit_on_ruled_lines(paper: PaperStyle) -> None:
    settings = RenderSettings(page=PageSettings(paper_style=paper))
    geometry = compute_page_geometry(settings)
    rules = set(round(y, 3) for y in geometry.rule_ys)
    assert geometry.baselines
    assert all(round(b, 3) in rules for b in geometry.baselines)
    spacing = mm_to_px(RULE_SPACING_MM[paper], settings.page.dpi)
    assert geometry.baselines[1] - geometry.baselines[0] == pytest.approx(spacing)


def test_every_other_line_halves_line_count() -> None:
    every = compute_page_geometry(RenderSettings(page=PageSettings(ruled_line_step=1)))
    other = compute_page_geometry(RenderSettings(page=PageSettings(ruled_line_step=2)))
    assert len(other.baselines) in (len(every.baselines) // 2, len(every.baselines) // 2 + 1)


def test_page_geometry_respects_format_and_margins() -> None:
    settings = RenderSettings(page=PageSettings(page_format=PageFormat.A4, paper_style=PaperStyle.BLANK,
                                                margins=Margins(left=30, right=30, top=30, bottom=30), dpi=200))
    geometry = compute_page_geometry(settings)
    assert (geometry.width, geometry.height) == PageFormat.A4.size_px(200) == (1654, 2339)
    assert geometry.text_left == pytest.approx(mm_to_px(30, 200))
    assert geometry.text_right == pytest.approx(1654 - mm_to_px(30, 200))
    bottom_limit = geometry.height - mm_to_px(30, 200)
    assert all(b < bottom_limit for b in geometry.baselines)


def test_impossible_margins_raise() -> None:
    settings = RenderSettings(page=PageSettings(margins=Margins(left=110, right=100)))
    with pytest.raises(ValueError):
        compute_page_geometry(settings)


def _words_are_whole(text: str, lines) -> bool:
    for line in lines:
        first, last = line.items[0].index, line.items[-1].index
        if first > 0 and not text[first - 1].isspace():
            return False
        if last + 1 < len(text) and not text[last + 1].isspace():
            return False
    return True


def test_lines_can_stop_short_without_splitting_words() -> None:
    text = " ".join(["word"] * 60)
    full = wrap_text(text, _advances(text), max_width=400)
    short = wrap_text(text, _advances(text), max_width=400, stop_short=lambda: 120.0)
    assert len(short) > len(full)
    assert all(line.width <= 400 - 120 + 1e-6 for line in short)
    assert "".join(text[i.index] for line in short for i in line.items) == text.replace(" ", "")
    assert _words_are_whole(text, short)


def test_stopping_short_never_breaks_a_word_that_fits_the_full_width() -> None:
    text = "tiny enormousword"
    advances = _advances(text)
    lines = wrap_text(text, advances, max_width=sum(advances[5:]) + 1, stop_short=lambda: 1e9)
    assert ["".join(text[i.index] for i in line.items) for line in lines] == ["tiny", "enormousword"]


def test_narrow_ruled_paper_has_quarter_inch_lines_and_no_margin_line() -> None:
    narrow = compute_page_geometry(RenderSettings(page=PageSettings(paper_style=PaperStyle.NARROW_RULED)))
    college = compute_page_geometry(RenderSettings(page=PageSettings(paper_style=PaperStyle.COLLEGE_RULED)))
    assert narrow.baselines[1] - narrow.baselines[0] == pytest.approx(mm_to_px(6.35, 300))
    assert len(narrow.baselines) > len(college.baselines)
    assert narrow.margin_rule_x is None and college.margin_rule_x is not None
    assert narrow.text_left == pytest.approx(mm_to_px(Margins().left, 300))  # text starts at the margin
