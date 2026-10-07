"""Write page: turn typed text into handwritten pages."""

from __future__ import annotations

import time

import streamlit as st

from handwriting.errors import HandwritingError
from handwriting.export import pdf_bytes, png_bytes, safe_file_stem, zip_of_png_bytes
from handwriting.models import GlyphSet
from handwriting.renderer import find_missing_characters, render_text
from handwriting.sample_store import ProfileStore
from handwriting.settings import (
    DEFAULT_INK,
    DEFAULT_PRESET,
    INK_COLORS,
    VARIATION_PRESETS,
    Margins,
    MarkdownStyle,
    MissingGlyphPolicy,
    PageFormat,
    PageSettings,
    PaperStyle,
    RenderSettings,
    VariantMode,
    VariationSettings,
    default_margins,
)
from handwriting.utils import parse_hex_color, to_hex_color

from .common import current_profile_id, dpi_control, friendly_errors, preview_image
from .shell import empty_state, page_header, welcome

RESULT_KEY = "generation_result"
_DEFAULT_FILE_NAME = "handwriting"
_VIEW_KEY = "result_page"
_SEED_KEY = "gen_seed"
_CUSTOM_INK = "Custom"
_PERSIST = "session"  # keep choices while visiting other pages
_PAPER_ORDER = [PaperStyle.NARROW_RULED, PaperStyle.COLLEGE_RULED, PaperStyle.WIDE_RULED, PaperStyle.GRAPH,
                PaperStyle.BLANK]
_PAPER_LABELS = {PaperStyle.NARROW_RULED: "Narrow ruled", PaperStyle.COLLEGE_RULED: "College ruled",
                 PaperStyle.WIDE_RULED: "Wide ruled", PaperStyle.GRAPH: "Graph", PaperStyle.BLANK: "Blank"}

# Variation sliders, grouped as shown: session key -> (attribute on VariationSettings,
# UI multiplier, label, slider max, step, help).
_VARIATION_GROUPS: dict[str, dict[str, tuple[str, float, str, float, float, str]]] = {
    "Letters": {
        "var_rotation": ("rotation_deg", 1.0, "Rotation (± degrees)", 6.0, 0.1, ""),
        "var_baseline": ("baseline_jitter", 100.0, "Baseline jitter (± % of x-height)", 20.0, 0.5, ""),
        "var_scale": ("scale_jitter", 100.0, "Size variation (± %)", 15.0, 0.5, ""),
        "var_letter_jitter": ("letter_spacing_jitter", 100.0, "Letter-spacing variation (± % of x-height)",
                              30.0, 0.5, ""),
        "var_overlap": ("letter_overlap", 100.0, "Touching letters (max overlap, % of x-height)", 25.0, 0.5,
                        "How far a letter may run into its neighbour, as in quick writing."),
        "var_warp": ("shape_warp", 100.0, "Shape distortion (% of x-height)", 15.0, 0.5,
                     "Bends every copy of a letter slightly differently, so repeated letters never match."),
    },
    "Words": {
        "var_word_jitter": ("word_spacing_jitter", 100.0, "Word-spacing variation (± %)", 60.0, 1.0, ""),
        "var_word_baseline": ("word_baseline", 100.0, "Word bounce (± % of x-height)", 30.0, 0.5,
                              "Whole words sit a little above or below the line."),
        "var_word_scale": ("word_scale", 100.0, "Word size variation (± %)", 20.0, 0.5, ""),
        "var_word_tilt": ("word_tilt_deg", 1.0, "Word tilt (± degrees)", 5.0, 0.1,
                          "Whole words run slightly uphill or downhill."),
        "var_slant": ("slant_jitter", 100.0, "Slant variation (± %)", 30.0, 0.5,
                      "Some words lean forward more, others stand more upright."),
        "var_size_drift": ("size_drift", 100.0, "Size drift (± %)", 20.0, 0.5,
                           "Handwriting slowly grows and shrinks as you write."),
    },
    "Lines": {
        "var_line_slope": ("line_slope_deg", 1.0, "Line slope (± degrees)", 2.5, 0.05,
                           "Lines run uphill or downhill instead of following the ruling exactly."),
        "var_line_wave": ("line_wave", 100.0, "Line wander (± % of x-height)", 25.0, 0.5,
                          "The baseline drifts up and down along the line."),
        "var_line_offset": ("line_offset", 100.0, "Line float (± % of x-height)", 30.0, 0.5,
                            "Whole lines sit above or sink below the rule."),
        "var_margin": ("margin_jitter", 100.0, "Ragged left edge (% of x-height)", 150.0, 5.0,
                       "Lines start at slightly different distances from the margin."),
    },
    "Layout": {
        "var_indent": ("indent_jitter", 1.0, "Uneven indents (± x-heights)", 2.0, 0.05,
                       "Bullets, list items and wrapped lines don't line up in perfect columns."),
        "var_ragged": ("ragged_right", 1.0, "Uneven line ends (x-heights)", 10.0, 0.25,
                       "Lines end at different distances from the right margin instead of filling every line."),
    },
    "Pen": {
        "var_ink": ("ink_fade", 100.0, "Pen pressure variation (%)", 50.0, 1.0,
                    "Some words come out lighter, as with a real pen."),
        "var_fatigue": ("fatigue", 100.0, "Gets messier towards the end (+%)", 100.0, 5.0,
                        "Extra messiness by the end of the text, like tired note-taking."),
    },
}
_VARIATION_FIELDS = {key: (spec[0], spec[1]) for group in _VARIATION_GROUPS.values() for key, spec in group.items()}

_SAMPLE_TEXT = "Machine learning is interesting.\n\nThe quick brown fox jumps over the lazy dog."

_MARKDOWN_HELP = """
| Type this | Written as |
|---|---|
| `# Title` · `## Section` · `### Sub` | larger headings |
| `- item` or `* item` | bullet point (indent 2 spaces to nest) |
| `1. item` | numbered item |
| `- [ ] task` · `- [x] done` | empty / ticked checkbox |
| `**bold**` · `*italic*` · `~~crossed out~~` | heavier pen · slanted · struck through |
| `<u>underlined</u>` | a hand-drawn line underneath (works inside lists, headings and quotes) |
| `> quote` | indented with a line in the margin |
| `---` | hand-drawn line across the page |
| `` `code` `` · `[text](link)` | written as plain text (links: text only) |
| `\\*` | a literal `*` (backslash escapes any symbol) |

Every line break is kept, as in handwritten notes, and every blank line skips one line.
"""


def _apply_preset() -> None:
    """Copy the chosen preset into the advanced-variation widgets."""
    preset = VARIATION_PRESETS[st.session_state["var_preset"]]
    st.session_state["var_mode"] = preset.variant_mode
    for key, (attribute, factor) in _VARIATION_FIELDS.items():
        st.session_state[key] = round(getattr(preset, attribute) * factor, 3)


def _init_state() -> None:
    if st.session_state.get("var_preset") not in VARIATION_PRESETS:
        st.session_state["var_preset"] = DEFAULT_PRESET
        _apply_preset()
    elif any(key not in st.session_state for key in _VARIATION_FIELDS):
        _apply_preset()  # e.g. sliders added since this session started


def _variation_from_state() -> VariationSettings:
    values = {attribute: float(st.session_state[key]) / factor
              for key, (attribute, factor) in _VARIATION_FIELDS.items()}
    return VariationSettings(variant_mode=st.session_state["var_mode"], **values)


def render_generate_tab(store: ProfileStore) -> None:
    _init_state()
    if not page_header("Write", "Type or paste text and get it back in your own handwriting.", store):
        welcome(store)
        return
    profile_id = current_profile_id()
    glyphs: GlyphSet | None = None
    with friendly_errors("loading the handwriting profile"):
        glyphs = store.load_glyph_set(profile_id)
    if glyphs is None:
        return
    if len(glyphs) == 0:
        _no_samples_yet()
        return

    controls, preview = st.columns([5, 7], gap="large")
    with controls:
        text, markdown = _text_editor(glyphs)
        settings, problem = _style_controls(markdown)
        if problem:
            st.error(problem)
        if st.button("Generate", type="primary", icon=":material/draw:", width="stretch",
                     disabled=settings is None):
            assert settings is not None
            with friendly_errors("generating the pages"):
                _generate(text, glyphs, settings)
    with preview:
        _show_result()


def _no_samples_yet() -> None:
    _, middle, _ = st.columns([1, 2.2, 1])
    with middle, st.container(border=True):
        empty_state("✎", "This profile has no handwriting yet",
                    "Fill in the printable sample sheet. It's the quickest way to teach the app every "
                    "character. You can also upload pictures of single characters.")
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.page_link("ui/pages/sample_sheet.py", label="Open the sample sheet", icon=":material/grid_on:")
            st.page_link("ui/pages/handwriting.py", label="Upload single characters", icon=":material/upload:")


def _text_editor(glyphs: GlyphSet) -> tuple[str, bool]:
    text = st.text_area("Your text", value=_SAMPLE_TEXT, height=240, key="gen_text", persist_state=_PERSIST,
                        placeholder="Type or paste anything…",
                        help="Exactly this text is written: nothing is corrected, added or left out.")
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        markdown = st.toggle("Markdown formatting", key="gen_markdown", persist_state=_PERSIST,
                             help="Headings, lists, checkboxes, bold, italic, strikethrough, underline, quotes and rules. "
                                  "Off: every character is written exactly as typed.")
        with st.popover("Formatting guide", icon=":material/help:", type="tertiary"):
            st.markdown(_MARKDOWN_HELP)
    _show_coverage(text, glyphs, markdown)
    return text, markdown


def _show_coverage(text: str, glyphs: GlyphSet, markdown: bool) -> None:
    missing = find_missing_characters(text, glyphs, markdown=markdown)
    if missing:
        shown = "  ".join(f"`{c}`×{n}" if c != "`" else f"``{c}``×{n}" for c, n in missing.items())
        st.warning(f"**Missing samples for {len(missing)} character(s):** {shown}\n\n"
                   "They're drawn as set under *More options → Other*. Add them on **My handwriting**.",
                   icon=":material/warning:")
    elif text.strip():
        st.caption(":green[:material/check_circle:] Every character is in your handwriting.")
    for problem in glyphs.problems:
        st.warning(problem)


def _style_controls(markdown: bool) -> tuple[RenderSettings | None, str | None]:
    """The style card. Returns the settings, or None and a message if something is invalid."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            ink_name = st.pills("Ink", [*INK_COLORS, _CUSTOM_INK], key="gen_ink", default=DEFAULT_INK,
                                required=True, persist_state=_PERSIST)
            if ink_name == _CUSTOM_INK:
                ink_color = parse_hex_color(st.color_picker(
                    "Ink colour", to_hex_color(INK_COLORS["Blue"]), key="gen_ink_custom",
                    persist_state=_PERSIST, label_visibility="collapsed"))
            else:
                ink_color = INK_COLORS[ink_name]
                swatch = f'<div class="hw-swatch" style="background:{to_hex_color(ink_color)}"></div>'
                st.html(swatch, width="content")
        paper_style = st.pills("Paper", _PAPER_ORDER, key="gen_paper", default=PageSettings().paper_style,
                               required=True, persist_state=_PERSIST, format_func=_PAPER_LABELS.get,
                               help="Narrow 1/4\", college 9/32\" and wide 11/32\" line spacing. "
                                    "Narrow-ruled paper has no red margin line.")
        st.pills("Messiness", list(VARIATION_PRESETS), key="var_preset", on_change=_apply_preset,
                 required=True, persist_state=_PERSIST,
                 help="How tidy the writing looks. Fine-tune it under More options.")
        x_height = st.slider("Size", 1.8, 6.0, RenderSettings().x_height_mm, 0.1, key="gen_size",
                             persist_state=_PERSIST, format="%.1f mm", help="Height of a lowercase x on the page.")
        dpi = dpi_control("gen_dpi", "Resolution (DPI)", help="300 prints crisply; 600 stays sharp when zoomed in but files "
                                                   "are about 4× larger. Also on the Settings page.")
        with st.expander("More options", icon=":material/tune:"):
            page_tab, spacing_tab, fine_tab, other_tab = st.tabs(["Page", "Spacing", "Fine-tune", "Other"])
            with page_tab:
                page = _page_options(paper_style, dpi)
            with spacing_tab:
                letter_spacing = st.slider("Letter spacing", -0.2, 0.6, RenderSettings().letter_spacing, 0.01,
                                           key="gen_letter_spacing",
                                           persist_state=_PERSIST,
                                           help="Extra space after each character, relative to the x-height.")
                word_spacing = st.slider("Word spacing", 0.3, 2.0, 0.85, 0.05, key="gen_word_spacing",
                                         persist_state=_PERSIST, help="Width of a space, relative to the x-height.")
            with fine_tab:
                _fine_tune_controls()
            with other_tab:
                underline = heading_gaps = False
                if markdown:
                    underline = st.toggle("Underline big headings", value=False, key="gen_underline",
                                          persist_state=_PERSIST,
                                          help="Markdown # and ## headings get a hand-drawn line underneath.")
                    heading_gaps = st.toggle("Free line above big headings", value=False, key="gen_heading_gaps",
                                             persist_state=_PERSIST,
                                             help="Off: spacing is exactly as typed; each blank line skips one "
                                                  "line. On: # to ### headings also get a free line above them.")
                policy = st.radio("Characters without a sample", list(MissingGlyphPolicy), key="gen_policy",
                                  persist_state=_PERSIST, format_func=lambda p: p.value)
                seed_text = st.text_input("Random seed", key=_SEED_KEY, persist_state=_PERSIST,
                                          placeholder="Empty: a new look every time",
                                          help="The same text, profile, settings and seed always give the "
                                               "same pages. Use “Keep this look” under the preview to fill it in.")

    seed: int | None = None
    if seed_text.strip():
        try:
            seed = int(seed_text.strip())
        except ValueError:
            return None, "The random seed (More options → Other) must be a whole number, or empty."

    settings = RenderSettings(
        page=page,
        markdown_style=MarkdownStyle(underline_levels=2 if underline else 0, heading_gaps=heading_gaps),
        variation=_variation_from_state(),
        ink_color=ink_color,
        x_height_mm=x_height,
        letter_spacing=letter_spacing,
        word_spacing=word_spacing,
        missing_policy=policy,
        markdown=markdown,
        seed=seed,
    )
    return settings, None


def _page_options(paper_style: PaperStyle, dpi: int) -> PageSettings:
    page_format = st.segmented_control("Page size", list(PageFormat), key="gen_page_format",
                                       default=PageFormat.LETTER, required=True, persist_state=_PERSIST,
                                       format_func=lambda f: f.value)
    if paper_style.is_ruled:
        line_step = st.segmented_control("Write on", [1, 2], key="gen_line_step", default=1, required=True,
                                         persist_state=_PERSIST,
                                         format_func=lambda n: "Every line" if n == 1 else "Every other line")
        line_spacing = PageSettings().line_spacing_mm
    else:
        line_step = 1
        line_spacing = st.slider("Line spacing", 5.0, 20.0, 8.5, 0.5, key="gen_line_spacing",
                                 persist_state=_PERSIST, format="%.1f mm",
                                 help="On graph paper, lines snap to the 5 mm grid.")

    # Each paper keeps its own margins: narrow ruled starts near the edge, the others leave room for a red line.
    defaults = default_margins(paper_style)
    st.caption("Margins (mm)")
    col_a, col_b = st.columns(2)
    margins = Margins(**{
        side: column.number_input(side.title(), 5.0, 80.0, getattr(defaults, side), 1.0, format="%.0f",
                                  key=f"gen_margin_{side}_{paper_style.name.lower()}", persist_state=_PERSIST)
        for side, column in (("left", col_a), ("right", col_b), ("top", col_a), ("bottom", col_b))
    })
    show_margin_rule = True
    if paper_style.has_margin_rule:
        show_margin_rule = st.toggle("Red margin line", value=True, key="gen_margin_rule", persist_state=_PERSIST)
    return PageSettings(
        page_format=page_format,
        paper_style=paper_style,
        margins=margins,
        dpi=dpi,
        line_spacing_mm=line_spacing,
        ruled_line_step=line_step,
        show_margin_rule=show_margin_rule,
    )


def _fine_tune_controls() -> None:
    st.caption("Starts from the Messiness choice above. Changing Messiness resets these.")
    st.selectbox("Which sample to use", list(VariantMode), key="var_mode", persist_state=_PERSIST,
                 format_func=lambda m: m.value,
                 help="When a character has several samples, how the next one is picked.")
    for group, sliders in _VARIATION_GROUPS.items():
        st.markdown(f"**{group}**")
        for key, (_, _, label, maximum, increment, help_text) in sliders.items():
            st.slider(label, 0.0, maximum, step=increment, key=key, persist_state=_PERSIST, help=help_text or None)
    st.button("Reset to preset", icon=":material/restart_alt:", on_click=_apply_preset)


def _generate(text: str, glyphs: GlyphSet, settings: RenderSettings) -> None:
    if len(glyphs) == 0:
        raise HandwritingError("This profile has no handwriting samples yet.")
    started = time.perf_counter()
    with st.spinner("Writing…"):
        result = render_text(text, glyphs, settings)
        dpi = result.dpi
        st.session_state[RESULT_KEY] = {
            "previews": [preview_image(page) for page in result.pages],
            "pngs": [png_bytes(page, dpi) for page in result.pages],
            "pdf": pdf_bytes(result.pages, dpi),
            "seed": result.seed,
            "missing": result.missing,
            "size": result.pages[0].size,
            "dpi": dpi,
            "seconds": time.perf_counter() - started,
        }
        st.session_state[_VIEW_KEY] = 1


def _keep_seed(seed: int) -> None:
    st.session_state[_SEED_KEY] = str(seed)


def _show_result() -> None:
    result = st.session_state.get(RESULT_KEY)
    if not result:
        with st.container(border=True, height=620, vertical_alignment="center"):
            empty_state("✍", "Your pages will appear here",
                        "Pick a style on the left and press Generate. Everything happens on this computer.")
        return

    pages = len(result["pngs"])
    if st.session_state.get(_VIEW_KEY) not in range(1, pages + 1):
        st.session_state[_VIEW_KEY] = 1
    current = st.session_state[_VIEW_KEY]

    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"**{pages} page{'s' if pages != 1 else ''}** · {result['dpi']} DPI · "
                    f"{result['seconds']:.1f} s", width="content")
        st.space("stretch")
        st.caption(f"Look #{result['seed']}", width="content")
        st.button("Keep this look", icon=":material/push_pin:", type="tertiary", on_click=_keep_seed,
                  args=(result["seed"],),
                  help="Fill in the random seed so Generate gives exactly these pages again.")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        typed_name = st.text_input("File name", key="gen_file_name", persist_state=_PERSIST,
                                   placeholder="Name your download (optional)", label_visibility="collapsed",
                                   width="stretch", icon=":material/edit:",
                                   help=f"Used for the PDF, PNG and zip downloads. Press Enter before "
                                        f"downloading. Left empty, files are named “{_DEFAULT_FILE_NAME}”.")
        stem = safe_file_stem(typed_name, fallback=_DEFAULT_FILE_NAME)
        st.download_button("PDF", result["pdf"], file_name=f"{stem}.pdf", mime="application/pdf",
                           on_click="ignore", type="primary", icon=":material/download:",
                           help=f"All pages in one PDF: {stem}.pdf")
        png_name = f"{stem}.png" if pages == 1 else f"{stem}_page_{current:02d}.png"
        st.download_button("PNG", result["pngs"][current - 1], file_name=png_name,
                           mime="image/png", on_click="ignore", icon=":material/image:",
                           help=f"Page {current} as an image: {png_name}")
        if pages > 1:
            pngs = result["pngs"]
            st.download_button("All PNGs", lambda: zip_of_png_bytes(pngs, f"{stem}_page"),
                               file_name=f"{stem}_pages.zip", mime="application/zip", on_click="ignore",
                               icon=":material/folder_zip:", help=f"Every page as a PNG, in {stem}_pages.zip")
    if pages > 1:
        st.pills("Page", list(range(1, pages + 1)), key=_VIEW_KEY, required=True, persist_state=_PERSIST,
                 label_visibility="collapsed")
    if result["missing"]:
        st.warning("Written with placeholders for: " + " ".join(result["missing"]), icon=":material/warning:")
    with st.container(key="page_preview"):
        st.image(result["previews"][current - 1], width="stretch")
