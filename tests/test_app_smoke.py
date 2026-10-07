"""End-to-end smoke test of the Streamlit app, driven headlessly with AppTest.

Covers the acceptance workflow: create a profile, upload samples manually,
import a photographed sample sheet, generate pages, and offer downloads.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from handwriting.sample_store import ProfileStore
from handwriting.settings import DEFAULT_DPI, VARIATION_PRESETS, PageFormat, PaperStyle

from tests.helpers import (
    PROJECT_ROOT,
    draw_char_image,
    filled_sheet,
    make_new_samples,
    png_bytes_of,
    simulate_photo,
)

APP = str(PROJECT_ROOT / "app.py")
TIMEOUT = 60


def _by_label(widgets, label: str):
    matches = [w for w in widgets if w.label.startswith(label)]
    assert matches, f"no widget labelled {label!r}; have {[w.label for w in widgets]}"
    return matches[0]


def _assert_no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, Path]:
    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("HANDWRITING_DATA_DIR", str(data_dir))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    _assert_no_errors(at)
    return at, data_dir


WRITE = "ui/pages/write.py"
HANDWRITING = "ui/pages/handwriting.py"
SAMPLE_SHEET = "ui/pages/sample_sheet.py"
SETTINGS = "ui/pages/settings.py"


def _open(at: AppTest, page: str) -> None:
    at.switch_page(page)
    at.run()
    _assert_no_errors(at)


def _create_profile(at: AppTest, name: str) -> None:
    _by_label(at.text_input, "Profile name").input(name)
    _by_label(at.button, "Create profile").click()
    at.run()
    _assert_no_errors(at)


def test_full_workflow(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Test Hand")
    store = ProfileStore(data_dir)
    summaries, _ = store.list_profiles()
    assert [s.name for s in summaries] == ["Test Hand"]
    profile_id = summaries[0].profile_id

    # Manual upload of two images of "a".
    _open(at, HANDWRITING)
    at.selectbox(key="manual_select").set_value("a")
    at.run()
    uploader = _by_label(at.file_uploader, "Images of this character")
    uploader.set_value([("a1.png", png_bytes_of(draw_char_image("a")), "image/png"),
                        ("a2.png", png_bytes_of(draw_char_image("a", offset=4)), "image/png")])
    at.run()
    _assert_no_errors(at)
    _by_label(at.button, "Add 2 sample(s)").click()
    at.run()
    _assert_no_errors(at)
    assert store.load_profile(profile_id).sample_counts() == {"a": 2}

    # Import a photographed sample-sheet page.
    _open(at, SAMPLE_SHEET)
    _, page = filled_sheet(PageFormat.LETTER, 0)
    photo = simulate_photo(page)
    buffer = io.BytesIO()
    Image.fromarray(photo).save(buffer, format="JPEG", quality=90)
    _by_label(at.file_uploader, "Scans or photos").set_value(("sheet.jpg", buffer.getvalue(), "image/jpeg"))
    at.run()
    _by_label(at.button, "Extract handwriting").click()
    at.run()
    _assert_no_errors(at)
    _by_label(at.button, "Save ").click()
    at.run()
    _assert_no_errors(at)
    counts = store.load_profile(profile_id).sample_counts()
    assert counts["a"] == 5  # 2 manual + 3 from the sheet
    assert counts["z"] == 3 and counts["G"] == 3

    # Generate pages and check the download buttons are offered.
    _open(at, WRITE)
    at.text_area(key="gen_text").set_value("abc defg\n\nGab")
    at.run()
    _by_label(at.button, "Generate").click()
    at.run()
    _assert_no_errors(at)
    assert any("1 page" in m.value for m in at.markdown)
    assert {b.label for b in at.get("download_button")} >= {"PDF", "PNG"}


def test_first_run_shows_welcome_and_empty_profile_points_to_sample_sheet(app: tuple[AppTest, Path]) -> None:
    at, _ = app
    assert any("set up your handwriting" in h.value for h in at.subheader)
    assert not at.text_area  # nothing to write with yet
    _create_profile(at, "Empty")
    assert not at.text_area
    assert any("no handwriting yet" in str(h.proto.body) for h in at.get("html"))


def test_missing_characters_are_listed_before_generation(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Partial")
    store = ProfileStore(data_dir)
    store.add_samples(store.list_profiles()[0][0].profile_id, make_new_samples("h", variants=1))
    at.run()
    at.text_area(key="gen_text").set_value("hi!")
    at.run()
    warnings = " ".join(w.value for w in at.warning)
    assert "Missing samples for 2 character(s)" in warnings


def test_bad_upload_shows_friendly_message(app: tuple[AppTest, Path]) -> None:
    at, _ = app
    _create_profile(at, "Errors")
    _open(at, HANDWRITING)
    _by_label(at.file_uploader, "Images of this character").set_value(
        ("broken.png", b"definitely not an image", "image/png"))
    at.run()
    assert not at.exception
    assert any("Unsupported or unrecognised image format" in e.value for e in at.error)


def test_multi_page_pdf_upload_imports_all_pages(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Pdf Hand")
    _open(at, SAMPLE_SHEET)
    pages = [Image.fromarray(filled_sheet(PageFormat.LETTER, index)[1]) for index in (0, 1)]
    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:], resolution=300.0)

    _by_label(at.file_uploader, "Scans or photos").set_value(("scan.pdf", buffer.getvalue(), "application/pdf"))
    at.run()
    _by_label(at.button, "Extract handwriting").click()
    at.run()
    _assert_no_errors(at)
    # AppTest lists expanders that have an icon as "status" blocks.
    expanders = " ".join(e.label for e in [*at.expander, *at.status])
    assert "scan.pdf (PDF page 1)" in expanders and "scan.pdf (PDF page 2)" in expanders

    _by_label(at.button, "Save ").click()
    at.run()
    _assert_no_errors(at)
    store = ProfileStore(data_dir)
    profile_id = store.list_profiles()[0][0].profile_id
    counts = store.load_profile(profile_id).sample_counts()
    assert counts["a"] == 3 and counts["Z"] == 3  # "a" is on page 1, "Z" on page 2


def test_clear_all_samples_requires_confirmation(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Clear Me")
    store = ProfileStore(data_dir)
    profile_id = store.list_profiles()[0][0].profile_id
    store.add_samples(profile_id, make_new_samples("ab", variants=2))
    _open(at, HANDWRITING)

    clear = _by_label(at.button, "Clear all samples")
    assert clear.disabled  # nothing happens without ticking the confirmation
    _by_label(at.checkbox, "Yes, remove all 4 samples").check()
    at.run()
    _by_label(at.button, "Clear all samples").click()
    at.run()
    _assert_no_errors(at)
    assert store.load_profile(profile_id).total_samples == 0
    assert [s.name for s in store.list_profiles()[0]] == ["Clear Me"]  # profile kept


def test_markdown_toggle_formats_instead_of_writing_syntax(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Notes")
    store = ProfileStore(data_dir)
    profile_id = store.list_profiles()[0][0].profile_id
    store.add_samples(profile_id, make_new_samples("ab", variants=1))
    at.run()  # the page now shows the editor
    at.text_area(key="gen_text").set_value("# ab\n- **a** b")
    at.run()
    assert any("Missing samples" in w.value for w in at.warning)  # '#', '*' and '-' as plain text

    at.toggle(key="gen_markdown").set_value(True)
    at.run()
    assert not any("Missing samples" in w.value for w in at.warning)
    _by_label(at.button, "Generate").click()
    at.run()
    _assert_no_errors(at)
    assert any("1 page" in m.value for m in at.markdown)


def test_messiness_preset_fills_the_advanced_sliders(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Messy")
    store = ProfileStore(data_dir)
    profile_id = store.list_profiles()[0][0].profile_id
    store.add_samples(profile_id, make_new_samples("ab", variants=2))
    at.run()  # the page now shows the editor
    at.text_area(key="gen_text").set_value("ab ba\nab")
    at.run()
    assert at.slider(key="var_line_slope").value < VARIATION_PRESETS["Rushed notes"].line_slope_deg

    at.pills(key="var_preset").set_value("Rushed notes")
    at.run()
    rushed = VARIATION_PRESETS["Rushed notes"]
    assert at.slider(key="var_line_slope").value == pytest.approx(rushed.line_slope_deg)
    assert at.slider(key="var_margin").value == pytest.approx(rushed.margin_jitter * 100)
    _by_label(at.button, "Generate").click()
    at.run()
    _assert_no_errors(at)
    assert any("1 page" in m.value for m in at.markdown)


def test_generate_writes_on_every_ruled_line(app: tuple[AppTest, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the variation sliders once overwrote the 'Write on' setting, spacing lines 5 rules apart."""
    import ui.generate_tab as generate_tab

    captured = []
    real_render = generate_tab.render_text

    def spy(text, glyphs, settings):
        result = real_render(text, glyphs, settings)
        captured.append((settings, result))
        return result

    monkeypatch.setattr(generate_tab, "render_text", spy)
    at, data_dir = app
    _create_profile(at, "Lines")
    store = ProfileStore(data_dir)
    profile_id = store.list_profiles()[0][0].profile_id
    store.add_samples(profile_id, make_new_samples("ab", variants=1))
    at.run()  # the page now shows the editor
    at.text_area(key="gen_text").set_value("ab\nba\nab")
    at.run()
    _by_label(at.button, "Generate").click()
    at.run()
    _assert_no_errors(at)
    settings, result = captured[-1]
    assert settings.page.ruled_line_step == 1
    assert sorted({p.line for p in result.placements}) == [0, 1, 2]

    at.segmented_control(key="gen_line_step").set_value(2)
    _by_label(at.button, "Generate").click()
    at.run()
    assert captured[-1][0].page.ruled_line_step == 2


def test_settings_survive_switching_pages(app: tuple[AppTest, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.generate_tab as generate_tab

    captured = []
    real_render = generate_tab.render_text
    monkeypatch.setattr(generate_tab, "render_text",
                        lambda text, glyphs, settings: captured.append(settings) or real_render(text, glyphs, settings))
    at, data_dir = app
    _create_profile(at, "Travel")
    store = ProfileStore(data_dir)
    store.add_samples(store.list_profiles()[0][0].profile_id, make_new_samples("ab", variants=1))
    at.run()
    at.text_area(key="gen_text").set_value("ab ba")
    at.pills(key="var_preset").set_value("Messy")
    at.run()

    _open(at, SETTINGS)
    at.segmented_control(key="render_dpi").set_value(150)
    at.run()
    _open(at, HANDWRITING)
    _open(at, WRITE)
    assert at.text_area(key="gen_text").value == "ab ba"
    assert at.pills(key="var_preset").value == "Messy"
    _by_label(at.button, "Generate").click()
    at.run()
    _assert_no_errors(at)
    assert captured[-1].page.dpi == 150
    assert captured[-1].variation == VARIATION_PRESETS["Messy"]

    seed = at.session_state["generation_result"]["seed"]
    _by_label(at.button, "Keep this look").click()
    at.run()
    assert at.text_input(key="gen_seed").value == str(seed)


def test_write_page_defaults_and_resolution_picker(app: tuple[AppTest, Path],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.generate_tab as generate_tab

    captured = []
    real_render = generate_tab.render_text
    monkeypatch.setattr(generate_tab, "render_text",
                        lambda text, glyphs, settings: captured.append(settings) or real_render(text, glyphs, settings))
    at, data_dir = app
    _create_profile(at, "Defaults")
    store = ProfileStore(data_dir)
    store.add_samples(store.list_profiles()[0][0].profile_id, make_new_samples("ab", variants=1))
    at.run()
    assert at.pills(key="gen_paper").value is PaperStyle.NARROW_RULED
    assert at.slider(key="gen_size").value == pytest.approx(2.2)
    assert at.segmented_control(key="gen_dpi").value == DEFAULT_DPI

    at.segmented_control(key="gen_dpi").set_value(600)  # chosen on the Write page...
    at.run()
    _open(at, SETTINGS)
    assert at.segmented_control(key="render_dpi").value == 600  # ...shows on Settings
    at.segmented_control(key="render_dpi").set_value(200)  # and back again
    at.run()
    _open(at, WRITE)
    assert at.segmented_control(key="gen_dpi").value == 200
    at.text_area(key="gen_text").set_value("ab ba")
    _by_label(at.button, "Generate").click()
    at.run()
    _assert_no_errors(at)
    assert captured[-1].page.dpi == 200
    assert captured[-1].page.paper_style is PaperStyle.NARROW_RULED
    assert captured[-1].x_height_mm == pytest.approx(2.2)


def test_multi_page_result_can_be_paged_through(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Pages")
    store = ProfileStore(data_dir)
    store.add_samples(store.list_profiles()[0][0].profile_id, make_new_samples("ab", variants=1))
    at.run()
    at.text_area(key="gen_text").set_value("ab\n" * 70)
    _by_label(at.button, "Generate").click()
    at.run()
    _assert_no_errors(at)
    pages = len(at.session_state["generation_result"]["pngs"])
    assert pages >= 2
    assert {b.label for b in at.get("download_button")} == {"PDF", "PNG", "All PNGs"}
    at.pills(key="result_page").set_value(2)
    at.run()
    png = next(b for b in at.get("download_button") if b.label == "PNG")
    assert "Page 2" in png.proto.help


def test_new_symbol_can_be_added_with_pictures(app: tuple[AppTest, Path]) -> None:
    at, data_dir = app
    _create_profile(at, "Symbols")
    _open(at, HANDWRITING)
    at.segmented_control(key="manual_mode").set_value("New symbol")
    at.run()
    at.text_input(key="symbol_typed").input("😀")
    at.run()
    assert any("emoji" in e.value for e in at.error)
    assert not at.button or not any(b.label.startswith("Add ") for b in at.button)

    at.pills(key="symbol_pick").set_value("→")  # a quick pick fills in the symbol
    at.run()
    assert at.text_input(key="symbol_typed").value == "→"
    _by_label(at.file_uploader, "Images of this character").set_value(
        [("arrow.png", png_bytes_of(draw_char_image("→")), "image/png")])
    at.run()
    _assert_no_errors(at)
    _by_label(at.button, "Add 1 sample(s)").click()
    at.run()
    _assert_no_errors(at)
    store = ProfileStore(data_dir)
    assert store.load_profile(store.list_profiles()[0][0].profile_id).sample_counts() == {"→": 1}
    assert any("Your symbols" in str(h.proto.body) for h in at.get("html"))

    # The new symbol is now offered next to the standard characters.
    at.segmented_control(key="manual_mode").set_value("Pick a character")
    at.run()
    assert "→" in at.selectbox(key="manual_select").options[-1]
