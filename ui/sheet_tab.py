"""Sample sheet page: print a template, then import the completed pages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np
import streamlit as st

from handwriting.charset import DEFAULT_CHARSET
from handwriting.errors import HandwritingError
from handwriting.export import pdf_bytes, zip_of_pngs
from handwriting.models import SampleSource
from handwriting.sample_store import NewSample, ProfileStore
from handwriting.settings import PageFormat
from handwriting.sheet_import import ExtractedSample, SheetImportResult, assign_x_height, import_sheet
from handwriting.template import (
    DEFAULT_VARIANTS,
    MAX_VARIANTS,
    MIN_VARIANTS,
    TEMPLATE_DPI,
    SheetIdentity,
    build_sheet_layouts,
    render_sample_sheet,
)
from handwriting.utils import decode_pages, is_pdf

from .common import (
    SHEET_UPLOAD_TYPES,
    char_label_html,
    current_profile_id,
    friendly_errors,
    glyph_thumbnail,
    preview_image,
)
from .shell import new_profile_form, page_header, step_title

IMPORT_KEY = "sheet_imports"
_UPLOAD_COUNTER = "sheet_upload_counter"
_MAX_PER_ROW = 8


@dataclass
class _ImportedPage:
    file_name: str
    result: SheetImportResult | None = None
    error: str | None = None
    samples: list[ExtractedSample] = field(default_factory=list)


def render_sheet_tab(store: ProfileStore) -> None:
    has_profiles = bool(page_header("Sample sheet", "Teach the app your handwriting: print, write, scan, import.",
                                    store))
    print_column, write_column = st.columns([3, 2], gap="medium")
    with print_column, st.container(border=True, height="stretch"):
        _create_section()
    with write_column, st.container(border=True, height="stretch"):
        step_title(2, "Write and scan")
        st.markdown(
            "- Write each character inside its boxes with a **dark pen**, one per box.\n"
            "- Vary them a little, as you naturally would.\n"
            "- Scan or photograph every page: **flat, well lit**, all four corner squares visible.\n"
            "- A **600 DPI** scan gives the sharpest letters."
        )
    with st.container(border=True):
        if has_profiles:
            _import_section(store)
        else:
            step_title(3, "Import your pages")
            st.caption("Create a profile to save your handwriting into first.")
            new_profile_form(store, "sheet_profile")


# ----------------------------------------------------------------- create sheet

@st.cache_data(show_spinner=False, max_entries=8)
def _sheet_files(page_format: PageFormat, variants: int) -> tuple[bytes, bytes, list]:
    pages = render_sample_sheet(page_format, variants)
    return (pdf_bytes(pages, TEMPLATE_DPI), zip_of_pngs(pages, TEMPLATE_DPI, stem="sample_sheet_page"),
            [preview_image(p, 600) for p in pages])


def _create_section() -> None:
    step_title(1, "Print the sheet")
    col_a, col_b = st.columns(2)
    page_format = col_a.segmented_control("Paper size", list(PageFormat), format_func=lambda f: f.value,
                                          key="sheet_format", default=PageFormat.LETTER, required=True)
    variants = col_b.number_input("Samples per character", MIN_VARIANTS, MAX_VARIANTS, DEFAULT_VARIANTS,
                                  key="sheet_variants",
                                  help="More samples give more natural variation between repeated letters.")
    page_count = len(build_sheet_layouts(page_format, int(variants)))
    st.caption(f"{len(DEFAULT_CHARSET)} characters × {int(variants)} samples → {page_count} pages. "
               "Print at 100% scale on any printer.")
    with st.spinner("Drawing the sample sheet…"):
        pdf, zipped, previews = _sheet_files(page_format, int(variants))
    with st.container(horizontal=True, vertical_alignment="center"):
        st.download_button("Download PDF", pdf, file_name="handwriting_sample_sheet.pdf", mime="application/pdf",
                           type="primary", on_click="ignore", icon=":material/download:")
        st.download_button("PNG images", zipped, file_name="handwriting_sample_sheet.zip",
                           mime="application/zip", on_click="ignore", icon=":material/folder_zip:")
        with st.popover("Preview", icon=":material/visibility:", type="tertiary"):
            columns = st.columns(min(4, len(previews)))
            for i, preview in enumerate(previews):
                columns[i % len(columns)].image(preview, caption=f"Page {i + 1}", width=220)


# ----------------------------------------------------------------- import sheet

def _manual_identity() -> SheetIdentity | None:
    if not st.toggle("Identify pages manually", key="sheet_manual",
                     help="Use this only if the page code next to the top-left corner square cannot be read."):
        return None
    col_a, col_b, col_c = st.columns(3)
    page_format = col_a.selectbox("Paper size of the sheet", list(PageFormat), format_func=lambda f: f.value,
                                  key="sheet_manual_format")
    variants = int(col_b.number_input("Samples per character", MIN_VARIANTS, MAX_VARIANTS, DEFAULT_VARIANTS,
                                      key="sheet_manual_variants"))
    page_count = len(build_sheet_layouts(page_format, variants))
    page = int(col_c.number_input("Page number", 1, page_count, 1, key="sheet_manual_page",
                                  help="For a multi-page PDF, this is the number of its first page; "
                                       "the following PDF pages count up from it."))
    return SheetIdentity(page_format, page - 1, variants)


def _import_section(store: ProfileStore) -> None:
    step_title(3, "Import your pages")
    profile_id = current_profile_id()
    counter = st.session_state.setdefault(_UPLOAD_COUNTER, 0)
    files = st.file_uploader("Scans or photos of completed pages (images or a multi-page PDF)",
                             type=SHEET_UPLOAD_TYPES, accept_multiple_files=True, key=f"sheet_files_{counter}")
    with st.expander("Pages not recognised?", icon=":material/help:"):
        override = _manual_identity()
    if st.button("Extract handwriting", type="primary", icon=":material/auto_fix_high:", disabled=not files):
        imported: list[_ImportedPage] = []
        progress = st.progress(0.0, text="Reading pages…")
        for i, uploaded in enumerate(files or []):
            imported.extend(_process_file(uploaded.name, uploaded.getvalue(), override))
            progress.progress((i + 1) / len(files), text=f"Processed {uploaded.name}")
        progress.empty()
        st.session_state[IMPORT_KEY] = imported

    imported = st.session_state.get(IMPORT_KEY)
    if imported:
        _review_and_save(store, profile_id, imported)


def _process_file(name: str, data: bytes, override: SheetIdentity | None) -> list[_ImportedPage]:
    """Import every page of an uploaded image or PDF."""
    try:
        images = decode_pages(data)
    except HandwritingError as exc:
        return [_ImportedPage(file_name=name, error=str(exc))]
    labelled = is_pdf(data) and len(images) > 1
    pages: list[_ImportedPage] = []
    for number, image in enumerate(images):
        label = f"{name} (PDF page {number + 1})" if labelled else name
        page_override = replace(override, page_index=override.page_index + number) if override else None
        pages.append(_process_page(label, image, page_override))
    return pages


def _process_page(label: str, image: np.ndarray, override: SheetIdentity | None) -> _ImportedPage:
    try:
        result = import_sheet(image, identity_override=override)
    except HandwritingError as exc:
        return _ImportedPage(file_name=label, error=str(exc))
    return _ImportedPage(file_name=label, result=result, samples=result.samples)


def _review_and_save(store: ProfileStore, profile_id: str, imported: list[_ImportedPage]) -> None:
    st.divider()
    st.markdown("**Review**")
    for page_index, page in enumerate(imported):
        if page.error or page.result is None:
            st.error(f"**{page.file_name}:** {page.error}\n\nTry another scan, or identify the page manually.")
            continue
        result = page.result
        identity = result.identity
        how = "page code" if result.identity_detected else "manual selection"
        with st.expander(f"{page.file_name} — page {identity.page_index + 1} "
                         f"({identity.page_format.value}, {identity.variants} per character, via {how}): "
                         f"{len(result.samples)} samples, {len(result.empty_cells)} empty boxes",
                         icon=":material/check_circle:"):
            st.image(result.preview, caption="Aligned page — green boxes contain handwriting", width="stretch")
        for warning in result.warnings:
            st.warning(f"{page.file_name}: {warning}")

    selected = _sample_picker(imported)
    if not selected:
        st.info("No samples selected.")
        return

    chars = {s.char for s in selected}
    with st.container(horizontal=True, vertical_alignment="center"):
        save = st.button(f"Save {len(selected)} samples for {len(chars)} characters", type="primary",
                         icon=":material/save:")
        replace = st.toggle("Replace existing samples of these characters", value=False,
                            help="Otherwise the new samples are added next to the existing ones.")
    if save:
        with friendly_errors("saving the samples"):
            if replace:
                store.delete_character_samples(profile_id, chars)
            known = store.template_x_height_reference(profile_id)
            final = assign_x_height(selected, known_reference=known)
            store.add_samples(profile_id, [
                NewSample(char=s.char, rgba=s.rgba, x_height_ref=s.x_height_ref,
                          baseline_from_top=s.baseline_from_top, source=SampleSource.TEMPLATE,
                          capture_dpi=s.capture_dpi)
                for s in final
            ])
            st.session_state.pop(IMPORT_KEY, None)
            st.session_state[_UPLOAD_COUNTER] = st.session_state.get(_UPLOAD_COUNTER, 0) + 1
            st.toast(f"Saved {len(final)} samples.")
            st.rerun()


def _sample_picker(imported: list[_ImportedPage]) -> list[ExtractedSample]:
    """Show every extracted glyph grouped by character with include/exclude checkboxes."""
    by_char: dict[str, list[ExtractedSample]] = defaultdict(list)
    for page in imported:
        for sample in page.samples:
            by_char[sample.char].append(sample)
    if not by_char:
        return []

    st.caption("Untick any sample that looks wrong (cut off, smudged, or the wrong character).")
    order = [c for c in DEFAULT_CHARSET if c in by_char] + sorted(set(by_char) - set(DEFAULT_CHARSET))
    with st.container(height=560, border=True):
        return _sample_rows(order, by_char)


def _sample_rows(order: list[str], by_char: dict[str, list[ExtractedSample]]) -> list[ExtractedSample]:
    selected: list[ExtractedSample] = []
    for char in order:
        samples = by_char[char]
        label_col, *cols = st.columns([1] + [1] * _MAX_PER_ROW)
        with label_col:
            st.html(char_label_html(char))
        for i, sample in enumerate(samples):
            with cols[i % _MAX_PER_ROW]:
                st.image(glyph_thumbnail(sample.rgba, height=70))
                if st.checkbox("Use", value=True, key=f"use_{sample.key}_{i}"):
                    selected.append(sample)
    return selected
