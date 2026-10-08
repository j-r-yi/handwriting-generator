"""My handwriting page: inspect, extend and clean up a profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from handwriting.baseline import estimate_x_height_ref
from handwriting.charset import (
    CHARACTER_GROUPS,
    COMMON_SYMBOLS,
    DEFAULT_CHARSET,
    describe_char,
    validate_symbol_char,
)
from handwriting.errors import HandwritingError
from handwriting.models import ProfileMetadata, SampleSource
from handwriting.preprocessing import extract_glyph
from handwriting.sample_store import NewSample, ProfileStore
from handwriting.utils import decode_image

from .common import (
    UPLOAD_TYPES,
    coverage_grid,
    current_profile_id,
    friendly_errors,
    glyph_thumbnail,
    select_profile_on_next_run,
)
from .shell import empty_state, page_header, welcome

_PICK, _NEW_SYMBOL = "Pick a character", "New symbol"
_SYMBOL_KEY = "symbol_typed"
_SYMBOL_PICK_KEY = "symbol_pick"
_UPLOAD_COUNTER = "manual_upload_counter"
_GRID_COLUMNS = 8


@dataclass
class _ProcessedUpload:
    name: str
    rgba: np.ndarray | None
    error: str | None
    ink_width: float = 0.0
    ink_height: float = 0.0


def render_profile_tab(store: ProfileStore) -> None:
    if not page_header("My handwriting", "See which characters you've written, add more, and tidy up.", store):
        welcome(store)
        return
    profile_id = current_profile_id()
    meta: ProfileMetadata | None = None
    with friendly_errors("loading the profile"):
        meta = store.load_profile(profile_id)
    if meta is None:
        return

    _summary(meta)
    characters, browse, upload, manage = st.tabs([":material/grid_view: Characters",
                                                  ":material/photo_library: Browse samples",
                                                  ":material/upload: Add images",
                                                  ":material/tune: Manage profile"])
    with characters:
        _coverage_section(meta)
    with browse:
        _browse_section(store, meta)
    with upload:
        _manual_upload_section(store, meta)
    with manage:
        _manage_section(store, meta)


def _summary(meta: ProfileMetadata) -> None:
    counts = meta.sample_counts()
    covered = sum(1 for c in DEFAULT_CHARSET if counts.get(c))
    samples, characters, missing = st.columns(3)
    samples.metric("Samples", meta.total_samples, border=True)
    characters.metric("Characters covered", f"{covered} / {len(DEFAULT_CHARSET)}", border=True)
    missing.metric("Still missing", len(DEFAULT_CHARSET) - covered, border=True)
    if meta.total_samples == 0:
        with st.container(border=True):
            empty_state("✎", "No handwriting yet",
                        "The sample sheet is the quickest start: print it, write every character, scan it.")
            with st.container(horizontal=True, horizontal_alignment="center"):
                st.page_link("ui/pages/sample_sheet.py", label="Open the sample sheet", icon=":material/grid_on:")


def _coverage_section(meta: ProfileMetadata) -> None:
    counts = meta.sample_counts()
    parts = []
    groups = dict(CHARACTER_GROUPS)
    extra = sorted(set(counts) - set(DEFAULT_CHARSET))
    if extra:
        groups["Your symbols"] = tuple(extra)
    for group, chars in groups.items():
        have = sum(1 for c in chars if counts.get(c))
        parts.append(f'<div class="hw-group">{group}<span>{have} of {len(chars)}</span></div>')
        parts.append(coverage_grid(list(chars), counts))
    st.caption("The number under each character is how many samples you have. Dashed red boxes have none yet.")
    st.html("".join(parts))
    if not extra:
        st.caption(":material/add_circle: Need arrows, bullets, maths signs or accented letters? "
                   "Add them under **Add images → New symbol**.")


def _choose_character(meta: ProfileMetadata) -> str | None:
    mode = st.segmented_control("Add pictures for", [_PICK, _NEW_SYMBOL], key="manual_mode", default=_PICK,
                                required=True)
    if mode == _PICK:
        options = list(dict.fromkeys([*DEFAULT_CHARSET, *sorted(meta.characters)]))
        return st.selectbox("Character", options, key="manual_select", format_func=describe_char, width=280,
                            help="The standard set plus any symbols you have added.")
    return _new_symbol(meta)


def _use_picked_symbol() -> None:
    st.session_state[_SYMBOL_KEY] = st.session_state.get(_SYMBOL_PICK_KEY) or ""
    st.session_state[_SYMBOL_PICK_KEY] = None


def _new_symbol(meta: ProfileMetadata) -> str | None:
    """A keyboard character beyond the standard set: typed, pasted or picked."""
    st.caption("Any character you can type on a keyboard: arrows, bullets, maths and currency signs, "
               "accented letters… Emoji can't be used.")
    st.pills("Common symbols", COMMON_SYMBOLS, key=_SYMBOL_PICK_KEY, on_change=_use_picked_symbol,
             help="Click one to fill it in, or type your own below.")
    typed = st.text_input("Symbol", key=_SYMBOL_KEY, max_chars=4, width=280,
                          placeholder="Type or paste one character")
    if not typed.strip():
        return None
    try:
        char = validate_symbol_char(typed.strip())
    except HandwritingError as exc:
        st.error(str(exc))
        return None
    count = len(meta.characters[char].samples) if char in meta.characters else 0
    if char in DEFAULT_CHARSET:
        st.caption(f"“{char}” is a standard character ({count} sample(s)). These pictures are added to it.")
    elif count:
        st.caption(f"You already have {count} sample(s) of {describe_char(char)}. These pictures are added.")
    else:
        st.caption(f":material/add_circle: New symbol: {describe_char(char)}")
    return char


def _manual_upload_section(store: ProfileStore, meta: ProfileMetadata) -> None:
    st.caption("Upload cropped pictures of one handwritten character (dark ink on light paper). "
               "Backgrounds and specks are removed automatically. For many characters at once, "
               "the sample sheet is faster.")
    char = _choose_character(meta)
    counter = st.session_state.setdefault(_UPLOAD_COUNTER, 0)
    files = st.file_uploader("Images of this character", type=UPLOAD_TYPES, accept_multiple_files=True,
                             key=f"manual_files_{counter}",
                             help="Use your own handwriting, or someone else's only with their permission.")
    if not files or char is None:
        return

    processed = [_process_upload(f.name, f.getvalue()) for f in files]
    selected: list[_ProcessedUpload] = []
    columns = st.columns(_GRID_COLUMNS)
    for i, item in enumerate(processed):
        with columns[i % _GRID_COLUMNS], st.container(border=True):
            if item.rgba is None:
                st.error(f"{item.name}: {item.error}")
                continue
            st.image(glyph_thumbnail(item.rgba), width="stretch", caption=item.name)
            if st.checkbox("Include", value=True, key=f"manual_inc_{counter}_{i}"):
                selected.append(item)

    if st.button(f"Add {len(selected)} sample(s) for “{char}”", type="primary", icon=":material/add:",
                 disabled=not selected):
        with friendly_errors("saving the samples"):
            store.add_samples(meta.profile_id, [
                NewSample(char=char, rgba=item.rgba,
                          x_height_ref=estimate_x_height_ref(char, item.ink_width, item.ink_height),
                          source=SampleSource.MANUAL)
                for item in selected if item.rgba is not None
            ])
            st.session_state[_UPLOAD_COUNTER] = counter + 1  # clears the uploader
            st.toast(f"Added {len(selected)} sample(s) for “{char}”.")
            st.rerun()


@st.cache_data(show_spinner=False, max_entries=256)
def _process_upload(name: str, data: bytes) -> _ProcessedUpload:
    try:
        glyph = extract_glyph(decode_image(data))
    except HandwritingError as exc:
        return _ProcessedUpload(name=name, rgba=None, error=str(exc))
    pad = 2
    return _ProcessedUpload(
        name=name,
        rgba=glyph.rgba,
        error=None,
        ink_width=max(1.0, glyph.bbox[2] - 2 * pad) * glyph.scale,
        ink_height=max(1.0, glyph.bbox[3] - 2 * pad) * glyph.scale,
    )


def _browse_section(store: ProfileStore, meta: ProfileMetadata) -> None:
    chars = [c for c in [*DEFAULT_CHARSET, *sorted(meta.characters)] if c in meta.characters]
    chars = list(dict.fromkeys(chars))
    if not chars:
        empty_state("🔍", "Nothing to browse yet", "Samples you add show up here.")
        return
    with st.container(horizontal=True, vertical_alignment="bottom"):
        char = st.selectbox("Character", chars, key="browse_char", format_func=describe_char, width=280)
        st.caption(f"{len(meta.characters[char].samples)} sample(s). Tick the ones to delete.")
    entry = meta.characters[char]
    marked: list[str] = []
    columns = st.columns(_GRID_COLUMNS)
    for i, sample in enumerate(entry.samples):
        with columns[i % _GRID_COLUMNS], st.container(border=True):
            try:
                image = store.load_sample_image(meta.profile_id, sample)
                st.image(glyph_thumbnail(image), width="stretch")
            except OSError:
                st.error(f"{sample.sample_id}: file missing or unreadable")
            if st.checkbox("Select", key=f"del_{meta.profile_id}_{sample.sample_id}",
                           help=f"Sample {sample.sample_id} · from {sample.source}"):
                marked.append(sample.sample_id)

    if marked:
        with st.container(horizontal=True, vertical_alignment="center"):
            confirm = st.checkbox(f"Yes, delete {len(marked)} selected sample(s)", key=f"confirm_del_{char}")
            delete = st.button("Delete selected samples", icon=":material/delete:", disabled=not confirm)
        if delete:
            with friendly_errors("deleting samples"):
                for sample_id in marked:
                    store.delete_sample(meta.profile_id, char, sample_id)
                st.toast(f"Deleted {len(marked)} sample(s). Files were moved to the trash folder.")
                st.rerun()


def _manage_section(store: ProfileStore, meta: ProfileMetadata) -> None:
    with st.container(border=True):
        st.markdown("**Name**")
        with st.form("rename_profile", border=False):
            with st.container(horizontal=True, vertical_alignment="bottom"):
                new_name = st.text_input("Rename profile", value=meta.name, label_visibility="collapsed")
                renamed = st.form_submit_button("Rename")
        if renamed:
            with friendly_errors("renaming the profile"):
                store.rename_profile(meta.profile_id, new_name)
                st.rerun()
        st.caption(f"Stored in `{store.profile_dir(meta.profile_id)}`")

    st.markdown("**Danger zone**")
    clear_column, delete_column = st.columns(2)
    with clear_column, st.container(border=True, height="stretch"):
        _clear_samples_control(store, meta)
    with delete_column, st.container(border=True, height="stretch"):
        st.markdown("**Delete this profile**")
        st.caption("The profile folder is moved to `data/trash` and disappears from the app.")
        typed = st.text_input(f"Type “{meta.name}” to confirm", key=f"delete_confirm_{meta.profile_id}")
        if st.button("Delete profile", icon=":material/delete_forever:", disabled=typed.strip() != meta.name):
            with friendly_errors("deleting the profile"):
                store.delete_profile(meta.profile_id)
                select_profile_on_next_run(None)
                st.rerun()


def _clear_samples_control(store: ProfileStore, meta: ProfileMetadata) -> None:
    """Reset the profile to zero samples while keeping the profile itself."""
    st.markdown("**Clear all samples**")
    total = meta.total_samples
    if total == 0:
        st.caption("This profile has no samples.")
        return
    st.caption(f"Removes all {total} samples so you can start again. The profile and its name stay. "
               "Removed images are moved to `data/trash`.")
    # The sample count in the key unticks the box once the profile has been cleared.
    confirmed = st.checkbox(f"Yes, remove all {total} samples", key=f"clear_confirm_{meta.profile_id}_{total}")
    if st.button("Clear all samples", icon=":material/layers_clear:", disabled=not confirmed):
        with friendly_errors("clearing the profile"):
            removed = store.clear_samples(meta.profile_id)
            st.toast(f"Removed {removed} samples. “{meta.name}” is now empty.")
            st.rerun()
