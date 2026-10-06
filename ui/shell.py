"""App shell shared by every page: data store, styling, page header and profile switching."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from handwriting.errors import HandwritingError, StorageError
from handwriting.sample_store import ProfileStore, ProfileSummary, default_data_dir

from .common import PROFILE_KEY, apply_pending_profile_selection, friendly_errors, select_profile_on_next_run

ASSETS = Path(__file__).parent / "assets"
_PICKER_KEY = "profile_picker"

# Small, self-contained style tweaks on top of the theme in .streamlit/config.toml.
_CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 2.5rem; padding-bottom: 5rem; }
h1, h2, h3, h4 { letter-spacing: -0.015em; }
.hw-subtitle { margin: -0.4rem 0 0.4rem; opacity: 0.62; font-size: 0.98rem; }
.hw-swatch { width: 2.1rem; height: 2.1rem; border-radius: 50%; margin-bottom: 0.15rem;
             box-shadow: inset 0 0 0 1px rgba(128, 128, 128, 0.35); }
.hw-step { display: inline-flex; align-items: center; justify-content: center; width: 1.6rem; height: 1.6rem;
           margin-right: 0.55rem; border-radius: 50%; background: #3451D1; color: #fff; font-size: 0.85rem;
           font-weight: 650; vertical-align: 0.1rem; }
.hw-empty { text-align: center; padding: 1rem; }
.hw-empty .hw-empty-icon { font-size: 2.6rem; line-height: 1; margin-bottom: 0.6rem; opacity: 0.35; }
.hw-empty .hw-empty-title { font-weight: 600; font-size: 1.05rem; }
.hw-empty .hw-empty-text { opacity: 0.6; font-size: 0.92rem; max-width: 26rem; margin: 0.3rem auto 0; }
.hw-grid { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.2rem 0 1.1rem; }
.hw-cell { width: 44px; padding: 5px 0 4px; border-radius: 8px; text-align: center; line-height: 1.15;
           border: 1px solid rgba(128, 128, 128, 0.22); background: rgba(128, 128, 128, 0.04); }
.hw-cell b { display: block; font: 500 18px ui-monospace, SFMono-Regular, Menlo, monospace; }
.hw-cell span { font-size: 10.5px; opacity: 0.6; }
.hw-cell.hw-missing { border: 1px dashed rgba(214, 64, 64, 0.6); background: rgba(214, 64, 64, 0.07);
                      color: rgb(196, 52, 52); }
.hw-group { font-weight: 600; font-size: 0.92rem; margin-top: 0.3rem; }
.hw-group span { font-weight: 400; opacity: 0.55; margin-left: 0.35rem; }
.hw-char { font: 500 30px ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.1; }
.hw-char-hint { font-size: 11px; opacity: 0.6; }
.st-key-page_preview [data-testid="stImage"] img {
    border-radius: 4px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06), 0 8px 28px rgba(16, 24, 40, 0.12); }
</style>
"""


@st.cache_resource(show_spinner=False)
def _store_for(data_dir: str) -> ProfileStore:
    """One store per data directory, shared across reruns (keeps the image cache warm)."""
    return ProfileStore(Path(data_dir))


def get_store() -> ProfileStore:
    """The profile store, or a clear message (and stop) if the data folder can't be used."""
    try:
        return _store_for(str(default_data_dir()))
    except StorageError as exc:
        st.error(str(exc))
    except HandwritingError as exc:
        st.error(f"Could not open the data directory: {exc}")
    st.stop()


def apply_style() -> None:
    st.logo(str(ASSETS / "logo.svg"), size="large")
    st.html(_CSS)


def page_header(title: str, subtitle: str, store: ProfileStore | None = None) -> list[ProfileSummary]:
    """Title row of a page, with the profile switcher on the right when *store* is given.

    Returns the existing profiles (empty when there are none, or no store was given).
    """
    summaries: list[ProfileSummary] = []
    problems: list[str] = []
    if store is not None:
        apply_pending_profile_selection()
        summaries, problems = store.list_profiles()
        ids = [s.profile_id for s in summaries]
        if st.session_state.get(PROFILE_KEY) not in ids:
            st.session_state[PROFILE_KEY] = ids[0] if ids else None

    title_column, menu_column = st.columns([3, 1.3], vertical_alignment="center")
    with title_column:
        st.header(title, anchor=False)
        st.html(f'<p class="hw-subtitle">{subtitle}</p>')
    if summaries:
        with menu_column, st.container(horizontal=True, horizontal_alignment="right"):
            _profile_menu(store, summaries)
    for problem in problems:
        st.warning(problem)
    return summaries


def _profile_menu(store: ProfileStore, summaries: list[ProfileSummary]) -> None:
    by_id = {s.profile_id: s for s in summaries}
    current = by_id[st.session_state[PROFILE_KEY]]
    # The picker has its own key and copies its choice into PROFILE_KEY, so the
    # current profile never depends on a widget that other pages don't draw.
    st.session_state[_PICKER_KEY] = current.profile_id
    with st.popover(current.name, icon=":material/person:", help="Switch or create a handwriting profile"):
        # Labels must not contain changing numbers: Streamlit keeps a selectbox's
        # displayed label when only its format_func output changes.
        st.selectbox("Handwriting profile", list(by_id), key=_PICKER_KEY, on_change=_picked,
                     format_func=lambda pid: by_id[pid].name if pid in by_id else pid)
        st.caption(f"{current.total_samples} samples · {current.character_count} characters")
        st.divider()
        st.markdown("**New profile**")
        new_profile_form(store, "create_profile")


def _picked() -> None:
    if st.session_state.get(_PICKER_KEY):
        st.session_state[PROFILE_KEY] = st.session_state[_PICKER_KEY]


def new_profile_form(store: ProfileStore, key: str) -> None:
    with st.form(key, clear_on_submit=True, border=False):
        name = st.text_input("Profile name", placeholder="e.g. My handwriting")
        submitted = st.form_submit_button("Create profile", type="primary", width="stretch")
    if submitted:
        with friendly_errors("creating the profile"):
            meta = store.create_profile(name)
            select_profile_on_next_run(meta.profile_id)
            st.rerun()


def welcome(store: ProfileStore) -> None:
    """First-run screen: create a profile, and what happens next."""
    _, middle, _ = st.columns([1, 2.2, 1])
    with middle, st.container(border=True):
        st.subheader("Let's set up your handwriting", anchor=False)
        st.markdown("A **profile** holds the characters you write by hand. "
                    "Give it a name to get started. You can have several, e.g. one neat and one messy.")
        new_profile_form(store, "welcome_profile")
    st.space("small")
    steps = [
        (":material/person_add:", "Create a profile", "One per handwriting style."),
        (":material/print:", "Fill in the sample sheet", "Print it, write each character, scan or photograph it."),
        (":material/edit_note:", "Write anything", "Type or paste text and download the pages as PDF or PNG."),
    ]
    for column, (icon, heading, text) in zip(st.columns(3), steps):
        with column, st.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{heading}**")
            st.caption(text)


def step_title(number: int, title: str) -> None:
    st.html(f'<h4 style="margin:0 0 .25rem"><span class="hw-step">{number}</span>{title}</h4>')


def empty_state(icon: str, title: str, text: str) -> None:
    st.html(f'<div class="hw-empty"><div class="hw-empty-icon">{icon}</div>'
            f'<div class="hw-empty-title">{title}</div><div class="hw-empty-text">{text}</div></div>')
