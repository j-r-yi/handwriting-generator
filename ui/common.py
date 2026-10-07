"""Helpers shared by the Streamlit tabs."""

from __future__ import annotations

import html
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import streamlit as st
from PIL import Image

from handwriting.charset import describe_char
from handwriting.errors import HandwritingError
from handwriting.settings import DEFAULT_DPI

PROFILE_KEY = "profile_id"
_PENDING_PROFILE_KEY = "pending_profile_id"
DPI_KEY = "render_dpi_value"
DPI_OPTIONS = [150, 200, 300, 600]
_DPI_LABELS = {150: "150 · Draft", 200: "200", 300: "300 · Print", 600: "600 · Ultra sharp"}
UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif"]
SHEET_UPLOAD_TYPES = [*UPLOAD_TYPES, "pdf"]
THUMBNAIL_HEIGHT = 90


@contextmanager
def friendly_errors(action: str) -> Iterator[None]:
    """Show expected errors as plain messages; unexpected ones with details tucked away."""
    try:
        yield
    except HandwritingError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - last-resort UI guard
        st.error(f"Something went wrong while {action}: {exc}")
        with st.expander("Technical details"):
            st.exception(exc)


def dpi_control(key: str, label: str, help: str | None = None, label_visibility: str = "visible") -> int:
    """An output resolution picker. All of them share one value (``DPI_KEY``), so each page's picker
    has its own widget *key*, is filled from the shared value before it is drawn, and copies its
    choice back when changed."""
    st.session_state.setdefault(DPI_KEY, DEFAULT_DPI)
    st.session_state[key] = st.session_state[DPI_KEY]

    def picked() -> None:
        st.session_state[DPI_KEY] = st.session_state[key]

    st.segmented_control(label, DPI_OPTIONS, key=key, required=True, on_change=picked,
                         format_func=_DPI_LABELS.get, help=help, label_visibility=label_visibility)
    return st.session_state[DPI_KEY]


def current_profile_id() -> str | None:
    return st.session_state.get(PROFILE_KEY)


def select_profile_on_next_run(profile_id: str | None) -> None:
    """Ask the sidebar to select *profile_id* on the next rerun.

    A widget's session-state value cannot be changed after the widget has been
    drawn in the current run, so the change is applied before it is drawn.
    """
    st.session_state[_PENDING_PROFILE_KEY] = profile_id


def apply_pending_profile_selection() -> None:
    if _PENDING_PROFILE_KEY in st.session_state:
        st.session_state[PROFILE_KEY] = st.session_state.pop(_PENDING_PROFILE_KEY)


def glyph_thumbnail(glyph: Image.Image | np.ndarray, height: int = THUMBNAIL_HEIGHT) -> Image.Image:
    """A glyph centred on a white square tile, so it is visible in any theme and grids line up."""
    if isinstance(glyph, np.ndarray):
        glyph = Image.fromarray(glyph)
    glyph = glyph.convert("RGBA")
    scale = min(1.0, height / max(1, glyph.height), height / max(1, glyph.width))
    if scale < 1.0:
        glyph = glyph.resize((max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale))),
                             Image.Resampling.LANCZOS)
    side = height + 12
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.alpha_composite(glyph, ((side - glyph.width) // 2, (side - glyph.height) // 2))
    return canvas.convert("RGB")


def preview_image(page: Image.Image, width: int = 2000) -> Image.Image:
    """Downscaled copy of a page for on-screen display (sharp on high-DPI screens)."""
    if page.width <= width:
        return page
    return page.resize((width, round(page.height * width / page.width)), Image.Resampling.LANCZOS)


def coverage_grid(chars: list[str], counts: dict[str, int]) -> str:
    """HTML grid showing how many samples each character has (styled by ui.shell)."""
    cells = []
    for char in chars:
        count = counts.get(char, 0)
        css = "hw-cell" if count else "hw-cell hw-missing"
        cells.append(f'<div class="{css}"><b>{html.escape(char)}</b><span>{count or "–"}</span></div>')
    return '<div class="hw-grid">' + "".join(cells) + "</div>"


def char_label_html(char: str) -> str:
    """Large, escaped character label with its description (safe for any character)."""
    description = describe_char(char)
    hint = description[len(char):].strip(" ()")
    hint_html = f'<div class="hw-char-hint">{html.escape(hint)}</div>' if hint else ""
    return f'<div class="hw-char">{html.escape(char)}</div>{hint_html}'
