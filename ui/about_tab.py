"""Settings page: output quality, storage, privacy and responsible-use information."""

from __future__ import annotations

import platform

import cv2
import numpy as np
import PIL
import streamlit as st

from handwriting.renderer import max_pages
from handwriting.sample_store import ProfileStore
from handwriting.settings import DEFAULT_DPI

from .common import dpi_control
from .shell import page_header


def render_about_tab(store: ProfileStore) -> None:
    page_header("Settings", "Output quality, where your data lives, and privacy.")

    with st.container(border=True):
        st.markdown("**Output resolution**")
        dpi = dpi_control("render_dpi", "Output resolution (DPI)", label_visibility="collapsed")
        st.caption("300 DPI prints crisply. 600 DPI stays sharp when zoomed in, especially with samples "
                   f"scanned at 600 DPI, but files are about 4× larger and at most {max_pages(600)} pages are "
                   "written at once. Lower values are faster.")
        if dpi > DEFAULT_DPI:
            st.caption(f"At {dpi} DPI: up to {max_pages(dpi)} pages per generation.")

    with st.container(border=True):
        st.markdown("**Storage**")
        st.code(str(store.data_dir), language=None)
        st.caption("Profiles live in `profiles/`. Deleted profiles and samples are moved to `trash/` and can be "
                   "restored by moving them back. Set the `HANDWRITING_DATA_DIR` environment variable to store "
                   "data elsewhere.")
        if st.button("Clear image cache", icon=":material/cached:"):
            store.clear_cache()
            st.cache_data.clear()
            st.toast("Cache cleared.")

    with st.container(border=True):
        st.markdown("**About, privacy & responsible use**")
        st.markdown(
            """
Handwriting Generator builds pages of text from *your own* handwritten character samples.
It is not an AI model: it cuts out the characters you wrote and arranges them, choosing
between your variants and adding small, controlled variation.

- **Private.** Everything runs on this computer. Your samples, scans and text are never sent
  anywhere. The app makes no network requests of its own and has no telemetry. Streamlit's
  usage statistics are switched off and the server only listens on `localhost`.
- **Exact text.** Nothing is rewritten, corrected or left out. A character without samples
  is drawn as a red placeholder box (or a typed fallback, if you choose).
- **Use it responsibly.** Only use your own handwriting, or someone else's with their permission.
  Don't pass off generated pages as handwritten where that matters (e.g. schoolwork or exams that
  must be handwritten), and never use it to imitate someone else or to forge signatures or documents.
"""
        )
        st.caption(f"Python {platform.python_version()} · Streamlit {st.__version__} · Pillow {PIL.__version__} · "
                   f"OpenCV {cv2.__version__} · NumPy {np.__version__}")
