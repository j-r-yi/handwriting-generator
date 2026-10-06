"""Handwriting Generator – local Streamlit interface.

Run with:  python -m streamlit run app.py

All processing happens on this computer; nothing is uploaded anywhere.
The rendering/extraction logic lives in the ``handwriting`` package and does
not depend on Streamlit. Each page is a small script in ``ui/pages`` that calls
the matching ``ui`` module.
"""

from __future__ import annotations

import streamlit as st

from ui.shell import ASSETS, apply_style, get_store

PAGES = [
    st.Page("ui/pages/write.py", title="Write", icon=":material/edit_note:", default=True),
    st.Page("ui/pages/handwriting.py", title="My handwriting", icon=":material/gesture:"),
    st.Page("ui/pages/sample_sheet.py", title="Sample sheet", icon=":material/grid_on:"),
    st.Page("ui/pages/settings.py", title="Settings", icon=":material/settings:"),
]


def main() -> None:
    st.set_page_config(page_title="Handwriting Generator", page_icon=str(ASSETS / "icon.svg"), layout="wide")
    apply_style()
    get_store()  # surfaces a broken data folder before any page runs
    st.navigation(PAGES, position="top").run()


if __name__ == "__main__":
    main()
