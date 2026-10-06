"""Handwriting Generator core library.

Composes typed text from a person's own handwritten character samples.
Everything here is independent of the Streamlit UI and runs fully offline.

Typical use::

    from handwriting import ProfileStore, RenderSettings, render_text

    store = ProfileStore()
    glyphs = store.load_glyph_set("my-handwriting")
    result = render_text("Hello world", glyphs, RenderSettings(seed=7))
    result.pages[0].save("page.png")
"""

from .models import GlyphSet, LoadedGlyph
from .renderer import RenderResult, find_missing_characters, render_text
from .sample_store import NewSample, ProfileStore
from .settings import (
    MissingGlyphPolicy,
    PageFormat,
    PageSettings,
    PaperStyle,
    RenderSettings,
    VariantMode,
    VariationSettings,
)

__all__ = [
    "GlyphSet",
    "LoadedGlyph",
    "MissingGlyphPolicy",
    "NewSample",
    "PageFormat",
    "PageSettings",
    "PaperStyle",
    "ProfileStore",
    "RenderResult",
    "RenderSettings",
    "VariantMode",
    "VariationSettings",
    "find_missing_characters",
    "render_text",
]

__version__ = "1.0.0"
