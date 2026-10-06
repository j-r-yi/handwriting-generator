"""Exception hierarchy.

Every error a user can reasonably trigger (bad upload, missing profile, empty
text, ...) derives from :class:`HandwritingError` and carries a message that is
safe to show directly in the UI.
"""

from __future__ import annotations


class HandwritingError(Exception):
    """Base class for expected, user-facing errors."""


class StorageError(HandwritingError):
    """The data directory cannot be read from or written to."""


class ProfileNotFoundError(HandwritingError):
    """The requested handwriting profile does not exist."""


class ProfileExistsError(HandwritingError):
    """A profile with the same name already exists."""


class InvalidProfileError(HandwritingError):
    """Profile metadata is missing, corrupted, or has an unexpected structure."""


class InvalidCharacterError(HandwritingError):
    """A character cannot be stored as a glyph (whitespace, control code, ...)."""


class ImageDecodeError(HandwritingError):
    """An uploaded file is corrupted or not a supported image format."""


class NoInkFoundError(HandwritingError):
    """No visible handwriting could be found in an image."""


class SheetDetectionError(HandwritingError):
    """A completed sample sheet could not be located or aligned in a photo/scan."""


class EmptyTextError(HandwritingError):
    """The text to render is empty or contains only whitespace."""


class MissingGlyphError(HandwritingError):
    """Text contains characters with no samples and the policy forbids fallbacks."""

    def __init__(self, missing: dict[str, int]) -> None:
        self.missing = missing
        shown = " ".join(repr(c) for c in sorted(missing))
        super().__init__(f"No handwriting samples for: {shown}")
