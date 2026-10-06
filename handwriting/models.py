"""Data models for handwriting profiles and renderer-ready glyphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .charset import GLYPH_ALIASES, char_to_id, validate_glyph_char
from .errors import InvalidProfileError

SCHEMA_VERSION = 1


class SampleSource:
    """Where a glyph sample came from (stored as plain strings in JSON)."""

    MANUAL = "manual"
    TEMPLATE = "template"


@dataclass
class GlyphSample:
    """Metadata for one stored handwriting sample (one PNG file).

    Attributes:
        sample_id: unique id inside the profile, equal to the file stem.
        file: path of the PNG relative to the profile directory (POSIX style).
        width / height: size of the cropped glyph image in pixels.
        x_height_ref: the writer's x-height in this image's pixel scale; used to
            normalise samples captured at different resolutions.
        baseline_from_top: baseline position from the top of the glyph, when
            known from a sample-sheet guide line.
        capture_dpi: resolution the sheet was read at (sheet samples only;
            older profiles without it were read at 300 DPI).
    """

    sample_id: str
    file: str
    width: int
    height: int
    x_height_ref: float
    baseline_from_top: float | None = None
    source: str = SampleSource.MANUAL
    added_at: str = ""
    capture_dpi: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "x_height_ref": round(self.x_height_ref, 3),
            "baseline_from_top": (
                None if self.baseline_from_top is None else round(self.baseline_from_top, 3)
            ),
            "source": self.source,
            "added_at": self.added_at,
            "capture_dpi": None if self.capture_dpi is None else round(self.capture_dpi, 1),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GlyphSample:
        try:
            sample = cls(
                sample_id=str(data["sample_id"]),
                file=str(data["file"]),
                width=int(data["width"]),
                height=int(data["height"]),
                x_height_ref=float(data["x_height_ref"]),
                baseline_from_top=(
                    None if data.get("baseline_from_top") is None
                    else float(data["baseline_from_top"])
                ),
                source=str(data.get("source", SampleSource.MANUAL)),
                added_at=str(data.get("added_at", "")),
                capture_dpi=None if data.get("capture_dpi") is None else float(data["capture_dpi"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidProfileError(f"Invalid glyph sample entry: {exc}") from exc
        if sample.capture_dpi is not None and sample.capture_dpi <= 0:
            raise InvalidProfileError(f"Glyph sample {sample.sample_id!r} has an invalid capture resolution.")
        if sample.width <= 0 or sample.height <= 0 or sample.x_height_ref <= 0:
            raise InvalidProfileError(f"Glyph sample {sample.sample_id!r} has invalid dimensions.")
        if ".." in sample.file.split("/") or sample.file.startswith("/"):
            raise InvalidProfileError(f"Glyph sample {sample.sample_id!r} has an unsafe path.")
        return sample


@dataclass
class CharacterEntry:
    """All samples for one character."""

    char: str
    char_id: str
    samples: list[GlyphSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "char_id": self.char_id,
            "samples": [s.to_dict() for s in self.samples],
        }


@dataclass
class ProfileMetadata:
    """Contents of a profile's ``profile.json``."""

    profile_id: str
    name: str
    created_at: str
    updated_at: str
    characters: dict[str, CharacterEntry] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def sample_counts(self) -> dict[str, int]:
        return {c: len(e.samples) for c, e in self.characters.items() if e.samples}

    @property
    def total_samples(self) -> int:
        return sum(len(e.samples) for e in self.characters.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.profile_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "characters": {c: e.to_dict() for c, e in sorted(self.characters.items())},
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProfileMetadata:
        if not isinstance(data, Mapping):
            raise InvalidProfileError("Profile metadata must be a JSON object.")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise InvalidProfileError(f"Unsupported profile schema version: {version!r}.")
        try:
            profile_id = str(data["id"])
            name = str(data["name"])
            raw_chars = data.get("characters", {})
        except KeyError as exc:
            raise InvalidProfileError(f"Profile metadata is missing field {exc}.") from exc
        if not isinstance(raw_chars, Mapping):
            raise InvalidProfileError("Profile 'characters' must be a JSON object.")

        characters: dict[str, CharacterEntry] = {}
        for char, entry in raw_chars.items():
            try:
                char = validate_glyph_char(char)
            except Exception as exc:
                raise InvalidProfileError(f"Invalid character key {char!r} in profile.") from exc
            if not isinstance(entry, Mapping) or not isinstance(entry.get("samples", []), list):
                raise InvalidProfileError(f"Invalid entry for character {char!r}.")
            samples = [GlyphSample.from_dict(s) for s in entry.get("samples", [])]
            characters[char] = CharacterEntry(char=char, char_id=char_to_id(char), samples=samples)

        return cls(
            profile_id=profile_id,
            name=name,
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            characters=characters,
        )


@dataclass(frozen=True)
class LoadedGlyph:
    """A glyph image ready for rendering.

    ``image`` is RGBA with the ink coverage in the alpha channel. ``descent``
    is the distance from the baseline to the glyph's bottom edge, in the
    image's own pixels (see :mod:`handwriting.baseline`).
    """

    char: str
    sample_id: str
    image: Image.Image
    x_height_ref: float
    descent: float

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


class GlyphSet:
    """Immutable mapping from characters to their loaded glyph variants."""

    def __init__(self, glyphs: Mapping[str, Iterable[LoadedGlyph]], name: str = "",
                 problems: Iterable[str] = ()) -> None:
        self._glyphs: dict[str, tuple[LoadedGlyph, ...]] = {}
        for char, variants in glyphs.items():
            variants = tuple(variants)
            if variants:
                self._glyphs[char] = variants
        self.name = name
        self.problems: tuple[str, ...] = tuple(problems)

    def variants(self, char: str) -> tuple[LoadedGlyph, ...]:
        """Variants for *char*, falling back to a visual alias (e.g. ’ -> ')."""
        found = self._glyphs.get(char)
        if found:
            return found
        alias = GLYPH_ALIASES.get(char)
        if alias is not None:
            return self._glyphs.get(alias, ())
        return ()

    def has(self, char: str) -> bool:
        return bool(self.variants(char))

    @property
    def characters(self) -> frozenset[str]:
        return frozenset(self._glyphs)

    def __len__(self) -> int:
        return len(self._glyphs)
