"""Local, file-based storage of handwriting profiles.

Layout on disk::

    <data_dir>/
        profiles/
            <profile-id>/
                profile.json                 # metadata (see models.ProfileMetadata)
                glyphs/
                    lower_a/
                        lower_a_001.png      # transparent RGBA glyph
                        lower_a_002.png
                    upper_a/ ...
        trash/                               # deleted profiles/samples (recoverable)

No database is used. Metadata writes are atomic (write temp file, then rename),
so a crash cannot leave a half-written ``profile.json``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .baseline import descent_px
from .charset import char_to_id, validate_glyph_char
from .template import TEMPLATE_DPI
from .errors import (
    InvalidProfileError,
    NoInkFoundError,
    ProfileExistsError,
    ProfileNotFoundError,
    StorageError,
)
from .models import CharacterEntry, GlyphSample, GlyphSet, LoadedGlyph, ProfileMetadata, SampleSource
from .utils import ensure_writable_dir, rgba_array_to_image, slugify

ENV_DATA_DIR = "HANDWRITING_DATA_DIR"
PROFILE_FILE = "profile.json"
GLYPH_DIR = "glyphs"
PROFILES_DIR = "profiles"
TRASH_DIR = "trash"
_IMAGE_CACHE_SIZE = 5000
_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def default_data_dir() -> Path:
    """Data directory: ``$HANDWRITING_DATA_DIR`` or ``<project>/data``."""
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ProfileSummary:
    """Lightweight description of a profile for selection lists."""

    profile_id: str
    name: str
    total_samples: int
    character_count: int


@dataclass(frozen=True)
class NewSample:
    """A glyph waiting to be saved into a profile."""

    char: str
    rgba: np.ndarray
    x_height_ref: float
    baseline_from_top: float | None = None
    source: str = SampleSource.MANUAL
    capture_dpi: float | None = None


class _ImageCache:
    """Small thread-safe LRU cache of decoded glyph images.

    Keys include the file's modification time and size, so edits on disk are
    picked up automatically.
    """

    def __init__(self, max_items: int) -> None:
        self._items: OrderedDict[tuple[str, int, int], Image.Image] = OrderedDict()
        self._max_items = max_items
        self._lock = threading.Lock()

    def get(self, path: Path) -> Image.Image:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached
        with Image.open(path) as img:
            img.load()
            image = img.convert("RGBA")
        with self._lock:
            self._items[key] = image
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return image

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class ProfileStore:
    """Create, read, update and delete handwriting profiles on disk."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.profiles_dir = self.data_dir / PROFILES_DIR
        self.trash_dir = self.data_dir / TRASH_DIR
        try:
            ensure_writable_dir(self.profiles_dir)
        except OSError as exc:
            raise StorageError(
                f"Cannot write to the data directory {self.profiles_dir}. "
                f"Check its permissions or set {ENV_DATA_DIR} to another folder. ({exc})"
            ) from exc
        self._cache = _ImageCache(_IMAGE_CACHE_SIZE)

    # ------------------------------------------------------------------ profiles

    def list_profiles(self) -> tuple[list[ProfileSummary], list[str]]:
        """Return valid profiles (sorted by name) and messages for broken ones."""
        summaries: list[ProfileSummary] = []
        problems: list[str] = []
        for directory in sorted(p for p in self.profiles_dir.iterdir() if p.is_dir()):
            try:
                meta = self.load_profile(directory.name)
            except InvalidProfileError as exc:
                problems.append(f"Skipped profile folder '{directory.name}': {exc}")
                continue
            except ProfileNotFoundError:
                continue
            summaries.append(ProfileSummary(
                profile_id=meta.profile_id,
                name=meta.name,
                total_samples=meta.total_samples,
                character_count=len(meta.sample_counts()),
            ))
        summaries.sort(key=lambda s: s.name.lower())
        return summaries, problems

    def create_profile(self, name: str) -> ProfileMetadata:
        name = name.strip()
        if not name:
            raise InvalidProfileError("Please enter a profile name.")
        self._ensure_unique_name(name)
        base = slugify(name)
        profile_id, suffix = base, 2
        while (self.profiles_dir / profile_id).exists():
            profile_id, suffix = f"{base}-{suffix}", suffix + 1
        now = _now()
        meta = ProfileMetadata(profile_id=profile_id, name=name, created_at=now, updated_at=now)
        try:
            (self.profiles_dir / profile_id / GLYPH_DIR).mkdir(parents=True)
        except OSError as exc:
            raise StorageError(f"Could not create the profile folder: {exc}") from exc
        self._write_metadata(meta)
        return meta

    def load_profile(self, profile_id: str) -> ProfileMetadata:
        path = self.profile_dir(profile_id) / PROFILE_FILE
        if not path.is_file():
            raise ProfileNotFoundError(f"Profile '{profile_id}' was not found.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidProfileError(f"profile.json could not be read: {exc}") from exc
        meta = ProfileMetadata.from_dict(data)
        if meta.profile_id != profile_id:
            raise InvalidProfileError(
                f"profile.json id '{meta.profile_id}' does not match its folder '{profile_id}'."
            )
        return meta

    def rename_profile(self, profile_id: str, new_name: str) -> ProfileMetadata:
        new_name = new_name.strip()
        if not new_name:
            raise InvalidProfileError("Please enter a profile name.")
        meta = self.load_profile(profile_id)
        if new_name.lower() != meta.name.lower():
            self._ensure_unique_name(new_name)
        meta.name = new_name
        self._write_metadata(meta)
        return meta

    def delete_profile(self, profile_id: str) -> Path:
        """Move a profile to the trash folder and return its new location."""
        source = self.profile_dir(profile_id)
        if not source.is_dir():
            raise ProfileNotFoundError(f"Profile '{profile_id}' was not found.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.trash_dir / f"{profile_id}-{stamp}"
        try:
            self.trash_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        except OSError as exc:
            raise StorageError(f"Could not delete profile: {exc}") from exc
        self._cache.clear()
        return target

    def clear_samples(self, profile_id: str) -> int:
        """Remove every sample but keep the (now empty) profile and its name.

        The glyph images and a copy of the old ``profile.json`` are moved to
        ``trash/<profile-id>/cleared-<timestamp>/`` so the reset can be undone
        by copying them back. Returns the number of samples removed.
        """
        meta = self.load_profile(profile_id)
        removed = meta.total_samples
        directory = self.profile_dir(profile_id)
        glyph_dir = directory / GLYPH_DIR
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = self.trash_dir / profile_id / f"cleared-{stamp}"
        try:
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory / PROFILE_FILE, backup / PROFILE_FILE)
        except OSError as exc:
            raise StorageError(f"Could not back up the profile before clearing it: {exc}") from exc

        # Update metadata first: an empty profile is always a consistent state.
        meta.characters.clear()
        self._write_metadata(meta)
        try:
            if glyph_dir.exists():
                shutil.move(str(glyph_dir), str(backup / GLYPH_DIR))
            glyph_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Samples were removed from the profile, but their files could not be "
                               f"moved to the trash: {exc}") from exc
        self._cache.clear()
        return removed

    def profile_dir(self, profile_id: str) -> Path:
        if not _PROFILE_ID_PATTERN.match(profile_id or ""):
            raise ProfileNotFoundError(f"Invalid profile id {profile_id!r}.")
        return self.profiles_dir / profile_id

    def _ensure_unique_name(self, name: str) -> None:
        summaries, _ = self.list_profiles()
        if any(s.name.lower() == name.lower() for s in summaries):
            raise ProfileExistsError(f"A profile named '{name}' already exists.")

    def _write_metadata(self, meta: ProfileMetadata) -> None:
        meta.updated_at = _now()
        directory = self.profile_dir(meta.profile_id)
        payload = json.dumps(meta.to_dict(), ensure_ascii=False, indent=2)
        _atomic_write(directory / PROFILE_FILE, payload.encode("utf-8"))

    # ------------------------------------------------------------------ samples

    def add_samples(self, profile_id: str, samples: Iterable[NewSample]) -> list[GlyphSample]:
        """Save glyph images into a profile and update its metadata."""
        meta = self.load_profile(profile_id)
        directory = self.profile_dir(profile_id)
        added: list[GlyphSample] = []
        try:
            for sample in samples:
                char = validate_glyph_char(sample.char)
                rgba = np.asarray(sample.rgba)
                if rgba.ndim != 3 or rgba.shape[2] != 4 or not rgba[:, :, 3].any():
                    raise NoInkFoundError(f"Sample for {char!r} contains no visible ink.")
                entry = meta.characters.setdefault(
                    char, CharacterEntry(char=char, char_id=char_to_id(char))
                )
                char_dir = directory / GLYPH_DIR / entry.char_id
                char_dir.mkdir(parents=True, exist_ok=True)
                sample_id = _next_sample_id(entry, char_dir)
                relative = f"{GLYPH_DIR}/{entry.char_id}/{sample_id}.png"
                _save_png(rgba, directory / relative)
                glyph = GlyphSample(
                    sample_id=sample_id,
                    file=relative,
                    width=int(rgba.shape[1]),
                    height=int(rgba.shape[0]),
                    x_height_ref=float(sample.x_height_ref),
                    baseline_from_top=sample.baseline_from_top,
                    source=sample.source,
                    added_at=_now(),
                    capture_dpi=sample.capture_dpi,
                )
                entry.samples.append(glyph)
                added.append(glyph)
        except OSError as exc:
            raise StorageError(f"Could not save samples: {exc}") from exc
        finally:
            # Persist whatever was written, even if a later sample failed.
            if added:
                self._write_metadata(meta)
        return added

    def delete_sample(self, profile_id: str, char: str, sample_id: str) -> None:
        """Remove one sample; its PNG is moved to the trash folder."""
        meta = self.load_profile(profile_id)
        entry = meta.characters.get(char)
        sample = next((s for s in entry.samples if s.sample_id == sample_id), None) if entry else None
        if entry is None or sample is None:
            raise InvalidProfileError(f"Sample {sample_id!r} for {char!r} was not found.")
        entry.samples.remove(sample)
        if not entry.samples:
            del meta.characters[char]
        self._write_metadata(meta)

        path = self.profile_dir(profile_id) / sample.file
        if path.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            target = self.trash_dir / profile_id / f"{stamp}-{path.name}"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
            except OSError as exc:
                raise StorageError(f"Sample removed from profile, but its file could not be moved: {exc}") from exc

    def delete_character_samples(self, profile_id: str, chars: Iterable[str]) -> int:
        """Remove every sample of the given characters (files go to the trash)."""
        removed = 0
        for char in set(chars):
            meta = self.load_profile(profile_id)
            entry = meta.characters.get(char)
            for sample in list(entry.samples) if entry else []:
                self.delete_sample(profile_id, char, sample.sample_id)
                removed += 1
        return removed

    def template_x_height_reference(self, profile_id: str) -> float | None:
        """Median x-height of samples imported from sample sheets, if any.

        Expressed in 300 DPI template pixels (samples read at a higher
        resolution are converted), so a later import of e.g. only the
        punctuation page keeps consistent sizes.
        """
        meta = self.load_profile(profile_id)
        values = sorted(
            s.x_height_ref * TEMPLATE_DPI / (s.capture_dpi or TEMPLATE_DPI)
            for entry in meta.characters.values()
            for s in entry.samples
            if s.source == SampleSource.TEMPLATE
        )
        return values[len(values) // 2] if values else None

    def load_sample_image(self, profile_id: str, sample: GlyphSample) -> Image.Image:
        """Load a sample's RGBA image (cached)."""
        return self._cache.get(self.profile_dir(profile_id) / sample.file)

    def load_glyph_set(self, profile_id: str) -> GlyphSet:
        """Load every sample of a profile into a renderer-ready :class:`GlyphSet`.

        Unreadable sample files are skipped and reported in ``GlyphSet.problems``.
        """
        meta = self.load_profile(profile_id)
        glyphs: dict[str, list[LoadedGlyph]] = {}
        problems: list[str] = []
        for char, entry in meta.characters.items():
            for sample in entry.samples:
                try:
                    image = self.load_sample_image(profile_id, sample)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    problems.append(f"{char!r} sample {sample.sample_id}: unreadable ({exc.__class__.__name__})")
                    continue
                scale = image.height / sample.height if sample.height else 1.0
                x_height = sample.x_height_ref * scale
                baseline = None if sample.baseline_from_top is None else sample.baseline_from_top * scale
                glyphs.setdefault(char, []).append(LoadedGlyph(
                    char=char,
                    sample_id=sample.sample_id,
                    image=image,
                    x_height_ref=x_height,
                    descent=descent_px(char, image.height, x_height, baseline),
                ))
        return GlyphSet(glyphs, name=meta.name, problems=problems)

    def clear_cache(self) -> None:
        self._cache.clear()


def _next_sample_id(entry: CharacterEntry, char_dir: Path) -> str:
    used: set[int] = set()
    prefix = f"{entry.char_id}_"
    names = [s.sample_id for s in entry.samples] + [p.stem for p in char_dir.glob("*.png")]
    for name in names:
        suffix = name[len(prefix):] if name.startswith(prefix) else ""
        if suffix.isdigit():
            used.add(int(suffix))
    number = max(used, default=0) + 1
    return f"{entry.char_id}_{number:03d}"


def _save_png(rgba: np.ndarray, path: Path) -> None:
    buffer_path = path.with_suffix(".tmp")
    rgba_array_to_image(rgba).save(buffer_path, format="PNG")
    os.replace(buffer_path, path)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except OSError as exc:
        Path(tmp_name).unlink(missing_ok=True)
        raise StorageError(f"Could not write {path.name}: {exc}") from exc
