"""Profile persistence, sample loading, safe file names and error handling."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from handwriting.charset import DEFAULT_CHARSET, char_to_id
from handwriting.errors import (
    InvalidCharacterError,
    InvalidProfileError,
    NoInkFoundError,
    ProfileExistsError,
    ProfileNotFoundError,
    StorageError,
)
from handwriting.sample_store import NewSample, ProfileStore
from tests.helpers import make_new_samples


def test_create_list_and_reload_profile(store: ProfileStore, tmp_path: Path) -> None:
    meta = store.create_profile("My Handwriting")
    assert meta.profile_id == "my-handwriting"
    store.add_samples(meta.profile_id, make_new_samples("ab", variants=2))

    # A brand-new store on the same folder sees everything (simulates restarting the app).
    reopened = ProfileStore(tmp_path / "data")
    summaries, problems = reopened.list_profiles()
    assert problems == []
    assert [(s.name, s.total_samples, s.character_count) for s in summaries] == [("My Handwriting", 4, 2)]
    loaded = reopened.load_profile(meta.profile_id)
    assert loaded.sample_counts() == {"a": 2, "b": 2}


def test_glyph_set_loads_rgba_images_with_metrics(store: ProfileStore) -> None:
    meta = store.create_profile("Glyphs")
    store.add_samples(meta.profile_id, make_new_samples("ag", variants=3))
    glyphs = store.load_glyph_set(meta.profile_id)
    assert glyphs.characters == {"a", "g"}
    variants = glyphs.variants("g")
    assert len(variants) == 3
    assert all(v.image.mode == "RGBA" and v.x_height_ref > 0 for v in variants)
    assert all(v.descent > 0 for v in variants)  # g hangs below the baseline
    assert all(v.descent == 0 for v in glyphs.variants("a"))


def test_case_sensitive_characters_get_distinct_safe_folders(store: ProfileStore) -> None:
    meta = store.create_profile("Case")
    store.add_samples(meta.profile_id, make_new_samples("aA/", variants=1))
    glyph_root = store.profile_dir(meta.profile_id) / "glyphs"
    assert sorted(p.name for p in glyph_root.iterdir()) == ["lower_a", "slash", "upper_a"]
    assert (glyph_root / "upper_a" / "upper_a_001.png").is_file()


def test_char_ids_are_unique_and_filesystem_safe() -> None:
    ids = [char_to_id(c) for c in DEFAULT_CHARSET]
    assert len(set(i.lower() for i in ids)) == len(ids)  # unique even on case-insensitive disks
    assert all(i.replace("_", "").isalnum() and i.isascii() for i in ids)
    assert char_to_id("é") == "u00e9"
    for bad in [" ", "\n", "ab", ""]:
        with pytest.raises(InvalidCharacterError):
            char_to_id(bad)


def test_sample_numbers_continue_after_deletion(store: ProfileStore) -> None:
    meta = store.create_profile("Numbers")
    added = store.add_samples(meta.profile_id, make_new_samples("e", variants=3))
    assert [s.sample_id for s in added] == ["lower_e_001", "lower_e_002", "lower_e_003"]
    store.delete_sample(meta.profile_id, "e", "lower_e_003")
    more = store.add_samples(meta.profile_id, make_new_samples("e", variants=1))
    assert more[0].sample_id == "lower_e_003"  # file was moved away, number is free again
    assert store.load_profile(meta.profile_id).sample_counts() == {"e": 3}


def test_delete_sample_moves_file_to_trash(store: ProfileStore) -> None:
    meta = store.create_profile("Trash")
    added = store.add_samples(meta.profile_id, make_new_samples("q", variants=2))
    path = store.profile_dir(meta.profile_id) / added[0].file
    store.delete_sample(meta.profile_id, "q", added[0].sample_id)
    assert not path.exists()
    assert list((store.trash_dir / meta.profile_id).glob("*lower_q_001.png"))
    assert store.load_profile(meta.profile_id).sample_counts() == {"q": 1}


def test_rename_and_duplicate_names(store: ProfileStore) -> None:
    first = store.create_profile("Alpha")
    store.create_profile("Beta")
    with pytest.raises(ProfileExistsError):
        store.create_profile("alpha")
    with pytest.raises(ProfileExistsError):
        store.rename_profile(first.profile_id, "BETA")
    renamed = store.rename_profile(first.profile_id, "Gamma")
    assert renamed.profile_id == first.profile_id and renamed.name == "Gamma"
    assert store.load_profile(first.profile_id).name == "Gamma"


def test_delete_profile_moves_it_to_trash(store: ProfileStore) -> None:
    meta = store.create_profile("Gone")
    store.add_samples(meta.profile_id, make_new_samples("z", variants=1))
    target = store.delete_profile(meta.profile_id)
    assert target.is_dir() and (target / "profile.json").is_file()
    assert store.list_profiles()[0] == []
    with pytest.raises(ProfileNotFoundError):
        store.load_profile(meta.profile_id)


def test_invalid_metadata_is_reported_not_crashing(store: ProfileStore) -> None:
    good = store.create_profile("Good")
    bad = store.create_profile("Bad")
    (store.profile_dir(bad.profile_id) / "profile.json").write_text("{ not json", encoding="utf-8")
    summaries, problems = store.list_profiles()
    assert [s.profile_id for s in summaries] == [good.profile_id]
    assert len(problems) == 1 and "bad" in problems[0]
    with pytest.raises(InvalidProfileError):
        store.load_profile(bad.profile_id)


@pytest.mark.parametrize("payload", [
    [],
    {"schema_version": 99, "id": "x", "name": "x"},
    {"schema_version": 1, "id": "wrong-id", "name": "x", "characters": {}},
    {"schema_version": 1, "id": "p", "name": "x", "characters": {"a": {"samples": [{"file": "../evil.png"}]}}},
])
def test_structurally_invalid_metadata_raises(store: ProfileStore, payload) -> None:
    directory = store.profiles_dir / "p"
    directory.mkdir()
    (directory / "profile.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InvalidProfileError):
        store.load_profile("p")


def test_corrupted_sample_file_is_skipped_with_message(store: ProfileStore) -> None:
    meta = store.create_profile("Corrupt")
    added = store.add_samples(meta.profile_id, make_new_samples("c", variants=2))
    (store.profile_dir(meta.profile_id) / added[0].file).write_bytes(b"garbage")
    glyphs = store.load_glyph_set(meta.profile_id)
    assert len(glyphs.variants("c")) == 1
    assert len(glyphs.problems) == 1 and "lower_c_001" in glyphs.problems[0]


def test_empty_sample_is_rejected(store: ProfileStore) -> None:
    meta = store.create_profile("Empty")
    blank = np.zeros((10, 10, 4), dtype=np.uint8)
    with pytest.raises(NoInkFoundError):
        store.add_samples(meta.profile_id, [NewSample("a", blank, 10.0)])


def test_unknown_profile_ids_are_rejected(store: ProfileStore) -> None:
    for bad in ["missing", "../etc", "UPPER", ""]:
        with pytest.raises(ProfileNotFoundError):
            store.load_profile(bad)


def test_template_x_height_reference(store: ProfileStore) -> None:
    meta = store.create_profile("Ref")
    assert store.template_x_height_reference(meta.profile_id) is None
    rgba = np.zeros((20, 10, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    store.add_samples(meta.profile_id, [NewSample("a", rgba, 40.0, source="template"),
                                        NewSample("b", rgba, 44.0, source="template"),
                                        NewSample("c", rgba, 99.0, source="manual")])
    assert store.template_x_height_reference(meta.profile_id) == 44.0


@pytest.mark.skipif(sys.platform.startswith("win") or os.geteuid() == 0,
                    reason="POSIX permissions required (and root ignores them)")
def test_unwritable_storage_raises_storage_error(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(StorageError):
            ProfileStore(locked)
    finally:
        locked.chmod(stat.S_IRWXU)


def test_clear_samples_empties_profile_but_keeps_it(store: ProfileStore, tmp_path: Path) -> None:
    meta = store.create_profile("Reset Me")
    store.add_samples(meta.profile_id, make_new_samples("abA", variants=2))
    assert store.clear_samples(meta.profile_id) == 6

    reopened = ProfileStore(tmp_path / "data")  # survives an app restart
    cleared = reopened.load_profile(meta.profile_id)
    assert cleared.name == "Reset Me" and cleared.total_samples == 0
    assert len(reopened.load_glyph_set(meta.profile_id)) == 0
    glyph_dir = reopened.profile_dir(meta.profile_id) / "glyphs"
    assert glyph_dir.is_dir() and not any(glyph_dir.iterdir())

    # Everything removed is recoverable from the trash.
    backups = list((reopened.trash_dir / meta.profile_id).glob("cleared-*"))
    assert len(backups) == 1
    assert len(list((backups[0] / "glyphs").rglob("*.png"))) == 6
    old = json.loads((backups[0] / "profile.json").read_text(encoding="utf-8"))
    assert set(old["characters"]) == {"a", "b", "A"}

    # The profile is immediately usable again, numbering restarts at 001.
    added = reopened.add_samples(meta.profile_id, make_new_samples("a", variants=1))
    assert added[0].sample_id == "lower_a_001"


def test_clear_samples_leaves_other_profiles_alone(store: ProfileStore) -> None:
    keep = store.create_profile("Keep")
    wipe = store.create_profile("Wipe")
    store.add_samples(keep.profile_id, make_new_samples("k", variants=2))
    store.add_samples(wipe.profile_id, make_new_samples("w", variants=2))
    store.clear_samples(wipe.profile_id)
    assert store.load_profile(keep.profile_id).sample_counts() == {"k": 2}
    assert store.load_profile(wipe.profile_id).sample_counts() == {}


def test_clear_unknown_profile_raises(store: ProfileStore) -> None:
    with pytest.raises(ProfileNotFoundError):
        store.clear_samples("does-not-exist")
