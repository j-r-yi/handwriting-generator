"""Supported characters and filesystem-safe character identifiers.

Glyph folders are named by a *character id* rather than the character itself,
because characters such as ``/``, ``?`` or ``:`` are illegal in file names on
some systems, and because macOS/Windows file systems are case-insensitive by
default (``a`` and ``A`` would collide).
"""

from __future__ import annotations

import unicodedata

from .errors import InvalidCharacterError

LOWERCASE: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")
UPPERCASE: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS: tuple[str, ...] = tuple("0123456789")
PUNCTUATION: tuple[str, ...] = tuple(".,!?;:'\"-_()[]{}/\\|&@#$%*+=<>^~`")

CHARACTER_GROUPS: dict[str, tuple[str, ...]] = {
    "Lowercase": LOWERCASE,
    "Uppercase": UPPERCASE,
    "Digits": DIGITS,
    "Punctuation": PUNCTUATION,
}

DEFAULT_CHARSET: tuple[str, ...] = LOWERCASE + UPPERCASE + DIGITS + PUNCTUATION

_PUNCTUATION_IDS: dict[str, str] = {
    ".": "period",
    ",": "comma",
    "!": "exclamation",
    "?": "question",
    ";": "semicolon",
    ":": "colon",
    "'": "apostrophe",
    '"': "double_quote",
    "-": "hyphen",
    "_": "underscore",
    "(": "paren_left",
    ")": "paren_right",
    "[": "bracket_left",
    "]": "bracket_right",
    "{": "brace_left",
    "}": "brace_right",
    "/": "slash",
    "\\": "backslash",
    "|": "vertical_bar",
    "&": "ampersand",
    "@": "at_sign",
    "#": "hash",
    "$": "dollar",
    "%": "percent",
    "*": "asterisk",
    "+": "plus",
    "=": "equals",
    "<": "less_than",
    ">": "greater_than",
    "^": "caret",
    "~": "tilde",
    "`": "backtick",
}

# Human-friendly hints for characters that are easy to confuse on a sample sheet.
_DESCRIPTIONS: dict[str, str] = {
    "l": "lowercase L",
    "I": "capital i",
    "1": "digit one",
    "0": "digit zero",
    "O": "capital o",
    "o": "lowercase o",
    "|": "vertical bar",
    "'": "apostrophe",
    '"': "double quote",
    "`": "backtick",
    "-": "hyphen",
    "_": "underscore",
    ",": "comma",
    ".": "period",
    "\\": "backslash",
}

# Typographic characters that are visually rendered with an ASCII handwritten
# sample when the profile has no dedicated sample for them. The *text* is never
# changed; this only decides which handwritten image represents the character.
GLYPH_ALIASES: dict[str, str] = {
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark / apostrophe
    "‚": ",",  # single low-9 quotation mark
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
}


def validate_glyph_char(char: str) -> str:
    """Return *char* if it can have handwriting samples, else raise.

    Text is normalised to NFC so that composed and decomposed forms of the same
    accented letter share samples.
    """
    if not isinstance(char, str):
        raise InvalidCharacterError("Character must be a string.")
    normalized = unicodedata.normalize("NFC", char)
    if len(normalized) != 1:
        raise InvalidCharacterError(
            f"Expected exactly one character, got {len(normalized)}: {char!r}."
        )
    if normalized.isspace():
        raise InvalidCharacterError("Whitespace characters do not need samples.")
    if unicodedata.category(normalized).startswith("C"):
        raise InvalidCharacterError(f"Control/unassigned character {char!r} cannot be used.")
    return normalized


# Ready-made choices for adding a symbol beyond the standard set: characters
# commonly typed with Option/AltGr, produced by autocorrect, or in other languages.
COMMON_SYMBOLS: tuple[str, ...] = tuple(
    "•→←↑↓°±×÷≈≠≤≥√∞πµ€£¥¢©®™§¶…–—“”‘’«»¿¡éèêëàâäáçñöôóüûúïîíßæø"
)

# Emoji and picture symbols: not keyboard characters, and often several code points.
_PICTOGRAPH_RANGES: tuple[tuple[int, int], ...] = (
    (0x2600, 0x27BF),  # miscellaneous symbols, dingbats
    (0x2B00, 0x2BFF),  # arrows/shapes used as emoji (⭐ ⬛)
    (0xFE00, 0xFE0F),  # emoji variation selectors
    (0x1F000, 0x1FFFF),  # emoji, pictographs, playing cards, …
    (0xE0000, 0xE007F),  # tag characters (flag sequences)
)


def validate_symbol_char(char: str) -> str:
    """Like :func:`validate_glyph_char`, but only for characters typed on a keyboard.

    Used when a user adds a new symbol: arrows, bullets, maths, currency and
    accented letters are fine; emoji, pictographs and lone combining marks are not.
    """
    if not isinstance(char, str):
        raise InvalidCharacterError("Character must be a string.")
    if any(low <= ord(c) <= high for c in char for low, high in _PICTOGRAPH_RANGES) or "\u200d" in char:
        raise InvalidCharacterError(
            f"“{char}” is an emoji or picture symbol. Only characters you can type on a keyboard "
            "can be added, such as • → ° € or é."
        )
    char = validate_glyph_char(char)
    if unicodedata.category(char).startswith("M"):
        raise InvalidCharacterError(f"“{char}” is an accent on its own. Type the accented letter instead, e.g. é.")
    return char


def char_to_id(char: str) -> str:
    """Map a character to a stable, filesystem-safe, case-insensitive-safe id."""
    char = validate_glyph_char(char)
    if "a" <= char <= "z":
        return f"lower_{char}"
    if "A" <= char <= "Z":
        return f"upper_{char.lower()}"
    if "0" <= char <= "9":
        return f"digit_{char}"
    if char in _PUNCTUATION_IDS:
        return _PUNCTUATION_IDS[char]
    return f"u{ord(char):04x}"


def describe_char(char: str) -> str:
    """Short label used in the UI and on the sample sheet."""
    hint = _DESCRIPTIONS.get(char)
    if hint is None and not char.isascii():
        hint = unicodedata.name(char, "").lower() or None
    return f"{char}  ({hint})" if hint else char


def char_group(char: str) -> str:
    """Name of the display group a character belongs to."""
    for group, chars in CHARACTER_GROUPS.items():
        if char in chars:
            return group
    return "Other"
