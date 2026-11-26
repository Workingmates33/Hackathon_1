"""Safe wrappers around the official syllabify() function and Fongbe text validators."""
from __future__ import annotations

import re
import unicodedata

from syllabify import LETTERS, TONE_DIGIT, syllabify

VALID_TONE_DIGITS = frozenset(TONE_DIGIT.values())
_TONE_DIGIT_RE = re.compile(r"[1-5]+$")
_SYLLABIFIED_TOKEN_RE = re.compile(r"^[a-zɔɖɛə]+[1-5]*$")
_HAS_SYLLABIFIED_TONE_RE = re.compile(r"[a-zɔɖɛə][1-5](?:\b|\s|$)")


def split_syllable(token: str) -> tuple[str, str]:
    """Split a syllabified token into (base, tone_digits).

    Examples:
        'nɔ1' -> ('nɔ', '1')
        'un'  -> ('un', '')
    """
    m = _TONE_DIGIT_RE.search(token)
    if not m:
        return token, ""
    return token[: m.start()], m.group(0)


def strip_tones(syllabified_text: str) -> str:
    """Remove tone digits from a syllabified string ('un nɔ1 ɖu nu3' -> 'un nɔ ɖu nu')."""
    if not syllabified_text:
        return ""
    return " ".join(split_syllable(tok)[0] for tok in syllabified_text.split())


def is_already_syllabified(text: str) -> bool:
    """Heuristically detect text that already contains syllabified tone digits (e.g. 'nɔ1').

    Calling official `syllabify()` on such text would silently strip all tone digits.
    """
    if not text:
        return False
    return bool(_HAS_SYLLABIFIED_TONE_RE.search(text))


def safe_syllabify(text: str, *, allow_already_syllabified: bool = False) -> str:
    """Apply official `syllabify()` while guarding against accidental double syllabification.

    Args:
        text: Raw Fongbe orthographic text (or syllabified text if allow_already_syllabified=True).
        allow_already_syllabified: If True and `text` already has tone digits, normalize spaces
            and return it unchanged instead of stripping tones. If False, raise ValueError.
    """
    if not text:
        return ""
    if is_already_syllabified(text):
        if allow_already_syllabified:
            return " ".join(unicodedata.normalize("NFC", text).lower().split())
        raise ValueError(
            "Input appears to be already syllabified (contains tone digits like 'nɔ1'). "
            "Calling syllabify() twice strips all tone digits! "
            "Pass allow_already_syllabified=True if this is intentional."
        )
    return syllabify(text)


def validate_submission_target(target: str) -> list[str]:
    """Return a list of formatting issues for a single submission Target string."""
    issues: list[str] = []
    if target is None:
        return ["target is None"]
    if target != target.strip():
        issues.append("leading or trailing whitespace")
    if "  " in target:
        issues.append("consecutive spaces")
    if target != unicodedata.normalize("NFC", target):
        issues.append("not NFC-normalized")

    for tok in target.split():
        if not _SYLLABIFIED_TOKEN_RE.match(tok):
            issues.append(f"invalid syllable token: {tok!r}")
    return issues
