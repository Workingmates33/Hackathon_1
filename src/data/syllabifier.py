"""Syllabification and tone conversion module for Fongbe.

Follows the challenge's official evaluation formatting:
e.g.: "Un nɔ́ ɖu nǔ" -> "un nɔ1 ɖu nu3"
"""

import re
import unicodedata
from typing import List


# Standard mapping of diacritics to numeric tone tags
DIACRITIC_TO_TONE = {
    "\u0301": "1",  # Acute / High tone (ex: nɔ́ -> nɔ1)
    "\u0302": "2",  # Circumflex / Falling
    "\u030c": "3",  # Caron / Rising (ex: nǔ -> nu3)
    "\u0300": "4",  # Grave / Low
}


def syllabify(text: str) -> str:
    """Simulate the challenge's official syllabify function.

    Decomposes characters, extracts the tone diacritic, appends tone number
    to the syllable/word, and returns the normalized string.

    Example:
        >>> syllabify("Un nɔ́ ɖu nǔ")
        'un nɔ1 ɖu nu3'
    """
    if not text:
        return ""

    words = text.strip().split()
    processed_words = []

    for word in words:
        # NFD decomposes precomposed characters into base letter + diacritics
        decomposed = unicodedata.normalize("NFD", word.lower())
        
        # Check for tone marks
        tones_found = []
        base_chars = []
        
        for ch in decomposed:
            if ch in DIACRITIC_TO_TONE:
                tones_found.append(DIACRITIC_TO_TONE[ch])
            elif not unicodedata.combining(ch):
                base_chars.append(ch)

        base_word = unicodedata.normalize("NFC", "".join(base_chars))
        
        # If a tone was found, append it (or the predominant tone)
        if tones_found:
            tone_suffix = "".join(tones_found)
            processed_words.append(f"{base_word}{tone_suffix}")
        else:
            processed_words.append(base_word)

    return " ".join(processed_words)
