"""Linguistic utilities and character mappings for Fongbe language.

Fongbe (ISO 639-3: fon) is a Niger-Congo language spoken primarily in Benin.
It is an isolating, tonal language with grammatical and lexical tone distinctions.
"""

import unicodedata
from typing import Dict, List, Set

# Vowels (oral and nasal)
FONGBE_ORAL_VOWELS = {"a", "e", "ɛ", "i", "o", "ɔ", "u"}
FONGBE_NASAL_VOWELS = {"an", "ɛn", "in", "ɔn", "un"}

# Specific Fongbe characters (including letters with diacritics / special IPA-like symbols)
FONGBE_SPECIAL_CHARS = {"ɖ", "ɔ", "ɛ"}

# Consonant clusters / labiovelars
FONGBE_DIGRAPHS = {"gb", "kp", "hw", "xw", "ny"}

# Tone marks (Unicode combining diacritics and precomposed)
# Example: High tone (´) -> tone 1, Falling (ˆ) -> tone 2, Rising (ˇ) -> tone 3, Low (`) -> tone 4
TONE_ACCENTS = {
    "\u0301": "1",  # Combining acute (high tone)
    "\u0300": "4",  # Combining grave (low tone)
    "\u0302": "2",  # Combining circumflex (falling tone)
    "\u030c": "3",  # Combining caron (rising tone)
    "\u0304": "0",  # Combining macron (mid tone)
}


def normalize_fongbe_text(text: str) -> str:
    """Normalize text into standard Unicode NFD decomposition to separate letters and tone diacritics,

    then lowercase standardly while preserving tone identities.
    """
    if not text:
        return ""
    # Strip excessive whitespaces
    text = " ".join(text.strip().split())
    # Decompose unicode (NFD)
    nfd_text = unicodedata.normalize("NFD", text)
    return nfd_text


def extract_vocabulary(transcriptions: List[str]) -> Set[str]:
    """Extract all unique characters present in the transcriptions."""
    vocab = set()
    for t in transcriptions:
        for char in unicodedata.normalize("NFD", t):
            vocab.add(char)
    return vocab
