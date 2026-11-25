"""Aperçu syllabique de contrôle pour le Data Engineering (validation des métadonnées).

Permet à la Personne 1 de vérifier que les transcriptions préparées sont propres et compatibles
avec le format attendu par la fonction `syllabify()` du challenge (ex: "Un nɔ́ ɖu nǔ" -> "un nɔ1 ɖu nu3").
"""

import unicodedata
from typing import List

DIACRITIC_TO_TONE = {
    "\u0301": "1",  # Ton haut (´)
    "\u0302": "2",  # Ton descendant (ˆ)
    "\u030c": "3",  # Ton montant (ˇ)
    "\u0300": "4",  # Ton bas (`)
}

VOWELS = {"a", "e", "ɛ", "i", "o", "ɔ", "u"}
FONGBE_ONSET_CLUSTERS = {
    "gb", "kp", "hw", "xw", "ny",
    "bl", "pl", "fl", "vl", "kl", "gl", "hl", "wl", "xl", "sl", "zl", "ml",
    "dl", "tl", "ɖl", "by", "py", "fy", "vy", "ky", "gy", "xy", "sy", "zy",
}


def syllabify_word(word: str) -> List[str]:
    """Découpe un mot Fongbé en syllabes tonales pour vérifier la cohérence des annotations."""
    nfd = unicodedata.normalize("NFD", word.lower().strip())
    units = []
    for ch in nfd:
        if ch in DIACRITIC_TO_TONE:
            if units:
                b, t = units[-1]
                units[-1] = (b, DIACRITIC_TO_TONE[ch] if not t else t)
        elif unicodedata.combining(ch):
            continue
        elif ch.isalpha() or ch in {"ɖ", "ɔ", "ɛ"}:
            units.append((ch, ""))

    syllables: List[str] = []
    i, n = 0, len(units)
    while i < n:
        chars: List[str] = []
        tone = ""

        if units[i][0] in {"m", "n"} and units[i][1] != "" and (i + 1 == n or units[i + 1][0] not in VOWELS):
            syllables.append(f"{units[i][0]}{units[i][1]}")
            i += 1
            continue

        if units[i][0] not in VOWELS:
            if i + 1 < n and (units[i][0] + units[i + 1][0]) in FONGBE_ONSET_CLUSTERS:
                chars.extend([units[i][0], units[i + 1][0]])
                i += 2
            else:
                chars.append(units[i][0])
                i += 1

        if i < n and units[i][0] in VOWELS:
            v_char, v_tone = units[i]
            chars.append(v_char)
            if v_tone:
                tone = v_tone
            i += 1
            while i < n and units[i][0] == v_char and (units[i][1] == "" or units[i][1] == tone):
                chars.append(units[i][0])
                if units[i][1] and not tone:
                    tone = units[i][1]
                i += 1
            if i < n and units[i][0] == "n" and units[i][1] == "":
                if not (i + 1 < n and units[i + 1][0] in VOWELS):
                    chars.append("n")
                    i += 1

        base = "".join(chars)
        if base:
            syllables.append(f"{base}{tone}" if tone else base)
    return syllables


def syllabify(text: str) -> str:
    """Transforme une transcription Fongbé en syllabes numérotées (ex: 'Un nɔ́ ɖu nǔ' -> 'un nɔ1 ɖu nu3')."""
    if not text or not isinstance(text, str):
        return ""
    out: List[str] = []
    for w in text.strip().split():
        out.extend(syllabify_word(w))
    return " ".join(out)
