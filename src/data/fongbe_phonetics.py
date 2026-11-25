"""Analyse linguistique et phonétique du Fongbé pour le Data Engineering (Personne 1).

Couvre la tâche : "Analyser les caractéristiques linguistiques du Fongbé" :
- Voyelles orales (a, e, ɛ, i, o, ɔ, u) et voyelles nasales (an, ɛn, in, ɔn, un)
- Caractères spéciaux de l'alphabet national béninois (ɖ, ɔ, ɛ)
- Digraphes labiovélaires et palataux (gb, kp, hw, xw, ny)
- Inventaire et distribution statistique des diacritiques tonals (haut, bas, montant, descendant)
- Détection des caractères hors-alphabet ou confusions d'encodage Unicode
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Set, Any

FONGBE_ORAL_VOWELS = {"a", "e", "ɛ", "i", "o", "ɔ", "u"}
FONGBE_NASAL_VOWELS = ("an", "ɛn", "in", "ɔn", "un")
FONGBE_SPECIAL_CHARS = {"ɖ", "ɔ", "ɛ"}
FONGBE_DIGRAPHS = ("gb", "kp", "hw", "xw", "ny")

ALLOWED_BASE_LETTERS = set("abcdeɛfghijklmnoɔpqrstuvwxyzɖ ")

TONE_NAMES = {
    "\u0301": "Ton Haut (´)",
    "\u0300": "Ton Bas (`)",
    "\u0302": "Ton Descendant (ˆ)",
    "\u030c": "Ton Montant (ˇ)",
    "\u0304": "Ton Moyen (¯)",
}

TONE_ACCENTS = {
    "\u0301": "1",
    "\u0302": "2",
    "\u030c": "3",
    "\u0300": "4",
    "\u0304": "0",
}


def normalize_fongbe_text(text: str) -> str:
    """Normalise les espaces et décompose en Unicode NFD tout en préservant lettres et tons."""
    if not text or not isinstance(text, str):
        return ""
    text = " ".join(text.strip().split())
    return unicodedata.normalize("NFD", text)


def extract_vocabulary(transcriptions: List[str]) -> Set[str]:
    """Extrait l'ensemble des caractères uniques (décomposés NFD) présents dans le corpus."""
    vocab = set()
    for t in transcriptions:
        if isinstance(t, str):
            for char in unicodedata.normalize("NFD", t.lower()):
                vocab.add(char)
    return vocab


def detect_unexpected_characters(text: str) -> List[str]:
    """Détecte les caractères parasites ou confusions d'encodage (ex: chiffres, epsilon grec ε, symboles)."""
    if not text or not isinstance(text, str):
        return []
    nfd = unicodedata.normalize("NFD", text.lower())
    unexpected = []
    for ch in nfd:
        if ch in TONE_NAMES:
            continue
        if ch in ALLOWED_BASE_LETTERS or ch in {"'", "-", ",", ".", "?", "!"}:
            continue
        unexpected.append(ch)
    return sorted(set(unexpected))


def analyze_fongbe_linguistics(transcriptions: List[str]) -> Dict[str, Any]:
    """Produit un bilan statistique complet des propriétés linguistiques d'un corpus Fongbé."""
    char_counter: Counter = Counter()
    tone_counter: Counter = Counter()
    special_char_counter: Counter = Counter()
    digraph_counter: Counter = Counter()
    nasal_vowel_counter: Counter = Counter()
    word_lengths: List[int] = []
    sentence_word_counts: List[int] = []
    unmarked_vowel_syllables = 0

    vowels_regex = re.compile(r"[aeɛioɔu]")

    for raw in transcriptions:
        if not isinstance(raw, str) or not raw.strip():
            continue
        low = raw.lower().strip()
        words = low.split()
        sentence_word_counts.append(len(words))

        # Comptage des digraphes et voyelles nasales sur texte sans accents
        nfd = unicodedata.normalize("NFD", low)
        base_only = "".join(c for c in nfd if not unicodedata.combining(c))

        for dg in FONGBE_DIGRAPHS:
            cnt = len(re.findall(dg, base_only))
            if cnt:
                digraph_counter[dg] += cnt

        for nv in FONGBE_NASAL_VOWELS:
            # Voyelle + n en fin de mot ou devant consonne
            cnt = len(re.findall(rf"{nv}(?![aeɛioɔu])", base_only))
            if cnt:
                nasal_vowel_counter[nv] += cnt

        for w in words:
            w_nfd = unicodedata.normalize("NFD", w)
            w_base = "".join(c for c in w_nfd if not unicodedata.combining(c) and c.isalpha())
            if w_base:
                word_lengths.append(len(w_base))

            # Vérifie chaque voyelle pour savoir si elle porte un diacritique tonal ou un ton moyen implicite
            i = 0
            while i < len(w_nfd):
                ch = w_nfd[i]
                char_counter[ch] += 1
                if ch in FONGBE_SPECIAL_CHARS:
                    special_char_counter[ch] += 1
                if ch in TONE_NAMES:
                    tone_counter[TONE_NAMES[ch]] += 1
                elif vowels_regex.match(ch):
                    has_mark = (i + 1 < len(w_nfd)) and (w_nfd[i + 1] in TONE_NAMES)
                    if not has_mark:
                        unmarked_vowel_syllables += 1
                i += 1

    tone_distribution = dict(tone_counter)
    tone_distribution["Ton Moyen / Non-marqué"] = unmarked_vowel_syllables

    return {
        "total_sentences": len(sentence_word_counts),
        "total_words": sum(sentence_word_counts),
        "avg_words_per_sentence": round(sum(sentence_word_counts) / max(1, len(sentence_word_counts)), 2),
        "avg_chars_per_word": round(sum(word_lengths) / max(1, len(word_lengths)), 2),
        "unique_nfd_symbols": len(char_counter),
        "special_fongbe_chars": dict(special_char_counter),
        "digraph_frequencies": dict(digraph_counter),
        "nasal_vowel_frequencies": dict(nasal_vowel_counter),
        "tone_distribution": tone_distribution,
    }
