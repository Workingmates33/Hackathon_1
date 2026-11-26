"""Audit raw Fongbe transcriptions against official syllabify.py rules (standard library only).

CLI usage:
    python3 -m src.nlp.audit_text train.csv [--column transcription]
"""
from __future__ import annotations

import argparse
import csv
import unicodedata
from collections import Counter
from typing import Any, Iterable

from syllabify import LETTERS, MARKUP, SPURIOUS, TONES, normalize, syllabify
from src.nlp.normalize import split_syllable

EMPTY_TONE_LABEL = "∅"


def audit_transcriptions(texts: Iterable[str], *, top_k: int = 20) -> dict[str, Any]:
    """Audit raw Fongbe transcriptions for encoding, character coverage, and tone statistics."""
    total_rows = 0
    empty_raw = 0
    empty_after_syllabify = 0
    non_nfc_rows = 0
    multi_tone_syllables: Counter[str] = Counter()
    dropped_chars: Counter[str] = Counter()
    tone_counts: Counter[str] = Counter()
    syllable_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    total_syllables = 0
    max_syllables = 0

    for raw in texts:
        total_rows += 1
        text = raw or ""
        if not text.strip():
            empty_raw += 1

        if text != unicodedata.normalize("NFC", text):
            non_nfc_rows += 1

        # Inspect characters dropped by syllabify.tonal_target() that are not standard punctuation/digits/markup
        for ch in normalize(text):
            if ch in TONES or ch in LETTERS or ch == " " or ch.isspace():
                continue
            if ch.isdigit() or ch in SPURIOUS or ch in MARKUP or unicodedata.category(ch).startswith("P"):
                continue
            dropped_chars[ch] += 1

        syl_str = syllabify(text)
        toks = syl_str.split()
        if not toks:
            empty_after_syllabify += 1
        total_syllables += len(toks)
        if len(toks) > max_syllables:
            max_syllables = len(toks)

        for tok in toks:
            syllable_counts[tok] += 1
            base, tone = split_syllable(tok)
            base_counts[base] += 1
            tone_counts[tone or EMPTY_TONE_LABEL] += 1
            if len(tone) > 1:
                multi_tone_syllables[tok] += 1

    avg_syllables = (total_syllables / total_rows) if total_rows > 0 else 0.0
    return {
        "total_rows": total_rows,
        "empty_raw_rows": empty_raw,
        "empty_after_syllabify_rows": empty_after_syllabify,
        "non_nfc_rows": non_nfc_rows,
        "total_syllables": total_syllables,
        "avg_syllables_per_row": avg_syllables,
        "max_syllables_per_row": max_syllables,
        "unique_syllables": len(syllable_counts),
        "unique_base_syllables": len(base_counts),
        "tone_distribution": dict(sorted(tone_counts.items())),
        "multi_tone_syllables": multi_tone_syllables.most_common(top_k),
        "unexpected_dropped_chars": [
            (ch, f"U+{ord(ch):04X}", unicodedata.name(ch, "UNKNOWN"), cnt)
            for ch, cnt in dropped_chars.most_common(top_k)
        ],
        "top_syllables": syllable_counts.most_common(top_k),
    }


def format_audit_report(report: dict[str, Any]) -> str:
    """Format a transcription audit report as human-readable text."""
    lines = [
        "=== Fongbe Transcription Audit ===",
        f"Total rows                 : {report['total_rows']}",
        f"Empty raw rows             : {report['empty_raw_rows']}",
        f"Empty after syllabify()    : {report['empty_after_syllabify_rows']}",
        f"Non-NFC raw rows           : {report['non_nfc_rows']}",
        f"Total syllables            : {report['total_syllables']} "
        f"(avg {report['avg_syllables_per_row']:.1f}, max {report['max_syllables_per_row']})",
        f"Unique syllables (tonal)   : {report['unique_syllables']}",
        f"Unique base syllables      : {report['unique_base_syllables']}",
    ]

    if report["tone_distribution"]:
        total = max(report["total_syllables"], 1)
        lines.append("\n--- Tone Distribution ---")
        for tone, cnt in report["tone_distribution"].items():
            lines.append(f"  tone {tone:<3} : {cnt:>8} ({cnt / total:.2%})")

    if report["unexpected_dropped_chars"]:
        lines.append("\n--- Unexpected Dropped Characters (check for unmapped look-alikes!) ---")
        for ch, cp, name, cnt in report["unexpected_dropped_chars"]:
            lines.append(f"  {ch!r} ({cp} {name}) : {cnt}")

    if report["multi_tone_syllables"]:
        lines.append("\n--- Multi-Tone Syllables Detected ---")
        for tok, cnt in report["multi_tone_syllables"][:10]:
            lines.append(f"  {tok} : {cnt}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw Fongbe transcriptions.")
    parser.add_argument("csv_path", help="Path to CSV containing transcriptions.")
    parser.add_argument(
        "--column",
        default=None,
        help="Column name to audit (auto-detects 'transcription', 'text', or 'Target').",
    )
    args = parser.parse_args()

    with open(args.csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file has no headers.")
        col = args.column
        if col is None:
            for cand in ("transcription", "text", "Target", "target"):
                if cand in reader.fieldnames:
                    col = cand
                    break
        if col is None or col not in reader.fieldnames:
            raise ValueError(f"Column {col!r} not found in headers: {list(reader.fieldnames)}")
        texts = [row.get(col) or "" for row in reader]

    report = audit_transcriptions(texts)
    print(format_audit_report(report))


if __name__ == "__main__":
    main()
