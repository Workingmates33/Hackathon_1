"""Automatic tonal quality filter for Train_uncertain.csv (compliant with conventions.md).

Generates a clean `--extra-train-csv` ready for Personne 2's `src/train_ctc.py`:
  - Calibrates reference tonal statistics (syllable vocabulary, word vocabulary, expected tone
    density) from `Train_reliable.csv` / `splits/train_clean.csv` (excluding validation rows).
  - Scores each row of `Train_uncertain.csv` automatically (NO manual correction):
      1. Tone density (catches utterances where annotators omitted tone marks)
      2. Tonal syllable in-vocabulary rate (catches systematic tone/spelling errors)
      3. Speaking rate plausibility in syllables/second (if `duration` is present)
      4. Encoding cleanliness (no unexpected dropped characters or stacked multi-tones)
  - Outputs a filtered CSV preserving all input columns + normalized `target` + `quality_score`.

CLI usage:
    python3 -m src.nlp.filter_uncertain \
        --reliable-csv splits/train_clean.csv \
        --uncertain-csv data/raw/Train_uncertain.csv \
        --out-csv splits/train_uncertain_filtered.csv \
        --min-score 0.65
"""
from __future__ import annotations

import argparse
import csv
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from syllabify import LETTERS, MARKUP, SPURIOUS, TONES, normalize, syllabify, tonal_target
from src.nlp.normalize import split_syllable


@dataclass(frozen=True)
class ReferenceStats:
    """Reference tonal & lexical statistics calibrated on clean training data."""

    known_syllables: frozenset[str]
    known_words: frozenset[str]
    mean_tone_density: float


def calibrate_reference_stats(
    texts: Iterable[str],
    *,
    min_syllable_freq: int = 1,
    min_word_freq: int = 1,
) -> ReferenceStats:
    """Compute reference tonal vocabulary and mean tone density from reliable transcriptions."""
    syl_counts: Counter[str] = Counter()
    word_counts: Counter[str] = Counter()
    total_syl = 0
    toned_syl = 0

    for raw in texts:
        cleaned = tonal_target(raw or "")
        if not cleaned:
            continue
        word_counts.update(cleaned.split())
        for tok in syllabify(raw or "").split():
            syl_counts[tok] += 1
            total_syl += 1
            _, tone = split_syllable(tok)
            if tone:
                toned_syl += 1

    known_syls = frozenset(s for s, c in syl_counts.items() if c >= min_syllable_freq)
    known_words = frozenset(w for w, c in word_counts.items() if c >= min_word_freq)
    mean_density = (toned_syl / total_syl) if total_syl > 0 else 0.0
    return ReferenceStats(
        known_syllables=known_syls,
        known_words=known_words,
        mean_tone_density=mean_density,
    )


def _count_unexpected_chars(text: str) -> int:
    bad = 0
    for ch in normalize(text):
        if ch in TONES or ch in LETTERS or ch == " " or ch.isspace():
            continue
        if ch.isdigit() or ch in SPURIOUS or ch in MARKUP or unicodedata.category(ch).startswith("P"):
            continue
        bad += 1
    return bad


def score_utterance(
    transcription: str,
    ref: ReferenceStats,
    *,
    duration: float | None = None,
    min_syllables: int = 2,
    min_tone_density: float = 0.25,
    min_syl_per_sec: float = 1.0,
    max_syl_per_sec: float = 12.0,
) -> dict[str, Any]:
    """Score a single uncertain utterance on tonal, lexical, encoding, and pacing quality."""
    text = transcription or ""
    target = tonal_target(text)
    syls = syllabify(text).split()
    n_syl = len(syls)

    if not target or n_syl < min_syllables:
        return {
            "accepted": False,
            "reason": "too_short_or_empty",
            "quality_score": 0.0,
            "target": target,
            "n_syllables": n_syl,
            "tone_density": 0.0,
            "syl_iv_rate": 0.0,
        }

    bad_chars = _count_unexpected_chars(text)
    multi_tones = sum(1 for tok in syls if len(split_syllable(tok)[1]) > 1)
    if bad_chars > 0 or multi_tones > 0:
        return {
            "accepted": False,
            "reason": "encoding_or_multitone_noise",
            "quality_score": 0.0,
            "target": target,
            "n_syllables": n_syl,
            "tone_density": 0.0,
            "syl_iv_rate": 0.0,
        }

    toned = sum(1 for tok in syls if split_syllable(tok)[1])
    tone_density = toned / n_syl
    if tone_density < min_tone_density:
        return {
            "accepted": False,
            "reason": "low_tone_density",
            "quality_score": round(0.25 * tone_density, 4),
            "target": target,
            "n_syllables": n_syl,
            "tone_density": round(tone_density, 4),
            "syl_iv_rate": 0.0,
        }

    if duration is not None and duration > 0:
        rate = n_syl / duration
        if rate < min_syl_per_sec or rate > max_syl_per_sec:
            return {
                "accepted": False,
                "reason": "implausible_speaking_rate",
                "quality_score": 0.0,
                "target": target,
                "n_syllables": n_syl,
                "tone_density": round(tone_density, 4),
                "syl_iv_rate": 0.0,
            }

    syl_iv = sum(1 for tok in syls if tok in ref.known_syllables) / n_syl
    words = target.split()
    word_iv = (sum(1 for w in words if w in ref.known_words) / len(words)) if words else 0.0

    # Closeness to reference tone density (1.0 if at or near reference mean, decays gently if far)
    if ref.mean_tone_density > 0:
        density_ratio = min(tone_density / ref.mean_tone_density, 1.0)
    else:
        density_ratio = 1.0

    # Weighted quality score in [0, 1]
    score = 0.50 * syl_iv + 0.25 * word_iv + 0.25 * density_ratio
    return {
        "accepted": True,
        "reason": "ok",
        "quality_score": round(score, 4),
        "target": target,
        "n_syllables": n_syl,
        "tone_density": round(tone_density, 4),
        "syl_iv_rate": round(syl_iv, 4),
    }


def _detect_col(fieldnames: Sequence[str], candidates: Sequence[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Column {explicit!r} not found in CSV headers: {list(fieldnames)}")
        return explicit
    for c in candidates:
        if c in fieldnames:
            return c
    raise ValueError(f"Could not find any of {candidates} in CSV headers: {list(fieldnames)}")


def filter_uncertain_csv(
    reliable_csv: Path,
    uncertain_csv: Path,
    out_csv: Path,
    *,
    min_score: float = 0.65,
    min_tone_density: float = 0.25,
    min_syllables: int = 2,
    min_syl_per_sec: float = 1.0,
    max_syl_per_sec: float = 12.0,
    text_column: str | None = None,
) -> dict[str, Any]:
    """Filter an uncertain CSV using statistics calibrated from a reliable training CSV."""
    # 1. Calibrate on reliable training rows (skip recommended_validation == 1)
    reliable_texts: list[str] = []
    with open(reliable_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{reliable_csv} has no headers.")
        rel_col = _detect_col(reader.fieldnames, ("transcription", "target", "text", "Target"), text_column)
        has_rec_val = "recommended_validation" in reader.fieldnames
        for row in reader:
            if has_rec_val and str(row.get("recommended_validation") or "").strip() == "1":
                continue
            reliable_texts.append(row.get(rel_col) or "")

    ref_stats = calibrate_reference_stats(reliable_texts)

    # 2. Score and filter uncertain rows
    kept_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    total_rows = 0
    total_hours = 0.0
    kept_hours = 0.0

    with open(uncertain_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{uncertain_csv} has no headers.")
        unc_col = _detect_col(reader.fieldnames, ("transcription", "target", "text", "Target"), text_column)
        base_fields = list(reader.fieldnames)
        for row in reader:
            total_rows += 1
            dur_raw = row.get("duration")
            dur = float(dur_raw) if dur_raw not in (None, "") else None
            if dur:
                total_hours += dur / 3600.0

            res = score_utterance(
                row.get(unc_col) or "",
                ref_stats,
                duration=dur,
                min_syllables=min_syllables,
                min_tone_density=min_tone_density,
                min_syl_per_sec=min_syl_per_sec,
                max_syl_per_sec=max_syl_per_sec,
            )
            if res["accepted"] and res["quality_score"] >= min_score:
                out_row = dict(row)
                out_row["target"] = res["target"]
                out_row["quality_score"] = res["quality_score"]
                out_row.setdefault("domain", "general")
                kept_rows.append(out_row)
                if dur:
                    kept_hours += dur / 3600.0
                reasons["kept"] += 1
            else:
                reason = res["reason"] if not res["accepted"] else "below_min_score"
                reasons[reason] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fields = list(base_fields)
    for extra in ("target", "domain", "quality_score"):
        if extra not in out_fields:
            out_fields.append(extra)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(kept_rows)

    return {
        "total_rows": total_rows,
        "kept_rows": len(kept_rows),
        "kept_ratio": round(len(kept_rows) / max(total_rows, 1), 4),
        "total_hours": round(total_hours, 2),
        "kept_hours": round(kept_hours, 2),
        "reference_mean_tone_density": round(ref_stats.mean_tone_density, 4),
        "reference_unique_syllables": len(ref_stats.known_syllables),
        "breakdown": dict(reasons),
        "out_csv": str(out_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Train_uncertain.csv by tonal quality.")
    parser.add_argument("--reliable-csv", type=Path, required=True, help="Clean reference CSV.")
    parser.add_argument("--uncertain-csv", type=Path, required=True, help="Uncertain CSV to filter.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Filtered output CSV.")
    parser.add_argument("--min-score", type=float, default=0.65, help="Minimum quality_score (0..1).")
    parser.add_argument("--min-tone-density", type=float, default=0.25, help="Min fraction of toned syllables.")
    parser.add_argument("--min-syllables", type=int, default=2, help="Minimum syllables per utterance.")
    parser.add_argument("--text-column", default=None, help="Explicit transcription column name.")
    args = parser.parse_args()

    summary = filter_uncertain_csv(
        args.reliable_csv,
        args.uncertain_csv,
        args.out_csv,
        min_score=args.min_score,
        min_tone_density=args.min_tone_density,
        min_syllables=args.min_syllables,
        text_column=args.text_column,
    )
    print("=== Train_uncertain Tonal Filter Summary ===")
    for k, v in summary.items():
        print(f"  {k:<28}: {v}")


if __name__ == "__main__":
    main()
