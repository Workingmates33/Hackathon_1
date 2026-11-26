"""Detailed tonal vs segmental ASR error analysis for Fongbe (standard library only).

CLI usage:
    python3 -m src.nlp.error_analysis predictions.csv [--ref-col ref] [--hyp-col hyp] [--already-syllabified]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from typing import Any, Sequence

from src.nlp.metrics import align_sequences, syllable_wer
from src.nlp.normalize import safe_syllabify, split_syllable

EMPTY_TONE_LABEL = "∅"


def analyze_errors(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    already_syllabified: bool = False,
    top_k: int = 20,
) -> dict[str, Any]:
    """Break down ASR errors into purely tonal vs segmental errors and tone confusions."""
    metrics = syllable_wer(refs, hyps, already_syllabified=already_syllabified)

    if already_syllabified:
        ref_syl = [r.split() for r in refs]
        hyp_syl = [h.split() for h in hyps]
    else:
        ref_syl = [safe_syllabify(r).split() for r in refs]
        hyp_syl = [safe_syllabify(h).split() for h in hyps]

    pure_tone_subs = 0
    segmental_subs = 0
    tone_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    tonal_pairs: Counter[tuple[str, str]] = Counter()
    segmental_pairs: Counter[tuple[str, str]] = Counter()
    deleted_syllables: Counter[str] = Counter()
    inserted_syllables: Counter[str] = Counter()

    for r, h in zip(ref_syl, hyp_syl):
        for op, rt, ht in align_sequences(r, h):
            if op == "equal" and rt is not None:
                _, tone = split_syllable(rt)
                label = tone or EMPTY_TONE_LABEL
                tone_confusion[label][label] += 1
            elif op == "sub" and rt is not None and ht is not None:
                rb, r_tone = split_syllable(rt)
                hb, h_tone = split_syllable(ht)
                if rb == hb:
                    pure_tone_subs += 1
                    tonal_pairs[(rt, ht)] += 1
                    tone_confusion[r_tone or EMPTY_TONE_LABEL][h_tone or EMPTY_TONE_LABEL] += 1
                else:
                    segmental_subs += 1
                    segmental_pairs[(rt, ht)] += 1
            elif op == "del" and rt is not None:
                deleted_syllables[rt] += 1
            elif op == "ins" and ht is not None:
                inserted_syllables[ht] += 1

    return {
        "metrics": metrics,
        "pure_tone_substitutions": pure_tone_subs,
        "segmental_substitutions": segmental_subs,
        "tone_confusion_matrix": {k: dict(v) for k, v in sorted(tone_confusion.items())},
        "top_tonal_confusions": tonal_pairs.most_common(top_k),
        "top_segmental_confusions": segmental_pairs.most_common(top_k),
        "top_deletions": deleted_syllables.most_common(top_k),
        "top_insertions": inserted_syllables.most_common(top_k),
    }


def format_error_report(report: dict[str, Any]) -> str:
    """Format an error analysis report as human-readable text."""
    m = report["metrics"]
    lines = [
        "=== Fongbe Tonal ASR Error Analysis ===",
        f"Utterances              : {m['num_utterances']}",
        f"Reference syllables     : {m['ref_syllables']}",
        f"Syllable WER (official) : {m['syllable_wer']:.4f}",
        f"Base Syllable WER       : {m['base_syllable_wer']:.4f}",
        f"Tone error gap          : +{m['tone_error_gap']:.4f}",
        f"Tone acc (matched base) : {m['tone_accuracy_on_matched_bases']:.2%}",
        f"Substitutions           : {m['substitutions']} "
        f"(purely tonal: {report['pure_tone_substitutions']}, "
        f"segmental: {report['segmental_substitutions']})",
        f"Deletions / Insertions  : {m['deletions']} / {m['insertions']}",
    ]

    matrix = report["tone_confusion_matrix"]
    if matrix:
        all_tones = sorted(set(matrix.keys()) | {t for row in matrix.values() for t in row})
        lines.append("\n--- Tone Confusion Matrix (rows=ref, cols=hyp) ---")
        header = "ref\\hyp | " + " | ".join(f"{t:>5}" for t in all_tones)
        lines.append(header)
        lines.append("-" * len(header))
        for rt in all_tones:
            row = matrix.get(rt, {})
            vals = " | ".join(f"{row.get(ht, 0):>5}" for ht in all_tones)
            lines.append(f"{rt:>7} | {vals}")

    if report["top_tonal_confusions"]:
        lines.append("\n--- Top Purely Tonal Confusions (ref -> hyp) ---")
        for (rt, ht), cnt in report["top_tonal_confusions"][:10]:
            lines.append(f"  {rt} -> {ht} : {cnt}")

    if report["top_segmental_confusions"]:
        lines.append("\n--- Top Segmental Confusions (ref -> hyp) ---")
        for (rt, ht), cnt in report["top_segmental_confusions"][:10]:
            lines.append(f"  {rt} -> {ht} : {cnt}")

    return "\n".join(lines)


def _detect_column(fieldnames: Sequence[str], candidates: Sequence[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Column {explicit!r} not found in CSV headers: {list(fieldnames)}")
        return explicit
    for cand in candidates:
        if cand in fieldnames:
            return cand
    raise ValueError(f"Could not find any of {candidates} in CSV headers: {list(fieldnames)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tonal vs segmental ASR errors.")
    parser.add_argument("csv_path", help="CSV file containing reference and prediction columns.")
    parser.add_argument("--ref-col", default=None, help="Reference column name.")
    parser.add_argument("--hyp-col", default=None, help="Hypothesis/prediction column name.")
    parser.add_argument(
        "--already-syllabified",
        action="store_true",
        help="Set if both columns are already in syllabified format ('un nɔ1 ɖu nu3').",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Number of top confusions to show.")
    args = parser.parse_args()

    with open(args.csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file has no headers.")
        ref_col = _detect_column(reader.fieldnames, ("ref", "reference", "target", "Target", "transcription"), args.ref_col)
        hyp_col = _detect_column(reader.fieldnames, ("hyp", "hypothesis", "pred", "prediction"), args.hyp_col)
        refs, hyps = [], []
        for row in reader:
            refs.append(row.get(ref_col) or "")
            hyps.append(row.get(hyp_col) or "")

    report = analyze_errors(
        refs,
        hyps,
        already_syllabified=args.already_syllabified,
        top_k=args.top_k,
    )
    print(format_error_report(report))


if __name__ == "__main__":
    main()
