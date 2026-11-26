"""Official Syllable WER and tone-agnostic Base Syllable WER (standard library only)."""
from __future__ import annotations

from typing import Sequence

from src.nlp.normalize import safe_syllabify, split_syllable, strip_tones


def align_sequences(
    ref: Sequence[str], hyp: Sequence[str]
) -> list[tuple[str, str | None, str | None]]:
    """Compute optimal Levenshtein alignment between ref and hyp token sequences.

    Returns a list of (op, ref_tok, hyp_tok) where op in {'equal', 'sub', 'del', 'ins'}.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        r_i = ref[i - 1]
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r_i == hyp[j - 1] else 1
            sub_cost = prev_row[j - 1] + cost
            del_cost = prev_row[j] + 1
            ins_cost = row[j - 1] + 1
            best = sub_cost
            if del_cost < best:
                best = del_cost
            if ins_cost < best:
                best = ins_cost
            row[j] = best

    # Backtrack
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                op = "equal" if cost == 0 else "sub"
                ops.append((op, ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def _wer_counts(refs_toks: list[list[str]], hyps_toks: list[list[str]]) -> dict[str, int | float]:
    subs = dels = ins = total_ref = 0
    for r, h in zip(refs_toks, hyps_toks):
        total_ref += len(r)
        for op, _, _ in align_sequences(r, h):
            if op == "sub":
                subs += 1
            elif op == "del":
                dels += 1
            elif op == "ins":
                ins += 1
    errors = subs + dels + ins
    wer = (errors / total_ref) if total_ref > 0 else (0.0 if errors == 0 else 1.0)
    return {
        "wer": wer,
        "errors": errors,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "ref_syllables": total_ref,
    }


def syllable_wer(
    refs: Sequence[str],
    hyps: Sequence[str],
    *,
    already_syllabified: bool = False,
) -> dict[str, int | float]:
    """Compute official Syllable WER alongside tone-agnostic Base Syllable WER.

    Args:
        refs: Ground truth strings.
        hyps: Predicted strings.
        already_syllabified: Set True if inputs are already in 'un nɔ1 ɖu nu3' format.

    Returns:
        Dictionary with:
            - syllable_wer: Official WER on syllabified tokens (lower is better).
            - base_syllable_wer: WER after stripping all tone digits.
            - tone_error_gap: syllable_wer - base_syllable_wer (exact penalty caused by tones).
            - tone_accuracy_on_matched_bases: Accuracy of tones when the base syllable is right.
            - substitutions, deletions, insertions, ref_syllables, num_utterances.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) must have the same length.")

    if already_syllabified:
        ref_syl = [r.split() for r in refs]
        hyp_syl = [h.split() for h in hyps]
    else:
        ref_syl = [safe_syllabify(r).split() for r in refs]
        hyp_syl = [safe_syllabify(h).split() for h in hyps]

    full = _wer_counts(ref_syl, hyp_syl)

    ref_base = [strip_tones(" ".join(r)).split() for r in ref_syl]
    hyp_base = [strip_tones(" ".join(h)).split() for h in hyp_syl]
    base = _wer_counts(ref_base, hyp_base)

    matched_bases = 0
    matched_tones = 0
    for r, h in zip(ref_syl, hyp_syl):
        for op, rt, ht in align_sequences(r, h):
            if op == "equal" and rt is not None:
                matched_bases += 1
                matched_tones += 1
            elif op == "sub" and rt is not None and ht is not None:
                rb, r_tone = split_syllable(rt)
                hb, h_tone = split_syllable(ht)
                if rb == hb:
                    matched_bases += 1
                    if r_tone == h_tone:
                        matched_tones += 1

    tone_acc = (matched_tones / matched_bases) if matched_bases > 0 else 0.0

    return {
        "syllable_wer": full["wer"],
        "base_syllable_wer": base["wer"],
        "tone_error_gap": float(full["wer"]) - float(base["wer"]),
        "tone_accuracy_on_matched_bases": tone_acc,
        "substitutions": full["substitutions"],
        "deletions": full["deletions"],
        "insertions": full["insertions"],
        "ref_syllables": full["ref_syllables"],
        "matched_bases": matched_bases,
        "num_utterances": len(refs),
    }
