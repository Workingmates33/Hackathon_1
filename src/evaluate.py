#!/usr/bin/env python
"""Local evaluation mirroring the Test set composition (67.8% general + 32.2% religious).

Computes official `score.py` metrics (`TSER`, `segmental_SER_without_tones`, `TER_conditional`, `TER_per_tone`)
separately on `general` and `religious`, plus the weighted mirror TSER matching Test.csv proportions.

Usage:
    python src/evaluate.py --reference splits/val_mirror.csv --predictions outputs/val_pred.csv [--output report.json] [--show-errors 10]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from score import id_and_text, measure, syllables  # noqa: E402
except ImportError as exc:
    sys.exit(f"Missing dependency for official score.py ({exc}). Install with: pip install rapidfuzz==3.14.5")

# Official Test.csv split from README.md: 6,382 general / 3,025 religious (total 9,407)
TEST_WEIGHT_GENERAL = 6382 / 9407
TEST_WEIGHT_RELIGIOUS = 3025 / 9407
_DIGITS_RE = re.compile(r"[1-5]")


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate plain Fongbe predictions on mirror validation set.")
    ap.add_argument("--reference", type=Path, required=True, help="Reference CSV (e.g. splits/val_mirror.csv)")
    ap.add_argument("--predictions", type=Path, required=True, help="Predictions CSV (id,transcription)")
    ap.add_argument("--output", type=Path, help="Optional JSON report path")
    ap.add_argument("--show-errors", type=int, default=0, help="Print N worst utterances for P3 tonal analysis")
    a = ap.parse_args()

    refs = read_csv(a.reference)
    preds = dict(id_and_text(r) for r in read_csv(a.predictions))

    # Guard against accidentally scoring already-syllabified predictions (digits would be stripped!)
    sample_hyps = [preds.get(id_and_text(r)[0], "") for r in refs[:50]]
    digit_hits = sum(1 for h in sample_hyps if _DIGITS_RE.search(h))
    if digit_hits > 10:
        print(
            "WARNING: Predictions appear to contain tone digits (1-5). "
            "score.py expects PLAIN Fongbe text with diacritics, NOT syllabify() output!",
            file=sys.stderr,
        )

    all_pairs = [(id_and_text(r)[1], preds.get(id_and_text(r)[0], "")) for r in refs]
    missing = sum(1 for r in refs if id_and_text(r)[0] not in preds)
    overall = measure(all_pairs)

    report: dict[str, object] = {
        "utterances": len(refs),
        "missing": missing,
        "overall_pooled": overall,
    }

    has_domain = bool(refs and "domain" in refs[0] and {r["domain"] for r in refs} >= {"general", "religious"})
    if has_domain:
        gen_pairs = [(id_and_text(r)[1], preds.get(id_and_text(r)[0], "")) for r in refs if r["domain"] == "general"]
        rel_pairs = [(id_and_text(r)[1], preds.get(id_and_text(r)[0], "")) for r in refs if r["domain"] == "religious"]
        m_gen = measure(gen_pairs)
        m_rel = measure(rel_pairs)
        mirror_tser = TEST_WEIGHT_GENERAL * m_gen["TSER"] + TEST_WEIGHT_RELIGIOUS * m_rel["TSER"]
        report["general"] = m_gen
        report["religious"] = m_rel
        report["TSER_mirror"] = round(mirror_tser, 6)

        print("=" * 72)
        print(f"  TSER_mirror (67.8% gen + 32.2% rel) : {mirror_tser * 100:6.2f} %")
        print("-" * 72)
        print(
            f"  general   ({len(gen_pairs):4d} utts) : TSER {m_gen['TSER'] * 100:6.2f}% | "
            f"seg_SER {m_gen['segmental_SER_without_tones'] * 100:6.2f}% | "
            f"cond_TER {m_gen['TER_conditional'] * 100:6.2f}%"
        )
        print(
            f"  religious ({len(rel_pairs):4d} utts) : TSER {m_rel['TSER'] * 100:6.2f}% | "
            f"seg_SER {m_rel['segmental_SER_without_tones'] * 100:6.2f}% | "
            f"cond_TER {m_rel['TER_conditional'] * 100:6.2f}%"
        )
        print(f"  TER_per_tone (general)  : {m_gen['TER_per_tone']}")
        print(f"  TER_per_tone (religious): {m_rel['TER_per_tone']}")
        print("=" * 72)
    else:
        print(
            f"TSER {overall['TSER'] * 100:.2f}% | seg_SER {overall['segmental_SER_without_tones'] * 100:.2f}% | "
            f"cond_TER {overall['TER_conditional'] * 100:.2f}% | missing={missing}"
        )
        print("TER_per_tone:", overall["TER_per_tone"])

    if a.show_errors > 0:
        from rapidfuzz.distance import Levenshtein

        scored = []
        for r in refs:
            uid, ref_txt = id_and_text(r)
            hyp_txt = preds.get(uid, "")
            rs = [b + "|" + t for b, t in syllables(ref_txt)]
            hs = [b + "|" + t for b, t in syllables(hyp_txt)]
            err = Levenshtein.distance(rs, hs) / max(1, len(rs))
            scored.append((err, uid, r.get("domain", ""), ref_txt, hyp_txt))
        scored.sort(reverse=True)
        print(f"\nTop {a.show_errors} worst utterances:")
        for err, uid, dom, ref_txt, hyp_txt in scored[: a.show_errors]:
            print(f"  [{uid} | {dom} | TSER={err * 100:.1f}%]\n    REF: {ref_txt}\n    HYP: {hyp_txt}")

    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {a.output}")


if __name__ == "__main__":
    main()
