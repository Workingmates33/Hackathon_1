#!/usr/bin/env python
"""Generate leak-free, test-mirroring train/validation splits and CTC vocab.

Test set composition (from official README.md):
  - General speech   : 6,382 utts (67.8%), 60 unseen voices
  - Religious speech : 3,025 utts (32.2%), same reader (BIBLE_L01), held-out Bible books

This script mirrors that exact structure locally:
  - val_general.csv   : recommended_validation == 1 from Train_reliable.csv (2,396 utts, 14 unseen voices)
  - val_religious.csv : one held-out Bible book from Train_religious.csv (default: JDG, 1,194 utts, 2.81 h)
  - val_mirror.csv    : combined validation set (~66.7% general / ~33.3% religious) with `domain` column
  - train_clean.csv   : Train_reliable (rec_val=0) + remaining 8 books of Train_religious (~138 h)
  - vocab.json        : full NFD tonal CTC vocabulary (29 base letters + 5 tones + | + [UNK] + [PAD])

Usage:
    python src/make_splits.py --data-root . --out-dir splits --val-book JDG
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from normalize import LETTERS, TONES, tonal_target  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_row(r: dict[str, str], source: str, domain: str) -> dict[str, object]:
    raw = r.get("transcription") or ""
    tgt = tonal_target(raw, digits=False)
    return {
        "id": r["id"],
        "audio_path": r["audio_path"],
        "transcription": raw,
        "target": tgt,
        "speaker_id": r.get("speaker_id", ""),
        "duration": float(r["duration"]),
        "source": source,
        "domain": domain,
        "meta": r.get("book", "") or r.get("protocol", ""),
    }


def build_vocab() -> dict[str, int]:
    """Deterministic NFD character vocab covering all official letters and tones."""
    chars = sorted(LETTERS) + sorted(TONES)
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def main() -> None:
    ap = argparse.ArgumentParser(description="Create leak-free mirror splits and CTC vocab.")
    ap.add_argument("--data-root", type=Path, default=ROOT, help="Folder containing Train_*.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "splits", help="Output folder for split CSVs")
    ap.add_argument("--val-book", default="JDG", help="Held-out Bible book code for val_religious (default: JDG)")
    a = ap.parse_args()

    reliable_raw = read_csv(a.data_root / "Train_reliable.csv")
    religious_raw = read_csv(a.data_root / "Train_religious.csv")

    train_rel = [make_row(r, "reliable", "general") for r in reliable_raw if r["recommended_validation"] == "0"]
    val_gen = [make_row(r, "reliable", "general") for r in reliable_raw if r["recommended_validation"] == "1"]

    available_books = sorted({r["book"] for r in religious_raw})
    if a.val_book not in available_books:
        sys.exit(f"--val-book {a.val_book!r} not found in Train_religious.csv. Available: {available_books}")

    train_bib = [make_row(r, "religious", "religious") for r in religious_raw if r["book"] != a.val_book]
    val_bib = [make_row(r, "religious", "religious") for r in religious_raw if r["book"] == a.val_book]

    # Drop any empty normalized targets from training only (never drop from validation)
    train_clean = [r for r in (train_rel + train_bib) if r["target"]]
    val_mirror = val_gen + val_bib

    # Integrity checks: zero speaker leakage on general, zero book leakage on religious
    trn_spks = {r["speaker_id"] for r in train_rel}
    val_spks = {r["speaker_id"] for r in val_gen}
    assert not (trn_spks & val_spks), f"Speaker leakage detected: {trn_spks & val_spks}"
    assert {r["meta"] for r in train_bib}.isdisjoint({r["meta"] for r in val_bib}), "Bible book leakage!"

    fields = ["id", "audio_path", "transcription", "target", "speaker_id", "duration", "source", "domain", "meta"]
    write_csv(a.out_dir / "train_clean.csv", train_clean, fields)
    write_csv(a.out_dir / "train_reliable_only.csv", [r for r in train_rel if r["target"]], fields)
    write_csv(a.out_dir / "val_general.csv", val_gen, fields)
    write_csv(a.out_dir / "val_religious.csv", val_bib, fields)
    write_csv(a.out_dir / "val_mirror.csv", val_mirror, fields)

    vocab = build_vocab()
    (a.out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(name: str, rows: list[dict[str, object]]) -> str:
        hrs = sum(float(r["duration"]) for r in rows) / 3600.0
        spks = len({r["speaker_id"] for r in rows})
        return f"  {name:22s}: {len(rows):6,d} utts | {hrs:6.2f} h | {spks:4d} speakers"

    print(f"Splits written to {a.out_dir} (val_book={a.val_book}):")
    print(stats("train_clean", train_clean))
    print(stats("  - reliable (train)", [r for r in train_rel if r["target"]]))
    print(stats("  - religious (train)", [r for r in train_bib if r["target"]]))
    print(stats("val_general", val_gen))
    print(stats("val_religious", val_bib))
    print(stats("val_mirror", val_mirror))
    gen_pct = 100.0 * len(val_gen) / len(val_mirror)
    rel_pct = 100.0 * len(val_bib) / len(val_mirror)
    print(f"  Mirror ratio          : {gen_pct:.1f}% general / {rel_pct:.1f}% religious (Test = 67.8% / 32.2%)")
    print(f"  vocab.json            : {len(vocab)} tokens")


if __name__ == "__main__":
    main()
