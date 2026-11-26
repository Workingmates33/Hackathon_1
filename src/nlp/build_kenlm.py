"""Prepare a leak-free NFD Fongbe corpus and optionally compile a frugal KenLM n-gram model.

Designed to plug directly into Personne 2's `src/infer_onnx.py --kenlm <model.binary>`:
  - Normalizes every training line with `syllabify.tonal_target()` (Unicode NFD, lowercase,
    letters + combining tone marks, spaces only), matching the 37-token CTC vocabulary.
  - Guards against validation leakage via `--exclude-csv` and `recommended_validation == 1`.
  - Exports `corpus.txt` (1 sentence per line) and `unigrams.txt` (word lexicon for pyctcdecode).
  - Optionally compiles a quantized KenLM trie (`lmplz` + `build_binary -q 8 -b 8 trie`)
    for minimal disk/RAM footprint (1 CPU, <= 2 GB RAM, <= 1000 MB total).

CLI usage:
    python3 -m src.nlp.build_kenlm \
        --train-csv splits/train_clean.csv \
        --exclude-csv splits/val_mirror.csv \
        --out-dir outputs/kenlm \
        --order 4 \
        --compile
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from syllabify import tonal_target


def _detect_text_column(fieldnames: Sequence[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Column {explicit!r} not found in CSV headers: {list(fieldnames)}")
        return explicit
    for cand in ("target", "transcription", "text", "Target"):
        if cand in fieldnames:
            return cand
    raise ValueError(f"Could not detect text column in CSV headers: {list(fieldnames)}")


def _load_excluded_ids(csv_paths: Sequence[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in csv_paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            id_col = "id" if "id" in reader.fieldnames else ("ID" if "ID" in reader.fieldnames else None)
            if not id_col:
                continue
            for row in reader:
                val = (row.get(id_col) or "").strip()
                if val:
                    excluded.add(val)
    return excluded


def prepare_kenlm_corpus(
    train_csvs: Sequence[Path],
    out_dir: Path,
    *,
    exclude_csvs: Sequence[Path] = (),
    text_column: str | None = None,
    min_words: int = 1,
    skip_recommended_val: bool = True,
    deduplicate: bool = False,
) -> dict[str, Any]:
    """Build NFD `corpus.txt` and `unigrams.txt` from training CSVs without validation leakage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded_ids = _load_excluded_ids(exclude_csvs)

    lines: list[str] = []
    seen_lines: set[str] = set()
    word_counts: Counter[str] = Counter()
    stats = {
        "rows_read": 0,
        "skipped_excluded_id": 0,
        "skipped_recommended_val": 0,
        "skipped_too_short": 0,
        "skipped_duplicate": 0,
        "lines_written": 0,
        "total_words": 0,
        "unique_words": 0,
    }

    for csv_path in train_csvs:
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            col = _detect_text_column(reader.fieldnames, text_column)
            id_col = "id" if "id" in reader.fieldnames else ("ID" if "ID" in reader.fieldnames else None)
            has_rec_val = "recommended_validation" in reader.fieldnames

            for row in reader:
                stats["rows_read"] += 1
                if id_col:
                    row_id = (row.get(id_col) or "").strip()
                    if row_id and row_id in excluded_ids:
                        stats["skipped_excluded_id"] += 1
                        continue
                if skip_recommended_val and has_rec_val:
                    if str(row.get("recommended_validation") or "").strip() == "1":
                        stats["skipped_recommended_val"] += 1
                        continue

                cleaned = tonal_target(row.get(col) or "")
                words = cleaned.split()
                if len(words) < min_words:
                    stats["skipped_too_short"] += 1
                    continue

                if deduplicate:
                    if cleaned in seen_lines:
                        stats["skipped_duplicate"] += 1
                        continue
                    seen_lines.add(cleaned)

                lines.append(cleaned)
                word_counts.update(words)

    corpus_path = out_dir / "corpus.txt"
    unigrams_path = out_dir / "unigrams.txt"

    with open(corpus_path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")

    with open(unigrams_path, "w", encoding="utf-8", newline="\n") as f:
        for word, _ in word_counts.most_common():
            f.write(word + "\n")

    stats["lines_written"] = len(lines)
    stats["total_words"] = sum(word_counts.values())
    stats["unique_words"] = len(word_counts)
    stats["corpus_path"] = str(corpus_path)
    stats["unigrams_path"] = str(unigrams_path)
    return stats


def compile_kenlm(
    corpus_path: Path,
    out_dir: Path,
    *,
    order: int = 4,
    prune: str = "0 0 1",
    quant_bits: int = 8,
) -> dict[str, Any]:
    """Compile `corpus.txt` into a quantized KenLM `.binary` trie if `lmplz`/`build_binary` exist."""
    lmplz = shutil.which("lmplz")
    build_binary = shutil.which("build_binary")

    arpa_path = out_dir / f"fongbe_{order}gram.arpa"
    binary_path = out_dir / f"fongbe_{order}gram.binary"

    lmplz_cmd = [
        "lmplz",
        "-o",
        str(order),
        "--discount_fallback",
        "--prune",
        *prune.split(),
        "--text",
        str(corpus_path),
        "--arpa",
        str(arpa_path),
    ]
    bin_cmd = [
        "build_binary",
        "-q",
        str(quant_bits),
        "-b",
        str(quant_bits),
        "trie",
        str(arpa_path),
        str(binary_path),
    ]

    if not lmplz or not build_binary:
        return {
            "compiled": False,
            "reason": "KenLM binaries ('lmplz' / 'build_binary') not found in PATH.",
            "commands": [" ".join(lmplz_cmd), " ".join(bin_cmd)],
        }

    subprocess.run(lmplz_cmd, check=True)
    subprocess.run(bin_cmd, check=True)
    size_mb = binary_path.stat().st_size / (1024 * 1024)
    return {
        "compiled": True,
        "arpa_path": str(arpa_path),
        "binary_path": str(binary_path),
        "binary_size_mb": round(size_mb, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leak-free NFD Fongbe corpus & KenLM.")
    parser.add_argument(
        "--train-csv",
        nargs="+",
        required=True,
        type=Path,
        help="One or more training CSVs (e.g., splits/train_clean.csv).",
    )
    parser.add_argument(
        "--exclude-csv",
        nargs="*",
        default=[],
        type=Path,
        help="Validation/test CSVs whose IDs must never enter the LM corpus.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--text-column", default=None, help="Explicit text column name.")
    parser.add_argument("--order", type=int, default=4, help="N-gram order (default: 4).")
    parser.add_argument(
        "--prune",
        default="0 0 1",
        help="KenLM pruning thresholds per order (default: '0 0 1').",
    )
    parser.add_argument("--deduplicate", action="store_true", help="Drop exact duplicate lines.")
    parser.add_argument("--compile", action="store_true", help="Run lmplz + build_binary.")
    args = parser.parse_args()

    stats = prepare_kenlm_corpus(
        args.train_csv,
        args.out_dir,
        exclude_csvs=args.exclude_csv,
        text_column=args.text_column,
        deduplicate=args.deduplicate,
    )
    print("=== KenLM Corpus Prepared ===")
    for k, v in stats.items():
        print(f"  {k:<24}: {v}")

    if args.compile:
        res = compile_kenlm(
            Path(stats["corpus_path"]),
            args.out_dir,
            order=args.order,
            prune=args.prune,
        )
        print("\n=== KenLM Compilation ===")
        if res["compiled"]:
            print(f"  Binary written          : {res['binary_path']} ({res['binary_size_mb']} MB)")
        else:
            print(f"  Skipped                 : {res['reason']}")
            print("  Run these commands on Kaggle/Linux with KenLM installed:")
            for cmd in res["commands"]:
                print(f"    {cmd}")


if __name__ == "__main__":
    main()
