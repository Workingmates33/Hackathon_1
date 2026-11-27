#!/usr/bin/env python
"""Ultra-frugal, ZERO-PyTorch, 100% offline 1-thread CPU ONNX INT8 inference.

Designed specifically to pass `measure_frugality.py`:
  - Does NOT import `torch` or `transformers` (saves ~400-500 MB RAM + fast startup)
  - Strict single-thread ONNX Runtime (`intra_op=1`, `inter_op=1`, `ORT_SEQUENTIAL`)
  - Safe long-audio chunking (`--max-chunk-sec`) to cap peak RAM on 50s+ recordings
  - Supports greedy CTC decoding OR optional KenLM (`--kenlm`) via `pyctcdecode`
  - Writes plain Fongbe `id,transcription` (compatible with `measure_frugality.py` & `score.py`)
  - Optional `--syllabified-output` writes ready-to-upload Zindi `ID,Target` via `syllabify()`

Usage with measure_frugality.py (Linux / Kaggle CPU / WSL2):
    python measure_frugality.py \
        --weights weights/w2vbert_12L_int8/ \
        --command "python src/infer_onnx.py --weights weights/w2vbert_12L_int8 --csv {csv} --root {root} --output {output}" \
        --csv splits/val_mirror.csv --root /tmp/fongbe --limit 300
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import wave
from pathlib import Path

# Enforce single-thread BLAS/OMP before importing numpy/onnxruntime
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ONNXRUNTIME_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features_numpy import extract_w2vbert_features  # noqa: E402
from syllabify import syllabify  # noqa: E402


def read_wav_16k(path: Path) -> np.ndarray:
    """Read 16 kHz PCM WAV using stdlib `wave` (fast, zero extra dependency), with soundfile fallback."""
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n_ch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sr != 16000 or sw != 2:
            raise ValueError("fallback")
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            audio = audio.reshape(-1, n_ch).mean(axis=1)
        return audio
    except Exception:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected 16000 Hz audio, got {sr} Hz for {path}")
        return audio if audio.ndim == 1 else audio.mean(axis=1)


def greedy_ctc_decode(logits: np.ndarray, id2tok: list[str], pad_id: int, unk_id: int) -> str:
    """Standard CTC collapse (merge consecutive identical IDs, drop blank/pad)."""
    ids = np.argmax(logits, axis=-1)
    chars: list[str] = []
    prev = -1
    for i in ids:
        ii = int(i)
        if ii != prev and ii != pad_id and ii != unk_id and 0 <= ii < len(id2tok):
            tok = id2tok[ii]
            chars.append(" " if tok == "|" else tok)
        prev = ii
    return "".join(chars).strip()


def build_kenlm_decoder(vocab: dict[str, int], kenlm_path: Path, alpha: float, beta: float):
    from pyctcdecode import build_ctcdecoder

    # Order labels strictly by token index 0..V-1
    labels = [""] * len(vocab)
    for tok, idx in vocab.items():
        if tok == "[PAD]":
            labels[idx] = ""
        elif tok == "|":
            labels[idx] = " "
        elif tok == "[UNK]":
            labels[idx] = ""
        else:
            labels[idx] = tok
    return build_ctcdecoder(labels=labels, kenlm_model_path=str(kenlm_path), alpha=alpha, beta=beta)


def transcribe_waveform(
    session: ort.InferenceSession,
    audio: np.ndarray,
    id2tok: list[str],
    pad_id: int,
    unk_id: int,
    max_chunk_sec: float,
    kenlm_decoder=None,
    beam_width: int = 32,
) -> str:
    max_samples = int(max_chunk_sec * 16000)
    if len(audio) <= max_samples:
        chunks = [audio]
    else:
        chunks = [audio[i : i + max_samples] for i in range(0, len(audio), max_samples) if len(audio[i : i + max_samples]) >= 800]

    parts: list[str] = []
    for chunk in chunks:
        feats = extract_w2vbert_features(chunk, sample_rate=16000)[None, :, :]
        mask = np.ones((1, feats.shape[1]), dtype=np.int64)
        logits = session.run(None, {"input_features": feats, "attention_mask": mask})[0][0]
        if kenlm_decoder is not None:
            text = kenlm_decoder.decode(logits, beam_width=beam_width).strip()
        else:
            text = greedy_ctc_decode(logits, id2tok, pad_id, unk_id)
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _infer_worker(args: tuple) -> list[tuple[str, str]]:
    """Run bit-exact single-thread ONNX INT8 inference on a shard of rows."""
    (
        worker_idx,
        weights_dir,
        root_dir,
        rows_shard,
        id2tok,
        pad_id,
        unk_id,
        max_chunk_sec,
        kenlm_path,
        alpha,
        beta,
        beam_width,
        vocab,
    ) = args
    import time

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(Path(weights_dir) / "model_int8.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    kenlm_decoder = build_kenlm_decoder(vocab, Path(kenlm_path), alpha, beta) if kenlm_path else None
    out: list[tuple[str, str]] = []
    t0 = time.perf_counter()
    total = len(rows_shard)
    for idx, r in enumerate(rows_shard, 1):
        audio = read_wav_16k(Path(root_dir) / r["audio_path"])
        text = transcribe_waveform(
            session,
            audio,
            id2tok=id2tok,
            pad_id=pad_id,
            unk_id=unk_id,
            max_chunk_sec=max_chunk_sec,
            kenlm_decoder=kenlm_decoder,
            beam_width=beam_width,
        )
        out.append((r["id"], text))
        if idx % 200 == 0 or idx == total:
            dt = time.perf_counter() - t0
            print(f"[worker {worker_idx}] {idx}/{total} utts ({dt / 60:.1f} min)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Frugal 1-thread CPU ONNX INT8 inference.")
    ap.add_argument("--weights", type=Path, required=True, help="Directory containing model_int8.onnx and vocab.json")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV with id, audio_path")
    ap.add_argument("--root", type=Path, required=True, help="Root folder for audio_path")
    ap.add_argument("--output", type=Path, required=True, help="Output plain Fongbe CSV (id, transcription)")
    ap.add_argument("--syllabified-output", type=Path, help="Optional Zindi submission CSV (ID, Target)")
    ap.add_argument("--kenlm", type=Path, help="Optional KenLM .bin/.arpa language model")
    ap.add_argument("--alpha", type=float, default=0.5, help="KenLM weight")
    ap.add_argument("--beta", type=float, default=1.0, help="KenLM word insertion bonus")
    ap.add_argument("--beam-width", type=int, default=32)
    ap.add_argument("--max-chunk-sec", type=float, default=25.0, help="Chunk audio longer than this (RAM safety)")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel 1-thread shard workers for fast offline Test.csv generation (ALWAYS keep 1 for measure_frugality.py)",
    )
    a = ap.parse_args()

    vocab: dict[str, int] = json.loads((a.weights / "vocab.json").read_text(encoding="utf-8"))
    id2tok = [""] * len(vocab)
    for tok, idx in vocab.items():
        id2tok[idx] = tok
    pad_id = vocab["[PAD]"]
    unk_id = vocab["[UNK]"]

    with open(a.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if a.workers <= 1:
        results = _infer_worker((
            0,
            str(a.weights),
            str(a.root),
            rows,
            id2tok,
            pad_id,
            unk_id,
            a.max_chunk_sec,
            str(a.kenlm) if a.kenlm else None,
            a.alpha,
            a.beta,
            a.beam_width,
            vocab,
        ))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        n_w = min(a.workers, max(1, len(rows)))
        # Round-robin sharding balances audio durations across workers
        shards = [rows[i::n_w] for i in range(n_w)]
        worker_args = [
            (
                w_idx,
                str(a.weights),
                str(a.root),
                shard,
                id2tok,
                pad_id,
                unk_id,
                a.max_chunk_sec,
                str(a.kenlm) if a.kenlm else None,
                a.alpha,
                a.beta,
                a.beam_width,
                vocab,
            )
            for w_idx, shard in enumerate(shards)
        ]
        by_id: dict[str, str] = {}
        # Use 'spawn' context on Linux to avoid ONNX Runtime fork() deadlocks!
        with ProcessPoolExecutor(max_workers=n_w, mp_context=mp.get_context("spawn")) as ex:
            for shard_res in ex.map(_infer_worker, worker_args):
                for uid, txt in shard_res:
                    by_id[uid] = txt
        results = [(r["id"], by_id[r["id"]]) for r in rows]

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "transcription"])
        w.writerows(results)

    if a.syllabified_output:
        a.syllabified_output.parent.mkdir(parents=True, exist_ok=True)
        with open(a.syllabified_output, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ID", "Target"])
            for uid, txt in results:
                w.writerow([uid, syllabify(txt)])


if __name__ == "__main__":
    main()
