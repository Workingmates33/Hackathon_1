#!/usr/bin/env python
"""Train or fine-tune a CTC Tonal ASR model (facebook/w2v-bert-2.0, MIT licence).

Supports:
  - Full 24-layer Teacher OR layer-pruned frugal Student (`--num-layers 12` / `16` / `24`)
  - Multi-domain training (`train_clean.csv` = reliable + religious, plus optional `--extra-train-csv`
    for filtered/pseudo-labeled Train_uncertain)
  - Automatic filtering to WAV files actually extracted on disk under `--data-dir`
  - Tonal-safe SpecAugment (`--spec-augment`, never shifts pitch!)
  - Resuming across Kaggle/Colab sessions (`--resume-from`)
  - Mirror validation (`val_general` + `val_religious` -> `TSER_mirror`) saving the best checkpoint

Usage (Kaggle / Colab / Linux GPU):
    python src/train_ctc.py \
        --data-dir /tmp/fongbe \
        --train-csv splits/train_clean.csv \
        --val-csv splits/val_mirror.csv \
        --vocab splits/vocab.json \
        --out-dir outputs/w2vbert_12L \
        --num-layers 12 \
        --max-steps 6000 \
        --eval-steps 500
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from normalize import tonal_target  # noqa: E402
from score import measure  # noqa: E402

TEST_WEIGHT_GENERAL = 6382 / 9407
TEST_WEIGHT_RELIGIOUS = 3025 / 9407


def read_rows(path: Path, data_dir: Path, max_seconds: float | None = None, require_target: bool = True) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    out = []
    for r in raw_rows:
        wav = data_dir / r["audio_path"]
        if not wav.exists():
            continue
        dur = float(r.get("duration") or 0.0)
        if max_seconds is not None and dur > max_seconds:
            continue
        tgt = (r.get("target") or tonal_target(r.get("transcription") or "")).strip()
        if require_target and not tgt:
            continue
        out.append({
            "id": r["id"],
            "wav": str(wav),
            "transcription": r.get("transcription") or "",
            "target": tgt,
            "duration": dur,
            "domain": r.get("domain", "general"),
        })
    return out


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz for {path}")
    return audio if audio.ndim == 1 else audio.mean(axis=1)


class SpeechDataset(Dataset):
    def __init__(self, rows: list[dict], feature_extractor, tokenizer, with_labels: bool = True) -> None:
        self.rows = rows
        self.fe = feature_extractor
        self.tok = tokenizer
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        feats = self.fe(load_audio(row["wav"]), sampling_rate=16000).input_features[0]
        item = {"input_features": feats}
        if self.with_labels:
            item["labels"] = self.tok(row["target"]).input_ids
        return item


def make_collate(processor):
    def collate(items: list[dict]) -> dict:
        batch = processor.pad(
            [{"input_features": x["input_features"]} for x in items],
            padding=True,
            return_tensors="pt",
        )
        if "labels" in items[0]:
            labels = processor.pad(
                labels=[{"input_ids": x["labels"]} for x in items],
                padding=True,
                return_tensors="pt",
            )
            batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        return batch

    return collate


def length_batches(durations: list[float], batch_size: int, shuffle: bool) -> list[list[int]]:
    order = np.argsort(durations)
    batches = [order[i : i + batch_size].tolist() for i in range(0, len(order), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def prune_encoder_layers(model: Wav2Vec2BertForCTC, num_layers: int) -> None:
    """Keep the first `num_layers` encoder layers for frugal Student models."""
    total = len(model.wav2vec2_bert.encoder.layers)
    if num_layers < 1 or num_layers > total:
        raise ValueError(f"--num-layers must be between 1 and {total}, got {num_layers}")
    if num_layers < total:
        model.wav2vec2_bert.encoder.layers = model.wav2vec2_bert.encoder.layers[:num_layers]
        model.config.num_hidden_layers = num_layers
        print(f"Pruned encoder layers: {total} -> {num_layers}")


@torch.no_grad()
def evaluate_model(
    model: Wav2Vec2BertForCTC,
    processor: Wav2Vec2BertProcessor,
    val_rows: list[dict],
    device: str,
    batch_size: int = 8,
    use_amp: bool = True,
) -> dict:
    model.eval()
    ds = SpeechDataset(val_rows, processor.feature_extractor, processor.tokenizer, with_labels=False)
    collate = make_collate(processor)
    batches = length_batches([r["duration"] for r in val_rows], batch_size, shuffle=False)
    preds = [""] * len(val_rows)

    for idxs in batches:
        batch = collate([ds[i] for i in idxs])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device == "cuda"):
            logits = model(
                input_features=batch["input_features"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
        texts = processor.batch_decode(logits.argmax(-1).cpu(), skip_special_tokens=True)
        for i, txt in zip(idxs, texts):
            preds[i] = txt

    gen_pairs = [(r["transcription"], p) for r, p in zip(val_rows, preds) if r["domain"] == "general"]
    rel_pairs = [(r["transcription"], p) for r, p in zip(val_rows, preds) if r["domain"] == "religious"]
    res: dict[str, object] = {"predictions": preds}

    if gen_pairs and rel_pairs:
        mg = measure(gen_pairs)
        mr = measure(rel_pairs)
        res["TSER_general"] = mg["TSER"]
        res["TSER_religious"] = mr["TSER"]
        res["TER_cond_general"] = mg["TER_conditional"]
        res["TER_cond_religious"] = mr["TER_conditional"]
        res["TSER_mirror"] = TEST_WEIGHT_GENERAL * mg["TSER"] + TEST_WEIGHT_RELIGIOUS * mr["TSER"]
    else:
        mo = measure([(r["transcription"], p) for r, p in zip(val_rows, preds)])
        res["TSER_mirror"] = mo["TSER"]
        res["TER_cond"] = mo["TER_conditional"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CTC Tonal ASR (w2v-bert-2.0 full or layer-pruned).")
    ap.add_argument("--data-dir", type=Path, required=True, help="Root folder containing extracted audio/")
    ap.add_argument("--train-csv", type=Path, required=True, help="Primary train CSV (e.g. splits/train_clean.csv)")
    ap.add_argument("--extra-train-csv", type=Path, help="Optional extra train CSV (e.g. filtered Train_uncertain)")
    ap.add_argument("--val-csv", type=Path, required=True, help="Validation CSV (e.g. splits/val_mirror.csv)")
    ap.add_argument("--vocab", type=Path, required=True, help="CTC vocab.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for checkpoints and logs")
    ap.add_argument("--base-model", default="facebook/w2v-bert-2.0", help="Commercial-licence HF backbone")
    ap.add_argument("--resume-from", type=Path, help="Resume model weights from a saved checkpoint directory")
    ap.add_argument("--num-layers", type=int, default=24, help="Number of encoder layers to keep (24, 16, 12, 8)")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--eval-limit", type=int, default=600, help="Stratified val cap during training (0 = full val)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8, help="Effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seconds", type=float, default=16.0)
    ap.add_argument("--spec-augment", action="store_true", help="Enable tonal-safe SpecAugment (no pitch shift)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_rows = read_rows(a.train_csv, a.data_dir, max_seconds=a.max_seconds, require_target=True)
    if a.extra_train_csv:
        extra = read_rows(a.extra_train_csv, a.data_dir, max_seconds=a.max_seconds, require_target=True)
        print(f"Added {len(extra):,} utterances from {a.extra_train_csv}")
        train_rows.extend(extra)
    val_rows = read_rows(a.val_csv, a.data_dir, max_seconds=None, require_target=False)

    if not train_rows:
        sys.exit(f"No training WAVs found under {a.data_dir}. Extract audio archives first!")
    if not val_rows:
        sys.exit(f"No validation WAVs found under {a.data_dir}.")

    if a.eval_limit and len(val_rows) > a.eval_limit:
        # Stratified sample preserving general/religious domains
        rng = random.Random(a.seed)
        gen = [r for r in val_rows if r["domain"] == "general"]
        rel = [r for r in val_rows if r["domain"] == "religious"]
        if gen and rel:
            n_gen = int(round(a.eval_limit * TEST_WEIGHT_GENERAL))
            n_rel = a.eval_limit - n_gen
            val_eval = rng.sample(gen, min(n_gen, len(gen))) + rng.sample(rel, min(n_rel, len(rel)))
        else:
            val_eval = rng.sample(val_rows, a.eval_limit)
    else:
        val_eval = val_rows

    trn_h = sum(r["duration"] for r in train_rows) / 3600.0
    print(f"Train available: {len(train_rows):,} utts ({trn_h:.2f} h)")
    print(f"Val available  : {len(val_rows):,} utts (scoring {len(val_eval):,} during training)")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(a.vocab),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
    )
    fe_src = str(a.resume_from) if a.resume_from else a.base_model
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(fe_src)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    if a.resume_from:
        print(f"Resuming weights from {a.resume_from}")
        model = Wav2Vec2BertForCTC.from_pretrained(str(a.resume_from))
    else:
        model = Wav2Vec2BertForCTC.from_pretrained(
            a.base_model,
            attention_dropout=0.05 if a.spec_augment else 0.0,
            hidden_dropout=0.05 if a.spec_augment else 0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05 if a.spec_augment else 0.0,
            mask_feature_prob=0.04 if a.spec_augment else 0.0,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            add_adapter=True,
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=len(tokenizer),
        )
        prune_encoder_layers(model, a.num_layers)

    model.gradient_checkpointing_enable()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f} M ({model.config.num_hidden_layers} encoder layers)")

    (a.out_dir / "training_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2),
        encoding="utf-8",
    )

    train_ds = SpeechDataset(train_rows, feature_extractor, tokenizer, with_labels=True)
    collate = make_collate(processor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    warmup_steps = int(a.max_steps * a.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, a.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    step, micro, best_tser = 0, 0, float("inf")
    running_loss: list[float] = []
    history: list[dict] = []
    t0 = time.time()

    model.train()
    while step < a.max_steps:
        sampler = length_batches([r["duration"] for r in train_rows], a.batch_size, shuffle=True)
        loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=a.num_workers)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss / a.grad_accum
            scaler.scale(loss).backward()
            running_loss.append(loss.item() * a.grad_accum)
            micro += 1
            if micro % a.grad_accum:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % 100 == 0:
                avg_l = float(np.mean(running_loss))
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step {step:5d}/{a.max_steps} | loss {avg_l:.4f} | lr {lr_now:.2e} | {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
                running_loss = []

            if step % a.eval_steps == 0 or step == a.max_steps:
                metrics = evaluate_model(model, processor, val_eval, device=device)
                tser = float(metrics["TSER_mirror"])
                g_str = f" gen={metrics['TSER_general'] * 100:.2f}%" if "TSER_general" in metrics else ""
                r_str = f" rel={metrics['TSER_religious'] * 100:.2f}%" if "TSER_religious" in metrics else ""
                print(f"  [EVAL step {step}] TSER_mirror={tser * 100:.2f}%{g_str}{r_str}", flush=True)
                entry = {"step": step, **{k: v for k, v in metrics.items() if k != "predictions"}}
                history.append(entry)
                (a.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

                # Always save latest for resumability
                latest_dir = a.out_dir / "latest"
                model.save_pretrained(latest_dir)
                processor.save_pretrained(latest_dir)

                if tser < best_tser:
                    best_tser = tser
                    best_dir = a.out_dir / "best"
                    model.save_pretrained(best_dir)
                    processor.save_pretrained(best_dir)
                    with open(a.out_dir / "best_val_predictions.csv", "w", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["id", "transcription"])
                        for r, p in zip(val_eval, metrics["predictions"]):
                            w.writerow([r["id"], p])
                    print(f"  >>> New best checkpoint saved to {best_dir} (TSER_mirror={best_tser * 100:.2f}%)", flush=True)

                model.train()

            if step >= a.max_steps:
                break

    print(f"Done. Best TSER_mirror = {best_tser * 100:.2f}% -> {a.out_dir / 'best'}")


if __name__ == "__main__":
    main()
