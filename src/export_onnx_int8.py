#!/usr/bin/env python
"""Export a Wav2Vec2BertForCTC checkpoint to ONNX INT8 for frugal 1-thread CPU inference.

Features:
  - Exports CTC model to ONNX FP32 -> dynamic INT8 quantization (`model_int8.onnx`), then deletes FP32
  - Copies only minimal offline inference files (`model_int8.onnx`, `vocab.json`) to `--out-dir`
  - Reports exact size in decimal MB (`bytes / 1e6`) vs the eliminatory 1,000 MB limit
  - Verifies pure-NumPy feature extractor parity against Hugging Face `SeamlessM4TFeatureExtractor`
  - Optional `--export-isolated-encoder` to produce `encoder_int8.onnx` for official verification
  - Optional `--dry-run-pretrained` to benchmark 24L / 16L / 12L / 8L CPU frugality BEFORE training!

Usage (benchmark BEFORE training):
    python src/export_onnx_int8.py --dry-run-pretrained --num-layers 12 --vocab splits/vocab.json --out-dir weights/bench_12L

Usage (after training):
    python src/export_onnx_int8.py --model-dir outputs/w2vbert_12L/best --out-dir weights/w2vbert_12L_int8 --export-isolated-encoder
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC, Wav2Vec2CTCTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features_numpy import extract_w2vbert_features  # noqa: E402

MAX_SIZE_BYTES = 1000 * 10**6  # 1,000 MB decimal (measure_frugality.py)


class _CTCWrapper(torch.nn.Module):
    def __init__(self, model: Wav2Vec2BertForCTC) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_features=input_features, attention_mask=attention_mask).logits


class _EncoderWrapper(torch.nn.Module):
    def __init__(self, model: Wav2Vec2BertForCTC) -> None:
        super().__init__()
        self.encoder = model.wav2vec2_bert

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features=input_features, attention_mask=attention_mask).last_hidden_state


def check_feature_parity(fe_source: str) -> None:
    """Verify pure-NumPy feature extractor matches SeamlessM4TFeatureExtractor."""
    rng = np.random.default_rng(42)
    wav = (0.1 * rng.standard_normal(16000 * 4)).astype(np.float32)
    hf_fe = SeamlessM4TFeatureExtractor.from_pretrained(fe_source)
    ref = hf_fe(wav, sampling_rate=16000).input_features[0]
    ours = extract_w2vbert_features(wav, sample_rate=16000)
    min_t = min(ref.shape[0], ours.shape[0])
    max_diff = float(np.max(np.abs(ref[:min_t] - ours[:min_t])))
    mean_diff = float(np.mean(np.abs(ref[:min_t] - ours[:min_t])))
    print(f"Feature parity check: ref={ref.shape}, numpy={ours.shape}, mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}")
    if ref.shape != ours.shape or mean_diff > 5e-3:
        print("WARNING: Feature mismatch detected! Inspect src/features_numpy.py before final inference.", file=sys.stderr)


def export_and_quantize(wrapper: torch.nn.Module, out_onnx: Path, opset: int = 17) -> None:
    wrapper.eval().cpu()
    dummy_feats = torch.randn(1, 150, 160, dtype=torch.float32)
    dummy_mask = torch.ones(1, 150, dtype=torch.long)

    with tempfile.TemporaryDirectory(prefix="onnx_fp32_") as tmp:
        fp32_path = Path(tmp) / "model_fp32.onnx"
        export_kwargs = dict(
            input_names=["input_features", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_features": {0: "batch", 1: "time"},
                "attention_mask": {0: "batch", 1: "time"},
                "output": {0: "batch", 1: "out_time"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        try:
            torch.onnx.export(wrapper, (dummy_feats, dummy_mask), str(fp32_path), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(wrapper, (dummy_feats, dummy_mask), str(fp32_path), **export_kwargs)
        print(f"Exported FP32 ONNX -> quantizing to INT8...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(out_onnx),
            weight_type=QuantType.QInt8,
        )
    print(f"Wrote INT8 ONNX: {out_onnx} ({out_onnx.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Wav2Vec2BertForCTC to ONNX INT8.")
    ap.add_argument("--model-dir", type=Path, help="Trained HF checkpoint directory")
    ap.add_argument("--dry-run-pretrained", action="store_true", help="Load pretrained base model to benchmark frugality")
    ap.add_argument("--base-model", default="facebook/w2v-bert-2.0")
    ap.add_argument("--num-layers", type=int, default=24, help="Encoder layers to keep (only for --dry-run-pretrained)")
    ap.add_argument("--vocab", type=Path, help="vocab.json (required for --dry-run-pretrained)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Minimal offline weights directory")
    ap.add_argument("--export-isolated-encoder", action="store_true", help="Also export encoder_int8.onnx outside out-dir")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.dry_run_pretrained:
        if not a.vocab:
            sys.exit("--vocab is required with --dry-run-pretrained")
        tok = Wav2Vec2CTCTokenizer(
            str(a.vocab),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            bos_token=None,
            eos_token=None,
        )
        model = Wav2Vec2BertForCTC.from_pretrained(
            a.base_model,
            add_adapter=True,
            pad_token_id=tok.pad_token_id,
            vocab_size=len(tok),
        )
        total = len(model.wav2vec2_bert.encoder.layers)
        if a.num_layers < total:
            model.wav2vec2_bert.encoder.layers = model.wav2vec2_bert.encoder.layers[: a.num_layers]
            model.config.num_hidden_layers = a.num_layers
        shutil.copy2(a.vocab, a.out_dir / "vocab.json")
        check_feature_parity(a.base_model)
    else:
        if not a.model_dir:
            sys.exit("Provide --model-dir or --dry-run-pretrained")
        model = Wav2Vec2BertForCTC.from_pretrained(str(a.model_dir))
        vocab_src = a.model_dir / "vocab.json"
        if not vocab_src.exists() and a.vocab:
            vocab_src = a.vocab
        if not vocab_src.exists():
            sys.exit(f"vocab.json not found in {a.model_dir}; pass --vocab")
        shutil.copy2(vocab_src, a.out_dir / "vocab.json")
        check_feature_parity(str(a.model_dir))

    model.eval().cpu()
    export_and_quantize(_CTCWrapper(model), a.out_dir / "model_int8.onnx", opset=a.opset)

    if a.export_isolated_encoder:
        enc_path = a.out_dir.parent / f"{a.out_dir.name}_isolated_encoder_int8.onnx"
        export_and_quantize(_EncoderWrapper(model), enc_path, opset=a.opset)
        print(f"Isolated encoder saved outside --weights dir (so it doesn't count twice): {enc_path}")

    total_bytes = sum(f.stat().st_size for f in a.out_dir.rglob("*") if f.is_file())
    verdict = "COMPLIANT" if total_bytes <= MAX_SIZE_BYTES else "EXCEEDS 1,000 MB LIMIT!"
    print(f"Total --weights size ({a.out_dir}): {total_bytes / 1e6:.2f} MB / 1000.00 MB -> {verdict}")


if __name__ == "__main__":
    main()
