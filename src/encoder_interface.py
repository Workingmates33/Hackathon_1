"""Isolated audio encoder interface for official verification & tone probing.

Competition verification rule:
  "top-ranked teams provide their inference code, weights, a Docker container or a
   reproducible script, the list of external resources with their licences, and
   their isolated encoder (for the tone analysis published in the summary paper)."

This module cleanly isolates the acoustic encoder from the CTC classification head
for both PyTorch checkpoints and exported ONNX INT8 models.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from src.features_numpy import extract_w2vbert_features


class IsolatedOnnxEncoder:
    """Zero-PyTorch isolated encoder running on 1 CPU thread via ONNX Runtime."""

    def __init__(self, encoder_onnx_path: str | Path, num_threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(encoder_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def encode(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return frame-level encoder hidden states of shape (T_enc, D)."""
        feats = extract_w2vbert_features(waveform, sample_rate=sample_rate)[None, :, :]
        mask = np.ones((1, feats.shape[1]), dtype=np.int64)
        hidden = self.session.run(
            None,
            {"input_features": feats, "attention_mask": mask},
        )[0]
        return hidden[0]


class IsolatedTorchEncoder:
    """PyTorch isolated encoder exposing final and intermediate layer states."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        import torch
        from transformers import Wav2Vec2BertForCTC

        self.device = device
        full = Wav2Vec2BertForCTC.from_pretrained(str(model_dir), local_files_only=True)
        # Keep ONLY the acoustic encoder backbone (no lm_head / CTC projection)
        self.encoder = full.wav2vec2_bert.eval().to(device)
        self._torch = torch

    def encode(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        output_hidden_states: bool = False,
    ):
        feats = extract_w2vbert_features(waveform, sample_rate=sample_rate)[None, :, :]
        x = self._torch.from_numpy(feats).to(self.device)
        mask = self._torch.ones((1, feats.shape[1]), dtype=self._torch.long, device=self.device)
        with self._torch.no_grad():
            out = self.encoder(
                input_features=x,
                attention_mask=mask,
                output_hidden_states=output_hidden_states,
            )
        if output_hidden_states:
            return tuple(h[0].cpu().numpy() for h in out.hidden_states)
        return out.last_hidden_state[0].cpu().numpy()
