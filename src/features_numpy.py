"""Pure-NumPy Kaldi log-Mel + stride-2 feature extractor for facebook/w2v-bert-2.0.

Zero PyTorch / Transformers dependency at inference time (saves ~400-500 MB RAM
and several seconds of startup time under `measure_frugality.py`).

Matches Hugging Face `SeamlessM4TFeatureExtractor` (80 Kaldi mel bins, 16 kHz,
25 ms frame / 10 ms shift, Povey window, preemphasis 0.97, utterance CMVN,
stride-2 stacking -> 160-dim features). Parity is checked automatically by
`src/export_onnx_int8.py`.
"""
from __future__ import annotations

import math
import numpy as np


def _mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _kaldi_mel_filterbank(
    num_mel_bins: int = 80,
    n_fft: int = 512,
    sample_rate: int = 16000,
    low_freq: float = 20.0,
    high_freq: float = 8000.0,
) -> np.ndarray:
    """Kaldi triangular mel filterbank matrix of shape (n_fft // 2 + 1, num_mel_bins)."""
    num_fft_bins = n_fft // 2
    fft_bin_width = sample_rate / n_fft
    mel_low = float(_mel_scale(np.array(low_freq)))
    mel_high = float(_mel_scale(np.array(high_freq)))
    mel_delta = (mel_high - mel_low) / (num_mel_bins + 1)

    bins = np.arange(num_mel_bins, dtype=np.float64)
    left_mel = mel_low + bins * mel_delta
    center_mel = mel_low + (bins + 1.0) * mel_delta
    right_mel = mel_low + (bins + 2.0) * mel_delta

    freqs = np.arange(num_fft_bins, dtype=np.float64) * fft_bin_width
    mel = _mel_scale(freqs)[:, None]  # (num_fft_bins, 1)

    up_slope = (mel - left_mel[None, :]) / (center_mel[None, :] - left_mel[None, :])
    down_slope = (right_mel[None, :] - mel) / (right_mel[None, :] - center_mel[None, :])
    fb = np.maximum(0.0, np.minimum(up_slope, down_slope))

    # Append Nyquist bin (zeroed, matching Kaldi / HF audio_utils)
    fb = np.vstack([fb, np.zeros((1, num_mel_bins), dtype=np.float64)])
    return fb.astype(np.float32)


_MEL_FB_80 = _kaldi_mel_filterbank(num_mel_bins=80, n_fft=512, sample_rate=16000, low_freq=20.0, high_freq=8000.0)
_POVEY_400 = np.power(0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(400, dtype=np.float64) / 399.0), 0.85).astype(
    np.float32
)


def extract_w2vbert_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    stride: int = 2,
    padding_value: float = 1.0,
) -> np.ndarray:
    """Extract (T_out, 160) float32 features matching SeamlessM4TFeatureExtractor.

    Args:
        waveform: 1D float32 audio in [-1, 1] at 16 kHz.
        sample_rate: Must be 16000.
        stride: Frame stacking stride (2 for w2v-bert-2.0 -> 80 * 2 = 160 dims).
        padding_value: Pad value if odd frame count (unused when dropping/zero-padding).
    """
    if sample_rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sample_rate}")
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)

    # SeamlessM4TFeatureExtractor multiplies waveform by 2**15 (Kaldi int16 scale)
    wav = wav * 32768.0

    frame_length = 400  # 25 ms
    frame_shift = 160   # 10 ms
    n_fft = 512

    if len(wav) < frame_length:
        wav = np.pad(wav, (0, frame_length - len(wav)), mode="constant")

    num_frames = 1 + (len(wav) - frame_length) // frame_shift
    # Strided framing view (num_frames, 400)
    itemsize = wav.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(num_frames, frame_length),
        strides=(frame_shift * itemsize, itemsize),
    ).copy()

    # Remove DC offset per frame
    frames -= frames.mean(axis=1, keepdims=True)

    # Pre-emphasis (0.97)
    preemph = 0.97
    frames[:, 1:] = frames[:, 1:] - preemph * frames[:, :-1]
    frames[:, 0] = frames[:, 0] * (1.0 - preemph)

    # Povey window + 512-pt rFFT power spectrum
    frames *= _POVEY_400[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    # Mel filterbank + log
    mel_energies = np.matmul(power, _MEL_FB_80)
    log_mel = np.log(np.maximum(mel_energies, np.float32(1.1920929e-7)))

    # Per-utterance zero-mean unit-variance normalisation (ddof=1 matches HF SeamlessM4T)
    mean = log_mel.mean(axis=0, keepdims=True)
    var = log_mel.var(axis=0, ddof=1 if len(log_mel) > 1 else 0, keepdims=True)
    log_mel = (log_mel - mean) / np.sqrt(var + np.float32(1e-7))

    # Stride-2 stacking (SeamlessM4T trims trailing odd frame: `num_frames // stride * stride`)
    n_stacked = (log_mel.shape[0] // stride) * stride
    if n_stacked == 0:
        pad_n = stride - log_mel.shape[0]
        log_mel = np.pad(log_mel, ((0, pad_n), (0, 0)), constant_values=padding_value)
        n_stacked = stride
    stacked = log_mel[:n_stacked].reshape(n_stacked // stride, log_mel.shape[1] * stride)
    return np.ascontiguousarray(stacked, dtype=np.float32)
