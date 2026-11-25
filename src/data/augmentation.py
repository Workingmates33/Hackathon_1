"""Module de Data Augmentation Audio & Spectrogramme pour l'ASR Fongbé Tonal.

Attention critique pour une langue tonale :
- NE JAMAIS appliquer de Pitch Shifting fort sans modifier les tons, car la hauteur mélodique (F0)
  porte l'information tonale lexicale et grammaticale en Fongbé !
- Privilégier : ajout de bruit blanc/ambiant léger, perturbation légère de vitesse (0.95x-1.05x
  avec préservation de hauteur ou très faible facteur), variation de gain, et SpecAugment (masquage
  temporel et fréquentiel modéré).
"""

import numpy as np
from typing import Tuple


def add_gaussian_noise(waveform: np.ndarray, snr_db: float = 20.0, rng: np.random.Generator = None) -> np.ndarray:
    """Ajoute un bruit gaussien contrôlé par le rapport signal-sur-bruit (SNR en dB)."""
    if rng is None:
        rng = np.random.default_rng(42)
    sig_power = np.mean(waveform ** 2) + 1e-12
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape)
    augmented = waveform + noise
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


def random_gain(waveform: np.ndarray, min_gain_db: float = -6.0, max_gain_db: float = 6.0, rng: np.random.Generator = None) -> np.ndarray:
    """Applique une variation aléatoire de volume sans écrêtage."""
    if rng is None:
        rng = np.random.default_rng(42)
    gain_db = rng.uniform(min_gain_db, max_gain_db)
    factor = 10 ** (gain_db / 20.0)
    return np.clip(waveform * factor, -1.0, 1.0).astype(np.float32)


def spec_augment(
    log_mel: np.ndarray,
    freq_mask_param: int = 8,
    time_mask_param: int = 15,
    num_freq_masks: int = 1,
    num_time_masks: int = 2,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Applique SpecAugment sur un spectrogramme Log-Mel de forme (n_frames, n_mels).
    Masquage fréquentiel modéré pour ne pas détruire les harmoniques de F0 (tons).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    aug = log_mel.copy()
    n_frames, n_mels = aug.shape
    fill_val = float(np.mean(aug))

    for _ in range(num_freq_masks):
        f = int(rng.integers(0, min(freq_mask_param, n_mels) + 1))
        if f > 0 and n_mels - f > 0:
            f0 = int(rng.integers(0, n_mels - f))
            aug[:, f0:f0 + f] = fill_val

    for _ in range(num_time_masks):
        t = int(rng.integers(0, min(time_mask_param, n_frames) + 1))
        if t > 0 and n_frames - t > 0:
            t0 = int(rng.integers(0, n_frames - t))
            aug[t0:t0 + t, :] = fill_val

    return aug
