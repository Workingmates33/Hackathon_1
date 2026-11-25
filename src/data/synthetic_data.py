"""Générateur de jeu d'essai réaliste Fongbé (incluant cas propres ET anomalies typiques)
pour valider le pipeline de détection et de nettoyage de la Personne 1 (Data Engineering).
"""

import wave
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.data.config import TARGET_SAMPLE_RATE
from src.data.syllabifier import syllabify


SAMPLE_FONGBE_CORPUS = [
    {"text": "Un nɔ́ ɖu nǔ", "speaker": "spk_01", "source": "validated_multispeaker", "anomaly": None},
    {"text": "A fɔ́n à", "speaker": "spk_01", "source": "validated_multispeaker", "anomaly": None},
    {"text": "É nyɔ́ ɖésú", "speaker": "spk_02", "source": "validated_multispeaker", "anomaly": None},
    {"text": "Mǐ ná yì kutɔ́nu", "speaker": "spk_02", "source": "validated_multispeaker", "anomaly": None},
    {"text": "Nɔví ce wá xwé gbe", "speaker": "spk_03", "source": "validated_multispeaker", "anomaly": None},
    {"text": "Kú ɖó azɔ̌", "speaker": "spk_03", "source": "validated_multispeaker", "anomaly": None},
    {"text": "Mawu jíɖótó ɔ́ bló jǐnukúnsin kpódó ayikúngban kpó", "speaker": "spk_04", "source": "biblical_reading", "anomaly": None},
    {"text": "12:4 Xó ɔ́ ko ɖè sín bǐbɛ́mɛ ε", "speaker": "spk_04", "source": "biblical_reading", "anomaly": "dirty_text"},
    {"text": "Gbɛtɔ́ lɛ́ɛ bǐ wɛ nyí nɔví", "speaker": "spk_05", "source": "validated_multispeaker", "anomaly": None},
    {"text": "Un jló ná kplɔ́n fɔngbe", "speaker": "spk_05", "source": "validated_multispeaker", "anomaly": None},
    {"text": "E yi axi me egbe", "speaker": "spk_06", "source": "uncertain_tones", "anomaly": None},
    {"text": "Vi lɛ ɖo wemaxɔmɛ", "speaker": "spk_06", "source": "uncertain_tones", "anomaly": None},
    # Cas problématiques volontaires pour démontrer la détection d'anomalies de Personne 1 :
    {"text": "Ó", "speaker": "spk_07", "source": "validated_multispeaker", "anomaly": "too_short"},
    {"text": "Un nɔ́ se fɔngbe", "speaker": "spk_07", "source": "validated_multispeaker", "anomaly": "silent_audio"},
    {"text": "", "speaker": "spk_08", "source": "uncertain_tones", "anomaly": "empty_transcript"},
]


def synthesize_tonal_utterance(text: str, sr: int = TARGET_SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """Synthétise un signal audio WAV 16kHz modulé selon les tons Fongbé."""
    rng = np.random.default_rng(seed)
    sylls = syllabify(text).split() if text else ["a"]
    segments = [np.zeros(int(0.06 * sr), dtype=np.float32)]

    for syl in sylls:
        dur = float(rng.uniform(0.18, 0.28))
        n = int(dur * sr)
        t = np.linspace(0, dur, n, endpoint=False)
        if syl.endswith("1"):
            f0 = np.full(n, 210.0)
        elif syl.endswith("2"):
            f0 = np.linspace(210.0, 130.0, n)
        elif syl.endswith("3"):
            f0 = np.linspace(130.0, 215.0, n)
        elif syl.endswith("4"):
            f0 = np.full(n, 125.0)
        else:
            f0 = np.full(n, 165.0)

        vowel_hash = (sum(ord(c) for c in syl) % 5) + 1
        phase = 2.0 * np.pi * np.cumsum(f0) / sr
        sig = (0.55 * np.sin(phase) + 0.20 * np.sin(2.0 * np.pi * (350.0 + 180.0 * vowel_hash) * t)) * np.hanning(n)
        sig += rng.normal(0, 0.004, size=n)
        segments.append(sig.astype(np.float32))
        segments.append(np.zeros(int(0.04 * sr), dtype=np.float32))

    full = np.concatenate(segments)
    return (0.8 * full / (np.max(np.abs(full)) + 1e-8)).astype(np.float32)


def write_wav_pcm16(filepath: Path, waveform: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def generate_demo_dataset(output_root: Path, test_ids: List[str] = None) -> Dict[str, Any]:
    """Génère un mini-corpus d'exploration (avec cas sains + cas problématiques) pour le Notebook Data."""
    train_audio_dir = output_root / "demo_audio" / "train"
    test_audio_dir = output_root / "demo_audio" / "test"
    rows = []

    for i, item in enumerate(SAMPLE_FONGBE_CORPUS, start=1):
        sample_id = f"train_{i:06d}"
        wav_path = train_audio_dir / f"{sample_id}.wav"

        if item["anomaly"] == "too_short":
            wav = 0.4 * np.sin(2 * np.pi * 180 * np.linspace(0, 0.20, int(0.20 * TARGET_SAMPLE_RATE))).astype(np.float32)
        elif item["anomaly"] == "silent_audio":
            wav = np.zeros(int(1.2 * TARGET_SAMPLE_RATE), dtype=np.float32)
        else:
            wav = synthesize_tonal_utterance(item["text"], seed=100 + i)

        write_wav_pcm16(wav_path, wav)
        rows.append({
            "ID": sample_id,
            "Target": item["text"],
            "speaker_id": item["speaker"],
            "source": item["source"],
            "audio_filepath": str(wav_path),
        })

    train_csv = output_root / "demo_train_raw.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)

    if test_ids is None:
        test_ids = [f"test_{i:06d}" for i in range(1, 6)]
    for idx, tid in enumerate(test_ids[:5]):
        wav = synthesize_tonal_utterance(SAMPLE_FONGBE_CORPUS[idx]["text"], seed=500 + idx)
        write_wav_pcm16(test_audio_dir / f"{tid}.wav", wav)

    return {
        "train_csv": str(train_csv),
        "train_count": len(rows),
        "test_audio_dir": str(test_audio_dir),
    }
