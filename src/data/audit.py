"""Audit technique approfondi des fichiers audio et détection d'anomalies (Personne 1).

Couvre les tâches Data Engineering :
- Identifier les formats, taux d'échantillonnage, canaux, profondeur en bits et durées
- Détecter les fichiers audio corrompus, manquants, silencieux, saturés (clipping) ou hors-plage (<0.5s / >20s)
- Vérifier la cohérence entre durée audio et longueur de transcription (ratio caractères/seconde)
"""

import wave
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.data.config import MIN_DURATION_SEC, MAX_DURATION_SEC, TARGET_SAMPLE_RATE, TARGET_CHANNELS
from src.data.fongbe_phonetics import detect_unexpected_characters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_audio_dependencies() -> str:
    """Identifie le backend audio disponible."""
    try:
        import soundfile  # noqa: F401
        return "soundfile"
    except ImportError:
        return "wave"


def inspect_audio_file(filepath: str, backend: Optional[str] = None) -> Dict[str, Any]:
    """Extrait les propriétés techniques et acoustiques (RMS, silence, saturation) d'un fichier audio."""
    p = Path(filepath)
    if not p.exists():
        return {"exists": False, "corrupted": True, "error": "File not found"}

    if backend is None:
        backend = check_audio_dependencies()

    try:
        if backend == "soundfile":
            import soundfile as sf
            info = sf.info(str(p))
            samples, rate = sf.read(str(p), dtype="float32", always_2d=True)
            channels = info.channels
            duration = info.duration
            bit_depth = 16 if "16" in (info.subtype or "") else (24 if "24" in (info.subtype or "") else 32)
            fmt = info.format
            mono = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
        else:
            with wave.open(str(p), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                duration = frames / float(rate) if rate > 0 else 0.0
                raw_bytes = wf.readframes(frames)
            bit_depth = sampwidth * 8
            fmt = "WAV"
            if sampwidth == 2 and len(raw_bytes) > 0:
                mono = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                mono = np.zeros(1, dtype=np.float32)

        rms_energy = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
        peak_amp = float(np.max(np.abs(mono))) if len(mono) else 0.0
        silence_ratio = float(np.mean(np.abs(mono) < 0.01)) if len(mono) else 1.0
        clipping_ratio = float(np.mean(np.abs(mono) >= 0.995)) if len(mono) else 0.0

        return {
            "exists": True,
            "corrupted": False,
            "duration": round(duration, 4),
            "samplerate": rate,
            "channels": channels,
            "bit_depth": bit_depth,
            "format": fmt,
            "rms_energy": round(rms_energy, 5),
            "peak_amplitude": round(peak_amp, 4),
            "silence_ratio": round(silence_ratio, 4),
            "clipping_ratio": round(clipping_ratio, 5),
            "needs_resampling": (rate != TARGET_SAMPLE_RATE) or (channels != TARGET_CHANNELS) or (bit_depth != 16),
        }
    except Exception as exc:
        return {"exists": True, "corrupted": True, "error": str(exc)}


def audit_dataset_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse une liste d'enregistrements {ID, audio_filepath, Target, ...} et détecte toutes les anomalies."""
    detailed_rows: List[Dict[str, Any]] = []
    anomalies_summary = {
        "missing_or_corrupted_audio": 0,
        "too_short_audio": 0,
        "too_long_audio": 0,
        "near_silent_audio": 0,
        "clipped_audio": 0,
        "empty_transcription": 0,
        "unexpected_chars_in_text": 0,
        "abnormal_speech_rate": 0,
        "needs_audio_standardization": 0,
    }

    for row in records:
        rec_id = row.get("ID", "unknown")
        wav_path = str(row.get("audio_filepath", ""))
        text = str(row.get("Target", row.get("text", "")) or "").strip()

        info = inspect_audio_file(wav_path)
        flags: List[str] = []

        if not info.get("exists") or info.get("corrupted"):
            flags.append("corrupted_or_missing_audio")
            anomalies_summary["missing_or_corrupted_audio"] += 1
            dur = 0.0
        else:
            dur = info["duration"]
            if dur < MIN_DURATION_SEC:
                flags.append("too_short")
                anomalies_summary["too_short_audio"] += 1
            if dur > MAX_DURATION_SEC:
                flags.append("too_long")
                anomalies_summary["too_long_audio"] += 1
            if info["rms_energy"] < 0.005 or info["silence_ratio"] > 0.92:
                flags.append("near_silent")
                anomalies_summary["near_silent_audio"] += 1
            if info["clipping_ratio"] > 0.01:
                flags.append("audio_clipping")
                anomalies_summary["clipped_audio"] += 1
            if info["needs_resampling"]:
                flags.append("needs_resampling")
                anomalies_summary["needs_audio_standardization"] += 1

        if not text:
            flags.append("empty_transcript")
            anomalies_summary["empty_transcription"] += 1

        bad_chars = detect_unexpected_characters(text)
        if bad_chars:
            flags.append(f"unexpected_chars:{''.join(bad_chars)}")
            anomalies_summary["unexpected_chars_in_text"] += 1

        chars_per_sec = round(len(text) / dur, 2) if dur > 0 else 0.0
        if dur > 0 and text and (chars_per_sec < 2.0 or chars_per_sec > 30.0):
            flags.append("abnormal_chars_per_sec")
            anomalies_summary["abnormal_speech_rate"] += 1

        # Un échantillon est directement utilisable s'il n'a aucune anomalie bloquante
        blocking = {"corrupted_or_missing_audio", "too_short", "too_long", "near_silent", "empty_transcript"}
        is_clean = not any(f in blocking for f in flags)

        detailed_rows.append({
            "ID": rec_id,
            "audio_filepath": wav_path,
            "Target": text,
            "speaker_id": row.get("speaker_id", "unknown"),
            "source": row.get("source", "unknown"),
            **{k: v for k, v in info.items() if k != "error"},
            "chars_per_sec": chars_per_sec,
            "anomaly_flags": flags,
            "is_clean_for_training": is_clean,
        })

    clean_count = sum(1 for r in detailed_rows if r["is_clean_for_training"])
    valid_durs = [r["duration"] for r in detailed_rows if r.get("duration", 0) > 0]

    return {
        "total_records_audited": len(detailed_rows),
        "clean_records_count": clean_count,
        "rejected_records_count": len(detailed_rows) - clean_count,
        "total_hours": round(sum(valid_durs) / 3600.0, 5),
        "duration_stats_sec": {
            "min": round(min(valid_durs), 3) if valid_durs else 0.0,
            "mean": round(float(np.mean(valid_durs)), 3) if valid_durs else 0.0,
            "median": round(float(np.median(valid_durs)), 3) if valid_durs else 0.0,
            "max": round(max(valid_durs), 3) if valid_durs else 0.0,
        },
        "anomalies_summary": anomalies_summary,
        "records": detailed_rows,
    }
