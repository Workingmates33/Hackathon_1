"""Constructeur des Manifests (Train / Validation / Test) et Métadonnées pour l'équipe.

Génère les livrables officiels de la Personne 1 (Data Engineering) :
- `train_manifest.jsonl` : échantillons nettoyés d'entraînement (sans fuite locuteur)
- `val_manifest.jsonl`   : échantillons nettoyés de validation (locuteurs disjoints)
- `test_manifest.jsonl`  : les 9 407 entrées exactes issues de `SampleSubmission.csv`
- `dataset_metadata_report.json` : documentation statistique complète du dataset préparé
"""

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from src.data.syllabifier import syllabify
from src.data.submission_parser import categorize_subset_quality, find_audio_for_id


def clean_text_for_manifest(text: str) -> str:
    """Nettoie les artefacts textuels évidents (glyphes confondus, versets chiffrés) en gardant les tons."""
    if not text or not isinstance(text, str):
        return ""
    replacements = {"Ɖ": "ɖ", "Ɔ": "ɔ", "Ɛ": "ɛ", "ε": "ɛ", "đ": "ɖ", "ɗ": "ɖ", "«": " ", "»": " "}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"\b\d+[:.]\d+\b", " ", text.lower())
    text = re.sub(r"[\[\]\(\)\{\}<>0-9]", " ", text)
    nfd = unicodedata.normalize("NFD", text)
    allowed_tones = {"\u0301", "\u0300", "\u0302", "\u030c", "\u0304"}
    kept = []
    for ch in nfd:
        if ch in allowed_tones or ch.isalpha() or ch in {"ɖ", "ɔ", "ɛ", " "}:
            kept.append(ch)
        elif not unicodedata.combining(ch):
            kept.append(" ")
    return " ".join(unicodedata.normalize("NFC", "".join(kept)).split())


def create_manifest_entry(
    audio_path: str,
    raw_transcript: str,
    duration: float,
    speaker_id: str = "unknown",
    sample_id: Optional[str] = None,
    quality_tier: str = "validated_multispeaker",
) -> Dict[str, Any]:
    """Crée une entrée standardisée pour les manifests d'entraînement et validation."""
    cleaned = clean_text_for_manifest(raw_transcript)
    entry = {
        "id": sample_id or Path(audio_path).stem,
        "audio_filepath": str(audio_path),
        "duration": round(float(duration), 3),
        "text_raw": raw_transcript.strip(),
        "text_clean": cleaned,
        "text_syllabified": syllabify(cleaned),
        "speaker_id": speaker_id,
        "quality_tier": quality_tier,
    }
    return entry


def split_by_speaker(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partitionne le dataset sans fuite de locuteur (intersection des speaker_id vide)."""
    rng = random.Random(seed)
    speakers = sorted({r.get("speaker_id", "unknown") for r in records})
    rng.shuffle(speakers)

    val_speaker_count = max(1, int(round(len(speakers) * val_ratio)))
    val_speakers = set(speakers[:val_speaker_count])

    train_records = [r for r in records if r.get("speaker_id") not in val_speakers]
    val_records = [r for r in records if r.get("speaker_id") in val_speakers]

    if not train_records or not val_records:
        shuffled = list(records)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1.0 - val_ratio)))
        train_records = shuffled[:split_idx]
        val_records = shuffled[split_idx:]

    return train_records, val_records


def build_test_manifest_from_submission(
    sample_submission_csv: Path,
    test_audio_dir: Optional[Path] = None,
    output_manifest_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Construit `test_manifest.jsonl` à partir des 9 407 IDs exacts de `SampleSubmission.csv`."""
    df = pd.read_csv(sample_submission_csv, dtype={"ID": str})
    entries: List[Dict[str, Any]] = []

    for order_idx, sample_id in enumerate(df["ID"].tolist()):
        wav_path = find_audio_for_id(sample_id, test_audio_dir)
        entries.append({
            "submission_order": order_idx,
            "id": sample_id,
            "audio_filepath": str(wav_path) if wav_path else f"data/raw/test/{sample_id}.wav",
            "audio_exists": wav_path is not None,
        })

    if output_manifest_path is not None:
        save_manifest(entries, output_manifest_path)
    return entries


def save_manifest(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Sauvegarde une liste d'enregistrements au format JSON Lines UTF-8."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
