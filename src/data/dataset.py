"""Dataset manifest builder and split manager."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.data.syllabifier import syllabify


def create_manifest_entry(
    audio_path: str,
    raw_transcript: str,
    duration: float,
    speaker_id: str = "unknown"
) -> Dict[str, Any]:
    """Create a standardized manifest entry dictionary."""
    return {
        "audio_filepath": str(audio_path),
        "duration": round(duration, 3),
        "text_raw": raw_transcript.strip(),
        "text_syllabified": syllabify(raw_transcript),
        "speaker_id": speaker_id
    }


def split_by_speaker(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset ensuring speaker-disjoint partition (no speaker leakage between train and val)."""
    random.seed(seed)
    speakers = list({r.get("speaker_id", "unknown") for r in records})
    random.shuffle(speakers)

    val_speaker_count = max(1, int(len(speakers) * val_ratio))
    val_speakers = set(speakers[:val_speaker_count])

    train_records = [r for r in records if r.get("speaker_id") not in val_speakers]
    val_records = [r for r in records if r.get("speaker_id") in val_speakers]

    # Fallback if all records have the same speaker
    if not train_records or not val_records:
        split_idx = int(len(records) * (1 - val_ratio))
        random.shuffle(records)
        train_records = records[:split_idx]
        val_records = records[split_idx:]

    return train_records, val_records


def save_manifest(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Save records as JSON Lines manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
