"""Analyse et gestion des métadonnées CSV du challenge (Train.csv & SampleSubmission.csv)."""

import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


def find_audio_for_id(sample_id: str, audio_dir: Optional[Path]) -> Optional[Path]:
    """Recherche le fichier audio (.wav, .flac, .mp3) correspondant à un identifiant donné."""
    if audio_dir is None or not audio_dir.exists():
        return None
    for sub in ("", "test", "train", "audios"):
        base = audio_dir / sub if sub else audio_dir
        for ext in (".wav", ".flac", ".mp3", ".ogg"):
            cand = base / f"{sample_id}{ext}"
            if cand.exists():
                return cand
    return None


def audit_sample_submission(csv_path: Path, audio_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Effectue un diagnostic complet du fichier SampleSubmission.csv."""
    df = pd.read_csv(csv_path, dtype={"ID": str})
    if "Target" not in df.columns:
        df["Target"] = ""

    total_rows = len(df)
    unique_ids = df["ID"].nunique()
    duplicates = int(df["ID"].duplicated().sum())

    pattern = re.compile(r"^test_(\d+)$")
    seq_nums: List[int] = []
    malformed_ids: List[str] = []

    for id_val in df["ID"].astype(str):
        m = pattern.match(id_val.strip())
        if m:
            seq_nums.append(int(m.group(1)))
        else:
            malformed_ids.append(id_val)

    min_seq = min(seq_nums) if seq_nums else 0
    max_seq = max(seq_nums) if seq_nums else 0
    expected_range = set(range(min_seq, max_seq + 1)) if seq_nums else set()
    missing_seq = sorted(expected_range - set(seq_nums))
    missing_ids = [f"test_{n:06d}" for n in missing_seq]

    found_audios = 0
    if audio_dir and audio_dir.exists():
        for id_val in df["ID"].astype(str):
            if find_audio_for_id(id_val, audio_dir) is not None:
                found_audios += 1

    consecutive_gaps: List[List[str]] = []
    if missing_seq:
        curr = [missing_seq[0]]
        for n in missing_seq[1:]:
            if n == curr[-1] + 1:
                curr.append(n)
            else:
                consecutive_gaps.append([f"test_{x:06d}" for x in curr])
                curr = [n]
        consecutive_gaps.append([f"test_{x:06d}" for x in curr])

    return {
        "file_path": str(csv_path),
        "total_rows": total_rows,
        "unique_ids": unique_ids,
        "duplicate_ids": duplicates,
        "columns": list(df.columns),
        "min_id": f"test_{min_seq:06d}" if seq_nums else None,
        "max_id": f"test_{max_seq:06d}" if seq_nums else None,
        "sequence_span": max_seq - min_seq + 1 if seq_nums else 0,
        "missing_sequence_count": len(missing_ids),
        "missing_sequence_ids": missing_ids,
        "consecutive_gap_groups": [g for g in consecutive_gaps if len(g) > 1],
        "malformed_ids": malformed_ids,
        "is_strictly_monotonic": seq_nums == sorted(seq_nums),
        "audio_files_found": found_audios if audio_dir else 0,
    }


def categorize_subset_quality(row: Dict[str, Any]) -> str:
    """Classe un enregistrement dans l'une des 3 catégories décrites dans le sujet du challenge :
    - 'validated_multispeaker' : enregistrements validés multi-locuteurs
    - 'biblical_reading'       : lectures de passages bibliques
    - 'uncertain_tones'        : annotations tonales incertaines ou absentes
    """
    source = str(row.get("source", row.get("subset", ""))).lower()
    text = str(row.get("Target", row.get("text", "")) or "")

    if "bible" in source or "biblique" in source:
        return "biblical_reading"
    if "uncertain" in source or "incertain" in source:
        return "uncertain_tones"

    nfd = unicodedata.normalize("NFD", text)
    has_tone = any(ch in {"\u0301", "\u0300", "\u0302", "\u030c", "\u0304"} for ch in nfd)
    if len(text.split()) >= 4 and not has_tone:
        return "uncertain_tones"
    return "validated_multispeaker"
