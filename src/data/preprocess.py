"""Audio preprocessing pipeline for Fongbe ASR.

Standardizes raw audio files into 16kHz, 1-channel (mono), 16-bit PCM WAV
using ffmpeg or standard library fallback.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.config import TARGET_SAMPLE_RATE, TARGET_CHANNELS

logger = logging.getLogger(__name__)


def convert_audio_ffmpeg(
    input_path: Path,
    output_path: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS
) -> bool:
    """Converts audio file to standardized WAV format using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-y",               # Overwrite output
        "-v", "error",       # Quiet
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le", # 16-bit PCM
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {input_path}: {e.stderr.decode('utf-8')}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg binary not found in PATH.")
        return False


def get_audio_duration_ffprobe(filepath: Path) -> Optional[float]:
    """Retrieves duration in seconds of an audio file using ffprobe/ffmpeg."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filepath)
    ]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(res.stdout.strip())
    except Exception:
        return None
