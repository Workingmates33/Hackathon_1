"""Dataset audit and exploration tool for Fongbe ASR.

This script scans audio directories and transcript manifests to inspect:
- Audio sample rates, channels, bit depth, format
- Audio duration distribution (min, max, mean, outliers)
- Corrupted audio files
- Missing audio files vs transcript records
- Character and tone vocabulary analysis
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_audio_dependencies():
    """Verify presence of soundfile or torchaudio or wave."""
    try:
        import soundfile as sf
        return "soundfile"
    except ImportError:
        try:
            import wave
            return "wave"
        except ImportError:
            return None


def inspect_audio_file(filepath: str, backend: str = "soundfile") -> Dict[str, Any]:
    """Inspect duration, sample rate and channels of a single audio file."""
    if backend == "soundfile":
        import soundfile as sf
        info = sf.info(filepath)
        return {
            "duration": info.duration,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype
        }
    elif backend == "wave":
        import wave
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            return {
                "duration": frames / float(rate),
                "samplerate": rate,
                "channels": channels,
                "format": "WAV",
                "subtype": f"{wf.getsampwidth() * 8}bit"
            }
    return {}


if __name__ == "__main__":
    logger.info("Fongbe ASR Data Engineering Audit Script initialized.")
    backend = check_audio_dependencies()
    logger.info(f"Audio backend available: {backend}")
