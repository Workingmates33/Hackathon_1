from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"

# Audio constraints & standard parameters
TARGET_SAMPLE_RATE = 16000  # 16 kHz standard for Wav2Vec2 / Whisper / Conformer
TARGET_CHANNELS = 1         # Mono
TARGET_SAMPLE_WIDTH = 2     # 16-bit PCM

# Audio filtering thresholds
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 20.0

# Memory & model constraints for challenge submission
MAX_MODEL_SIZE_MB = 1000
MAX_RAM_MB = 2048
TARGET_DEVICE = "cpu"
SINGLE_THREAD = True
