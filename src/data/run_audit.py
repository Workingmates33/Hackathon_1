"""CLI Entrypoint for Person 1 to run audits and manifest creation.

Usage:
    python3 -m src.data.run_audit --help
    python3 -m src.data.run_audit --raw-dir data/raw --output-dir data/manifests
"""

import argparse
import logging
from pathlib import Path

from src.data.config import RAW_DATA_DIR, MANIFESTS_DIR, PROCESSED_DATA_DIR
from src.data.audit import check_audio_dependencies

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Audit and prepare Fongbe ASR dataset")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Path to raw data directory")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR, help="Path to processed audio directory")
    parser.add_argument("--manifests-dir", type=Path, default=MANIFESTS_DIR, help="Path to output manifests")
    args = parser.parse_args()

    logger.info("=== Audit du Dataset Fongbe ASR (Pôle Data Engineering) ===")
    logger.info(f"Dossier source brut : {args.raw_dir}")
    logger.info(f"Dossier standardisé : {args.processed_dir}")
    logger.info(f"Dossier manifests   : {args.manifests_dir}")

    backend = check_audio_dependencies()
    logger.info(f"Backend audio détecté : {backend}")

    # Check for raw files
    if not args.raw_dir.exists() or not any(args.raw_dir.iterdir()):
        logger.warning(
            f"Le dossier '{args.raw_dir}' est actuellement vide.\n"
            "Placez les archives/audios du challenge dans 'data/raw/' puis relancez ce script."
        )
        return

    logger.info("Analyse des fichiers en cours...")


if __name__ == "__main__":
    main()
