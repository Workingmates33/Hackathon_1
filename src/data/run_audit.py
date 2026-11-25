"""Point d'entrée CLI de la Personne 1 (Data Engineering).

Exécute :
1. L'audit complet de `SampleSubmission.csv` (9 407 lignes, 31 IDs absents) et la création de `test_manifest.jsonl`
2. L'exploration et le diagnostic d'anomalies sur les données d'entraînement (`Train.csv` ou corpus démo)
3. L'analyse linguistique Fongbé (tons, voyelles nasales, digraphes, caractères spéciaux)
4. Le nettoyage, le découpage sans fuite locuteur et la sauvegarde de `train_manifest.jsonl`,
   `val_manifest.jsonl` et `dataset_metadata_report.json`.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd

from src.data.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MANIFESTS_DIR
from src.data.submission_parser import audit_sample_submission, categorize_subset_quality
from src.data.audit import audit_dataset_records
from src.data.fongbe_phonetics import analyze_fongbe_linguistics
from src.data.dataset import (
    create_manifest_entry,
    split_by_speaker,
    build_test_manifest_from_submission,
    save_manifest,
)
from src.data.synthetic_data import generate_demo_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pipeline complet Pôle 1 — Data Engineering Fongbé")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="Dossier contenant SampleSubmission.csv et Train.csv")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR, help="Dossier de sortie des audios standardisés")
    parser.add_argument("--manifests-dir", type=Path, default=MANIFESTS_DIR, help="Dossier de sortie des manifests JSONL")
    args = parser.parse_args()

    logger.info("=== PÔLE 1 (PERSONNE 1) : PIPELINE DATA ENGINEERING FONGBÉ ===")

    # 1. Traitement de SampleSubmission.csv
    sample_sub_csv = args.raw_dir / "SampleSubmission.csv"
    sub_report = {}
    if sample_sub_csv.exists():
        sub_report = audit_sample_submission(sample_sub_csv, audio_dir=args.raw_dir)
        logger.info(
            f"SampleSubmission.csv audité : {sub_report['total_rows']} lignes exactes "
            f"({sub_report['min_id']} -> {sub_report['max_id']}), "
            f"{sub_report['missing_sequence_count']} IDs absents de la numérotation."
        )

    # 2. Chargement de Train.csv réel ou création du jeu d'essai Data Engineering
    real_train_csv = args.raw_dir / "Train.csv"
    if real_train_csv.exists():
        logger.info(f"Chargement du fichier réel : {real_train_csv}")
        raw_df = pd.read_csv(real_train_csv)
        test_audio_dir = args.raw_dir / "test"
    else:
        logger.info("Train.csv non détecté dans data/raw/ -> Génération du corpus d'essai Fongbé (avec anomalies types).")
        sample_ids = pd.read_csv(sample_sub_csv)["ID"].head(5).tolist() if sample_sub_csv.exists() else None
        demo = generate_demo_dataset(args.processed_dir / "demo_data", test_ids=sample_ids)
        raw_df = pd.read_csv(demo["train_csv"])
        test_audio_dir = Path(demo["test_audio_dir"])

    if sample_sub_csv.exists():
        test_manifest_path = args.manifests_dir / "test_manifest.jsonl"
        build_test_manifest_from_submission(sample_sub_csv, test_audio_dir, test_manifest_path)
        logger.info(f"Manifest de test généré : {test_manifest_path} ({sub_report['total_rows']} entrées)")

    # 3. Audit technique audio/texte et filtrage des données problématiques
    audit_res = audit_dataset_records(raw_df.to_dict("records"))
    logger.info(
        f"Audit qualité terminé : {audit_res['clean_records_count']} propres / "
        f"{audit_res['total_records_audited']} inspectés ({audit_res['rejected_records_count']} écartés pour anomalie)."
    )

    clean_rows = [r for r in audit_res["records"] if r["is_clean_for_training"]]
    manifest_entries = []
    for r in clean_rows:
        tier = categorize_subset_quality(r)
        manifest_entries.append(
            create_manifest_entry(
                audio_path=r["audio_filepath"],
                raw_transcript=r["Target"],
                duration=r["duration"],
                speaker_id=r["speaker_id"],
                sample_id=r["ID"],
                quality_tier=tier,
            )
        )

    # 4. Analyse linguistique Fongbé sur les transcriptions nettoyées
    ling_report = analyze_fongbe_linguistics([m["text_clean"] for m in manifest_entries])

    # 5. Split Train / Validation sans fuite locuteur & Sauvegarde des livrables
    train_entries, val_entries = split_by_speaker(manifest_entries, val_ratio=0.2, seed=42)
    train_path = args.manifests_dir / "train_manifest.jsonl"
    val_path = args.manifests_dir / "val_manifest.jsonl"
    save_manifest(train_entries, train_path)
    save_manifest(val_entries, val_path)

    metadata_doc = {
        "pole": "Personne 1 — Data Engineering",
        "sample_submission_audit": sub_report,
        "training_data_audit": {k: v for k, v in audit_res.items() if k != "records"},
        "fongbe_linguistic_analysis": ling_report,
        "splits": {
            "train_count": len(train_entries),
            "val_count": len(val_entries),
            "train_speakers": sorted({x["speaker_id"] for x in train_entries}),
            "val_speakers": sorted({x["speaker_id"] for x in val_entries}),
        },
    }
    report_path = args.manifests_dir / "dataset_metadata_report.json"
    report_path.write_text(json.dumps(metadata_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Livrables Data Engineering générés dans : {args.manifests_dir}")


if __name__ == "__main__":
    main()
