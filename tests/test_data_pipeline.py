"""Tests unitaires complets du Pôle 1 — Data Engineering (Personne 1)."""

import tempfile
import unittest
from pathlib import Path

from src.data.syllabifier import syllabify
from src.data.fongbe_phonetics import (
    normalize_fongbe_text,
    analyze_fongbe_linguistics,
    detect_unexpected_characters,
)
from src.data.submission_parser import audit_sample_submission, categorize_subset_quality
from src.data.dataset import (
    create_manifest_entry,
    split_by_speaker,
    build_test_manifest_from_submission,
)
from src.data.synthetic_data import generate_demo_dataset
from src.data.audit import audit_dataset_records


class TestPole1DataEngineering(unittest.TestCase):

    def test_sample_submission_audit_and_test_manifest(self):
        sample_csv = Path("/home/hilarie/Hackathon_1/data/raw/SampleSubmission.csv")
        report = audit_sample_submission(sample_csv)
        self.assertEqual(report["total_rows"], 9407)
        self.assertEqual(report["missing_sequence_count"], 31)
        self.assertIn("test_000186", report["missing_sequence_ids"])
        self.assertTrue(report["is_strictly_monotonic"])

        test_entries = build_test_manifest_from_submission(sample_csv)
        self.assertEqual(len(test_entries), 9407)
        self.assertEqual(test_entries[0]["id"], "test_000001")
        self.assertEqual(test_entries[-1]["id"], "test_009438")

    def test_syllabification_and_special_chars(self):
        self.assertEqual(syllabify("Un nɔ́ ɖu nǔ"), "un nɔ1 ɖu nu3")
        self.assertEqual(syllabify("kutɔ́nu"), "ku tɔ1 nu")
        norm = normalize_fongbe_text("ɖɔ ɛ")
        self.assertIn("ɖ", norm)
        self.assertIn("ɔ", norm)
        self.assertIn("ɛ", norm)

    def test_linguistic_analysis_and_unexpected_chars(self):
        bad = detect_unexpected_characters("12:4 Xó ɔ́ ε")
        self.assertIn("ε", bad)  # Epsilon grec détecté comme anomalie d'encodage

        stats = analyze_fongbe_linguistics(["Un nɔ́ ɖu nǔ", "Gbɛtɔ́ kpódó xwé kpó"])
        self.assertEqual(stats["total_sentences"], 2)
        self.assertGreater(stats["special_fongbe_chars"].get("ɔ", 0), 0)
        self.assertGreater(stats["digraph_frequencies"].get("gb", 0), 0)
        self.assertGreater(stats["nasal_vowel_frequencies"].get("un", 0), 0)

    def test_anomaly_detection_and_speaker_disjoint_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            demo = generate_demo_dataset(tmp)
            import pandas as pd
            raw_df = pd.read_csv(demo["train_csv"])
            audit = audit_dataset_records(raw_df.to_dict("records"))

            # Le jeu démo contient 15 lignes dont 3 anomalies bloquantes (too_short, silent, empty_transcript)
            self.assertEqual(audit["total_records_audited"], 15)
            self.assertEqual(audit["clean_records_count"], 12)
            self.assertEqual(audit["rejected_records_count"], 3)

            clean_entries = [
                create_manifest_entry(
                    r["audio_filepath"],
                    r["Target"],
                    r["duration"],
                    r["speaker_id"],
                    r["ID"],
                    categorize_subset_quality(r),
                )
                for r in audit["records"]
                if r["is_clean_for_training"]
            ]
            train, val = split_by_speaker(clean_entries, val_ratio=0.25, seed=42)
            train_spks = {r["speaker_id"] for r in train}
            val_spks = {r["speaker_id"] for r in val}
            self.assertEqual(len(train_spks.intersection(val_spks)), 0)


if __name__ == "__main__":
    unittest.main()
