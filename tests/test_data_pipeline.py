"""Unit tests for the Data Engineering pipeline."""

import unittest
from pathlib import Path
from src.data.syllabifier import syllabify
from src.data.fongbe_phonetics import normalize_fongbe_text, FONGBE_SPECIAL_CHARS
from src.data.dataset import create_manifest_entry, split_by_speaker


class TestDataEngineeringPipeline(unittest.TestCase):

    def test_syllabification_baseline(self):
        # Test case directly mentioned in challenge specification:
        # "Un nɔ́ ɖu nǔ" -> "un nɔ1 ɖu nu3"
        raw_text = "Un nɔ́ ɖu nǔ"
        expected = "un nɔ1 ɖu nu3"
        result = syllabify(raw_text)
        self.assertEqual(result, expected)

    def test_fongbe_special_characters(self):
        text = "ɖɔ"
        norm = normalize_fongbe_text(text)
        self.assertIn("ɖ", norm)
        self.assertIn("ɔ", norm)

    def test_manifest_entry(self):
        entry = create_manifest_entry(
            audio_path="data/processed/sample.wav",
            raw_transcript="Un nɔ́ ɖu nǔ",
            duration=3.14159,
            speaker_id="spk_01"
        )
        self.assertEqual(entry["text_syllabified"], "un nɔ1 ɖu nu3")
        self.assertEqual(entry["duration"], 3.142)
        self.assertEqual(entry["speaker_id"], "spk_01")

    def test_speaker_disjoint_split(self):
        records = [
            {"speaker_id": "spk_1", "id": 1},
            {"speaker_id": "spk_1", "id": 2},
            {"speaker_id": "spk_2", "id": 3},
            {"speaker_id": "spk_3", "id": 4},
        ]
        train, val = split_by_speaker(records, val_ratio=0.33, seed=42)
        train_spks = {r["speaker_id"] for r in train}
        val_spks = {r["speaker_id"] for r in val}
        # Intersection between train and validation speakers must be empty
        self.assertEqual(len(train_spks.intersection(val_spks)), 0)


if __name__ == "__main__":
    unittest.main()
