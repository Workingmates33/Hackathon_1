"""Unit tests for the Persona 3 Fongbe NLP & tonal analysis module."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.nlp.audit_text import audit_transcriptions, format_audit_report
from src.nlp.build_kenlm import prepare_kenlm_corpus
from src.nlp.error_analysis import analyze_errors, format_error_report
from src.nlp.filter_uncertain import calibrate_reference_stats, filter_uncertain_csv, score_utterance
from src.nlp.metrics import syllable_wer
from src.nlp.normalize import (
    is_already_syllabified,
    safe_syllabify,
    split_syllable,
    strip_tones,
    validate_submission_target,
)
from syllabify import syllabify


class TestNormalize(unittest.TestCase):
    def test_official_examples(self) -> None:
        self.assertEqual(safe_syllabify("Un nɔ́ ɖu nǔ"), "un nɔ1 ɖu nu3")
        self.assertEqual(safe_syllabify("Kɔ́xojí, ɖ'é mɛ !"), "kɔ1 xo ji1 ɖe1 mɛ")
        self.assertEqual(safe_syllabify("Mǐ ná yi gbɛ̀ ń"), "mi3 na1 yi gbɛ2 n1")

    def test_double_syllabify_guard(self) -> None:
        already = syllabify("Un nɔ́ ɖu nǔ")
        self.assertTrue(is_already_syllabified(already))
        # Official syllabify() strips digits if called twice:
        self.assertEqual(syllabify(already), "un nɔ ɖu nu")
        # safe_syllabify() guards against this bug:
        with self.assertRaises(ValueError):
            safe_syllabify(already)
        self.assertEqual(
            safe_syllabify(already, allow_already_syllabified=True),
            "un nɔ1 ɖu nu3",
        )

    def test_split_and_strip_tones(self) -> None:
        self.assertEqual(split_syllable("nɔ1"), ("nɔ", "1"))
        self.assertEqual(split_syllable("un"), ("un", ""))
        self.assertEqual(strip_tones("un nɔ1 ɖu nu3"), "un nɔ ɖu nu")

    def test_validate_submission_target(self) -> None:
        self.assertEqual(validate_submission_target("un nɔ1 ɖu nu3"), [])
        self.assertEqual(validate_submission_target(""), [])
        self.assertTrue(validate_submission_target("Un nɔ1"))
        self.assertTrue(validate_submission_target("un  nɔ1"))
        self.assertTrue(validate_submission_target("un nɔ́"))


class TestMetricsAndErrorAnalysis(unittest.TestCase):
    def test_perfect_match(self) -> None:
        res = syllable_wer(["Un nɔ́ ɖu nǔ"], ["Un nɔ́ ɖu nǔ"])
        self.assertEqual(res["syllable_wer"], 0.0)
        self.assertEqual(res["base_syllable_wer"], 0.0)
        self.assertEqual(res["tone_error_gap"], 0.0)
        self.assertEqual(res["tone_accuracy_on_matched_bases"], 1.0)

    def test_pure_tone_errors(self) -> None:
        # 'nɔ́' (1) predicted as 'nɔ̀' (2), 'nǔ' (3) predicted without tone
        res = syllable_wer(["Un nɔ́ ɖu nǔ"], ["Un nɔ̀ ɖu nu"])
        self.assertAlmostEqual(res["syllable_wer"], 0.5)
        self.assertAlmostEqual(res["base_syllable_wer"], 0.0)
        self.assertAlmostEqual(res["tone_error_gap"], 0.5)
        self.assertAlmostEqual(res["tone_accuracy_on_matched_bases"], 0.5)

        report = analyze_errors(["Un nɔ́ ɖu nǔ"], ["Un nɔ̀ ɖu nu"])
        self.assertEqual(report["pure_tone_substitutions"], 2)
        self.assertEqual(report["segmental_substitutions"], 0)
        self.assertEqual(report["tone_confusion_matrix"]["1"]["2"], 1)
        self.assertEqual(report["tone_confusion_matrix"]["3"]["∅"], 1)
        text = format_error_report(report)
        self.assertIn("Tone Confusion Matrix", text)

    def test_mixed_errors_already_syllabified(self) -> None:
        report = analyze_errors(
            ["un nɔ1 ɖu nu3"],
            ["un nɔ2 gba nu3 mɛ"],
            already_syllabified=True,
        )
        self.assertEqual(report["pure_tone_substitutions"], 1)
        self.assertEqual(report["segmental_substitutions"], 1)
        self.assertEqual(report["metrics"]["insertions"], 1)


class TestAuditText(unittest.TestCase):
    def test_audit_transcriptions(self) -> None:
        samples = [
            "Un nɔ́ ɖu nǔ",
            "Kɔ́xojí, ɖ'é mɛ !",
            "Test ŋ symbol",  # 'ŋ' is not in syllabify.LETTERS -> flagged as dropped
            "",
        ]
        report = audit_transcriptions(samples)
        self.assertEqual(report["total_rows"], 4)
        self.assertEqual(report["empty_raw_rows"], 1)
        self.assertEqual(report["empty_after_syllabify_rows"], 1)
        dropped = {ch for ch, _, _, _ in report["unexpected_dropped_chars"]}
        self.assertIn("ŋ", dropped)
        text = format_audit_report(report)
        self.assertIn("Tone Distribution", text)


class TestBuildKenLM(unittest.TestCase):
    def test_prepare_corpus_excludes_val_and_normalizes_nfd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_csv = tmp_dir / "train.csv"
            val_csv = tmp_dir / "val.csv"
            out_dir = tmp_dir / "kenlm"

            with open(train_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "transcription", "recommended_validation"])
                w.writerow(["u1", "Un nɔ́ ɖu nǔ!", "0"])
                w.writerow(["u2", "Kɔ́xojí, ɖ'é mɛ ?", "1"])  # recommended_validation -> excluded
                w.writerow(["u3", "Mǐ ná yi gbɛ̀ ń", "0"])  # in val.csv -> excluded
                w.writerow(["u4", "Azɔ̌mɛ 2026", "0"])

            with open(val_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "transcription"])
                w.writerow(["u3", "Mǐ ná yi gbɛ̀ ń"])

            stats = prepare_kenlm_corpus([train_csv], out_dir, exclude_csvs=[val_csv])
            self.assertEqual(stats["lines_written"], 2)
            self.assertEqual(stats["skipped_recommended_val"], 1)
            self.assertEqual(stats["skipped_excluded_id"], 1)

            corpus_lines = (out_dir / "corpus.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(corpus_lines), 2)
            # Digits and punctuation stripped, tones kept in NFD
            self.assertNotIn("2026", corpus_lines[1])
            self.assertTrue((out_dir / "unigrams.txt").exists())


class TestFilterUncertain(unittest.TestCase):
    def test_score_and_filter_uncertain(self) -> None:
        reliable = [
            "Un nɔ́ ɖu nǔ",
            "Kɔ́xojí ɖé mɛ",
            "Mǐ ná yi gbɛ̀ ń",
        ]
        ref = calibrate_reference_stats(reliable)
        self.assertGreater(ref.mean_tone_density, 0.5)

        good = score_utterance("Un nɔ́ ɖu nǔ", ref, duration=1.2)
        self.assertTrue(good["accepted"])
        self.assertGreaterEqual(good["quality_score"], 0.9)

        # Missing all tone marks -> rejected for low_tone_density
        no_tones = score_utterance("un nɔ ɖu nu", ref, duration=1.2)
        self.assertFalse(no_tones["accepted"])
        self.assertEqual(no_tones["reason"], "low_tone_density")

        # Implausible speaking rate (4 syllables in 10 seconds -> 0.4 syl/s)
        too_slow = score_utterance("Un nɔ́ ɖu nǔ", ref, duration=10.0)
        self.assertFalse(too_slow["accepted"])
        self.assertEqual(too_slow["reason"], "implausible_speaking_rate")

        # End-to-end CSV filtering
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rel_csv = tmp_dir / "rel.csv"
            unc_csv = tmp_dir / "unc.csv"
            out_csv = tmp_dir / "out.csv"

            with open(rel_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "transcription", "recommended_validation"])
                for idx, txt in enumerate(reliable):
                    w.writerow([f"r{idx}", txt, "0"])

            with open(unc_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "audio_path", "transcription", "duration"])
                w.writerow(["u1", "a.wav", "Un nɔ́ ɖu nǔ", "1.2"])
                w.writerow(["u2", "b.wav", "un nɔ ɖu nu", "1.2"])

            summary = filter_uncertain_csv(rel_csv, unc_csv, out_csv, min_score=0.7)
            self.assertEqual(summary["total_rows"], 2)
            self.assertEqual(summary["kept_rows"], 1)
            with open(out_csv, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "u1")
            self.assertIn("target", rows[0])
            self.assertIn("quality_score", rows[0])


if __name__ == "__main__":
    unittest.main()
