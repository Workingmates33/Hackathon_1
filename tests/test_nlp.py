"""Unit tests for the Persona 3 Fongbe NLP & tonal analysis module."""
from __future__ import annotations

import unittest

from src.nlp.audit_text import audit_transcriptions, format_audit_report
from src.nlp.error_analysis import analyze_errors, format_error_report
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


if __name__ == "__main__":
    unittest.main()
