"""NLP and tonal analysis utilities for the BENIN Fongbe Tonal ASR Challenge (Persona 3)."""

from src.nlp.audit_text import audit_transcriptions, format_audit_report
from src.nlp.error_analysis import analyze_errors, format_error_report
from src.nlp.metrics import align_sequences, syllable_wer
from src.nlp.normalize import (
    is_already_syllabified,
    safe_syllabify,
    split_syllable,
    strip_tones,
    validate_submission_target,
)

__all__ = [
    "align_sequences",
    "analyze_errors",
    "audit_transcriptions",
    "format_audit_report",
    "format_error_report",
    "is_already_syllabified",
    "safe_syllabify",
    "split_syllable",
    "strip_tones",
    "syllable_wer",
    "validate_submission_target",
]
