"""Autograder tests for Drill 6A — Text Processing & NLP Basics."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drill import preprocess_text, extract_linguistic_annotations, extract_entities


SAMPLE_TEXT = (
    "The IPCC released its latest report in Geneva on March 20, 2024. "
    "Dr. Ahmad presented findings on Jordan's climate adaptation strategy "
    "at the COP28 conference in Dubai."
)

STOP_WORDS = {"the", "a", "an", "in", "on", "at", "its", "of", "and", "is"}


def test_preprocess_text():
    """Verify preprocessing: lowercase, no punctuation, stop words removed."""
    result = preprocess_text(SAMPLE_TEXT, STOP_WORDS)
    assert result is not None, "preprocess_text returned None"
    assert isinstance(result, list), "preprocess_text must return a list"
    assert len(result) > 0, "preprocess_text returned empty list"

    # All tokens should be lowercase
    for token in result:
        assert token == token.lower(), f"Token '{token}' is not lowercase"

    # No punctuation-only tokens
    for token in result:
        assert not all(c in ".,;:!?()-'\"" for c in token), (
            f"Punctuation token '{token}' should be removed"
        )

    # Stop words should be removed
    for sw in ["the", "in", "on", "at", "its"]:
        assert sw not in result, f"Stop word '{sw}' should be removed"

    # Key content words should remain
    lower_result = [t.lower() for t in result]
    assert "ipcc" in lower_result or "report" in lower_result, (
        "Expected content words like 'ipcc' or 'report' to remain"
    )


def test_linguistic_annotations():
    """Verify (token, POS, dep) tuples from spaCy."""
    result = extract_linguistic_annotations(SAMPLE_TEXT)
    assert result is not None, "extract_linguistic_annotations returned None"
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "Returned empty list"

    # Each item should be a tuple/list of 3 elements
    for item in result:
        assert len(item) == 3, f"Expected 3 elements per annotation, got {len(item)}"

    # Check that POS tags are valid spaCy universal POS tags
    valid_pos = {
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
        "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM",
        "VERB", "X", "SPACE",
    }
    token_texts, pos_tags, dep_labels = zip(*result)
    for pos in pos_tags:
        assert pos in valid_pos, f"Unexpected POS tag: '{pos}'"


def test_extract_entities():
    """Verify NER extraction returns correct (text, label) tuples."""
    result = extract_entities(SAMPLE_TEXT)
    assert result is not None, "extract_entities returned None"
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "No entities extracted from text with known entities"

    # Each item should be a tuple/list of 2 elements
    for item in result:
        assert len(item) == 2, f"Expected 2 elements per entity, got {len(item)}"

    entity_texts = [ent[0] for ent in result]
    entity_labels = [ent[1] for ent in result]

    # At least one recognized entity type should be present
    known_labels = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "LOC",
                    "NORP", "EVENT", "FAC", "PRODUCT", "WORK_OF_ART",
                    "LAW", "LANGUAGE", "PERCENT", "TIME", "QUANTITY",
                    "ORDINAL", "CARDINAL"}
    for label in entity_labels:
        assert label in known_labels, f"Unexpected entity label: '{label}'"

    # The text contains obvious entities — at least one should be found
    all_entity_text = " ".join(entity_texts).lower()
    assert any(term in all_entity_text for term in ["ipcc", "geneva", "jordan", "dubai", "march"]), (
        f"Expected at least one known entity. Found: {entity_texts}"
    )


def test_preprocess_handles_unicode():
    """Verify preprocess_text handles non-ASCII characters without crashing."""
    unicode_text = "The café in Zürich hosted a résumé workshop on AI."
    stop_words = {"the", "a", "in", "on"}
    result = preprocess_text(unicode_text, stop_words)
    assert result is not None, "preprocess_text returned None on Unicode input"
    assert isinstance(result, list), "Must return a list"
    assert len(result) > 0, "Returned empty list for Unicode text"

    # Should not crash and should produce valid tokens
    for token in result:
        assert isinstance(token, str), f"Token must be a string, got {type(token)}"
        assert len(token) > 0, "Empty string token"
