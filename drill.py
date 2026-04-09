"""
Module 6 Week A — Core Skills Drill: Text Processing & NLP Basics

Complete the three functions below. Each function has a docstring
describing its inputs, outputs, and purpose.

Run your work: python drill.py
Test your work: the autograder runs automatically when you open a PR.
"""

import spacy

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text, stop_words):
    """Preprocess a raw text string for NLP analysis.

    Steps: lowercase, remove punctuation, tokenize into words,
    remove stop words from the provided list, return cleaned tokens.

    Args:
        text: Raw text string (may contain punctuation, mixed case,
              and non-ASCII characters).
        stop_words: A set of lowercase stop words to remove.

    Returns:
        List of cleaned, lowercase token strings with punctuation
        and stop words removed.
    """
    # TODO: Lowercase the text, tokenize with spaCy, filter out
    #       punctuation tokens and stop words
    pass


def extract_linguistic_annotations(text):
    """Extract linguistic annotations for each token in the text.

    Args:
        text: A text string to analyze.

    Returns:
        List of (token_text, pos_tag, dep_label) tuples — one per
        token. Use spaCy's .text, .pos_, and .dep_ attributes.
    """
    # TODO: Process text with spaCy and build annotation tuples
    pass


def extract_entities(text):
    """Extract named entities from text using spaCy NER.

    Args:
        text: A text string to analyze.

    Returns:
        List of (entity_text, entity_label) tuples sorted by
        appearance order. Use doc.ents and each entity's .text
        and .label_ attributes.
    """
    # TODO: Process text with spaCy and extract entity tuples
    pass


if __name__ == "__main__":
    sample = (
        "The IPCC released its latest report in Geneva on March 20, 2024. "
        "Dr. Ahmad presented findings on Jordan's climate adaptation strategy "
        "at the COP28 conference in Dubai."
    )
    stop_words = {"the", "a", "an", "in", "on", "at", "its", "of", "and", "is"}

    # Task 1: Preprocess
    tokens = preprocess_text(sample, stop_words)
    if tokens is not None:
        print(f"Cleaned tokens ({len(tokens)}): {tokens[:10]}...")

    # Task 2: Linguistic annotations
    annotations = extract_linguistic_annotations(sample)
    if annotations is not None:
        print(f"\nAnnotations ({len(annotations)} tokens):")
        for tok, pos, dep in annotations[:5]:
            print(f"  {tok:15s} {pos:8s} {dep}")

    # Task 3: Named entities
    entities = extract_entities(sample)
    if entities is not None:
        print(f"\nEntities ({len(entities)}):")
        for text, label in entities:
            print(f"  {text:25s} {label}")
