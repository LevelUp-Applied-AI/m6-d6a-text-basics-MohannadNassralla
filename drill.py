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

  
    doc = nlp(text)
    
    cleaned_tokens = []
    
    for token in doc:
      
        if not token.is_punct and not token.is_space:
      
            lowered = token.text.lower()
        
            if lowered not in stop_words:
                cleaned_tokens.append(lowered)
                
    return cleaned_tokens


def extract_linguistic_annotations(text):
   
    doc = nlp(text)
    annotations = []
    
    for token in doc:
   
        annotations.append((token.text, token.pos_, token.dep_))
        
    return annotations


def extract_entities(text):

    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
       
        entities.append((ent.text, ent.label_))
        
    return entities


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