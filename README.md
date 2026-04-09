# Core Skills Drill 6A — Text Processing & NLP Basics

Module 6 Week A drill for AI.SPIRE Applied AI & ML Systems.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

This installs spaCy and Hugging Face transformers for the first time in this course, and downloads the spaCy English language model.

## Tasks

Complete the three functions in `drill.py`:
1. `preprocess_text` — Lowercase, remove punctuation, filter stop words
2. `extract_linguistic_annotations` — Extract (token, POS, dependency) tuples with spaCy
3. `extract_entities` — Extract named entities using spaCy NER

## Submission

1. Create a branch: `drill-6a-text-basics`
2. Complete `drill.py`
3. Open a PR to `main`
4. Paste your PR URL into TalentLMS → Module 6 Week A → Drill 6A

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
