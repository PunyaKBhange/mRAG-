# mRAG+ — Multilingual Retrieval-Augmented Generation

> Zero-shot multilingual QA pipeline with Named Entity Transliteration Normalization (NETN) and noise robustness evaluation.

---

## Overview

**mRAG+** is a research pipeline for multilingual open-domain question answering. It combines retrieval-augmented generation (RAG) with specialized handling for multilingual queries, transliteration normalization, and character-level noise — all without any task-specific fine-tuning.

The system translates incoming queries from 200+ languages into English using Facebook's NLLB-200 model, retrieves relevant context, generates answers with GPT-2, and then normalizes named entities across transliteration variants using fuzzy string matching.

---

## Features

- **Zero-shot multilingual QA** — no labeled training data required
- **Query reformulation** — NLLB-200 translates queries from 200+ languages to English
- **NETN normalization** — fuzzy Unicode matching resolves transliteration variants (e.g. *Sofia Kovalevskaia* ↔ *Sofya Kovalevskaya*)
- **Noise robustness evaluation** — simulates character deletions, swaps, and substitutions
- **Language-agnostic evaluation** — character 3-gram recall metric works on any script
- **Graceful degradation** — fallback paths when models fail to load

---

## Project Structure

```
mrag_plus_project/
├── project/
│   ├── src/
│   │   ├── utils.py          # Data loading / saving helpers
│   │   ├── netn.py           # Named Entity Transliteration Normalization
│   │   ├── qr_baseline.py    # Query reformulation via NLLB-200
│   │   ├── noise_eval.py     # Character-level noise injection
│   │   ├── evaluate.py       # Char-3gram recall metric
│   │   ├── model.py          # GPT-2 answer generator wrapper
│   │   └── train.py          # Placeholder (zero-shot — no training needed)
│   ├── data/
│   │   └── mkqa_subset.jsonl # Synthetic MKQA-style QA dataset (300 entries)
│   ├── results/
│   │   ├── sample_output.json
│   │   └── last_plot.png
│   └── notebooks/
├── inference_examples.py     # End-to-end demo script
└── README.md
```

---

## Installation

**Requirements:** Python 3.8+, pip

```bash
git clone https://github.com/saanjjjj/mrag_plus_project.git
cd mrag_plus_project

pip install transformers rapidfuzz unidecode datasets sentencepiece \
            accelerate faiss-cpu langdetect python-Levenshtein \
            scikit-learn matplotlib pandas transliterate
```

---

## Quick Start

### Run the end-to-end demo

```python
from project.src.qr_baseline import reformulate_query
from project.src.model import load_generator, generate_answer
from project.src.netn import ne_transliteration_match

query = "Qui est la première femme à obtenir un doctorat en mathématiques ?"

# Step 1 — Reformulate
reformulated = reformulate_query(query, src_lang="fra")
print(reformulated)
# → "Find short factual answer: Who is the first woman to get a doctorate in mathematics?"

# Step 2 — Generate
gen = load_generator("gpt2")
answer = generate_answer(gen, f"Answer briefly: {reformulated}", max_new_tokens=50)

# Step 3 — NETN match against gold
matched, score = ne_transliteration_match(answer, "Sofya Kovalevskaya")
print(f"Match: {matched}, Score: {score}")
```

Or run the bundled script directly:

```bash
python inference_examples.py
```

---

## Pipeline

```
Multilingual query (any of 200+ languages)
        ↓
[Query Reformulation]  →  NLLB-200 translates to English
        ↓                  + wraps in retrieval template
[Retrieval]            →  MKQA dataset / FAISS vector store (planned)
        ↓
[Prompt Construction]  →  question + retrieved context
        ↓
[Answer Generation]    →  GPT-2 (zero-shot)
        ↓
[NETN Normalization]   →  Unicode normalize + fuzzy match (rapidfuzz)
        ↓
[Evaluation]           →  char-3gram recall, confusion matrix, ROC curve
```

---

## Modules

### Query Reformulation (`src/qr_baseline.py`)
Uses `facebook/nllb-200-distilled-600M` to translate multilingual queries to English, then wraps them in a retrieval-friendly template. Falls back gracefully if the model fails to load.

### NETN (`src/netn.py`)
Resolves named entity transliteration variants across scripts. Applies Unicode NFKD normalization, optional Latin transliteration, and `rapidfuzz.fuzz.partial_ratio` with a configurable threshold (default: 80).

### Noise Evaluation (`src/noise_eval.py`)
Injects character-level noise at a configurable probability `p` — random deletions, adjacent swaps, and Unicode substitutions. Also includes diacritic stripping for multilingual robustness testing.

### Evaluation (`src/evaluate.py`)
Computes character 3-gram recall between predicted and gold answers — a language-agnostic metric that handles morphologically rich languages better than token-level overlap.

---

## Evaluation Results (Simulated)

| Language | Baseline (char-3gram %) | + NETN (%) |
|----------|------------------------|------------|
| English  | 70                     | 72         |
| French   | 55                     | 58         |
| Russian  | 36                     | 40         |
| Korean   | 29                     | 32         |

Noise robustness (English baseline):

| Noise level `p` | Performance (%) |
|-----------------|-----------------|
| 0.00            | 90              |
| 0.02            | 82              |
| 0.05            | 75              |
| 0.10            | 60              |

> Note: values above are illustrative. Replace with real evaluation runs against MKQA or XOR-TyDi QA for published results.

---

## Known Limitations & Future Work

- **GPT-2 is weak for factual QA** — replace with a dedicated QA model (e.g. Mistral-7B, Llama-3, or `deepset/roberta-base-squad2`) for meaningful answer quality
- **No real retrieval yet** — FAISS vector store is planned; current context is hardcoded per query
- **Language detection is manual** — `langdetect` is installed but not yet wired into `reformulate_query`; auto-detection would remove the need to pass `src_lang` explicitly
- **Simulated metrics** — evaluation plots in Steps 9A–9D use placeholder values; hook into real inference runs

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | NLLB-200 translation, GPT-2 generation |
| `rapidfuzz` | Fuzzy string matching for NETN |
| `datasets` | HuggingFace dataset loading |
| `faiss-cpu` | Vector similarity search (planned) |
| `langdetect` | Language identification (planned integration) |
| `transliterate` | Script-to-Latin transliteration |
| `scikit-learn` | Confusion matrix, ROC curve |
| `sentencepiece` | Tokenizer for NLLB-200 |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
