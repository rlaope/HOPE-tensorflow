# papers/

This directory holds reference papers used by `hope-tensorflow`. PDFs are **not committed** (see `.gitignore`).

## Primary reference

Behrouz, A., Razaviyayn, M., Zhong, P., Mirrokni, V.
*Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS 2025.
arXiv: [2512.24695](https://arxiv.org/abs/2512.24695)

## How to fetch

From the repo root:

```bash
bash scripts/download_paper.sh
```

The script drops `nested_learning_2512.24695.pdf` into this folder. Re-running is a no-op if the file already exists.

## Implementation rule

Every component in `hope/` cites the corresponding paper equation or section number in its docstring. If a piece of code references the paper, the PDF in this folder is the source of truth — not blog summaries, not the plan document, not memory.
