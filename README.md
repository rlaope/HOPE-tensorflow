# HOPE-tensorflow

A tensorflow implementation of *"Nested Learning: The Illusion of Deep Learning Architectures"* (Behrouz, Razaviyayn, Zhong, Mirrokni; Google Research; NeurIPS 2025).

> Status: **Phase 0 — scaffolding.** Components below are stubs and will be filled in phase by phase.

> Local working directory is `hope-architecture/`; published repo and importable package are `HOPE-tensorflow` / `hope`.

---

## Purpose

HOPE pairs the *Nested Learning* paradigm with a recurrent backbone: a self-modifying layer plus a **Continuum Memory System (CMS)** that updates memory banks at multiple frequencies. PyTorch reimplementations exist; this repo fills the TF / Keras gap and doubles as a study log.

- **Character**: learning / portfolio / shared-study
- **Differentiator**: HOPE has several PyTorch reimplementations; no TF / Keras implementation exists
- **Audience**: ML learners studying HOPE, TF / Keras users, the Korean ML community

### Goals

1. Translate the paper's core components (`AssociativeMemory`, `ContinuumMemorySystem`, `SelfModifyingLayer`, `HopeAttention`, `DGD` / `DeepOptimizer`) into readable TF / Keras code with paper equation numbers in every docstring.
2. Show, visually, that HOPE reduces catastrophic forgetting compared to a parameter-matched vanilla model in a continual-learning setting.
3. Provide one walkthrough notebook per component.
4. Every piece of code maps 1:1 to a paper section / equation in its docstring.

### Non-goals

- LLM-scale training. **nanoGPT scale (a few million to tens of millions of parameters) is the ceiling.**
- Multi-GPU / distributed training / custom CUDA / XLA optimization
- TFLite / TF Serving deployment
- 1:1 accuracy parity with the official PyTorch implementation
- Reproducing the paper's benchmark numbers (we only show direction)

---

## Tech Stack

- Python ≥ 3.10
- TensorFlow ≥ 2.15 (Keras 3 compatible)
- numpy, matplotlib, tqdm
- pytest (tests)
- jupyter (notebooks)

**Hardware**: single GPU. Minimum Colab T4 or 8 GB+ local VRAM recommended. Must remain runnable on CPU with a small config (smoke test).

No external ML libraries (transformers, flax, pytorch, ...). Pure TF / Keras.

---

## Baseline & Benchmarks

HOPE is a **sequence / language model architecture**, so the comparison stays in the LM domain.

### Baseline
- **Mini-Transformer** (nanoGPT style), **same parameter count** as the HOPE model
- Same dataset, same training schedule

### Datasets
- **TinyShakespeare** — char-level, lightest, used for fast iteration
- **WikiText-2** — token-level, general LM signal
- **Synthetic key-value retrieval** — random (key, value) pairs sprinkled across a long sequence; recall is queried at the end (long-context probe)

### Scenarios
1. **Long-context retrieval** — recall information seen far earlier in the sequence. The CMS-defining test.
2. **Continual LM (domain transfer)** — train on domain A, then on domain B, then measure A again. Transformer is expected to forget; HOPE is expected to retain.
3. **In-context adaptation** — how fast does the model pick up a new pattern shown only inside the prompt. Self-modifying-layer signal.

### Metrics
- Perplexity (per-domain and overall)
- Long-context recall accuracy (key-value retrieval)
- Forgetting curve (previous-domain perplexity over time)

---

## Install (planned)

```bash
git clone https://github.com/rlaope/HOPE-tensorflow.git
cd HOPE-tensorflow
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart (planned)

```python
from hope.model import HOPE
from hope.baseline import MiniTransformer

hope = HOPE(...)
baseline = MiniTransformer.matched_to(hope)   # same parameter budget
```

Actual usable API lands at Phase 3.

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Repo scaffolding, paper downloader, first push, spec sync (text-domain pivot) | done |
| 1 | `AssociativeMemory`, `SelfModifyingLayer` + unit tests | todo |
| 2 | `ContinuumMemorySystem` + visualization notebooks | todo |
| 3 | `HOPE` model assembly (LM head included) | todo |
| 4 | `MiniTransformer` baseline + training infra (`DGD`, `scripts/train.py`, text loaders) | todo |
| 5 | Benchmark experiments (long-context retrieval / continual LM / in-context adaptation) + result plots | todo |
| 6 | Documentation polish, notebook cleanup | todo |

---

## Paper

Behrouz, A., Razaviyayn, M., Zhong, P., Mirrokni, V.
*Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS 2025.
[arXiv:2512.24695](https://arxiv.org/abs/2512.24695)

Blog: <https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/>

The PDF is not committed. Fetch it locally:

```bash
bash scripts/download_paper.sh
```

See [`papers/README.md`](papers/README.md).

---

## License

MIT. See [LICENSE](LICENSE).
