# HOPE-tensorflow

A TensorFlow / Keras implementation of **HOPE** from *Nested Learning: The Illusion of Deep Learning Architectures* (Behrouz, Razaviyayn, Zhong, Mirrokni; Google Research; NeurIPS 2025).

> Status: **Phase 0 — scaffolding.** Components below are stubs and will be filled in phase by phase. See [`PLAN`](#roadmap) for the order.

> Note: the local working directory is named `hope-architecture/`, the published repository and the importable package are both named `hope-tensorflow` / `hope`.

---

## Why this repo

The HOPE architecture is published with the *Nested Learning* paradigm — a recurrent, self-modifying model with a Continuum Memory System (CMS) that updates memory at multiple frequencies. PyTorch reimplementations already exist; this repo fills the TF / Keras gap and doubles as a study log.

Goals:
- Translate the paper's core components into readable TF / Keras code, with paper equation numbers wired into docstrings.
- Show, on Split-MNIST, that HOPE retains more accuracy across tasks than a vanilla baseline (continual learning demo).
- Walk through each component in a dedicated notebook.

Non-goals: LLM-scale training, distributed training, custom CUDA, TFLite / TF Serving, exact parity with the official PyTorch numbers.

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

model = HOPE(...)
```

Actual usable API lands at Phase 3.

---

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Repo scaffolding, paper downloader, first push | in progress |
| 1 | `AssociativeMemory`, `SelfModifyingLayer` + tests | todo |
| 2 | `ContinuumMemorySystem` + visualization notebooks | todo |
| 3 | `HopeAttention`, full `HOPE` model | todo |
| 4 | `DGD` / `DeepOptimizer`, `scripts/train.py`, Split-MNIST loader | todo |
| 5 | Catastrophic-forgetting demo (vanilla vs HOPE) | todo |
| 6 | Documentation polish, notebook cleanup | todo |

---

## Paper

Behrouz, A., Razaviyayn, M., Zhong, P., Mirrokni, V.
*Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS 2025.
[arXiv:2512.24695](https://arxiv.org/abs/2512.24695)

The PDF is not committed. Fetch it locally:

```bash
bash scripts/download_paper.sh
```

See [`papers/README.md`](papers/README.md) for details.

---

## License

MIT. See [LICENSE](LICENSE).
