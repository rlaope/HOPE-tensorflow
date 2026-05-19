# Changelog

## 0.1.0 — first runnable cut

All six implementation phases are merged.

### Phase 0 — scaffolding
- Package layout (`hope/`, `scripts/`, `tests/`, `notebooks/`, `papers/`, `assets/`).
- `pyproject.toml`, MIT `LICENSE`, `.gitignore`, README skeleton, `papers/README.md`.
- `scripts/download_paper.sh` fetches arXiv 2512.24695 into `papers/`.

### Phase 1 — associative memory primitives
- `hope/memory.AssociativeMemory` with `hebbian` (Eq. 18), `delta` (Eq. 93 gradient), and `oja` (Eq. 88 with `α = 1`) update rules.
- `hope/layers.SelfModifyingLayer` with a per-token Hebbian fast weight (Eq. 18 inside §8.1's self-referential Titans).
- Unit tests for outer-product correctness, delta-rule error decay, and Oja-style norm boundedness.

### Phase 2 — Continuum Memory System
- `hope/memory.ContinuumMemorySystem` and `CMSBank` implementing §7.1 / Eq. 70-71 with per-bank update frequencies and decays.
- Notebooks 02 and 03 visualise memory norms over time and CMS bank-frequency behaviour.

### Phase 3 — HOPE model assembly
- `hope/layers.HopeAttention` causal multi-head softmax block (§8 "Hope-Attention").
- `hope/model.HOPE` Keras Model: embedding → SelfModifyingLayer → CMS → HopeAttention → LayerNorm → LM head, matching Figure 5 / Eq. 94-97.
- Notebooks 04 (self-modifying fast weight) and 05 (full HOPE + tiny training loop).
- Tests for output shape, copy-task loss curve, and CPU path.

### Phase 4 — baseline + training infra
- `hope/baseline.MiniTransformer` nanoGPT-class decoder-only transformer with `matched_to(hope_model, tolerance=0.05)` analytical sweep over `(mlp_ratio, n_layers, d_model)`.
- `hope/optimizers.DGD` (§4.5, Eq. 31) and `DeepOptimizer` (§4.2, Eq. 33-34) as plain Python objects.
- `scripts/download_data.py` for TinyShakespeare + char-level `tf.data` pipeline.
- `scripts/train.py` CLI driving both HOPE and MiniTransformer through the same Adam loop.

### Phase 5 — three-scenario benchmark
- `scripts/benchmark.py` with `--scenario {longctx, continual, incontext, all}`.
- Scenario 1 (long-context retrieval): plant a `(key, value)` pair at the start, recall query near the end.
- Scenario 2 (continual LM): train on TinyShakespeare, then on random alphabet sequences, then re-measure A's cross-entropy.
- Scenario 3 (in-context adaptation): few-shot character substitution inside the prompt.
- Outputs `assets/bench_longctx.png`, `assets/bench_continual.png`, `assets/bench_incontext.png` and is exercised by notebooks 06 and 07.

### Phase 6 — documentation polish
- Notebook 01: paper overview + repo concept map.
- README rewritten with install snippet, working quickstart, inline benchmark gallery, notebook index, paper BibTeX, hardware paragraph.
- Version bumped to 0.1.0.
- This `CHANGELOG.md`.
