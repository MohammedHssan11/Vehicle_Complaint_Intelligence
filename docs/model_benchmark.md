# Model Benchmark — Baseline vs. DistilBERT vs. DeBERTa-v3-base (Phase 10)

## Headline result

All three models trained on the full 188,056-row training split and evaluated once on the same 40,298-row held-out test set. **DeBERTa-v3-base wins on accuracy; the linear baseline wins on latency and explainability by a wide margin.**

| Metric | Baseline (TF-IDF + Linear SVC) | DistilBERT (fine-tuned) | DeBERTa-v3-base (fine-tuned) |
|---|---|---|---|
| Training data size | 188,056 rows (full) | 188,056 rows (full) | 188,056 rows (full) |
| Hardware | CPU | GPU (RTX 4050 Laptop, 6GB VRAM) | GPU (RTX 4050 Laptop, 6GB VRAM) |
| Training time | 228.2s (~3.8 min) | 3,625.7s (~60.4 min), 2 epochs | 27,493s (~7.6 hr), 4 epochs |
| Test accuracy | 0.641 | 0.674 | **0.679** |
| Test macro-F1 | 0.625 | 0.667 | **0.678** |
| Test weighted-F1 | 0.636 | 0.668 | **0.673** |
| Lenient accuracy (see below) | 0.731 | 0.766 | **0.770** |
| Inference latency (single sample) | **0.022 ms** (CPU) | 18.7 ms (GPU) | 23.6 ms (GPU) |
| Model size on disk | ~21.4 MB | 267.9 MB (66.97M params) | 737.8 MB (184.4M params) |
| Explainability | Full (SHAP linear + top terms) | Not implemented (see below) | Not implemented (see below) |

DeBERTa-v3-base (184M params, ~2.8x DistilBERT) edges out DistilBERT by +1.6% relative macro-F1 (0.667 → 0.678) — a real but modest gain for 2.8x the parameters and ~7.5x the training time. Both transformers substantially beat the linear baseline (+6.7% and +8.5% relative macro-F1 respectively) once given the same full data volume the linear model always had.

## Getting DeBERTa-v3-base to actually train correctly took real debugging

Two genuine bugs surfaced getting a bigger transformer to train reliably — both are now fixed in `src/training/train_transformer.py` and worth knowing about before running your own experiments here:

1. **Class-weighted loss destabilizes DeBERTa-v3's training (opt-in, off by default).** Matching the baseline's `class_weight="balanced"` sounds like a free win given the label imbalance (majority-class baseline accuracy is only 0.16 across 28 classes), but raw inverse-frequency weights span a ~160x ratio here, and that's wide enough to make DeBERTa-v3-base's loss get stuck near `ln(28)≈3.33` — i.e. it never learns anything — while the exact same setup with plain unweighted cross-entropy learns fine. Confirmed via a careful A/B at small scale (not a fluke): unweighted loss reached 0.348 macro-F1 in 2 epochs on an 8K-row subsample; raw-weighted loss stayed at ~0.01-0.07 macro-F1 under multiple learning-rate and gradient-clipping settings. `sqrt`-compressing the weights recovers some but not all of the gap. Net result: `TRANSFORMER_USE_WEIGHTED_LOSS` defaults to off; enabling it is a documented, deliberate tradeoff, not a default.
2. **Missing LR warmup causes a full-scale collapse to the majority class.** Independent of weighting, running the *unweighted* full 188K-row/4-epoch training with the HF `Trainer` defaults (`learning_rate=5e-5`, no warmup) collapsed after epoch 1 to predicting only the most common class — `eval_accuracy` landed exactly on the 0.1598 majority-class baseline, and `eval_macro_f1` was 0.007. This is a documented DeBERTa-v2/v3-specific instability (its own fine-tuning examples always include warmup) that a quick small-scale smoke test didn't catch (250-500 steps wasn't long enough to reach the collapse point). Fix: `warmup_ratio=0.1` and a lower `learning_rate=2e-5` (both now defaults) — verified stable across 8 eval checkpoints on an 8K-row/2-epoch run before committing to the full 7.6-hour run.

**Takeaway if you're extending this**: always validate a new base model or loss function on a moderate subsample (several thousand rows, multiple eval checkpoints across at least a full epoch's worth of steps) before committing hours to a full run — a 250-step smoke test caught neither of these bugs.

## Reading the latency gap

The baseline is ~1000x+ faster per single-sample prediction than either transformer, even on GPU. This is structural, not a tuning artifact: the linear model is one sparse dot product per class; a transformer runs a full forward pass per request, and single-sample (batch=1) inference doesn't amortize GPU overhead the way batched serving would. For high-QPS, latency-sensitive serving, the linear baseline remains the practical choice; a transformer is the choice when the accuracy ceiling matters more than p50 latency.

## Honest evaluation: how much of "error" is really the label taxonomy, not the model?

`components` is a genuinely multi-label field (e.g. `"POWER TRAIN,ENGINE"`) collapsed to the first-listed tag only (`POWER TRAIN` in that example) for this task's training and scoring. Every time a complaint really is about both and the model predicts the non-primary one, standard accuracy scores that as wrong even though it isn't really wrong. `lenient_accuracy` (`src/evaluation/metrics.py`) quantifies this: it credits a prediction if it matches *any* listed component, not just the first one.

| Backend | Strict accuracy | Lenient accuracy | Gap attributable to label collapse |
|---|---|---|---|
| Baseline | 0.641 | 0.731 | +9.0pp |
| DistilBERT | 0.674 | 0.766 | +9.1pp |
| DeBERTa-v3-base | 0.679 | 0.770 | +9.1pp |

The gap is ~9 points **for every backend**, model-independent — strong evidence this is a property of the label collapse itself, not something a bigger or better-tuned model fixes. A deeper break-down on DeBERTa-v3-base's errors specifically:

| Effect | Share of test set | Cumulative accuracy |
|---|---|---|
| Strict accuracy | — | 67.9% |
| + secondary-label matches (above) | 9.2% | 77.1% |
| + `UNKNOWN OR OTHER` as the true label (a data-entry catch-all with no real distinguishing content) | 6.2% | 83.3% |
| + confusion within the ENGINE / POWER TRAIN / FUEL SYSTEM family (genuinely overlapping categories) | 3.0% | 86.4% |

Practical takeaway: if the goal is a materially higher accuracy number, the highest-leverage lever is **not** a bigger model — it's reformulating the task as true multi-label prediction (score against the full component set) and deciding how to handle `UNKNOWN OR OTHER` (arguably shouldn't be a scored target class at all, since it has no distinguishing textual signal by construction). The genuine "model could do better" headroom, isolated from those effects, is closer to 3 points than 20.

## Which one is `models/production/`?

Still the baseline, for now — switching the default `SERVING_MODEL_BACKEND` to `transformer` trades a real accuracy gain for a >1000x latency increase and the complete loss of word-level explainability (SHAP explanations only exist for the linear baseline; see below). That's a product decision, not a modeling one, so it's left as a config choice (`SERVING_MODEL_BACKEND=transformer` in `.env`, pointed at whichever transformer checkpoint is in `models/transformer/`) rather than silently switched.

## Baseline bake-off: did XGBoost or char n-grams help?

`src/training/train.py` bakes off 2 featurizers × 3 models (6 combinations) on every run, all on the full 188,056-row training split, selected by validation macro-F1:

| Featurizer | Model | Val macro-F1 | Val accuracy |
|---|---|---|---|
| word TF-IDF (30K features) | Logistic Regression | 0.5838 | 0.6186 |
| word TF-IDF (30K features) | **Linear SVC** | **0.6190** | 0.6454 |
| word TF-IDF (30K features) | XGBoost (80 trees, depth 5) | 0.5835 | 0.6130 |
| word+char TF-IDF (50K features) | Logistic Regression | 0.5850 | 0.6182 |
| word+char TF-IDF (50K features) | Linear SVC | 0.6174 | 0.6467 |
| word+char TF-IDF (50K features) | XGBoost (80 trees, depth 5) | 0.5939 | 0.6196 |

**The original word-TF-IDF + Linear SVC combination remains the winner.** Two hypotheses from the original text/data audit didn't pan out in practice:

- **XGBoost doesn't beat a well-regularized linear model here.** This is a known pattern for sparse, extremely high-dimensional bag-of-words features — a linear separator with `class_weight="balanced"` already exploits the sparsity well, and tree splits don't add much on top. (Getting XGBoost to even train correctly took real debugging — its GPU `hist` path silently produced a near-random model, 0.042 val macro-F1 vs. CPU's 0.52 on a diagnostic subsample, apparently mishandling the scipy sparse CSR input. `src/models/xgboost_model.py` documents this and pins XGBoost to CPU deliberately.)
- **Word+char n-grams don't help**, despite the text audit finding ~46% of the vocabulary is hapax legomena (much of it typos). The char n-gram features may be adding noise/dimensionality without enough signal to offset it, or the word-level model was already robust enough to common misspelling patterns via other means (stopword/lemmatization normalization, or just enough training data that typo variants still get seen).

Both are legitimate negative results, not bugs — trying them and keeping the simpler winning model is the correct call here (Occam's razor over a modeling detour).

## Explainability gap for the transformer backends

`src/inference/predictor.py`'s `TransformerPredictor` returns `explanation: null` regardless of which transformer checkpoint is loaded. The SHAP/coefficient approach in `src/explainability/explain.py` is exact and cheap specifically because both baseline models are linear over a fixed TF-IDF vocabulary — it does not extend to a transformer's contextual embeddings. Adding transformer explainability (e.g. integrated gradients via `captum`, or attention-weight visualization) is a documented future extension, not implemented here.

## History: the original CPU subsample benchmark

The first version of this benchmark trained DistilBERT on a 4,000-row stratified subsample (2.1% of training data) because this development machine originally had CPU-only PyTorch, and full fine-tuning over 188K rows was estimated at ~8 hours/epoch on CPU. That run scored 0.364 macro-F1 — worse than the baseline — but the gap was **not** a fair transformer-vs-linear-model comparison; it was a training-data confound (188,056 rows vs. 4,000). Installing CUDA-enabled PyTorch and using the machine's RTX 4050 GPU (previously unused) cut full-data DistilBERT training time to ~60 minutes and produced the honest DistilBERT-vs-baseline comparison — DeBERTa-v3-base was trained afterward on the same GPU, as a further "how much does a bigger base model help" experiment (see above).

`src/training/train_transformer.py` is fully config-driven via environment variables (`TRANSFORMER_BASE_MODEL`, `TRANSFORMER_TRAIN_SUBSAMPLE_SIZE=0` for full data, etc. — see `docs/Training.md`), so reproducing any of these runs, or trying yet another base model, is a settings change, not a code edit.
