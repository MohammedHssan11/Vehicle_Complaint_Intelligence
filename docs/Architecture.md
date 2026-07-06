# Architecture

## Why this exists

The original project was a single Colab notebook operating on what its own markdown described as a small Kaggle sample. The actual local `complaints.csv` turned out to be the full 271,517-row NHTSA export (2016-2026 model years) — a very different scale problem. A full audit (data quality, leakage, text profiling, notebook review) found several issues that shaped this redesign; the most consequential is the label formulation below.

## The label formulation decision

`components` is a comma-joined, potentially multi-label field (`"POWER TRAIN,ENGINE"`). The original notebook kept raw compound strings as their own classes (2,341 raw / 21 arbitrarily grouped), producing near-duplicate classes that share vocabulary with their parent single-tag class — its own SVM baseline scored **F1=0.00** on several of these compound classes, proving the approach doesn't work.

This system classifies on the **primary (first-listed) component**, collapsed to a curated taxonomy:

- Splitting on comma gives 45 raw primary classes.
- `LabelConfig.MIN_SAMPLES_PER_CLASS = 200` (see `src/config/settings.py`) keeps 27 classes with enough support for a stable macro-F1 estimate, folding the remaining ~583 rows (mostly noisy/duplicate tags like `"NONE"`, `"COMMUNICATIONS"` vs `"COMMUNICATION"`) into `OTHER`.
- Multi-label classification (predicting *all* components) is a documented, straightforward v2 upgrade — swap `LabelEncoder` for `MultiLabelBinarizer` and the softmax head for sigmoid — not implemented here because it adds real complexity (per-label thresholds, harder-to-explain UI) that wasn't worth taking on for v1.

## Data flow

```
data/raw/complaints.csv (271,517 rows)
  -> src/data/loader.py       validate schema, parse dtypes (fast C-engine CSV read)
  -> src/data/cleaning.py     drop null/near-empty summaries, drop vin/products,
                              dedupe on summary TEXT ALONE (not summary+label —
                              see "Leakage guards" below)
  -> src/data/labels.py       extract primary component, collapse rare tags to OTHER
  -> src/preprocessing/       Phase 8 text cleaning (see "Preprocessing" below)
  -> src/data/splits.py       stratified 70/15/15 + a separate time-based holdout
  -> data/processed/*.parquet
```

Then, for training:

```
data/processed/{train,val,test}.parquet
  -> src/features/tfidf.py    fit TF-IDF on train only
  -> src/models/               LogisticRegression / LinearSVC (class_weight="balanced")
  -> src/training/train.py    trains both, selects winner by val macro-F1,
                              evaluates ONCE on test, logs to MLflow
  -> src/evaluation/           metrics, confusion matrix, error analysis
  -> models/production/        promoted winner: model.joblib, vectorizer.joblib, metadata.json
```

And for serving:

```
Request -> src/api (FastAPI) or app/streamlit_app (Streamlit)
  -> src/services/classification_service.py   shared logic, avoids duplicating
                                               prediction+logging between the two front doors
  -> src/inference/predictor.py               PredictorService: hides baseline vs.
                                               transformer backend differences
  -> src/explainability/                      SHAP linear explainer + confidence + top-3
  -> src/monitoring/prediction_log.py          counts, latency, class distribution
  -> Response
```

## Leakage guards

| Risk | Guard |
|---|---|
| Duplicate complaint text landing in both train and test | `cleaning.py` dedupes on `summary` text alone — a real-data check found identical text under *different* label codes, so keying on `(summary, components)` wasn't strict enough |
| The original notebook's dead upsampling code (`resample(replace=True)` before any split) | Removed entirely; class imbalance is handled via `class_weight="balanced"`, which reweights the loss instead of duplicating rows |
| Silent train/test contamination | `src/data/splits.py::assert_no_text_leakage` raises if any text appears in more than one split — called after every split in `src/data/pipeline.py` |
| Naive future joins against `recalls.csv`/`investigations.csv` | **Not currently used for modeling** — if a future feature engineering pass joins these tables (e.g. "has this make/model had a recall for this component"), the join must filter on `ReportReceivedDate`/`ODATE` strictly before `dateComplaintFiled`, or the model partially learns from information that didn't exist yet (recalls are often opened *because of* clustered complaints in that exact component) |

## Preprocessing

`src/preprocessing/text_cleaning.py` has two entry points because the two model families need different treatment:

- **`preprocess_for_classical`** (TF-IDF path): unicode NFKC normalize -> strip NHTSA transcriber codes (`TL*`, `*TR`, generic `*XX`) -> **expand contractions before stripping punctuation** -> lowercase -> strip non-alpha -> remove stopwords *except negation words* (`not`, `no`, `never`, ...) -> lemmatize.
- **`preprocess_for_transformer`** (DistilBERT path): unicode normalize -> strip transcriber codes -> expand contractions. That's it — DistilBERT's subword tokenizer handles casing/morphology, and stopword removal/lemmatization would break the sentence structure it relies on.

The contraction-expansion-before-punctuation-stripping order fixes a real bug in the original notebook: it stripped punctuation first, so `"doesn't"` collapsed to `"doesnt"` — a token matching neither the stopword list nor any negation rule. Negation is exactly the signal that distinguishes "brake did **not** engage" from "brake engaged."

## Folder rationale

| Folder | Why it exists |
|---|---|
| `data/raw` vs `data/interim` vs `data/processed` | Raw = untouched source CSVs. Interim is reserved for future partially-cleaned intermediates (not currently populated — cleaning is fast enough to run inline). Processed = the actual train/val/test parquet the models train on. |
| `models/production/` vs `artifacts/` | `models/production/` holds only the **currently promoted, servable** model (small, curated — vectorizer + model + metadata). `artifacts/` holds the full history of experiment output (classification reports, confusion matrices, per-run metrics) — every training run gets a timestamped subfolder here, but only the winner gets promoted to `models/production/`. |
| `src/services/` | Both `src/api` (FastAPI) and `app/streamlit_app` need the same predict-then-log behavior. Without this layer, that logic would live twice. |
| `src/inference/` vs `src/models/` | `src/models` defines what a model backend *is* (fit/predict/predict_proba). `src/inference/predictor.py` is the actual swappable abstraction — it hides the fact that the baseline backend takes a pre-vectorized TF-IDF matrix while the transformer backend tokenizes raw text internally, so callers only ever see `predict(text) -> PredictionResult`. |
| `experiments/` | MLflow's local SQLite tracking store (`mlflow.db`) — MLflow 3.x deprecated the plain filesystem backend. |

## Known scope limits (by design, not oversight)

- Auxiliary tables (`investigations.csv`, `recalls.csv`, `ratings.csv`, `car_models.csv`) are loaded and audited but not used for modeling — see the leakage guard above for why joining them safely requires a temporal cutoff that isn't implemented yet.
- Transformer explainability is not implemented — SHAP's `LinearExplainer` is exact and cheap for the linear baseline over a fixed TF-IDF vocabulary; it does not extend to DistilBERT's contextual embeddings. A documented future extension (e.g. `captum` integrated gradients), not built here.
- Multi-label classification (see above) is a documented v2 upgrade path, not v1 scope.
