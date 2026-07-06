"""Transformer classifier wrapper — inference-side interface only.

Training happens via the HuggingFace `Trainer` API in
`src/training/train_transformer.py` (a very different workflow from the
sklearn `.fit(X, y)` baseline: tokenized `datasets.Dataset`, batched
forward/backward passes, epoch-based training loop). This class wraps the
*result* of that training so `src/inference` can serve a transformer model
through the same shape of interface (`predict`, `predict_proba`) that the
baseline `BaseClassifier` subclasses expose — the difference is this class
tokenizes raw text internally (`predict(texts: list[str])`), whereas the
baseline models consume a pre-vectorized TF-IDF matrix. That asymmetry is
intentional and is exactly why `src/inference/predictor.py` — not this
class — is the true swappable serving abstraction.

Uses `AutoModelForSequenceClassification`/`AutoTokenizer` rather than
DistilBERT-specific classes so any encoder checkpoint (DistilBERT, BERT,
RoBERTa, DeBERTa-v3, ...) works without code changes — only
`TRANSFORMER_BASE_MODEL` (see `src/training/train_transformer.py`) changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MAX_LENGTH = 128


class TransformerClassifier:
    def __init__(self, model, tokenizer, labels: list[str], device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels  # index-aligned with the model's label ids (id2label)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def new_for_training(cls, base_model_name: str, labels: list[str]) -> "TransformerClassifier":
        labels_sorted = sorted(labels)
        id2label = {i: label for i, label in enumerate(labels_sorted)}
        label2id = {label: i for i, label in id2label.items()}
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=len(labels_sorted), id2label=id2label, label2id=label2id
        )
        return cls(model, tokenizer, labels_sorted)

    def _predict_logits(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**encoded).logits
                all_logits.append(logits.cpu().numpy())
        return np.vstack(all_logits)

    def predict_proba(
        self, texts: list[str], batch_size: int = 32, max_length: int = DEFAULT_MAX_LENGTH
    ) -> np.ndarray:
        logits = self._predict_logits(texts, batch_size, max_length)
        return F.softmax(torch.tensor(logits), dim=-1).numpy()

    def predict(
        self, texts: list[str], batch_size: int = 32, max_length: int = DEFAULT_MAX_LENGTH
    ) -> np.ndarray:
        proba = self.predict_proba(texts, batch_size=batch_size, max_length=max_length)
        idx = proba.argmax(axis=1)
        return np.array([self.labels[i] for i in idx])

    @property
    def classes_(self) -> np.ndarray:
        return np.array(self.labels)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        (directory / "labels.json").write_text(json.dumps(self.labels))

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> "TransformerClassifier":
        directory = Path(directory)
        tokenizer = AutoTokenizer.from_pretrained(directory)
        model = AutoModelForSequenceClassification.from_pretrained(directory)
        labels = json.loads((directory / "labels.json").read_text())
        return cls(model, tokenizer, labels, device=device)
