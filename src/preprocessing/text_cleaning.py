"""Text preprocessing pipeline (Phase 8).

Fixes the original notebook's preprocessing bug: it stripped punctuation
(`re.sub(r'[^a-z\\s]', '', text)`) *before* any contraction handling, so
`"doesn't"` collapsed to `"doesnt"` — a token that matches neither the
stopword list nor any negation rule. Negation is exactly the kind of signal
that matters here ("brake did **not** engage" vs "brake engaged"), so this
pipeline expands contractions first and explicitly keeps negation words out
of the stopword-removal step.

Two entry points are provided because the two model families need different
amounts of normalization:
- `preprocess_for_classical`: full pipeline (stopwords removed, lemmatized) —
  needed for TF-IDF, which has no other way to reduce sparse/noisy vocabulary.
- `preprocess_for_transformer`: light-touch (unicode/boilerplate/contractions
  only) — DistilBERT's own subword tokenizer handles casing and morphology
  better than a hand-rolled lemmatizer would, and stopword removal actively
  hurts transformer models by breaking natural sentence structure.
"""
from __future__ import annotations

import re
import unicodedata

import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NHTSA transcribers prefix/suffix complaints with internal shorthand codes
# (e.g. "TL*" = telephone contact, "*TR" = consumer report). These are
# metadata artifacts, not complaint content, and appear in ~5-7% of rows.
_BOILERPLATE_CODE_PATTERN = re.compile(r"\bTL\*|\*TR\b|\*[A-Z]{2}\b")
_NON_ALPHA_KEEP_DIGITS = re.compile(r"[^a-z0-9\s]")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")

_STOPWORDS = set(stopwords.words("english"))
_NEGATION_WORDS = {"not", "no", "nor", "never", "none", "neither", "without"}
_STOPWORDS_KEEP_NEGATION = _STOPWORDS - _NEGATION_WORDS

_lemmatizer = WordNetLemmatizer()


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _strip_boilerplate_codes(text: str) -> str:
    return _BOILERPLATE_CODE_PATTERN.sub(" ", text)


def _expand_contractions(text: str) -> str:
    return contractions.fix(text)


def preprocess_for_classical(text: str, keep_digits: bool = False) -> str:
    if not isinstance(text, str) or not text:
        return ""

    text = _normalize_unicode(text)
    text = _strip_boilerplate_codes(text)
    text = _expand_contractions(text)
    text = text.lower()
    text = (_NON_ALPHA_KEEP_DIGITS if keep_digits else _NON_ALPHA).sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()

    tokens = [t for t in text.split() if t not in _STOPWORDS_KEEP_NEGATION]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_for_transformer(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    text = _normalize_unicode(text)
    text = _strip_boilerplate_codes(text)
    text = _expand_contractions(text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
