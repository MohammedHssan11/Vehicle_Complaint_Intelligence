import pandas as pd

from src.features.tfidf import CombinedTfidfFeaturizer, TfidfFeaturizer

TEXTS = pd.Series(
    [
        "brake pedal not engage",
        "engine stall highway speed",
        "brake pedal fail stop",
    ]
)


class TestTfidfFeaturizer:
    def test_fit_transform_returns_matrix_with_correct_row_count(self):
        featurizer = TfidfFeaturizer(max_features=50, min_df=1)
        X = featurizer.fit_transform(TEXTS)
        assert X.shape[0] == len(TEXTS)
        assert X.shape[1] <= 50

    def test_transform_after_fit_uses_same_vocabulary(self):
        featurizer = TfidfFeaturizer(max_features=50, min_df=1)
        featurizer.fit_transform(TEXTS)
        X_new = featurizer.transform(pd.Series(["brake pedal not engage"]))
        assert X_new.shape[1] == len(featurizer.get_feature_names())

    def test_get_feature_names_nonempty_after_fit(self):
        featurizer = TfidfFeaturizer(max_features=50, min_df=1)
        featurizer.fit_transform(TEXTS)
        names = featurizer.get_feature_names()
        assert len(names) > 0
        assert "brake" in names

    def test_save_and_load_roundtrip(self, tmp_path):
        featurizer = TfidfFeaturizer(max_features=50, min_df=1)
        featurizer.fit_transform(TEXTS)
        save_path = tmp_path / "vectorizer.joblib"
        featurizer.save(save_path)

        loaded = TfidfFeaturizer.load(save_path)
        original_vec = featurizer.transform(TEXTS).toarray()
        loaded_vec = loaded.transform(TEXTS).toarray()
        assert (original_vec == loaded_vec).all()


class TestCombinedTfidfFeaturizer:
    def _make(self):
        return CombinedTfidfFeaturizer(
            word={"max_features": 30, "min_df": 1},
            char={"max_features": 30, "min_df": 1, "ngram_range": (3, 4)},
        )

    def test_fit_transform_hstacks_word_and_char_features(self):
        featurizer = self._make()
        X = featurizer.fit_transform(TEXTS)
        assert X.shape[0] == len(TEXTS)
        assert X.shape[1] == len(featurizer.get_feature_names())

    def test_char_feature_names_are_prefixed(self):
        featurizer = self._make()
        featurizer.fit_transform(TEXTS)
        names = featurizer.get_feature_names()
        assert any(n.startswith("char:") for n in names)

    def test_column_count_equals_word_plus_char_vocab(self):
        featurizer = self._make()
        X = featurizer.fit_transform(TEXTS)
        n_word = len(featurizer.word_vectorizer.get_feature_names_out())
        n_char = len(featurizer.char_vectorizer.get_feature_names_out())
        assert X.shape[1] == n_word + n_char

    def test_save_and_load_roundtrip(self, tmp_path):
        featurizer = self._make()
        featurizer.fit_transform(TEXTS)
        save_path = tmp_path / "vectorizer.joblib"
        featurizer.save(save_path)

        loaded = CombinedTfidfFeaturizer.load(save_path)
        original_vec = featurizer.transform(TEXTS).toarray()
        loaded_vec = loaded.transform(TEXTS).toarray()
        assert (original_vec == loaded_vec).all()
