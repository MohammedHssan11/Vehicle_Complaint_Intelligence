from src.preprocessing.text_cleaning import preprocess_for_classical, preprocess_for_transformer


class TestNegationPreservation:
    """Regression tests for the original notebook's bug: stripping punctuation
    before contraction expansion turned "doesn't" into the unmatched token
    "doesnt", silently destroying negation signal."""

    def test_contraction_negation_survives_as_not(self):
        result = preprocess_for_classical("The brake doesn't engage when pressed.")
        tokens = result.split()
        assert "not" in tokens
        assert "doesnt" not in tokens
        assert "doesn" not in tokens

    def test_wasnt_expands_correctly(self):
        result = preprocess_for_classical("The airbag wasn't deployed during the crash.")
        assert "not" in result.split()

    def test_transformer_preprocessing_also_expands_contractions(self):
        result = preprocess_for_transformer("The brake doesn't engage.")
        assert "does not" in result.lower()


class TestBoilerplateStripping:
    def test_tl_star_prefix_removed(self):
        result = preprocess_for_classical("TL* THE CONTACT OWNS A VEHICLE.")
        assert "tl" not in result.split()

    def test_star_tr_suffix_removed(self):
        result = preprocess_for_classical("SEAT BELT FAILED. *TR")
        assert "tr" not in result.split()


class TestBasicNormalization:
    def test_lowercases(self):
        result = preprocess_for_classical("ENGINE STALLED")
        assert result == result.lower()

    def test_digits_stripped_by_default(self):
        result = preprocess_for_classical("Vehicle traveling at 55 mph")
        assert "55" not in result

    def test_keep_digits_true_preserves_digits(self):
        result = preprocess_for_classical("Vehicle traveling at 55 mph", keep_digits=True)
        assert "55" in result

    def test_empty_string_returns_empty(self):
        assert preprocess_for_classical("") == ""
        assert preprocess_for_transformer("") == ""

    def test_non_string_input_returns_empty(self):
        assert preprocess_for_classical(None) == ""
        assert preprocess_for_classical(float("nan")) == ""

    def test_stopwords_removed_but_not_negation(self):
        result = preprocess_for_classical("The engine is not the problem")
        tokens = result.split()
        assert "the" not in tokens
        assert "is" not in tokens
        assert "not" in tokens
