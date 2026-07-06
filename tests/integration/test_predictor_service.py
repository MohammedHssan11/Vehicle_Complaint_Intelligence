import pytest

from src.config.settings import paths
from src.inference.predictor import InvalidInputError, PredictorService

BASELINE_AVAILABLE = (paths.production_model_dir / "model.joblib").exists()
TRANSFORMER_AVAILABLE = (paths.models_dir / "transformer" / "model.safetensors").exists()

requires_baseline = pytest.mark.skipif(
    not BASELINE_AVAILABLE,
    reason="models/production/ not built — run `python -m src.data.pipeline` and `python -m src.training.train` first",
)
requires_transformer = pytest.mark.skipif(
    not TRANSFORMER_AVAILABLE,
    reason="models/transformer/ not built — run `python -m src.training.train_transformer` first",
)


@pytest.fixture(scope="module")
def baseline_service():
    return PredictorService(backend="baseline")


@requires_baseline
class TestBaselinePredictor:
    def test_predicts_plausible_label_and_confidence(self, baseline_service):
        result = baseline_service.predict(
            "The brake pedal did not engage properly and the car would not stop in time."
        )
        assert result.predicted_label
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_backend == "baseline"

    def test_top_k_has_three_entries_sorted_descending(self, baseline_service):
        result = baseline_service.predict("Engine stalls at highway speed without warning.")
        assert len(result.top_k) == 3
        confidences = [item["confidence"] for item in result.top_k]
        assert confidences == sorted(confidences, reverse=True)
        assert result.top_k[0]["label"] == result.predicted_label

    def test_explanation_present_when_requested(self, baseline_service):
        result = baseline_service.predict("Airbag deployed without any collision.", explain=True)
        assert result.explanation is not None
        assert result.explanation_method in ("shap", "coefficient")
        assert len(result.explanation) > 0

    def test_explanation_absent_when_not_requested(self, baseline_service):
        result = baseline_service.predict("Airbag deployed without any collision.", explain=False)
        assert result.explanation is None
        assert result.explanation_method is None

    def test_batch_predict_returns_one_result_per_input(self, baseline_service):
        texts = ["Engine stalls at highway speed.", "Seat belt did not retract after use."]
        results = baseline_service.predict_batch(texts, explain=False)
        assert len(results) == len(texts)

    def test_empty_text_raises_invalid_input(self, baseline_service):
        with pytest.raises(InvalidInputError):
            baseline_service.predict("")

    def test_too_short_text_raises_invalid_input(self, baseline_service):
        with pytest.raises(InvalidInputError):
            baseline_service.predict("hi")

    def test_whitespace_only_text_raises_invalid_input(self, baseline_service):
        with pytest.raises(InvalidInputError):
            baseline_service.predict("     ")


@requires_transformer
class TestTransformerPredictor:
    def test_predicts_without_error(self):
        service = PredictorService(backend="transformer")
        result = service.predict("The steering wheel vibrates violently at highway speeds.")
        assert result.predicted_label
        assert result.model_backend == "transformer"

    def test_explanation_not_implemented_for_transformer(self):
        service = PredictorService(backend="transformer")
        result = service.predict("Engine stalls at highway speed.", explain=True)
        assert result.explanation is None


class TestBackendSelection:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown SERVING_MODEL_BACKEND"):
            PredictorService(backend="not_a_real_backend")
