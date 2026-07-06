import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.config.settings import paths

BASELINE_AVAILABLE = (paths.production_model_dir / "model.joblib").exists()

pytestmark = pytest.mark.skipif(
    not BASELINE_AVAILABLE,
    reason="models/production/ not built — run `python -m src.data.pipeline` and `python -m src.training.train` first",
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


class TestPredict:
    def test_predict_returns_valid_shape(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={"text": "The brake pedal did not engage properly.", "explain": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_label"]
        assert 0.0 <= body["confidence"] <= 1.0
        assert len(body["top_k"]) == 3
        assert body["explanation"] is not None
        assert body["model_backend"] == "baseline"

    def test_predict_without_explanation(self, client):
        resp = client.post("/api/v1/predict", json={"text": "Engine stalls at highway speed.", "explain": False})
        assert resp.status_code == 200
        assert resp.json()["explanation"] is None

    def test_predict_empty_text_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_too_short_text_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={"text": "hi"})
        assert resp.status_code == 422

    def test_predict_missing_text_field_returns_422(self, client):
        resp = client.post("/api/v1/predict", json={})
        assert resp.status_code == 422


class TestBatchPredict:
    def test_batch_predict_returns_one_result_per_input(self, client):
        resp = client.post(
            "/api/v1/batch-predict",
            json={"texts": ["Engine stalls at highway speed.", "Seat belt did not retract."], "explain": False},
        )
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 2

    def test_batch_predict_empty_list_returns_422(self, client):
        resp = client.post("/api/v1/batch-predict", json={"texts": []})
        assert resp.status_code == 422


class TestMetrics:
    def test_metrics_reflects_served_predictions(self, client):
        before = client.get("/api/v1/metrics").json()["total_predictions"]
        client.post("/api/v1/predict", json={"text": "Suspension makes a loud clunking noise.", "explain": False})
        after = client.get("/api/v1/metrics").json()["total_predictions"]
        assert after == before + 1


class TestRetrain:
    def test_retrain_starts_and_invokes_training(self, client, monkeypatch):
        """Starlette's TestClient drains BackgroundTasks synchronously before
        returning the response, so the mocked training job below has already
        run to completion by the time this call returns — this test can only
        verify the "started" branch and that training was actually invoked,
        not the live already-in-progress race (covered by the test below,
        which asserts the guard directly)."""
        started = {"count": 0}

        def _fake_train_and_select():
            started["count"] += 1

        monkeypatch.setattr("src.training.train.train_and_select", _fake_train_and_select)

        resp = client.post("/api/v1/retrain")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        assert started["count"] == 1

    def test_retrain_reports_already_running_when_job_in_progress(self, client):
        from src.api.routes import _retrain_state

        _retrain_state["in_progress"] = True
        try:
            resp = client.post("/api/v1/retrain")
            assert resp.json()["status"] == "already_running"
        finally:
            _retrain_state["in_progress"] = False
