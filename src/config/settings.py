"""Central configuration for the vehicle complaint classification system.

All paths are resolved relative to the project root so the code behaves the
same whether invoked from the repo root, a Docker container, or a test runner.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Paths:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    models_dir = PROJECT_ROOT / "models"
    logs_dir = PROJECT_ROOT / "logs"

    complaints_csv = raw_dir / "complaints.csv"

    train_parquet = processed_dir / "train.parquet"
    val_parquet = processed_dir / "val.parquet"
    test_parquet = processed_dir / "test.parquet"
    time_holdout_train_parquet = processed_dir / "time_holdout_train.parquet"
    time_holdout_test_parquet = processed_dir / "time_holdout_test.parquet"

    production_model_dir = models_dir / "production"


class LabelConfig:
    """Governs the Phase 3 label-formulation decision: single-label
    classification on the *primary* (first-listed) component, with rare
    tags folded into OTHER.

    MIN_SAMPLES_PER_CLASS=200 was chosen by inspecting the actual primary-tag
    distribution (45 raw classes): it keeps 27 semantically coherent classes
    with enough support for a stable macro-F1 estimate, and folds 583 rows
    (~0.2% of data) of noisy/near-duplicate tags (e.g. "NONE", "COMMUNICATIONS"
    vs "COMMUNICATION", free-text miscodes) into OTHER instead of preserving
    them as unlearnable single-digit-support classes.
    """

    MIN_SAMPLES_PER_CLASS = 200
    OTHER_LABEL = "OTHER"
    TARGET_COLUMN = "components"
    PRIMARY_LABEL_COLUMN = "primary_component"
    TEXT_COLUMN = "summary"
    CLEAN_TEXT_COLUMN = "clean_summary"


class SplitConfig:
    RANDOM_STATE = 42
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15  # of the full dataset; taken out of the train remainder
    # Time-based holdout: train on complaints filed through this year (inclusive),
    # test on everything filed after it. Used to sanity-check concept drift
    # (e.g. FORWARD COLLISION AVOIDANCE / LANE DEPARTURE are recent-tech categories).
    TIME_HOLDOUT_TRAIN_MAX_YEAR = 2023


class FeatureConfig:
    TFIDF_MAX_FEATURES = 30000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 3


class ModelConfig:
    BASELINE_MODELS = ("logistic_regression", "linear_svc")
    RANDOM_STATE = 42


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MLflow 3.x deprecated the plain filesystem backend; sqlite is the
    # recommended lightweight local backend going forward.
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'experiments' / 'mlflow.db').as_posix()}"
    log_level: str = "INFO"
    serving_model_backend: str = "baseline"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
paths = Paths()
label_config = LabelConfig()
split_config = SplitConfig()
feature_config = FeatureConfig()
model_config = ModelConfig()
