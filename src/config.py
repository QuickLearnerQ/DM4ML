"""Shared paths and data schema for the recommendation pipeline."""

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PIPELINE_CONFIG_PATH = ROOT / "configs" / "pipeline.json"
ROLL_NUMBER = "2025AG05563"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
REQUIRED_RATING_COLUMNS = ("userId", "movieId", "rating", "timestamp")
MIRROR_TIMESTAMP_START = 1_700_000_000


def load_pipeline_settings() -> dict[str, Any]:
    """Load and validate settings shared by local and Airflow runs."""
    try:
        settings = json.loads(PIPELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to load pipeline settings from {PIPELINE_CONFIG_PATH}."
        ) from exc

    if not isinstance(settings, dict):
        raise ValueError("Pipeline settings must be a JSON object.")

    models = settings.get("models")
    preparation = settings.get("preparation")
    evaluation = settings.get("evaluation")
    if not all(isinstance(section, dict) for section in (preparation, evaluation, models)):
        raise ValueError(
            "Pipeline settings must define preparation, evaluation, and model objects."
        )
    popularity = models.get("popularity")
    svd = models.get("truncated_svd")
    if not isinstance(popularity, dict) or not isinstance(svd, dict):
        raise ValueError("Pipeline settings must define preparation, evaluation, and SVD models.")
    if popularity.get("strategy") != "interaction_count":
        raise ValueError("models.popularity.strategy must be 'interaction_count'.")
    if popularity.get("tie_breaker") != "item_average_rating":
        raise ValueError("models.popularity.tie_breaker must be 'item_average_rating'.")

    test_size = preparation.get("test_size")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("preparation.test_size must be a finite number between 0 and 1.")

    for key in ("k", "max_ranking_users"):
        value = evaluation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"evaluation.{key} must be a positive integer.")

    for key in ("n_components", "n_iter"):
        value = svd.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"models.truncated_svd.{key} must be a positive integer.")
    if not isinstance(svd.get("random_state"), int) or isinstance(svd["random_state"], bool):
        raise ValueError("models.truncated_svd.random_state must be an integer.")

    return settings
