"""Shared paths and data schema for the recommendation pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ROLL_NUMBER = "2025AG05563"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
REQUIRED_RATING_COLUMNS = ("userId", "movieId", "rating", "timestamp")
MIRROR_TIMESTAMP_START = 1_700_000_000
