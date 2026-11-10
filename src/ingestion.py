"""MovieLens CSV ingestion and reproducible dataset acquisition."""

import logging
import zipfile
from pathlib import Path
from typing import TypeAlias
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from config import DATA_DIR, MIRROR_TIMESTAMP_START, MOVIELENS_URL, REQUIRED_RATING_COLUMNS

logger = logging.getLogger(__name__)
DataTables: TypeAlias = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _synthetic_timestamps(count: int) -> np.ndarray:
    """Generate deterministic past dates for mirrors without event timestamps."""
    generator = np.random.default_rng(42)
    return generator.integers(1_420_070_400, 1_704_067_200, size=count, dtype=np.int64)


def ensure_dataset(data_dir: str | Path = DATA_DIR) -> Path:
    """Download and extract MovieLens, falling back to its Hugging Face mirror."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / "ml-latest-small"
    ratings_path = dataset_dir / "ratings.csv"
    if ratings_path.exists():
        existing = pd.read_csv(ratings_path, usecols=["timestamp"])
        timestamps = pd.to_numeric(existing["timestamp"], errors="coerce").to_numpy()
        legacy_sequence = np.arange(len(existing), dtype=np.int64) + MIRROR_TIMESTAMP_START
        if np.array_equal(timestamps, legacy_sequence):
            logger.info("Replacing legacy sequential mirror timestamps with synthetic dates.")
            ratings = pd.read_csv(ratings_path)
            ratings["timestamp"] = _synthetic_timestamps(len(ratings))
            ratings.to_csv(ratings_path, index=False)
        return dataset_dir

    archive_path = root / "ml-latest-small.zip"
    try:
        if not archive_path.exists():
            logger.info("Downloading MovieLens from GroupLens.")
            urlretrieve(MOVIELENS_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)
    except (OSError, zipfile.BadZipFile) as exc:
        logger.warning("GroupLens dataset unavailable (%s); trying the Hugging Face mirror.", exc)

    if not ratings_path.exists():
        from datasets import load_dataset

        source = load_dataset("ashraq/movielens_ratings", split="train[:100000]").to_pandas()
        source = source.rename(columns={"user_id": "userId", "movie_id": "movieId"}).copy()
        source["timestamp"] = _synthetic_timestamps(len(source))
        for column in ("title", "genres", "imdbId", "tmdbId"):
            fill_value = "Unknown" if column in ("title", "genres") else "0"
            source[column] = source[column].fillna(fill_value)

        dataset_dir.mkdir(parents=True, exist_ok=True)
        source[list(REQUIRED_RATING_COLUMNS)].to_csv(ratings_path, index=False)
        source[["movieId", "title", "genres"]].drop_duplicates("movieId").to_csv(
            dataset_dir / "movies.csv", index=False
        )
        source[["movieId", "imdbId", "tmdbId"]].drop_duplicates("movieId").to_csv(
            dataset_dir / "links.csv", index=False
        )

    required_files = ("ratings.csv", "movies.csv", "links.csv")
    missing = [name for name in required_files if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MovieLens dataset is missing files: {missing}")
    return dataset_dir


class DataIngestion:
    """Load ratings, item catalog, and external item links from CSV files."""

    def __init__(self, data_dir: str | Path = DATA_DIR):
        self.data_dir = Path(data_dir)

    def load_raw_data(self) -> DataTables:
        dataset_dir = ensure_dataset(self.data_dir)
        paths = [dataset_dir / filename for filename in ("ratings.csv", "movies.csv", "links.csv")]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Required input file missing: {path.resolve()}")
        ratings = pd.read_csv(paths[0])
        movies = pd.read_csv(paths[1])
        links = pd.read_csv(paths[2])
        logger.info(
            "Loaded ratings=%s, movies=%s, links=%s",
            ratings.shape,
            movies.shape,
            links.shape,
        )
        return ratings, movies, links

    @staticmethod
    def derive_user_metadata(ratings: pd.DataFrame) -> pd.DataFrame:
        """Aggregate user activity from interaction history."""
        return (
            ratings.groupby("userId")
            .agg(
                total_ratings=("rating", "count"),
                avg_rating=("rating", "mean"),
                first_active=("timestamp", "min"),
                last_active=("timestamp", "max"),
            )
            .reset_index()
        )


def ingest_tables(data_dir: str | Path = DATA_DIR) -> DataTables:
    """Load the three source tables used by the full pipeline."""
    return DataIngestion(data_dir).load_raw_data()


def ingest(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Compatibility helper returning the interaction table only."""
    ratings, _, _ = ingest_tables(data_dir)
    return ratings
