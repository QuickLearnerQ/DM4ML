"""Chronological splitting and persisted training datasets."""

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix, save_npz

from config import REQUIRED_RATING_COLUMNS


def prepare(
    ratings: pd.DataFrame, test_size: float = 0.2, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split events chronologically; ties at the cutoff remain in the test set."""
    del seed
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(ratings) < 2:
        raise ValueError("At least two interactions are required for a train/test split.")
    ordered = (
        ratings[list(REQUIRED_RATING_COLUMNS)]
        .sort_values("timestamp", kind="mergesort")
        .reset_index(drop=True)
    )
    split_index = min(max(int(len(ordered) * (1 - test_size)), 1), len(ordered) - 1)
    cutoff = ordered.loc[split_index, "timestamp"]
    train = ordered.loc[ordered["timestamp"] < cutoff].copy()
    test = ordered.loc[ordered["timestamp"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def derive_user_metadata(ratings: pd.DataFrame) -> pd.DataFrame:
    """Summarize user activity using only the supplied interactions."""
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


def save_interaction_matrix(ratings: pd.DataFrame, output_dir: str | Path) -> None:
    """Write a sparse user-item matrix and matching stable ID maps."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    users = sorted(ratings["userId"].unique())
    items = sorted(ratings["movieId"].unique())
    user_index = {value: index for index, value in enumerate(users)}
    item_index = {value: index for index, value in enumerate(items)}
    events = ratings.groupby(["userId", "movieId"], as_index=False)["rating"].mean()
    matrix = csr_matrix(
        (
            events["rating"].to_numpy(),
            (
                events["userId"].map(user_index).to_numpy(),
                events["movieId"].map(item_index).to_numpy(),
            ),
        ),
        shape=(len(users), len(items)),
    )
    save_npz(destination / "user_item_matrix.npz", matrix)
    pd.DataFrame({"row_index": range(len(users)), "userId": users}).to_csv(
        destination / "matrix_users.csv", index=False
    )
    pd.DataFrame({"column_index": range(len(items)), "movieId": items}).to_csv(
        destination / "matrix_items.csv", index=False
    )
