"""Portable Parquet-backed offline feature store."""

from pathlib import Path

import pandas as pd


class OfflineFeatureStore:
    """Read and write named feature tables beneath a single store directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Feature table name must be a non-empty filename stem.")
        return self.root / f"{name}.parquet"

    def write(self, name: str, frame: pd.DataFrame) -> Path:
        """Persist a feature table as Parquet and return its path."""
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def read(self, name: str) -> pd.DataFrame:
        """Load a persisted feature table by name."""
        return pd.read_parquet(self._path(name))

    def read_training_features(self) -> pd.DataFrame:
        """Load the point-in-time feature rows prepared for model training."""
        return self.read("train_features")

    def read_prediction_features(self, user_id: int, item_ids: list[int]) -> pd.DataFrame:
        """Join stored user and item snapshots for a user's candidate items."""
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Prediction item_ids must be unique.")

        users = self.read("users")
        user_features = users.loc[users["userId"] == user_id]
        if user_features.empty:
            raise KeyError(f"No stored feature snapshot for userId={user_id}.")

        items = self.read("items")
        item_features = items.set_index("movieId").reindex(item_ids)
        if item_features.isna().all(axis=1).any():
            missing_items = item_features.index[item_features.isna().all(axis=1)].tolist()
            raise KeyError(f"No stored feature snapshot for movieIds={missing_items}.")

        user_values = user_features.drop(columns="userId").iloc[0].to_dict()
        result = item_features.reset_index()
        result.insert(0, "userId", user_id)
        for name, value in user_values.items():
            result[name] = value
        return result
