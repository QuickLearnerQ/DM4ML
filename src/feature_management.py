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
