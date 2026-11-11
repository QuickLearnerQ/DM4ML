"""Interaction validation, normalization, and invalid-row quarantine."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import REQUIRED_RATING_COLUMNS, ROOT


class DataValidationError(ValueError):
    """Raised when required input schema is absent or no valid rows remain."""


@dataclass
class ValidationReport:
    """Summary of validation checks and quarantined interactions."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    passed: bool = True
    status: str = "passed"
    input_interactions: int = 0
    valid_interactions: int = 0
    quarantined_interactions: int = 0
    duplicate_interactions: int = 0
    missing_values: int = 0
    invalid_rating_count: int = 0
    invalid_data_type_counts: dict[str, int] = field(default_factory=dict)
    invalid_timestamp_count: int = 0
    duplicate_movie_ids: int = 0
    duplicate_link_ids: int = 0
    missing_movie_references: int = 0
    missing_link_references: int = 0
    rating_dtypes_after_cleaning: dict[str, str] = field(default_factory=dict)
    referential_integrity: dict[str, bool] = field(default_factory=dict)
    cold_start_users_below_5: int = 0
    cold_start_items_below_5: int = 0
    cold_start_user_threshold: int = 5
    cold_start_item_threshold: int = 5
    quarantine_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _normalize_timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    converted = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = parsed.notna()
    converted.loc[valid] = parsed.loc[valid].astype("int64") / 1_000_000_000
    converted.loc[numeric.notna()] = numeric.loc[numeric.notna()]
    return converted


def validate_dataset(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    links: pd.DataFrame,
    report_path: str | Path,
    errors_path: str | Path,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return normalized valid ratings and write rejected rows with validation reasons."""
    if min_user_ratings < 1 or min_item_ratings < 1:
        raise ValueError("Cold-start thresholds must be positive integers.")
    original = ratings.copy()
    reasons = pd.Series("", index=original.index, dtype="object")

    def reject(mask: pd.Series, reason: str) -> None:
        selected = mask.fillna(True)
        reasons.loc[selected] = reasons.loc[selected].map(
            lambda prior: f"{prior}; {reason}" if prior else reason
        )

    missing_columns = sorted(set(REQUIRED_RATING_COLUMNS) - set(original.columns))
    if missing_columns:
        raise DataValidationError(f"Ratings table is missing columns: {missing_columns}")
    for table, table_name in ((movies, "movies"), (links, "links")):
        if "movieId" not in table.columns:
            raise DataValidationError(f"{table_name} table is missing movieId.")

    normalized = original[list(REQUIRED_RATING_COLUMNS)].copy()
    invalid_type_counts: dict[str, int] = {}
    for column in ("userId", "movieId", "rating"):
        converted = pd.to_numeric(normalized[column], errors="coerce")
        invalid_numeric = converted.isna() | ~np.isfinite(converted.fillna(0))
        invalid_type_counts[column] = int(
            (normalized[column].notna() & invalid_numeric).sum()
        )
        reject(invalid_numeric, f"invalid_or_missing_{column}")
        normalized[column] = converted

    timestamps = _normalize_timestamp(normalized["timestamp"])
    invalid_timestamps = timestamps.isna() | ~np.isfinite(timestamps.fillna(0))
    invalid_type_counts["timestamp"] = int(
        (normalized["timestamp"].notna() & invalid_timestamps).sum()
    )
    reject(invalid_timestamps, "invalid_or_missing_timestamp")
    reject(timestamps.notna() & (timestamps % 1 != 0), "invalid_timestamp_precision")
    normalized["timestamp"] = timestamps
    reject(normalized.isna().any(axis=1), "missing_required_value")
    reject(
        (normalized["userId"] <= 0)
        | (normalized["userId"] % 1 != 0)
        | (normalized["userId"] >= 2**63),
        "invalid_userId",
    )
    reject(
        (normalized["movieId"] <= 0)
        | (normalized["movieId"] % 1 != 0)
        | (normalized["movieId"] >= 2**63),
        "invalid_movieId",
    )
    reject(
        ~np.isfinite(normalized["rating"].fillna(0))
        | (normalized["rating"] < 0.5)
        | (normalized["rating"] > 5.0)
        | ((normalized["rating"] * 2) % 1 != 0),
        "invalid_rating",
    )
    reject(
        (normalized["timestamp"] < 0) | (normalized["timestamp"] >= 2**63),
        "invalid_timestamp",
    )
    now_seconds = datetime.now(timezone.utc).timestamp()
    reject(normalized["timestamp"] > now_seconds, "future_timestamp")

    duplicate = normalized[["userId", "movieId", "timestamp"]].duplicated(keep="first")
    reject(duplicate, "duplicate_interaction")
    movie_ids = pd.to_numeric(movies["movieId"], errors="coerce")
    link_ids = pd.to_numeric(links["movieId"], errors="coerce")
    valid_movie_ids = normalized["movieId"].where(reasons.eq(""))
    reject(
        valid_movie_ids.notna() & ~valid_movie_ids.isin(movie_ids.dropna()),
        "missing_movie_reference",
    )
    reject(
        valid_movie_ids.notna() & ~valid_movie_ids.isin(link_ids.dropna()),
        "missing_link_reference",
    )

    rejected_mask = reasons.ne("")
    rejected = original.loc[rejected_mask].copy()
    rejected["validation_errors"] = reasons.loc[rejected_mask]
    errors_path = Path(errors_path)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    rejected.to_csv(errors_path, index=False)

    clean = normalized.loc[~rejected_mask].copy()
    if clean.empty:
        raise DataValidationError(f"Validation rejected every interaction; see {errors_path}.")
    clean["userId"] = clean["userId"].astype("int64")
    clean["movieId"] = clean["movieId"].astype("int64")
    clean["rating"] = clean["rating"].astype("float64")
    clean["timestamp"] = clean["timestamp"].astype("int64")

    clean_movie_ids = set(movie_ids.dropna().astype("int64"))
    clean_link_ids = set(link_ids.dropna().astype("int64"))
    used_movie_ids = set(clean["movieId"])
    missing_movies = used_movie_ids - clean_movie_ids
    missing_links = used_movie_ids - clean_link_ids
    if missing_movies or missing_links:
        raise DataValidationError("Referential integrity failed after interaction filtering.")

    user_counts = clean["userId"].value_counts()
    item_counts = clean["movieId"].value_counts()
    try:
        quarantine_reference = str(errors_path.relative_to(ROOT))
    except ValueError:
        quarantine_reference = str(errors_path)
    report = ValidationReport(
        input_interactions=len(original),
        valid_interactions=len(clean),
        quarantined_interactions=len(rejected),
        duplicate_interactions=int(duplicate.sum()),
        missing_values=int(original[list(REQUIRED_RATING_COLUMNS)].isna().any(axis=1).sum()),
        invalid_rating_count=int(reasons.str.contains("invalid_rating", regex=False).sum()),
        invalid_data_type_counts=invalid_type_counts,
        invalid_timestamp_count=int(reasons.str.contains("timestamp", regex=False).sum()),
        duplicate_movie_ids=int(movie_ids.duplicated().sum()),
        duplicate_link_ids=int(link_ids.duplicated().sum()),
        missing_movie_references=int(
            reasons.str.contains("missing_movie_reference", regex=False).sum()
        ),
        missing_link_references=int(
            reasons.str.contains("missing_link_reference", regex=False).sum()
        ),
        rating_dtypes_after_cleaning=clean.dtypes.astype(str).to_dict(),
        referential_integrity={"movies": not missing_movies, "links": not missing_links},
        cold_start_users_below_5=int((user_counts < min_user_ratings).sum()),
        cold_start_items_below_5=int((item_counts < min_item_ratings).sum()),
        cold_start_user_threshold=min_user_ratings,
        cold_start_item_threshold=min_item_ratings,
        quarantine_path=quarantine_reference,
        status="passed" if rejected.empty else "passed_with_quarantine",
    )
    report.save_json(report_path)
    return clean, report.to_dict()


def validate_recommendation_data(
    df_ratings: pd.DataFrame,
    df_movies: pd.DataFrame,
    df_links: pd.DataFrame,
    output_report_path: str | Path = "outputs/validation_report.json",
    errors_path: str | Path = "data/errors/invalid_interactions.csv",
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
) -> ValidationReport:
    """Convenience API for callers that prefer a typed validation report."""
    _, payload = validate_dataset(
        df_ratings,
        df_movies,
        df_links,
        output_report_path,
        errors_path,
        min_user_ratings,
        min_item_ratings,
    )
    return ValidationReport(**payload)


def validate_ratings(
    df_ratings: pd.DataFrame,
    data_dir: str | Path = "data",
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
) -> dict[str, Any]:
    """Load item references and run the canonical validation workflow."""
    root = Path(data_dir)
    dataset_dir = root / "raw" / "ml-latest-small"
    report_path = (
        root.parent / "outputs" / "validation_report.json"
        if root.name == "data"
        else root / "outputs" / "validation_report.json"
    )
    report = validate_recommendation_data(
        df_ratings,
        pd.read_csv(dataset_dir / "movies.csv"),
        pd.read_csv(dataset_dir / "links.csv"),
        report_path,
        root / "errors" / "invalid_interactions.csv",
        min_user_ratings,
        min_item_ratings,
    )
    return report.to_dict()
