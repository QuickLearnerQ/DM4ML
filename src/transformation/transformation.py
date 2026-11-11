"""Reusable user, item, and interaction feature transformations."""

from pathlib import Path

import numpy as np
import pandas as pd


def build_feature_tables(
    train: pd.DataFrame, movies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build user/item cutoff snapshots and interaction event-time features."""
    cutoff = int(train["timestamp"].max())
    catalog = movies[["movieId", "title", "genres"]].drop_duplicates("movieId").copy()
    interactions = train.merge(catalog, on="movieId", how="left")

    users = (
        train.groupby("userId")
        .agg(
            user_activity_count=("rating", "size"),
            user_average_rating=("rating", "mean"),
            first_interaction=("timestamp", "min"),
            last_interaction=("timestamp", "max"),
        )
        .reset_index()
    )
    users["days_since_last_interaction"] = ((cutoff - users["last_interaction"]) / 86_400).astype(
        float
    )
    preferred = (
        interactions.assign(category=interactions["genres"].fillna("Unknown").str.split("|"))
        .explode("category")
        .groupby(["userId", "category"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["userId", "count", "category"], ascending=[True, False, True])
        .drop_duplicates("userId")
        .rename(columns={"category": "preferred_category"})
    )
    users = users.merge(preferred[["userId", "preferred_category"]], on="userId", how="left")

    item_stats = train.groupby("movieId").agg(
        item_popularity=("rating", "size"),
        item_average_rating=("rating", "mean"),
        item_rating_count=("rating", "count"),
    )
    items = catalog.merge(item_stats, on="movieId", how="left")
    items["item_popularity"] = items["item_popularity"].fillna(0).astype("int64")
    items["item_rating_count"] = items["item_rating_count"].fillna(0).astype("int64")
    items["item_average_rating"] = items["item_average_rating"].fillna(
        float(train["rating"].mean())
    )

    interaction_features = train.copy()
    event_time = pd.to_datetime(interaction_features["timestamp"], unit="s", utc=True)
    interaction_features["interaction_hour_utc"] = event_time.dt.hour.astype("int8")
    interaction_features["interaction_weekday_utc"] = event_time.dt.day_of_week.astype("int8")
    return users, items, interaction_features


def build_features(train: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Build leakage-safe training features using only strictly earlier events."""
    ordered = train.sort_values(["timestamp"], kind="mergesort").copy()
    global_mean = float(ordered["rating"].mean())
    timestamp_stats = ordered.groupby("timestamp")["rating"].agg(["sum", "count"])
    timestamp_stats["prior_sum"] = timestamp_stats["sum"].cumsum() - timestamp_stats["sum"]
    timestamp_stats["prior_count"] = (
        timestamp_stats["count"].cumsum() - timestamp_stats["count"]
    )
    prior_global_mean = np.divide(
        timestamp_stats["prior_sum"],
        timestamp_stats["prior_count"],
        out=np.full(len(timestamp_stats), 3.0, dtype=float),
        where=timestamp_stats["prior_count"].to_numpy() > 0,
    )
    ordered["global_mean"] = ordered["timestamp"].map(
        pd.Series(prior_global_mean, index=timestamp_stats.index)
    )
    for entity in ("userId", "movieId"):
        grouped = (
            ordered.groupby([entity, "timestamp"])["rating"].agg(["sum", "count"]).sort_index()
        )
        entity_groups = grouped.groupby(level=0, sort=False)
        grouped["prior_rating_sum"] = entity_groups["sum"].cumsum() - grouped["sum"]
        grouped["prior_rating_count"] = entity_groups["count"].cumsum() - grouped["count"]
        event_keys = pd.MultiIndex.from_frame(ordered[[entity, "timestamp"]])
        prior = grouped.reindex(event_keys)
        feature_prefix = "user" if entity == "userId" else "item"
        count_column = f"{feature_prefix}_prior_rating_count"
        average_column = f"{feature_prefix}_prior_average_rating"
        prior_counts = prior["prior_rating_count"].to_numpy(dtype="int64")
        prior_sums = prior["prior_rating_sum"].to_numpy(dtype=float)
        ordered[count_column] = prior_counts
        ordered[average_column] = np.divide(
            prior_sums,
            prior_counts,
            out=ordered["global_mean"].to_numpy(dtype=float).copy(),
            where=prior_counts > 0,
        )
    event_time = pd.to_datetime(ordered["timestamp"], unit="s", utc=True)
    ordered["interaction_hour_utc"] = event_time.dt.hour.astype("int8")
    ordered["interaction_weekday_utc"] = event_time.dt.day_of_week.astype("int8")
    return ordered, global_mean


def save_features(features: pd.DataFrame, path: str | Path) -> Path:
    """Persist a reusable feature table as Parquet."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    return destination
