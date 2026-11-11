"""Rating and top-K recommendation evaluation."""

from typing import Any

import numpy as np
import pandas as pd

from training.model_training import Recommender, evaluate_explicit_ratings


def _diversity(movie_ids: list[int], genres_by_item: dict[int, set[str]]) -> float:
    if len(movie_ids) < 2:
        return 0.0
    distances = []
    for left_index, left in enumerate(movie_ids):
        for right in movie_ids[left_index + 1 :]:
            left_genres = genres_by_item.get(left, set())
            right_genres = genres_by_item.get(right, set())
            union = left_genres | right_genres
            similarity = len(left_genres & right_genres) / len(union) if union else 1.0
            distances.append(1.0 - similarity)
    return float(np.mean(distances))


def evaluate_model(
    model: Recommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    max_ranking_users: int = 1000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate explicit ratings and top-K quality on a reproducible user sample."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if max_ranking_users <= 0:
        raise ValueError("max_ranking_users must be positive.")

    metrics: dict[str, Any] = evaluate_explicit_ratings(model, test)
    catalog = sorted(
        pd.to_numeric(movies["movieId"], errors="coerce").dropna().astype(int).unique()
    )
    seen_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()
    relevant = test.loc[test["rating"] >= 4].groupby("userId")["movieId"].apply(set).to_dict()
    population = sorted(relevant)
    ranking_users = population
    if len(population) > max_ranking_users:
        generator = np.random.default_rng(42)
        ranking_users = sorted(
            generator.choice(population, size=max_ranking_users, replace=False).tolist()
        )

    genre_lookup = movies.set_index("movieId")["genres"].fillna("").astype(str).to_dict()
    genres_by_item = {
        int(item): {genre for genre in value.split("|") if genre and genre != "(no genres listed)"}
        for item, value in genre_lookup.items()
    }
    recommendation_rows: list[dict[str, Any]] = []
    precisions, recalls, ndcgs, diversities = [], [], [], []
    for user_id in ranking_users:
        relevant_items = relevant[user_id]
        seen_items = seen_by_user.get(user_id, set())
        candidates = [item for item in catalog if item not in seen_items]
        scores = model.score_items(user_id, candidates)
        top_positions = np.argsort(-scores, kind="stable")[:k]
        top_items = [candidates[position] for position in top_positions]
        hits = [int(item in relevant_items) for item in top_items]
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant_items))

        dcg = sum(hit / np.log2(index + 2) for index, hit in enumerate(hits))
        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1 / np.log2(index + 2) for index in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        diversities.append(_diversity(top_items, genres_by_item))
        recommendation_rows.extend(
            {
                "model": type(model).__name__,
                "userId": user_id,
                "rank": rank,
                "movieId": item,
                "score": float(scores[position]),
            }
            for rank, (position, item) in enumerate(
                zip(top_positions, top_items, strict=True), start=1
            )
        )

    recommended_items = {row["movieId"] for row in recommendation_rows}
    metrics.update(
        {
            f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
            f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "Coverage": float(len(recommended_items) / len(catalog)) if catalog else 0.0,
            "Diversity": float(np.mean(diversities)) if diversities else 0.0,
            "K": k,
            "RankingUsers": len(ranking_users),
            "RankingPopulation": len(population),
            "RankingSampled": len(ranking_users) < len(population),
        }
    )
    return metrics, pd.DataFrame(recommendation_rows)
