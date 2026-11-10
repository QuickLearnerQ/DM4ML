"""Recommendation model implementations."""

from typing import Protocol

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Recommender(Protocol):
    """Minimal interface required by ranking evaluation."""

    def fit(self, train: pd.DataFrame) -> "Recommender": ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...

    def score_items(self, user_id: int, item_ids: list[int]) -> np.ndarray: ...


class MeanBaseline:
    """Personalized mean estimator with global, user, and item averages."""

    def fit(self, train: pd.DataFrame) -> "MeanBaseline":
        if train.empty:
            raise ValueError("Cannot fit the baseline on an empty dataset.")
        self.global_mean = float(train["rating"].mean())
        self.user_means = train.groupby("userId")["rating"].mean().to_dict()
        item_means = train.groupby("movieId")["rating"].mean().sort_index()
        self.item_means = item_means.to_dict()
        self.item_ids = item_means.index.tolist()
        self.item_index = pd.Index(self.item_ids)
        self.item_values = item_means.to_numpy()
        return self

    def predict_one(self, user_id: int, movie_id: int) -> float:
        values = [self.global_mean]
        if user_id in self.user_means:
            values.append(self.user_means[user_id])
        if movie_id in self.item_means:
            values.append(self.item_means[movie_id])
        return float(np.mean(values))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = np.full(len(frame), self.global_mean, dtype=float)
        counts = np.ones(len(frame), dtype=float)
        user_means = frame["userId"].map(self.user_means)
        item_means = frame["movieId"].map(self.item_means)
        has_user = user_means.notna().to_numpy()
        has_item = item_means.notna().to_numpy()
        predictions[has_user] += user_means[has_user].to_numpy()
        predictions[has_item] += item_means[has_item].to_numpy()
        counts += has_user.astype(float) + has_item.astype(float)
        return predictions / counts

    def score_items(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        indices = self.item_index.get_indexer(item_ids)
        known = indices >= 0
        scores = np.full(len(item_ids), self.global_mean, dtype=float)
        counts = np.ones(len(item_ids), dtype=float)
        if user_id in self.user_means:
            scores += self.user_means[user_id]
            counts += 1
        scores[known] += self.item_values[indices[known]]
        counts[known] += 1
        return scores / counts


class SVDRecommender:
    """Latent-factor explicit-rating recommender based on truncated SVD."""

    def __init__(self, n_components: int = 12, random_state: int = 42, n_iter: int = 3):
        if n_components < 1:
            raise ValueError("n_components must be positive.")
        if n_iter < 1:
            raise ValueError("n_iter must be positive.")
        self.n_components = n_components
        self.random_state = random_state
        self.n_iter = n_iter

    def fit(self, train: pd.DataFrame) -> "SVDRecommender":
        if train.empty:
            raise ValueError("Cannot fit SVD on an empty dataset.")
        self.global_mean = float(train["rating"].mean())
        self.user_means = train.groupby("userId")["rating"].mean().to_dict()
        self.item_means = train.groupby("movieId")["rating"].mean().to_dict()
        self.users = sorted(self.user_means)
        self.items = sorted(self.item_means)
        self.user_index = {value: index for index, value in enumerate(self.users)}
        self.item_index = {value: index for index, value in enumerate(self.items)}

        events = train.groupby(["userId", "movieId"], as_index=False)["rating"].mean()
        rows = events["userId"].map(self.user_index).to_numpy()
        columns = events["movieId"].map(self.item_index).to_numpy()
        matrix = csr_matrix(
            (events["rating"].to_numpy() - self.global_mean, (rows, columns)),
            shape=(len(self.users), len(self.items)),
        )
        max_components = min(matrix.shape) - 1
        self.model = None
        self.user_factors = None
        self.item_factors = None
        if max_components > 0:
            self.model = TruncatedSVD(
                n_components=min(self.n_components, max_components),
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.user_factors = self.model.fit_transform(matrix)
            self.item_factors = self.model.components_.T
        return self

    def predict_one(self, user_id: int, movie_id: int) -> float:
        if self.model is not None and user_id in self.user_index and movie_id in self.item_index:
            estimate = self.global_mean + (
                self.user_factors[self.user_index[user_id]]
                @ self.item_factors[self.item_index[movie_id]]
            )
        else:
            estimate = self.item_means.get(movie_id, self.user_means.get(user_id, self.global_mean))
        return float(np.clip(estimate, 0.5, 5.0))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                self.predict_one(user_id, movie_id)
                for user_id, movie_id in zip(frame["userId"], frame["movieId"], strict=True)
            ]
        )

    def score_items(self, user_id: int, item_ids: list[int]) -> np.ndarray:
        if self.model is None or user_id not in self.user_index:
            fallback = self.user_means.get(user_id, self.global_mean)
            item_means = pd.Series(item_ids).map(self.item_means).fillna(fallback)
            return item_means.to_numpy(dtype=float)
        indices = np.asarray(
            [self.item_index.get(item_id, -1) for item_id in item_ids], dtype=np.int64
        )
        known = indices >= 0
        scores = np.full(len(item_ids), self.user_means.get(user_id, self.global_mean))
        scores[known] = self.global_mean + (
            self.user_factors[self.user_index[user_id]] @ self.item_factors[indices[known]].T
        )
        return np.clip(scores, 0.5, 5.0)


def evaluate_explicit_ratings(model: Recommender, test: pd.DataFrame) -> dict[str, float]:
    """Calculate rating prediction error on a held-out test frame."""
    predictions = model.predict(test)
    return {
        "RMSE": float(mean_squared_error(test["rating"], predictions) ** 0.5),
        "MAE": float(mean_absolute_error(test["rating"], predictions)),
    }


def recommend_for_user(
    model: Recommender,
    user_id: int,
    candidate_movie_ids: list[int],
    seen_movie_ids: tuple[int, ...] | list[int] | set[int] = (),
    n: int = 10,
) -> list[tuple[int, float]]:
    """Return a user's highest-scored unseen items."""
    if n < 0:
        raise ValueError("n cannot be negative.")
    seen = set(seen_movie_ids)
    candidates = [item for item in candidate_movie_ids if item not in seen]
    scores = model.score_items(user_id, candidates)
    order = np.argsort(-scores, kind="stable")[:n]
    return [(candidates[index], float(scores[index])) for index in order]
