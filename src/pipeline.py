"""Dependency-ordered orchestration for the recommendation workflow."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA_DIR, ROLL_NUMBER, ROOT
from evaluation import evaluate_model
from feature_management import OfflineFeatureStore
from ingestion import DataIngestion, ingest, ingest_tables
from model_training import MeanBaseline, SVDRecommender
from preparation import derive_user_metadata, prepare, save_interaction_matrix
from reporting import (
    write_architecture_diagram,
    write_feature_catalog,
    write_final_report,
)
from tracking import track_experiment
from transformation import build_feature_tables, build_features
from validation import DataValidationError, validate_dataset

logger = logging.getLogger(__name__)

__all__ = [
    "DataIngestion",
    "DataValidationError",
    "MeanBaseline",
    "SVDRecommender",
    "build_features",
    "evaluate_model",
    "ingest",
    "prepare",
    "run",
    "validate_dataset",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON output with consistent encoding and formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    models_dir: Path,
    ranking_k: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Fit each configured recommender, evaluate it, and save its artifact."""
    specifications = [
        (
            "user_item_mean",
            MeanBaseline(),
            {"strategy": "global_user_item_mean"},
        ),
        (
            "truncated_svd",
            SVDRecommender(n_components=12, n_iter=3, random_state=42),
            {"n_components": 12, "n_iter": 3, "random_state": 42},
        ),
    ]
    results: list[dict[str, Any]] = []
    recommendation_frames = []
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model, parameters in specifications:
        logger.info("Training model %s", name)
        model.fit(train)
        metrics, recommendations = evaluate_model(model, train, test, movies, k=ranking_k)
        artifact = models_dir / f"{name}.pkl"
        with artifact.open("wb") as file:
            pickle.dump(model, file)
        track_experiment(ROOT, name, parameters, metrics, artifact)
        results.append({"model": name, **parameters, **metrics})
        recommendation_frames.append(recommendations)
    return results, recommendation_frames


def run() -> dict[str, Any]:
    """Run ingest, validate, prepare, transform, train, evaluate, and report."""
    ratings, movies, links = ingest_tables(DATA_DIR)
    logger.info("Validating %s interactions", len(ratings))
    clean, quality = validate_dataset(
        ratings,
        movies,
        links,
        ROOT / "outputs" / "validation_report.json",
        DATA_DIR / "errors" / "invalid_interactions.csv",
    )

    train, test = prepare(clean)
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)
    derive_user_metadata(train).to_csv(processed_dir / "users.csv", index=False)
    movies.to_csv(processed_dir / "items.csv", index=False)
    save_interaction_matrix(train, processed_dir)

    features_dir = DATA_DIR / "features"
    users, items, interaction_features = build_feature_tables(train, movies)
    feature_store = OfflineFeatureStore(features_dir)
    feature_store.write("users", users)
    feature_store.write("items", items)
    feature_store.write("interactions", interaction_features)
    feature_store.write("train_features", interaction_features)
    write_feature_catalog(ROOT / "docs" / "feature_catalog.csv")
    write_architecture_diagram(ROOT / "docs" / "pipeline_architecture.png")

    results, recommendation_frames = _train_and_evaluate(
        train, test, movies, ROOT / "models", ranking_k=10
    )
    comparison = pd.DataFrame(results)
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(outputs_dir / "model_comparison.csv", index=False)
    (outputs_dir / "model_comparison.json").write_text(
        comparison.to_json(orient="records", indent=2), encoding="utf-8"
    )
    pd.concat(recommendation_frames, ignore_index=True).to_csv(
        outputs_dir / "recommendations.csv", index=False
    )
    report_path = write_final_report(ROOT, comparison, quality, len(train), len(test))

    summary = {
        "status": "passed",
        "roll_number": ROLL_NUMBER,
        "validation": quality,
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_max_timestamp": int(train["timestamp"].max()),
            "test_min_timestamp": int(test["timestamp"].min()),
            "chronological": bool(train["timestamp"].max() < test["timestamp"].min()),
        },
        "models": results,
        "artifacts": {
            "validation": "outputs/validation_report.json",
            "comparison": "outputs/model_comparison.csv",
            "experiment_tracking": "outputs/experiment_tracking.jsonl",
            "feature_catalog": "docs/feature_catalog.csv",
            "final_report": str(report_path.relative_to(ROOT)),
        },
    }
    _write_json(outputs_dir / "metrics.json", summary)
    logger.info("Pipeline completed successfully.")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
