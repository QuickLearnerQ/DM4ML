import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_pipeline_settings  # noqa: E402
from evaluation import evaluate_model  # noqa: E402
from feature_management import OfflineFeatureStore  # noqa: E402
from model_training import MeanBaseline, PopularityRecommender, SVDRecommender  # noqa: E402
from preparation import prepare  # noqa: E402
from transformation import build_features  # noqa: E402
from validation import validate_dataset  # noqa: E402


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            [
                (1, 10, 5.0, 100),
                (1, 11, 4.0, 200),
                (2, 10, 4.5, 300),
                (2, 12, 3.0, 400),
                (3, 11, 4.0, 500),
                (3, 12, 2.5, 600),
                (1, 12, 4.5, 700),
                (2, 11, 3.5, 800),
                (3, 10, 4.0, 900),
                (1, 10, 4.5, 1000),
            ],
            columns=["userId", "movieId", "rating", "timestamp"],
        )
        self.movies = pd.DataFrame(
            [
                (10, "A", "Action"),
                (11, "B", "Drama"),
                (12, "C", "Action|Drama"),
            ],
            columns=["movieId", "title", "genres"],
        )
        self.links = pd.DataFrame({"movieId": [10, 11, 12]})

    def test_validation_quarantines_bad_rows_and_keeps_valid_rows(self):
        ratings = pd.concat(
            [
                self.ratings,
                pd.DataFrame(
                    [
                        (2, 12, 6.0, 1100),
                        (4, 99, 4.0, 1200),
                        (2, 12, 4.0, "not-a-time"),
                        (3, 10, 4.0, 900),
                    ],
                    columns=self.ratings.columns,
                ),
            ],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid, report = validate_dataset(
                ratings, self.movies, self.links, root / "report.json", root / "errors.csv"
            )
            quarantined = pd.read_csv(root / "errors.csv")
        self.assertEqual(len(valid), len(self.ratings))
        self.assertEqual(report["quarantined_interactions"], 4)
        self.assertEqual(len(quarantined), 4)
        self.assertIn("validation_errors", quarantined.columns)
        self.assertEqual(valid["rating"].dtype, "float64")

    def test_validation_quarantines_missing_values_and_incorrect_types(self):
        ratings = pd.concat(
            [
                self.ratings,
                pd.DataFrame(
                    [
                        (1, 10, None, 1100),
                        ("unknown-user", 10, 4.0, 1200),
                    ],
                    columns=self.ratings.columns,
                ),
            ],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid, report = validate_dataset(
                ratings, self.movies, self.links, root / "report.json", root / "errors.csv"
            )
            quarantined = pd.read_csv(root / "errors.csv")
        self.assertEqual(len(valid), len(self.ratings))
        self.assertEqual(report["quarantined_interactions"], 2)
        self.assertEqual(report["invalid_data_type_counts"]["userId"], 1)
        self.assertEqual(len(quarantined), 2)

    def test_split_is_strictly_chronological(self):
        train, test = prepare(self.ratings, test_size=0.3)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(len(train) + len(test), len(self.ratings))

    def test_pipeline_settings_load_and_validate(self):
        settings = load_pipeline_settings()
        self.assertEqual(settings["evaluation"]["k"], 10)
        self.assertEqual(settings["models"]["popularity"]["strategy"], "interaction_count")

    def test_popularity_model_ranks_by_interaction_count(self):
        model = PopularityRecommender().fit(self.ratings)
        scores = model.score_items(1, [11, 12, 10, 99])
        self.assertGreater(scores[2], scores[0])
        self.assertGreater(scores[2], scores[1])
        self.assertEqual(scores[3], 0)
        self.assertEqual(model.predict(pd.DataFrame({"movieId": [10, 99]}))[1], model.global_mean)

    def test_both_models_return_all_required_metrics(self):
        train, test = prepare(self.ratings, test_size=0.3)
        required = {
            "RMSE",
            "MAE",
            "Precision@5",
            "Recall@5",
            "NDCG@5",
            "Coverage",
            "Diversity",
        }
        for model in (
            PopularityRecommender(),
            MeanBaseline(),
            SVDRecommender(n_components=2),
        ):
            metrics, recommendations = evaluate_model(
                model.fit(train), train, test, self.movies, k=5
            )
            self.assertTrue(required.issubset(metrics))
            self.assertTrue(recommendations.empty or recommendations["rank"].max() <= 5)

    def test_feature_aggregates_use_only_previous_interactions(self):
        interactions = pd.DataFrame(
            [
                (1, 10, 5.0, 100),
                (1, 11, 1.0, 200),
                (1, 12, 2.0, 200),
                (1, 12, 4.0, 300),
            ],
            columns=["userId", "movieId", "rating", "timestamp"],
        )
        features, _ = build_features(interactions)
        first_event = features.iloc[0]
        second_event = features.iloc[1]
        self.assertEqual(first_event["user_prior_rating_count"], 0)
        self.assertEqual(first_event["user_prior_average_rating"], 3.0)
        self.assertEqual(second_event["user_prior_rating_count"], 1)
        self.assertEqual(second_event["user_prior_average_rating"], 5.0)
        self.assertEqual(features.iloc[2]["user_prior_rating_count"], 1)
        self.assertEqual(features.iloc[2]["user_prior_average_rating"], 5.0)
        self.assertEqual(features.iloc[3]["user_prior_rating_count"], 3)

    def test_vectorized_baseline_scores_match_single_predictions(self):
        model = MeanBaseline().fit(self.ratings)
        candidates = [10, 11, 12, 99]
        scores = model.score_items(1, candidates)
        expected = [model.predict_one(1, item) for item in candidates]
        self.assertEqual(scores.tolist(), expected)

    def test_svd_scores_match_single_predictions_including_cold_item(self):
        model = SVDRecommender(n_components=2).fit(self.ratings)
        candidates = [10, 11, 12, 99]
        scores = model.score_items(1, candidates)
        expected = [model.predict_one(1, item) for item in candidates]
        for actual, prediction in zip(scores, expected, strict=True):
            self.assertAlmostEqual(float(actual), prediction)

    def test_feature_store_round_trip_and_rejects_path_traversal(self):
        frame = pd.DataFrame({"userId": [1], "feature": [0.5]})
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OfflineFeatureStore(temp_dir)
            store.write("users", frame)
            pd.testing.assert_frame_equal(store.read("users"), frame)
            with self.assertRaises(ValueError):
                store.write("../outside", frame)


if __name__ == "__main__":
    unittest.main()
