import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_pipeline_settings  # noqa: E402
from evaluation.evaluation import evaluate_model  # noqa: E402
from features.feature_management import OfflineFeatureStore  # noqa: E402
from ingestion.ingestion import DataIngestion  # noqa: E402
from preparation.preparation import prepare  # noqa: E402
from training.model_training import MeanBaseline, PopularityRecommender, SVDRecommender  # noqa: E402
from transformation.transformation import build_features  # noqa: E402
from validation.validation import validate_dataset  # noqa: E402


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

    def test_validation_quarantines_non_finite_values_instead_of_failing(self):
        ratings = pd.concat(
            [
                self.ratings,
                pd.DataFrame(
                    [
                        (float("inf"), 10, 4.0, 1100),
                        (1, float("inf"), 4.0, 1200),
                        (1, 10, 4.0, float("inf")),
                        (1, 10, float("inf"), 1300),
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
        self.assertEqual(report["invalid_data_type_counts"]["userId"], 1)
        self.assertEqual(report["invalid_data_type_counts"]["movieId"], 1)
        self.assertEqual(report["invalid_data_type_counts"]["timestamp"], 1)
        self.assertEqual(report["invalid_rating_count"], 1)
        self.assertTrue(quarantined["validation_errors"].str.len().gt(0).all())

    def test_split_is_strictly_chronological(self):
        train, test = prepare(self.ratings, test_size=0.3)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(len(train) + len(test), len(self.ratings))

    def test_pipeline_settings_load_and_validate(self):
        settings = load_pipeline_settings()
        self.assertEqual(settings["evaluation"]["k"], 10)
        self.assertEqual(settings["models"]["popularity"]["strategy"], "interaction_count")

    def test_ingestion_loads_user_profiles_from_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "raw" / "ml-latest-small"
            dataset_dir.mkdir(parents=True)
            self.ratings.to_csv(dataset_dir / "ratings.csv", index=False)
            self.movies.to_csv(dataset_dir / "movies.csv", index=False)
            self.links.to_csv(dataset_dir / "links.csv", index=False)
            pd.DataFrame(
                {"userId": [1, 2, 3], "region": ["north", "south", "west"]}
            ).to_csv(dataset_dir / "users.csv", index=False)

            ratings, movies, links, users = DataIngestion(temp_dir).load_all_data()

        self.assertEqual(len(ratings), len(self.ratings))
        self.assertEqual(len(movies), len(self.movies))
        self.assertEqual(len(links), len(self.links))
        self.assertEqual(users["region"].tolist(), ["north", "south", "west"])

    def test_ingestion_derives_user_activity_when_profiles_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "raw" / "ml-latest-small"
            dataset_dir.mkdir(parents=True)
            self.ratings.to_csv(dataset_dir / "ratings.csv", index=False)
            self.movies.to_csv(dataset_dir / "movies.csv", index=False)
            self.links.to_csv(dataset_dir / "links.csv", index=False)

            _, _, _, users = DataIngestion(temp_dir).load_all_data()

        self.assertEqual(set(users["userId"]), {1, 2, 3})
        self.assertIn("total_ratings", users.columns)
        self.assertIn("avg_rating", users.columns)

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
        self.assertEqual(features.iloc[0]["global_mean"], 3.0)
        self.assertEqual(features.iloc[1]["global_mean"], 5.0)

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

    def test_feature_store_serves_training_and_prediction_features(self):
        from transformation.transformation import build_feature_tables

        train = self.ratings.iloc[:8].copy()
        catalog = pd.concat(
            [
                self.movies,
                pd.DataFrame({"movieId": [13], "title": ["D"], "genres": ["Comedy"]}),
            ],
            ignore_index=True,
        )
        users, items, interactions = build_feature_tables(train, catalog)
        training_features, _ = build_features(train)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OfflineFeatureStore(temp_dir)
            store.write("users", users)
            store.write("items", items)
            store.write("interactions", interactions)
            store.write("train_features", training_features)

            stored_training = store.read_training_features()
            prediction_features = store.read_prediction_features(1, [13, 10])

        self.assertIn("user_prior_average_rating", stored_training.columns)
        self.assertIn("item_prior_rating_count", stored_training.columns)
        self.assertIn("interaction_hour_utc", stored_training.columns)
        self.assertEqual(prediction_features["movieId"].tolist(), [13, 10])
        self.assertIn("user_average_rating", prediction_features.columns)
        self.assertIn("preferred_category", prediction_features.columns)
        self.assertIn("days_since_last_interaction", prediction_features.columns)
        self.assertIn("item_popularity", prediction_features.columns)
        self.assertEqual(prediction_features.loc[0, "item_popularity"], 0)
        self.assertEqual(prediction_features.loc[0, "item_rating_count"], 0)

    def test_prediction_features_reject_unknown_entities_and_duplicate_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = OfflineFeatureStore(temp_dir)
            store.write("users", pd.DataFrame({"userId": [1], "feature": [0.5]}))
            store.write("items", pd.DataFrame({"movieId": [10], "feature": [1.0]}))
            with self.assertRaisesRegex(KeyError, "userId"):
                store.read_prediction_features(99, [10])
            with self.assertRaisesRegex(KeyError, "movieIds"):
                store.read_prediction_features(1, [99])
            with self.assertRaisesRegex(ValueError, "unique"):
                store.read_prediction_features(1, [10, 10])


if __name__ == "__main__":
    unittest.main()
