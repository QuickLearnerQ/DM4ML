"""Pipeline documentation artifacts generated from executed workflow results."""

from pathlib import Path

import pandas as pd

FEATURE_CATALOG = [
    ("user_activity_count", "user", "Training interactions at the model cutoff", "count"),
    ("user_average_rating", "user", "Mean rating over training interactions", "rating"),
    ("preferred_category", "user", "Most-interacted movie genre in training data", "category"),
    ("days_since_last_interaction", "user", "Days from last training event to cutoff", "days"),
    ("item_popularity", "item", "Number of training interactions for the item", "count"),
    ("item_average_rating", "item", "Mean training rating for the item", "rating"),
    ("item_rating_count", "item", "Training rating count for the item", "count"),
    ("interaction_hour_utc", "interaction", "UTC hour derived from event timestamp", "hour"),
    (
        "interaction_weekday_utc",
        "interaction",
        "UTC weekday derived from event timestamp",
        "weekday",
    ),
]
GITHUB_REPOSITORY_URL = "https://github.com/QuickLearnerQ/DM4ML"


def write_feature_catalog(path: str | Path) -> Path:
    """Write feature definitions in a machine-readable table."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(FEATURE_CATALOG, columns=["feature", "entity", "description", "unit"]).to_csv(
        destination, index=False
    )
    return destination


def write_architecture_diagram(path: str | Path) -> Path:
    """Render the current pipeline stages to a PNG architecture diagram."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stages = [
        "Ingest",
        "Validate +\nquarantine",
        "Time split",
        "Feature\nsnapshots",
        "Train 2\nmodels",
        "Evaluate",
        "Track +\nreport",
        "Schedule",
    ]
    figure, axis = plt.subplots(figsize=(16, 3.5))
    axis.set_xlim(0, len(stages))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, stage in enumerate(stages):
        x_position = index + 0.08
        axis.add_patch(
            FancyBboxPatch(
                (x_position, 0.35),
                0.82,
                0.32,
                boxstyle="round,pad=0.03",
                facecolor="#e8f1fb",
                edgecolor="#315a7d",
                linewidth=1.5,
            )
        )
        axis.text(x_position + 0.41, 0.51, stage, ha="center", va="center", fontsize=9)
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(index + 1.05, 0.51),
                xytext=(index + 0.92, 0.51),
                arrowprops={"arrowstyle": "->", "color": "#315a7d", "lw": 1.5},
            )
    axis.text(
        4,
        0.13,
        "Parquet features | mean baseline + truncated SVD | ranking and rating metrics",
        ha="center",
        fontsize=10,
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_final_report(
    root: str | Path,
    results: pd.DataFrame,
    quality: dict,
    train_count: int,
    test_count: int,
) -> Path:
    """Create the three-page final project PDF from run results."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    project_root = Path(root)
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "final_report.pdf"
    pages = [
        (
            "Recommendation Pipeline - Design",
            [
                (
                    "Dataset: 100,000 public MovieLens interactions from the "
                    "ashraq/movielens_ratings mirror, plus its movie catalog."
                ),
                (
                    "The mirror has no authoritative event timestamps; timestamps are "
                    "deterministically generated (NumPy seed 42)."
                ),
                (
                    f"Interactions retained: {quality['valid_interactions']:,}; "
                    f"quarantined: {quality['quarantined_interactions']:,}."
                ),
                (
                    f"Chronological holdout: {train_count:,} older training interactions, "
                    f"{test_count:,} newer test interactions."
                ),
                (
                    "Stages: ingest, validate/quarantine, chronological preparation, "
                    "feature snapshots, two models, and ranking evaluation."
                ),
                (
                    "Features are stored as Parquet entity snapshots; the interaction matrix "
                    "is stored as sparse NPZ."
                ),
                (
                    "Models: interaction-count popularity baseline and personalized "
                    "truncated-SVD collaborative filtering."
                ),
                f"GitHub repository: {GITHUB_REPOSITORY_URL}",
            ],
        ),
        (
            "Recommendation Pipeline - Results",
            [
                "Ranking metrics use K=10 and held-out ratings of at least four stars.",
                "Ranking uses a deterministic sample of up to 1,000 relevant users.",
                (
                    "Coverage is the recommended share of the item catalog; diversity is "
                    "pairwise genre Jaccard distance."
                ),
                "The full comparison table is available in outputs/model_comparison.csv.",
            ],
        ),
        (
            "Recommendation Pipeline - Limitations",
            [
                "A time-based split better approximates future interactions than a random split.",
                "Invalid rows are retained in data/errors with rejection reasons.",
                (
                    "JSONL tracking stores model parameters, metrics, and artifact paths "
                    "without a tracking server."
                ),
                (
                    "The Python scheduler can execute this workflow once or at a configured "
                    "interval; an optional daily Airflow DAG runs the same workflow."
                ),
                (
                    "MovieLens cold-start users/items, popularity bias, and offline metrics "
                    "limit generalization."
                ),
                (
                    "Mirror timestamps are synthetic when the source mirror does not "
                    "provide event times; temporal evaluation does not represent real "
                    "event chronology for this input."
                ),
            ],
        ),
    ]
    with PdfPages(report_path) as pdf:
        for page_number, (title, bullets) in enumerate(pages):
            figure, axis = plt.subplots(figsize=(8.27, 11.69))
            axis.axis("off")
            axis.text(0.06, 0.94, title, fontsize=18, weight="bold", va="top")
            if page_number == 1:
                columns = [
                    "model",
                    "RMSE",
                    "MAE",
                    "Precision@10",
                    "Recall@10",
                    "NDCG@10",
                    "Coverage",
                    "Diversity",
                ]
                table = results[columns]
                axis.table(
                    cellText=table.round(4).values,
                    colLabels=table.columns,
                    loc="upper center",
                    bbox=[0.05, 0.63, 0.9, 0.22],
                    cellLoc="center",
                )
            y_position = 0.55 if page_number == 1 else 0.84
            for bullet in bullets:
                text = axis.text(0.08, y_position, f"- {bullet}", fontsize=11, va="top", wrap=True)
                if bullet.startswith("GitHub repository:"):
                    text.set_url(GITHUB_REPOSITORY_URL)
                y_position -= 0.09
            axis.text(
                0.06, 0.03, f"Roll number: 2025AG05563 | Page {page_number + 1} of 3", fontsize=9
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return report_path
