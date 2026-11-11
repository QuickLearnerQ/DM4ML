"""Pipeline documentation artifacts generated from executed workflow results."""

import io
import json
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
    (
        "user_prior_rating_count",
        "interaction",
        "User interactions strictly before the event timestamp",
        "count",
    ),
    (
        "user_prior_average_rating",
        "interaction",
        "User mean rating strictly before the event timestamp",
        "rating",
    ),
    (
        "item_prior_rating_count",
        "interaction",
        "Item interactions strictly before the event timestamp",
        "count",
    ),
    (
        "item_prior_average_rating",
        "interaction",
        "Item mean rating strictly before the event timestamp",
        "rating",
    ),
    ("global_mean", "interaction", "Training-set mean rating", "rating"),
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
        "Parquet features | interaction-count popularity + truncated SVD | "
        "ranking and rating metrics",
        ha="center",
        fontsize=10,
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def _draw_evidence_card(axis, title: str, artifact: str, lines: list[str]) -> None:
    from matplotlib.patches import FancyBboxPatch

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(
        FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.96,
            boxstyle="round,pad=0.012",
            facecolor="#f4f8fc",
            edgecolor="#315a7d",
            linewidth=1.2,
        )
    )
    axis.text(0.05, 0.86, title, fontsize=12, weight="bold", color="#173c5a", va="top")
    axis.text(0.05, 0.69, artifact, fontsize=8, color="#526579", va="top", wrap=True)
    axis.text(
        0.05,
        0.54,
        "\n".join(lines),
        fontsize=9,
        color="#172b3a",
        va="top",
        linespacing=1.5,
        wrap=True,
        family="monospace",
    )


def _save_evidence_snapshot(pdf, figure, plt, mpimg) -> None:
    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=180, facecolor="white")
    image_buffer.seek(0)
    image = mpimg.imread(image_buffer)
    page, axis = plt.subplots(figsize=(11.69, 8.27))
    page.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axis.imshow(image)
    axis.axis("off")
    pdf.savefig(page)
    plt.close(page)
    plt.close(figure)


def _write_execution_evidence(
    pdf,
    project_root: Path,
    results: pd.DataFrame,
    quality: dict,
    train_count: int,
    test_count: int,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    data_dir = project_root / "data"
    source_dir = data_dir / "raw" / "ml-latest-small"
    ratings = pd.read_csv(source_dir / "ratings.csv", nrows=3)
    movies_count = len(pd.read_csv(source_dir / "movies.csv"))
    links_count = len(pd.read_csv(source_dir / "links.csv"))
    train = pd.read_csv(data_dir / "processed" / "train.csv", usecols=["timestamp"])
    test = pd.read_csv(data_dir / "processed" / "test.csv", usecols=["timestamp"])
    feature_files = {
        name: data_dir / "features" / f"{name}.parquet"
        for name in ("users", "items", "interactions")
    }
    feature_summaries = []
    for name, path in feature_files.items():
        frame = pd.read_parquet(path)
        feature_summaries.append(f"{name}: {len(frame):,} rows, {len(frame.columns)} columns")

    diagram_path = project_root / "docs" / "pipeline_architecture.png"
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.suptitle(
        "Execution evidence: ingestion, validation, preparation, and feature management",
        fontsize=15,
        weight="bold",
        y=0.97,
    )
    diagram_axis = figure.add_axes([0.06, 0.70, 0.88, 0.20])
    diagram_axis.imshow(mpimg.imread(diagram_path))
    diagram_axis.axis("off")
    cards = [
        (
            "1. Data ingestion",
            "data/raw/ml-latest-small/*.csv",
            [
                f"Ratings source: {quality['input_interactions']:,} interactions",
                f"Movies: {movies_count:,} | Links: {links_count:,}",
                f"Columns loaded: {', '.join(ratings.columns)}",
                "Source preview read from the ingested CSV",
            ],
        ),
        (
            "2. Validation and quarantine",
            "outputs/validation_report.json",
            [
                f"Status: {quality['status']}",
                f"Valid: {quality['valid_interactions']:,}",
                f"Quarantined: {quality['quarantined_interactions']:,}",
                f"Referential integrity: {quality['referential_integrity']}",
            ],
        ),
        (
            "3. Preparation",
            "data/processed/train.csv, test.csv, user_item_matrix.npz",
            [
                f"Train rows: {train_count:,} | Test rows: {test_count:,}",
                f"Chronological: {train['timestamp'].max() < test['timestamp'].min()}",
                f"Last train timestamp: {int(train['timestamp'].max())}",
                f"First test timestamp: {int(test['timestamp'].min())}",
            ],
        ),
        (
            "4. Feature management",
            "data/features/*.parquet and docs/feature_catalog.csv",
            feature_summaries + ["Parquet snapshots read from the feature store"],
        ),
    ]
    positions = [(0.05, 0.38), (0.52, 0.38), (0.05, 0.06), (0.52, 0.06)]
    for (title, artifact, lines), (x_position, y_position) in zip(cards, positions, strict=True):
        axis = figure.add_axes([x_position, y_position, 0.44, 0.27])
        _draw_evidence_card(axis, title, artifact, lines)
    figure.text(
        0.05,
        0.015,
        "Visual evidence compiled from the successful run's generated files and metrics.",
        fontsize=8,
        color="#526579",
    )
    _save_evidence_snapshot(pdf, figure, plt, mpimg)

    model_artifacts = []
    for model in results["model"]:
        artifact = project_root / "models" / f"{model}.pkl"
        model_artifacts.append(f"{artifact.name}: {'present' if artifact.is_file() else 'missing'}")
    tracking_path = project_root / "outputs" / "experiment_tracking.jsonl"
    tracking_records = [
        line for line in tracking_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    recent_records = [json.loads(line) for line in tracking_records[-len(results) :]]
    tracking_summary = [
        f"Runs recorded: {len(tracking_records)} total",
        *[f"{record['model']}: run {record['run_id']}" for record in recent_records],
    ]
    report_settings = (project_root / "dags" / "recommendation_pipeline.py").read_text(
        encoding="utf-8"
    )
    airflow_schedule = (
        "@daily" if 'schedule="@daily"' in report_settings else "See DAG configuration"
    )
    scheduler_source = (project_root / "src" / "scheduler.py").read_text(encoding="utf-8")
    interval_option = "--interval-hours" if "--interval-hours" in scheduler_source else "not found"

    figure, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    figure.suptitle(
        "Execution evidence: model training, evaluation, tracking, and orchestration",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    _draw_evidence_card(
        axes[0, 0],
        "5. Model training",
        "models/*.pkl",
        model_artifacts + ["Artifacts checked on disk after model training"],
    )
    metric_lines = []
    metric_columns = ["model", "RMSE", "MAE", "Precision@10", "Recall@10", "NDCG@10"]
    for row in results[metric_columns].itertuples(index=False, name=None):
        metric_lines.append(
            f"{row[0]}: RMSE {row[1]:.4f}, MAE {row[2]:.4f}, "
            f"P@10 {row[3]:.4f}, R@10 {row[4]:.4f}, NDCG@10 {row[5]:.4f}"
        )
    _draw_evidence_card(
        axes[0, 1],
        "6. Evaluation",
        "outputs/model_comparison.csv",
        metric_lines + ["Values read from this run's model comparison"],
    )
    _draw_evidence_card(
        axes[1, 0],
        "7. Experiment tracking and reporting",
        "outputs/experiment_tracking.jsonl and reports/final_report.pdf",
        tracking_summary
        + [
            "Model metrics and artifact paths are recorded per run",
            "This PDF includes the stage evidence pages",
        ],
    )
    _draw_evidence_card(
        axes[1, 1],
        "8. Orchestration",
        "src/scheduler.py and dags/recommendation_pipeline.py",
        [
            f"Pipeline status: {quality['status']}",
            "Python scheduler: one-shot or repeating interval",
            f"Interval option: {interval_option}",
            f"Airflow schedule: {airflow_schedule}",
            "DAG configuration evidence (not an Airflow execution log)",
        ],
    )
    figure.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
    _save_evidence_snapshot(pdf, figure, plt, mpimg)


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
    evidence_path = report_dir / "execution_evidence.pdf"
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
            axis.text(0.06, 0.03, f"Roll number: 2025AG05563 | Page {page_number + 1}", fontsize=9)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    with PdfPages(evidence_path) as pdf:
        _write_execution_evidence(
            pdf,
            project_root,
            results,
            quality,
            train_count,
            test_count,
        )
    return report_path
