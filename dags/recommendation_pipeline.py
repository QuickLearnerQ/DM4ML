"""Optional Airflow entry point for the dependency-ordered recommendation workflow."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dag(
    dag_id="recommendation_pipeline",
    description="Run the complete configured MovieLens recommendation pipeline.",
    schedule="@daily",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["recommendation", "data-management"],
)
def recommendation_pipeline():
    @task
    def run_pipeline() -> dict:
        from pipeline import run

        return run()

    run_pipeline()


recommendation_pipeline()
