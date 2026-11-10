"""Local experiment record and model-artifact utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def track_experiment(
    root: str | Path,
    name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifact: str | Path,
) -> dict[str, Any]:
    """Append one reproducible model run record to JSON Lines."""
    project_root = Path(root)
    artifact_path = Path(artifact)
    record = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "model": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "metrics": metrics,
        "artifacts": [str(artifact_path.relative_to(project_root))],
    }
    tracking_path = project_root / "outputs" / "experiment_tracking.jsonl"
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    with tracking_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    return record
