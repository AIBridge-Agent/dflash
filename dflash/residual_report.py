# pyright: reportMissingImports=none
"""Pilot report generation for residual DFlash metrics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no metrics rows found in {metrics_path}")
    return rows


def _avg(rows: list[dict[str, Any]], field: str, default: float = 0.0) -> float:
    return mean(float(row.get(field, default)) for row in rows)


def generate_residual_report(
    metrics_path: str | Path,
    *,
    command: str,
    strong_speedup: float = 0.05,
    moderate_speedup: float = 0.0,
) -> str:
    """Generate a paper-facing residual pilot report from JSONL metrics."""

    path = Path(metrics_path)
    rows = _load_rows(path)
    baseline_tps = _avg(rows, "baseline_tps")
    residual_tps = _avg(rows, "residual_tps")
    speedup = (residual_tps / baseline_tps - 1.0) if baseline_tps > 0 else 0.0
    trigger_rate = _avg(rows, "trigger_rate")
    residual_accepted = _avg(rows, "residual_accepted_mean")
    predicted_gain = _avg(rows, "predicted_gain_mean")
    realized_gain = _avg(rows, "realized_gain_mean")
    fair_eval = all(bool(row.get("fair_eval", False)) for row in rows)

    if not fair_eval:
        decision = "weak"
        fairness_note = "Wall-clock evidence is not paper-valid because fair_eval is false or missing."
    elif speedup >= strong_speedup:
        decision = "strong"
        fairness_note = "Wall-clock evidence is marked fair by the metrics log."
    elif speedup >= moderate_speedup and speedup > 0:
        decision = "moderate"
        fairness_note = "Wall-clock evidence is marked fair, but speedup is below the strong threshold."
    else:
        decision = "weak"
        fairness_note = "Residual policy did not produce a positive fair speedup."

    return "\n".join(
        [
            "# Residual DFlash Pilot Report",
            "",
            f"Decision: {decision}",
            f"Metrics: {path}",
            f"Command: `{command}`",
            "",
            f"Baseline TPS: {baseline_tps:.3f}",
            f"Residual TPS: {residual_tps:.3f}",
            f"Speedup: {speedup:.2%}",
            f"Trigger rate: {trigger_rate:.3f}",
            f"Residual accepted mean: {residual_accepted:.3f}",
            f"Predicted gain mean: {predicted_gain:.3f}",
            f"Realized gain mean: {realized_gain:.3f}",
            "",
            f"Fairness: {fairness_note}",
            "",
        ]
    )
