# pyright: reportMissingImports=none
import json
from pathlib import Path

from dflash.residual_report import generate_residual_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_report_summarizes_strong_fair_pilot(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics,
        [
            {
                "baseline_tps": 100.0,
                "residual_tps": 108.0,
                "trigger_rate": 0.35,
                "residual_accepted_mean": 0.6,
                "predicted_gain_mean": 0.7,
                "realized_gain_mean": 0.6,
                "fair_eval": True,
            }
        ],
    )

    report = generate_residual_report(
        metrics, command="python -m dflash.benchmark --enable-residual"
    )

    assert "Decision: strong" in report
    assert "Speedup: 8.00%" in report
    assert "Trigger rate: 0.350" in report
    assert str(metrics) in report
    assert "python -m dflash.benchmark --enable-residual" in report


def test_report_marks_unfair_wallclock_as_not_paper_valid(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics,
        [
            {
                "baseline_tps": 100.0,
                "residual_tps": 120.0,
                "trigger_rate": 0.50,
                "residual_accepted_mean": 1.0,
                "predicted_gain_mean": 1.1,
                "realized_gain_mean": 1.0,
                "fair_eval": False,
            }
        ],
    )

    report = generate_residual_report(metrics, command="cmd")

    assert "Decision: weak" in report
    assert "not paper-valid" in report


def test_report_aggregates_multiple_rows(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics,
        [
            {
                "baseline_tps": 100.0,
                "residual_tps": 102.0,
                "trigger_rate": 0.2,
                "fair_eval": True,
            },
            {
                "baseline_tps": 100.0,
                "residual_tps": 104.0,
                "trigger_rate": 0.4,
                "fair_eval": True,
            },
        ],
    )

    report = generate_residual_report(
        metrics, command="cmd", strong_speedup=0.05, moderate_speedup=0.01
    )

    assert "Decision: moderate" in report
    assert "Speedup: 3.00%" in report
    assert "Trigger rate: 0.300" in report
