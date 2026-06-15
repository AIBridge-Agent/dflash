from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "dflash" / "benchmark.py"
CONFIG = ROOT / "configs" / "qwen3_8b_residual_a100.yaml"
SCRIPT = ROOT / "scripts" / "a100" / "dry_run_residual_pilot.sh"


def test_benchmark_exposes_residual_flags_and_passes_them_to_generate() -> None:
    source = BENCHMARK.read_text(encoding="utf-8")

    for flag in [
        "--enable-residual",
        "--residual-budget",
        "--residual-draft-seconds",
        "--residual-target-seconds",
        "--residual-diagnostics",
    ]:
        assert flag in source

    assert "enable_residual=args.enable_residual" in source
    assert "residual_budget=args.residual_budget" in source
    assert "residual_collect_diagnostics=args.residual_diagnostics" in source


def test_qwen3_8b_residual_config_records_dataset_schema_and_fair_eval() -> None:
    text = CONFIG.read_text(encoding="utf-8")

    assert "Qwen/Qwen3-8B" in text
    assert "z-lab/Qwen3-8B-DFlash-b16" in text
    assert "dataset: gsm8k" in text
    assert "hf_dataset_schema_verified: true" in text
    assert "policy_order_mode: alternate_by_sample" in text
    assert "tail_record_detail: minimal" in text


def test_a100_dry_run_script_is_git_only_and_tmux_safe() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "rsync" not in text
    assert "scp" not in text
    assert "git" in text
    assert "tmux attach -t jungyo-1" in text
    assert "nvidia-smi" in text
    assert "pgrep -af" in text
    assert "/root/jungyo/hf_cache/transformers" in text
    assert "--enable-residual" in text
