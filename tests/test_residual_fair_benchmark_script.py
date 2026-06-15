from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "a100"
    / "run_fair_residual_benchmark.py"
)


def test_fair_benchmark_runs_all_dflash_datasets() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for dataset in [
        "gsm8k",
        "math500",
        "aime25",
        "humaneval",
        "mbpp",
        "livecodebench",
        "mt_bench",
        "alpaca",
    ]:
        assert dataset in source


def test_fair_benchmark_alternates_policy_order_and_resets_cuda_cache() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "def policy_order" in source
    assert "pair_index % 2" in source
    assert "start_with_residual" in source
    assert "tuple(reversed(POLICIES))" in source
    assert "def reset_gpu_state" in source
    assert "torch.cuda.synchronize()" in source
    assert "torch.cuda.empty_cache()" in source
    assert "torch.cuda.ipc_collect()" in source
    assert "cuda_cache_reset_before_each_policy" in source


def test_fair_benchmark_supports_balanced_round_robin_rollout() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--dataset-samples" in source
    assert "--round-robin-datasets" in source
    assert "build_round_robin_plan" in source
    assert "pair_index" in source


def test_fair_benchmark_compares_only_block16_policies() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "dflash_block16" in source
    assert "dflash_block16_residual" in source
    assert "block_size=16" in source
    assert "baseline_block1" not in source


def test_fair_benchmark_logs_gate_diagnostics() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "residual_gate_records" in source
    assert "residual_edge_records" in source
    assert "residual_edge_accept_rate" in source
    assert "mean_edge_probability" in source
    assert "gate_off_count" in source
    assert "mean_delta_hat" in source
    assert "mean_theta_base" in source
    assert "mean_theta_res_hat" in source
    assert "mean_effective_delta_hat" in source
    assert "mean_t_res" in source
    assert "residual_gain_scale" in source
    assert "residual_min_gain" in source
    assert "residual_min_margin" in source
