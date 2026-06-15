#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-results/full_balanced_residual_eval_$(date +%Y%m%d_%H%M%S)}

python scripts/a100/run_fair_residual_benchmark.py \
	--datasets gsm8k math500 aime25 humaneval mbpp livecodebench mt_bench alpaca \
	--dataset-samples \
	gsm8k=110 \
	math500=110 \
	aime25=30 \
	humaneval=110 \
	mbpp=110 \
	livecodebench=110 \
	mt_bench=80 \
	alpaca=110 \
	--round-robin-datasets \
	--start-with-residual \
	--max-new-tokens 256 \
	--residual-budget 128 \
	--residual-tree-width 5 \
	--residual-gain-scale 0.6 \
	--residual-min-gain 3.0 \
	--residual-min-margin 0.2 \
	--residual-draft-seconds 0.0078 \
	--residual-target-seconds 0.0520 \
	--output-dir "${OUTPUT_DIR}"
