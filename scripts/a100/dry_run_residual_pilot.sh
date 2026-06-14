#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
	cat <<'USAGE'
Dry-run the Qwen3-8B residual DFlash A100 pilot.

This script prints the git/tmux/GPU-check commands that should be run on the A100.
It does not start a long job.

Required environment for real execution later:
  REMOTE_REPO=/root/jungyo/throughput-aware-on-specedge-er-eal
  BRANCH=<user-approved-branch>
USAGE
	exit 0
fi

REMOTE_REPO="${REMOTE_REPO:-/root/jungyo/throughput-aware-on-specedge-er-eal}"
BRANCH="${BRANCH:-}"

if [[ -z "${BRANCH}" ]]; then
	echo "DRY RUN: BRANCH is not set; real remote update is intentionally blocked."
fi

cat <<COMMANDS
# A100 safety checks
ssh -p 1122 root@10.201.135.123 'nvidia-smi'
ssh -p 1122 root@10.201.135.123 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,memory.total,power.draw --format=csv,noheader,nounits'
ssh -p 1122 root@10.201.135.123 'pgrep -af "python|train|eval|a100_l5|loradrop|dflash"'

# Git-only update once BRANCH is user-approved
ssh -p 1122 root@10.201.135.123 'cd ${REMOTE_REPO} && git fetch origin && git checkout ${BRANCH:-<branch>} && git pull --ff-only origin ${BRANCH:-<branch>}'

# Long job must be launched from tmux attach -t jungyo-1
ssh -p 1122 root@10.201.135.123 'tmux attach -t jungyo-1'

# Command to run inside tmux
cd ${REMOTE_REPO}
export HF_HOME=/root/jungyo/hf_cache
export HF_HUB_CACHE=/root/jungyo/hf_cache/transformers
export HUGGINGFACE_HUB_CACHE=/root/jungyo/hf_cache/transformers
export TRANSFORMERS_CACHE=/root/jungyo/hf_cache/transformers
python -m dflash.benchmark \
  --backend transformers \
  --model Qwen/Qwen3-8B \
  --draft-model z-lab/Qwen3-8B-DFlash-b16 \
  --dataset gsm8k \
  --max-samples 128 \
  --block-size 16 \
  --enable-residual \
  --residual-budget 64 \
  --residual-tree-width 4
COMMANDS
