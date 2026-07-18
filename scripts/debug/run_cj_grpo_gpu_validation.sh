#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <sdar|llada2> <model-path> [output-dir]" >&2
    exit 2
fi

model_kind=$1
model_path=$2
output_dir=${3:-tmp/cj_grpo_gpu_validation/${model_kind}}

if [[ "${model_kind}" != "sdar" && "${model_kind}" != "llada2" ]]; then
    echo "model kind must be sdar or llada2, got: ${model_kind}" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
mkdir -p "${output_dir}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

# Step 3: FULL_SHARD ranks receive different CJ active-step patterns.  The
# process-group timeout turns a collective mismatch into an explicit failure.
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    scripts/debug/check_cj_grpo_fsdp_schedule.py \
    --model-kind "${model_kind}" \
    --model-path "${model_path}" \
    --output "${output_dir}/fsdp_schedule.json"

# Step 4: compare the six-transition serial oracle, three-forward grouped
# replay, and SP=2 grouped replay with nonzero PPO gradients.
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    scripts/debug/compare_cj_grpo_sp_numerics.py \
    --model-kind "${model_kind}" \
    --model-path "${model_path}" \
    --output "${output_dir}/sp_numerics.json"

echo "CJ-GRPO GPU validation results: ${output_dir}"
