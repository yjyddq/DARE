#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_coupled_grpo_llada_8b_instruct.sh" "$@" --algorithm bridgeratio-grpo
