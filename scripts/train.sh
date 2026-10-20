#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
config="${1:-configs/pi05_piper.py}"
work_dir="${2:-work_dirs/pi05_piper}"
if (( $# >= 2 )); then shift 2; else shift "$#"; fi
exec "${PYTHON:-python}" -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE:-1}" \
  --nnodes="${NNODES:-1}" \
  --node-rank="${NODE_RANK:-0}" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-29500}" \
  scripts/train.py --config "$config" --work-dir "$work_dir" "$@"
