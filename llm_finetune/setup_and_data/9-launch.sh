#!/usr/bin/env bash
set -euo pipefail

echo $RANK 
NODE_RANK=$RANK

# --- configurable (via env or defaults) ---
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"                   # 0 for master, 1..NNODES-1 for workers
MASTER_ADDR="${MASTER_ADDR:-llama3-finetune-job-master-0}"  # K8s Service DNS or IP
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"

# Any extra args to your script can be passed via EXTRA_ARGS env or CLI:
# Example: EXTRA_ARGS="--epochs 3 --lr 3e-5"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[INFO] Starting torchrun:"
echo "  --nproc_per_node=${NPROC_PER_NODE}"
echo "  --nnodes=${NNODES}"
echo "  --node_rank=${NODE_RANK}"
echo "  --master_addr=${MASTER_ADDR}"
echo "  --master_port=${MASTER_PORT}"
echo "  script=${TRAIN_SCRIPT} ${EXTRA_ARGS}"

# Build command as an array (preserves arguments) and echo it in a shell-escaped form
cmd=(torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}")

# Split EXTRA_ARGS into words (if any) and append
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi

# Print the full command, with shell-quoting for clarity
printf '[INFO] Executing: '
printf '%q ' "${cmd[@]}"
printf '\n'

exec torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" ${EXTRA_ARGS}




#NODE_RANK=0 MASTER_ADDR=llama3-finetune-job-master-0 ./launch.sh
#NODE_RANK=1 MASTER_ADDR=llama3-finetune-job-master-0 ./launch.sh
#   NODE_RANK=1 MASTER_ADDR=llama3-finetune-sleep-job-master-0.default.svc.cluster.local ./launch.sh