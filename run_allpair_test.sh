#!/usr/bin/env bash
set -euo pipefail

# -------------------------- CONFIG (edit) --------------------------
HOSTFILE="${HOSTFILE:-/opt/hostfile}"
GEN_SCRIPT="${GEN_SCRIPT:-validation/allpair/generate_permutations.py}"  # path to your Python generator
NPERNODE="${NPERNODE:-8}"         # processes per node
NP_TOTAL="${NP_TOTAL:-$((2*NPERNODE))}"   # -np (total ranks across both nodes)
LOGDIR="${LOGDIR:-allpairs_logs}" # where per-pair logs go
MASTER_PORT_BASE="${MASTER_PORT_BASE:-45566}" # will use BASE + job_idx per round
EXTRA_MPI_ARGS="${EXTRA_MPI_ARGS:-}" # e.g., "--mca pml ucx --mca btl ^openib"
# Example NCCL/other envs; add/remove as needed:
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export LOCAL_WORLD="${LOCAL_WORLD:-$NPERNODE}"

APP_CMD=(
  python ~/npairs/npairs.py
)
# ------------------------------------------------------------------

mkdir -p "$LOGDIR"

# Cleanly stop all children on exit/ctrl-c
cleanup() {
  pkill -P $$ || true
}
trap cleanup EXIT INT TERM

# -------------------------- INPUTS --------------------------

# 1) Load nodes from hostfile (first column)
if [[ ! -f "$HOSTFILE" ]]; then
  echo "ERROR: Hostfile not found at: $HOSTFILE" >&2
  exit 1
fi

# shellcheck disable=SC2207
mapfile -t NODES < <(awk '{print $1}' "$HOSTFILE" | sed '/^\s*$/d')

N="${#NODES[@]}"
if (( N < 2 )); then
  echo "ERROR: Need at least 2 nodes in $HOSTFILE; found $N" >&2
  exit 1
fi

echo "Loaded $N nodes from $HOSTFILE"
# Optional: show them
# printf '  %s\n' "${NODES[@]}"

# 2) Generate the per-round all-pair schedule and load into combinations[]
#    We strip the leading spaces and surrounding quotes to get a clean line.
if [[ ! -x "$GEN_SCRIPT" && ! -f "$GEN_SCRIPT" ]]; then
  echo "ERROR: Generator script not found: $GEN_SCRIPT" >&2
  exit 1
fi

# shellcheck disable=SC2207
mapfile -t combinations < <(
  python3 "$GEN_SCRIPT" --nitems "$N" --format text \
    | sed 's/^[[:space:]]*//; s/^"//; s/"$//'
)

if (( ${#combinations[@]} == 0 )); then
  echo "ERROR: No combinations produced by generator." >&2
  exit 1
fi

echo "Schedule has ${#combinations[@]} rounds; ~$(($N/2)) pairs per round."

# -------------------------- RUN ROUNDS --------------------------
round_idx=0
for combo in "${combinations[@]}"; do
  echo
  echo "=== Round $round_idx ==="
  # combo format: '0 9 | 1 8 | 2 7 | ...'
  IFS='|' read -r -a pairs <<< "$combo"

  job_idx=0
  pids=()

  for pair in "${pairs[@]}"; do
    # trim spaces & split indices
    pair=$(echo "$pair" | xargs)
    # shellcheck disable=SC2206
    idx=($pair)
    if (( ${#idx[@]} != 2 )); then
      echo "WARN: malformed pair: '$pair' (skipping)" >&2
      continue
    fi

    i="${idx[0]}"
    j="${idx[1]}"
    node1="${NODES[$i]}"
    node2="${NODES[$j]}"

    if [[ -z "${node1:-}" || -z "${node2:-}" ]]; then
      echo "WARN: index out of range in pair '$pair' (skipping)" >&2
      continue
    fi

    # Unique-ish port per job in a round (not strictly required since nodes
    # do not repeat within a round, but keeps things tidy)
    master_port=$((MASTER_PORT_BASE + job_idx))

    log_file="${LOGDIR}/round${round_idx}_job${job_idx}_${node1}--${node2}.log"
    echo "Launching Job${job_idx}: $node1 & $node2  -> $log_file"

    # Kick off MPI job in background
    mpirun --tag-output --display-map --allow-run-as-root \
      -np "$NP_TOTAL" \
      -H "${node1}:${NPERNODE},${node2}:${NPERNODE}" \
      -x LOCAL_WORLD \
      -x NCCL_DEBUG \
      -x MASTER_ADDR="${node1}" \
      -x MASTER_PORT="${master_port}" \
      ${EXTRA_MPI_ARGS} \
      "${APP_CMD[@]}" \
      >"$log_file" 2>&1 &

    pids+=($!)
    ((job_idx++))
  done

  # Wait for all jobs in this round to finish
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done

  if (( fail != 0 )); then
    echo "Round $round_idx: one or more jobs failed (see logs in $LOGDIR)" >&2
    # Decide whether to exit or continue:
    # exit 1
  else
    echo "Round $round_idx: all jobs completed."
  fi

  ((round_idx++))
done

echo
echo "All rounds complete. Logs in: $LOGDIR"
