#!/usr/bin/env bash
set -euo pipefail

# Directories at local machine
hdir=/home/hari/b200/validation
dtools_dir=$hdir/distrbuted_training_tools/b200
dltest_dir=$hdir/deeplearning_unit_test

# Directories at job pod
job_dir=/opt/
log_dir=/data/cluster_validation/dltest/latest/


#variables
# --- config ---
NS="gcr-admin"
TEMPLATE=/home/hari/b200/validation/others/dltest.yml  # path to the YAML you pasted (save it once) /home/hari/b200/validation/distrbuted_training_tools/b200/dltest.yml
NODES_FILE=$dtools_dir/nodes.txt                # uses $hdir if you already set it
# OUTDIR="$dtools_dir/job_yamls"
OUTDIR="$hdir/job_yamls"            # where to write per-node yamls
APPLY="yes"                                        # set to "no" to only generate files
# --------------


#home dir
cd $hdir

#step 1: Get all node names
del "$NODES_FILE" 2>/dev/null || true
echo "Fetching GPU node names..."
kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name \
  | grep "gpu" > "$NODES_FILE"
echo "Node names written to NODES_FILE: $NODES_FILE"
wait
#######################step 2: Create job yamls for each node#########################
mkdir -p "$OUTDIR" # ensure output dir exists

# sanity checks
[[ -f "$TEMPLATE" ]] || { echo "Template not found: $TEMPLATE"; exit 1; }
[[ -s "$NODES_FILE" ]] || { echo "Nodes file empty/missing: $NODES_FILE"; exit 1; }

# normalize CRLF just in case
tmp_nodes="$(mktemp)"
tr -d '\r' < "$NODES_FILE" > "$tmp_nodes"

echo "Generating jobs from: $NODES_FILE"
echo "Available nodes:"
cat "$tmp_nodes"
echo "Using template: $TEMPLATE"
echo "Writing to: $OUTDIR"
echo

while IFS= read -r NODE; do
  # skip blanks / comments
  [[ -z "${NODE// }" ]] && continue
  [[ "$NODE" =~ ^# ]] && continue

  OUTFILE="${OUTDIR}/tmp_dltest_${NODE}.yml"
  # timestamp in los angeles timezone
  timestamp=$(TZ="America/Los_Angeles" date +%Y%m%d-%H%M%S)
  today=$(TZ="America/Los_Angeles" date +%Y%m%d)
  # replace the placeholder safely
  sed "s/gcr-node-name-placeholder/${NODE}/g" "$TEMPLATE" > "$OUTFILE"
  # replace any other placeholders if needed
  # replace job name placeholder "gcr-daily-validation-hari-placeholder-" with "gcrdcv-hari-NODENAME-"
  sed -i "s/gcr-daily-validation-hari-placeholder-/gcrdcv-hari-$timestamp-${NODE}-/g" "$OUTFILE"
  

  echo "✔ wrote $OUTFILE"

  if [[ "$APPLY" == "yes" ]]; then
    # create the job and print the created name
    created=$(kubectl create -f "$OUTFILE" -n "$NS" 2>&1)
    echo "→ $created"
  fi
done < "$tmp_nodes"

rm -f "$tmp_nodes"
echo "Done."