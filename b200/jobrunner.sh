#!/usr/bin/env bash
set -euo pipefail

# Directories at local machine
hdir=/home/hari/b200/validation/
dtools_dir=$hdir/distrbuted_training_tools/b200/
dltest_dir=$hdir/deeplearning_unit_test/

# Directories at job pod
job_dir=/opt/
log_dir=/data/cluster_validation/dltest/latest/


#variables
# --- config ---
NS="gcr-admin"
TEMPLATE=$dtools_dir/dltest.yml   # path to the YAML you pasted (save it once) /home/hari/b200/validation/distrbuted_training_tools/b200/dltest.yml
NODES_FILE=$dtools_dir/nodes.txt                # uses $hdir if you already set it
OUTDIR="$dtools_dir/job_yamls"                   # where to write per-node yamls
APPLY="yes"                                        # set to "no" to only generate files
# --------------


#home dir
cd $hdir

#step 1: Get all node names
kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name \
  | grep "gpu" > $dtools_dir/nodes.txt
echo "Node names written to $dtools_dir/nodes.txt"

#######################step 2: Create job yamls for each node#########################
mkdir -p "$OUTDIR" # ensure output dir exists

# sanity checks
[[ -f "$TEMPLATE" ]] || { echo "Template not found: $TEMPLATE"; exit 1; }
[[ -s "$NODES_FILE" ]] || { echo "Nodes file empty/missing: $NODES_FILE"; exit 1; }

# normalize CRLF just in case
tmp_nodes="$(mktemp)"
tr -d '\r' < "$NODES_FILE" > "$tmp_nodes"

echo "Generating jobs from: $NODES_FILE"
echo "Writing to: $OUTDIR"
echo

while IFS= read -r NODE; do
  # skip blanks / comments
  [[ -z "${NODE// }" ]] && continue
  [[ "$NODE" =~ ^# ]] && continue

  OUTFILE="${OUTDIR}/tmp_dltest_${NODE}.yml"

  # replace the placeholder safely
  sed "s/gcr-node-name-placeholder/${NODE}/g" "$TEMPLATE" > "$OUTFILE"

  echo "✔ wrote $OUTFILE"

  if [[ "$APPLY" == "yes" ]]; then
    # create the job and print the created name
    created=$(kubectl create -f "$OUTFILE" -n "$NS" 2>&1)
    echo "→ $created"
  fi
done < "$tmp_nodes"

rm -f "$tmp_nodes"
echo "Done."













#step 2: Create tmp_job.yml
export gcrjobname=$(kubectl create -f $dltest_dir/dltest.yml -n gcr-admin --dry-run=client -o yaml 2>/dev/null
  | awk '{print $1}' | cut -d'/' -f2)

podname=$gcrjobname-server-0
echo "GCR Job name: $gcrjobname"
echo "GCR Pod name: $podname"

# step 3: fetch node names from gcrjobname
gcrnode=$(kubectl get pod $podname -n gcr-admin -o jsonpath='{.spec.nodeName}{"\n"}')
echo "GCR Node name: $gcrnode"

#step 4: pass the node names to the jobpod using kubectl exec
# kubectl exec -n gcr-admin $podname -- bash -c "echo $gcrnode > /opt/gcrnode.txt"
# echo "GCR Node name written to /opt/gcrnode.txt in the job pod"
#### The node has been passed to the job config file using env variable in dltest.yml as gcrnode





