#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   K8S_NAMESPACE=my-namespace ./del_jobs.sh
# If K8S_NAMESPACE is not set, current kubectl context namespace is used.

NAMESPACE="${K8S_NAMESPACE:-}"
if [[ -n "$NAMESPACE" ]]; then NS_ARGS=(-n "$NAMESPACE"); else NS_ARGS=(); fi

JOBS=(
    "llama3-finetune-sleep-job"
)

PODS=(
    "llama3-finetune-sleep-job-master-0"
    "llama3-finetune-sleep-job-worker-0"
)

echo "Namespace: ${NAMESPACE:-<current>}"
echo "Deleting jobs and pods..."

for job in "${JOBS[@]}"; do
    echo "-> Deleting job: $job"
    kubectl delete job "$job" "${NS_ARGS[@]}" --ignore-not-found || true

    echo "-> Deleting pods with label job-name=$job"
    kubectl delete pods -l job-name="$job" "${NS_ARGS[@]}" --ignore-not-found || true
    kubectl delete pytorchjobs.kubeflow.org "$job"
    kubectl delete pytorchjobs.kubeflow.org llama3-finetune-sleep-job
done

for pod in "${PODS[@]}"; do
    echo "-> Deleting pod: $pod"
    kubectl delete pod "$pod" "${NS_ARGS[@]}" --ignore-not-found || true
done

echo "Done."