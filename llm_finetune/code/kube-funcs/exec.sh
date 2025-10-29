#!/usr/bin/env bash
# Usage: exec.sh [-n NAMESPACE] POD_PATTERN [-- COMMAND...]
# Examples:
#   ./exec.sh llama3-finetune-sleep-job-master-0
#   ./exec.sh -n ml ns-.*master   # regex match, picks first matching pod
#   ./exec.sh mypod -- /bin/sh -c "echo hi"
set -euo pipefail

namespace="$(kubectl config view --minify --output 'jsonpath={..namespace}' 2>/dev/null || true)"
namespace="${namespace:-default}"

while [[ $# -gt 0 && "$1" == -* ]]; do
    case "$1" in
        -n|--namespace) shift; namespace="$1"; shift ;;
        --) shift; break ;;
        -h|--help) cat <<'USAGE'
Usage: exec.sh [-n NAMESPACE] POD_PATTERN [-- COMMAND...]
Runs kubectl exec -it on a pod. If POD_PATTERN is not an exact pod name,
the script searches pods in the namespace and uses the first match.
USAGE
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "Error: missing POD_PATTERN" >&2
    echo "Usage: exec.sh [-n NAMESPACE] POD_PATTERN [-- COMMAND...]" >&2
    exit 2
fi

pod_pattern="$1"; shift

# If exact pod exists, use it
if kubectl get pod "$pod_pattern" -n "$namespace" >/dev/null 2>&1; then
    pod="$pod_pattern"
else
    # Find pods matching pattern (regex). Use grep -E for extended regex.
    mapfile -t matches < <(kubectl get pods -n "$namespace" --no-headers -o custom-columns=:metadata.name 2>/dev/null | grep -E -- "$pod_pattern" || true)
    if [ "${#matches[@]}" -eq 0 ]; then
        echo "No pods found matching pattern '$pod_pattern' in namespace '$namespace'" >&2
        exit 1
    fi
    if [ "${#matches[@]}" -gt 1 ]; then
        echo "Multiple pods match pattern; using first match: ${matches[0]}" >&2
    fi
    pod="${matches[0]}"
fi

# Ensure pod is in Running state (optional)
phase="$(kubectl get pod "$pod" -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
if [ -n "$phase" ] && [ "$phase" != "Running" ]; then
    echo "Warning: pod '$pod' is in phase '$phase' (not Running). kubectl exec may fail." >&2
fi

# Default command
if [ $# -eq 0 ]; then
    cmd=(/bin/bash)
else
    cmd=("$@")
fi

exec kubectl exec -it -n "$namespace" "$pod" -- "${cmd[@]}"