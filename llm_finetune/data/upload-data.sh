#!/bin/bash

# --- 1. Check Node Labels (New Step) ---
echo "Checking your node labels to ensure we schedule the pod correctly..."
echo "Running: kubectl get nodes --show-labels"
kubectl get nodes --show-labels

echo ""
echo "!!! ACTION REQUIRED !!!"
echo "Look for a label that is present on your two GPU nodes."
echo "The default in 'uploader-pod.yaml' is 'nebius.ai/gpu-node: \"true\"'."
echo "If your nodes have a different label, please edit the 'nodeSelector' in 'uploader-pod.yaml'."
read -p "Press [Enter] after you have verified/edited the nodeSelector..."

# --- 2. Create the Helper Pod ---
echo "Creating the 'uploader' pod..."
kubectl apply -f uploader-pod.yaml

echo "Waiting for the 'uploader' pod to be in 'Running' state..."
# This command waits until the pod's status is "Running"
kubectl wait --for=condition=Ready pod/uploader --timeout=300s

if [ $? -ne 0 ]; then
  echo "Pod 'uploader' failed to start. Please check pod status:"
  echo "  kubectl describe pod uploader"
  echo "  kubectl logs uploader"
  echo "This might be due to an incorrect 'nodeSelector' label."
  exit 1
fi

echo "Pod 'uploader' is running."

# --- 3. Copy Your Data ---
echo ""
echo "!!! ACTION REQUIRED !!!"
echo "Run the following 'kubectl cp' command in your terminal."
echo "Replace 'dataset.jsonl' with the path to your *local* dataset file:"
echo ""
echo "  kubectl cp dataset.jsonl uploader:/mnt/data/dataset.jsonl"
echo ""
read -p "Press [Enter] after you have successfully copied the file..."

# --- 4. Verify and Clean Up ---
echo "Verifying the file exists in the pod..."
# We will create a 'dataset' subdirectory as planned
kubectl exec uploader -- mkdir -p /mnt/data/dataset
kubectl exec uploader -- mv /mnt/data/dataset.jsonl /mnt/data/dataset/dataset.jsonl
echo "Listing contents of /mnt/data/dataset:"
kubectl exec uploader -- ls -lh /mnt/data/dataset

echo ""
echo "You can now clean up the helper pod by running:"
echo "  kubectl delete pod uploader"

echo "Your dataset is now on the shared volume, ready for training."

