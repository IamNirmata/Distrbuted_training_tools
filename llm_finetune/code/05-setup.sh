#!/bin/bash

# This script will guide you through setting up and launching the
# multi-node fine-tuning job on your Nebius K8s cluster.

# --- 1. Local Dependencies (Ensure these are installed) ---
echo "--- Step 1: Checking Local Dependencies ---"
echo "Please ensure you have 'docker', 'kubectl', and 'git' installed locally."
echo "You also need to be logged in to your container registry."
echo "-----------------------------------------------------"
echo ""

# --- 2. Configuration ---
# export IMAGE_TAG="ghcr.io/iamnirmata/nebius-llama3-poc:latest"
# echo "Image will be tagged as: $IMAGE_TAG"
# echo ""

# --- 3. Create Kubernetes Secrets ---
# echo "--- Step 3: Creating Kubernetes Secrets ---"
# echo "Creating 'hf-secret' with your Hugging Face token..."
# kubectl delete secret generic hf-secret --ignore-not-found
# kubectl create secret generic hf-secret \
#   --from-literal=token='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'

# echo "Creating 'wandb-secret' with your Weights & Biases API key..."
# kubectl delete secret generic wandb-secret --ignore-not-found
# kubectl create secret generic wandb-secret \
#   --from-literal=api_key='b66635c5b4b15ae1cd620492149ced7dc75e024d'
# echo "Secrets created successfully."
# echo "-----------------------------------------------------"
# echo ""

# --- 4. Build and Push Docker Image ---
# echo "--- Step 4: Building and Pushing Docker Image ---"
# echo "Building Docker image... this may take a few minutes."
# echo "Looking for '02-dockerfile.Dockerfile'..."
# docker build -t $IMAGE_TAG -f 02-dockerfile.Dockerfile .

# if [ $? -ne 0 ]; then
#   echo "!!! ERROR: Docker build failed."
#   exit 1
# fi
# echo "Build successful."

# echo "Pushing image to $IMAGE_TAG..."
# echo "You must be logged in to your registry for this to work."
# docker push $IMAGE_TAG

# if [ $? -ne 0 ]; then
#   echo "!!! ERROR: Docker push failed. Are you logged in?"
#   exit 1
# fi
# echo "Image pushed successfully."
# echo "-----------------------------------------------------"
# echo ""

# --- 5. Download Model and Dataset to Shared Storage ---
echo "--- Step 5: Downloading Assets to /mnt/data ---"
echo "Applying '00-prepare-environment.yaml' to download Llama-3 and the dataset."
echo "This Job will run on one of your nodes and may take 15-30 minutes."
kubectl apply -f 00-prepare-environment.yaml

echo "Waiting for the 'prepare-environment' job to complete..."
kubectl wait --for=condition=complete job/prepare-environment --timeout=30m

if [ $? -ne 0 ]; then
  echo "!!! ERROR: Asset download failed. Please check pod logs:"
  echo "  kubectl logs -l job-name=prepare-environment"
  exit 1
fi
echo "Asset download complete. Model and data are in /mnt/data."
kubectl delete job prepare-environment
echo "-----------------------------------------------------"
echo ""

# --- 6. Launch the Training Job ---
echo "--- Step 6: Launching the PyTorchJob ---"
echo "Applying '04-training-job.yaml' to start the 2-node, 16-GPU training."
echo "This will use torchrun as requested."
kubectl apply -f 04-training-job.yaml

echo "-----------------------------------------------------"
echo ""

# --- 7. Monitor the Job ---
echo "--- Step 7: Monitoring ---"
echo "The training job 'llama3-finetune-job' has been submitted."
echo "It may take a few minutes for the pods to be created and start running."
echo ""
echo "Monitor your W&B project: 'func_calls_llm' (Entity: 'iamnirmata-microsoft')"
echo ""
echo "To see the pods, run:"
echo "  kubectl get pods -l job-name=llama3-finetune-job"
echo ""
echo "To stream logs from the master (rank 0) pod, run:"
echo "  kubectl logs -f -l job-name=llama3-finetune-job,training-role=master"
echo ""
echo "Job launched successfully!"
