#step 1: Get all node names
kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name \
  | grep "gpu" > gpu_nodes.txt

#step 2: Create tmp_job.yml



kubectl create -f temp_job.yml
