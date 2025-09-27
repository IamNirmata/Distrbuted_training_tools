home_directory=/home/hari/b200/validation/distrbuted_training_tools
cd $home_directory || exit 1


#step 1: Get all node names
kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name \
  | grep "gpu" > gpu_nodes.txt


#step 2: Create tmp_job.yml
export gcrjobname=$(kubectl create -f /home/hari/b200/validation/deeplearning_unit_test/dltest.yml \
  | awk '{print $1}' | cut -d'/' -f2)



# step 3: fetch node names from gcrjobname
gcrnode=$(kubectl get pod $gcrjobname-server-0 -n gcr-admin -o jsonpath='{.spec.nodeName}{"\n"}')
echo "GCR Node name: $gcrnode"

#step 4: pass the node names to the jobpod using kubectl exec
kubectl exec -n gcr-admin $gcrjobname-server-0 -- bash -c "echo $gcrnode > /opt/gcrnode.txt"
echo "GCR Node name written to /opt/gcrnode.txt in the job pod"


#step 5: Monitor the job status
