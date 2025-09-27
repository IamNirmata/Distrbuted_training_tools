home_directory=/home/hari/b200/validation/distrbuted_training_tools
cd $home_directory || exit 1


#step 1: Get all node names
kubectl get nodes --no-headers -o custom-columns=NAME:.metadata.name \
  | grep "gpu" > gpu_nodes.txt

#step 2: Create tmp_job.yml

export gcrjobname=$(kubectl create -f /home/hari/b200/validation/deeplearning_unit_test/dltest.yml \
  | awk '{print $1}' | cut -d'/' -f2)

kubectl create -f /home/hari/b200/validation/deeplearning_unit_test/dltest.yml 