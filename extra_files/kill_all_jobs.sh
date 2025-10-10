kubectl get vcjob -n gcr-admin -o name \
  | grep 'gcrdcv-hari' \
  | xargs -r kubectl delete -n gcr-admin
