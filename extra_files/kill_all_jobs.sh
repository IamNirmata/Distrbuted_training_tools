kubectl get vcjob -n gcr-admin -o name \
  | grep 'gcr-daily-validation-hari-' \
  | xargs -r kubectl delete -n gcr-admin
