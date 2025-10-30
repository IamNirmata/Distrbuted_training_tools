# 0) Set vars
NS=default
JOB=llama3-finetune-sleep-job
OUT=/mnt/data/output/llama-3-8b-function-calling

# 1) find a running pod of the job (master or worker)
POD=$(kubectl -n $NS get pod -l training.kubeflow.org/job-name=$JOB -o jsonpath='{.items[0].metadata.name}')

# 2) stream-compress from pod to local (no temp files in pod)
kubectl -n $NS exec $POD -c pytorch -- tar -C "$OUT" -czf - . > llama3_adapters.tgz

# 3) inspect locally
tar -tzf llama3_adapters.tgz | head
