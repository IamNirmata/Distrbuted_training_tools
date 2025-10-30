# Reproducibility Guide — AI Assignment 16 (2× Nebius H100 nodes)

This guide reproduces the fine-tuning run for Meta Llama 3 8B Instruct on the XLAM-60k function-calling dataset across 2 nodes × 8 GPUs.

## Prerequisites
- Kubernetes cluster with Kubeflow PyTorch Operator installed
- Two Nebius nodes with 8× NVIDIA H100 each and a shared RWX storage class
- `kubectl` access and permissions to create PV/PVC, secrets, and jobs
- Container image access: `nvcr.io/nvidia/pytorch:24.07-py3`
- Hugging Face token with access to `meta-llama/Meta-Llama-3-8B-Instruct`
- Weights & Biases API key (optional but recommended)

## 1) Clone the repository
```bash
# on your workstation or an admin pod
git clone https://github.com/IamNirmata/distrbuted_training_tools.git
cd distrbuted_training_tools/llm_finetune
```

## 2) Provision shared storage (PV/PVC)
Confirm your cluster has a RWX storage class. Apply the sample manifest:
```bash
kubectl apply -f code/pvc.yaml
```
This creates:
- PV `external-storage-persistent-volume` (2Ti)
- PVC `external-storage-persistent-volumeclaim` mounted as `/mnt/data`

## 3) Create secrets
```bash
# Replace the example values with your actual tokens
kubectl create secret generic hf-secret \
  --from-literal=token="<HF_TOKEN>"

kubectl create secret generic wandb-secret \
  --from-literal=api_key="<WANDB_API_KEY>"
```

## 4) Stage model and dataset (option A: in-cluster job)
Use the one-off job to pre-download assets into `/mnt/data`:
```bash
kubectl apply -f code/old/00-prepare-environment.yaml
kubectl wait --for=condition=complete job/prepare-environment --timeout=30m
kubectl delete job prepare-environment
```
This populates:
- Model → `/mnt/data/models/Meta-Llama-3-8B-Instruct`
- Dataset → `/mnt/data/datasets/xlam-function-calling-60k`

### Alternative 4B) Stage assets via scripts from a pod (option B)
If you prefer to run setup inside a long-running pod:
```bash
kubectl apply -f code/sleep.yaml
# When the master/worker pods are Running, exec into one of them:
llm_finetune/code/kube-funcs/exec.sh llama3-finetune-sleep-job-master-0 -- bash -lc '
  cd /workspace/distrbuted_training_tools/llm_finetune/setup_and_data && \
  bash 0-setup.sh && bash 1-data.sh && bash 2-model.sh'
```

## 5) Launch training
Apply the PyTorchJob that runs torchrun over two pods (8 GPUs each):
```bash
kubectl apply -f code/04-training-job.yaml
```
Notes:
- The YAML uses `torchrun --nproc_per_node=8 --nnodes=2 --node_rank={0,1}` and rendezvous at `llama3-finetune-job-master-0:29500`.
- It installs Python requirements in both pods and runs `llm_finetune/setup_and_data/train.py`.

## 6) Monitor and logs
```bash
# Pods
kubectl get pods -l job-name=llama3-finetune-job

# Stream logs from the master (rank 0)
kubectl logs -f -l job-name=llama3-finetune-job,training-role=master

# Check Weights & Biases project: func_calls_llm (entity: iamnirmata-microsoft)
```

## 7) Collect artifacts (adapters)
Adapters are saved under `/mnt/data/output/llama-3-8b-function-calling`.
Use the helper to tar them to your local machine:
```bash
cd code/kube-funcs
bash download.sh    # creates ./llama3_adapters.tgz locally
```

## 8) Load adapters for inference (example)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE = "/mnt/data/models/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "/mnt/data/output/llama-3-8b-function-calling"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb_config, torch_dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

prompt = """<user>Find restaurants in Seattle open now for dinner</user>
<tools>[{"name": "searchRestaurants", "args": ["city", "open_now"], "returns": ["name", "address", "hours"]}]</tools>
<calls>"""

inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=False))
```

## 9) Configuration knobs
- Data/model paths via env (defaults in `train.py`):
  - `MODEL_PATH`, `DATASET_PATH`, `OUTPUT_DIR`
- Enable FlashAttention2: set `USE_FLASH_ATTENTION=1`
- Adjust batch size and accumulation depending on memory headroom
- NCCL tuning: if your NIC differs, update `NCCL_SOCKET_IFNAME` and (optionally) `NCCL_IB_HCA`

## Troubleshooting tips
- If rendezvous fails: ensure Master pod DNS `llama3-finetune-job-master-0` resolves in cluster and port 29500 is open in pod
- If OOM: lower per-device batch size or increase gradient accumulation; ensure LoRA + 4-bit config is intact
- If IB/NVLink issues: temporarily set `NCCL_IB_DISABLE=1` to fall back to TCP sockets for isolation tests

## Reference files
- Storage: `code/pvc.yaml`
- Prepare assets: `code/old/00-prepare-environment.yaml`
- Training job: `code/04-training-job.yaml`
- Training script: `setup_and_data/train.py`
- Setup scripts: `setup_and_data/0-setup.sh`, `1-data.sh`, `2-model.sh`, `8-pre-launch.sh`, `9-launch.sh`
- Helpers: `code/kube-funcs/*.sh`
