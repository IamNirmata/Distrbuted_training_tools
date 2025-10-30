# Fine-tuning Meta Llama 3 for Function Calling on 2×8 H100 Nodes

## Executive summary
- Goal: Fine-tune Meta Llama 3 8B Instruct for robust tool/function calling using the XLAM-60k dataset.
- Hardware: 2 Nebius nodes, each with 8× NVIDIA H100 (16 GPUs total) interconnected over high-speed networking.
- Orchestration: Kubeflow PyTorchJob on Kubernetes; shared PV/PVC mounted at `/mnt/data`.
- Distributed Method: Torchrun over 16 GPUs (2 nodes × 8 GPUs each) as PyTorchJob.
- Outcome: Successful 1000-step training run across 16 GPUs, runtime ~6,633s (~110.6 min), mean token accuracy peaked around 0.964, final per-step losses ~0.142–0.147. Adapters saved to `/mnt/data/output/llama-3-8b-function-calling` and logged to Weights & Biases.

## Cluster and topology
- 2× worker nodes; each node has 8× NVIDIA H100 GPUs
- Kubernetes with Kubeflow PyTorch Operator
- Shared storage
  - PV/PVC: `csi-mounted-fs-path-sc` (2Ti)
  - Mounted at `/mnt/data` for all pods
- Job layout
  - 1 Master + 1 Worker pod (8 GPUs each)
  - Rendezvous via `torchrun` using `llama3-finetune-job-master-0` and port `29500`

## Software stack
- Base container image: `nvcr.io/nvidia/pytorch:24.07-py3`
- Python libs (key pins from `setup_and_data/requirements.txt`):
  - transformers>=4.41.2, trl>=0.8.6, peft>=0.11.1, accelerate>=0.30.1, bitsandbytes>=0.43.1, datasets>=2.19.0, wandb>=0.16.6
- Training script: 
    - DDP `llm_finetune/setup_and_data/ddp.py`
    - FSDP - `llm_finetune/setup_and_data/fsdp_no4.py`
- Launcher: `torchrun` via PyTorchJob (`llm_finetune/code/04-training-job.yaml`)
- Observability: Weights & Biases (project `func_calls_llm`, entity `iamnirmata-microsoft`)

## Data and model
- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
  - Downloaded to: `/mnt/data/models/Meta-Llama-3-8B-Instruct`
- Dataset: `Salesforce/xlam-function-calling-60k`
  - Saved to: `/mnt/data/datasets/xlam-function-calling-60k`
- Prompt shaping (simplified)
  - For each row, construct a single `text` field:
    - `<user>{query}</user>\n\n<tools>{tools}</tools>\n\n<calls>{answers}</calls>{eos}`
  - Any parsing errors are made non-fatal and examples are filtered if empty

## Fine-tuning approach
- Quantization: 4-bit (bitsandbytes NF4, compute dtype fp16)
- LoRA (via PEFT):
  - r=16, alpha=16, dropout=0.05, bias=none
  - target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj
- Tokenizer: right-padding; adds custom pad token `<|eot_id|>` when needed
- Trainer: TRL SFTTrainer with Accelerate
- Hyperparameters (from `train.py`):
  - per_device_train_batch_size=4, per_device_eval_batch_size=4
  - gradient_accumulation_steps=8
  - optimizer=`paged_adamw_8bit`
  - fp16=True, bf16=False
  - max_steps=1000, warmup_ratio=0.1, logging_steps=10, save_steps=250
  - gradient_checkpointing_enable + reentrant=False
  - ddp_find_unused_parameters=False
- Output directory: `/mnt/data/output/llama-3-8b-function-calling`

## Distributed setup
- Process topology: 16 processes total (8 per node × 2 nodes)
- Launch method: `torchrun` with explicit rendezvous parameters passed in the YAML
- NCCL/networking (used in setup scripts and recommended for H100 IB):
  - `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,ENV,NET`
  - `NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
  - `NCCL_NVLS_ENABLE=0`, `NCCL_SHARP_DISABLE=1`, `NCCL_NET_GDR_LEVEL=PHB`
  - `NCCL_SOCKET_IFNAME=eth0` (adjust to your cluster if needed)

## Kubernetes workflow overview
- Storage: `llm_finetune/code/pvc.yaml` provisions the 2Ti PV and RWX PVC
- Optional staging job: `llm_finetune/code/old/00-prepare-environment.yaml` to prefetch model and dataset into `/mnt/data`
- Training job: `llm_finetune/code/04-training-job.yaml` (PyTorchJob with Master+Worker)
  - Both pods pull this repo and pip-install requirements
  - Both run `torchrun` pointing to the shared `train.py`

## Run-time logs and results
Source: `llm_finetune/logs/logs.txt` (2025-10-29 to 2025-10-30)
- Step-level signals (selected):
  - losses gradually improved to ~0.142–0.147 range in late training
  - mean_token_accuracy improved to ~0.964 at peak
  - entropy trended down from ~0.186 to ~0.158–0.162 range
  - tokens processed reached ~97.5M by the end
- Final training summary from trainer:
  - train_runtime: 6633.5201 s (~110.6 min)
  - train_steps_per_second: 0.151
  - train_samples_per_second: 38.592
  - train_loss (reported average): 0.2815267630815506
- Checkpoints: `save_steps=250` produced artifacts (e.g., `checkpoint-1000`), and adapters were saved to `OUTPUT_DIR`

## Artifacts and model usage
- Adapters are stored in `/mnt/data/output/llama-3-8b-function-calling`
- A helper to export adapters from a running pod is provided: `llm_finetune/code/kube-funcs/download.sh`
- To load the fine-tuned adapter in Python (example):
  1) Load base model with 4-bit and apply PEFT adapters from the saved dir
  2) Generate on prompts with function-calling formatting used during SFT

## Observability
- W&B configured via env variables; logging every 10 steps
- Entity/project: `iamnirmata-microsoft/func_calls_llm`
- Checkpoints automatically logged via `WANDB_LOG_MODEL=checkpoint`

## Reliability and mitigations
- NCCL tuning applied to avoid IB/NVLink surprises on H100
- Gradient checkpointing enabled to reduce memory pressure
- 4-bit QLoRA reduces VRAM footprint, enabling 16-GPU utilization with headroom

## Risks and follow-ups
- Accuracy: Consider continued training beyond 1000 steps and/or curriculum mixing for harder function-calling samples
- Eval: Add a structured evaluation set with schema-aware scoring (e.g., tool selection accuracy and argument correctness)
- Throughput: Investigate `flash_attention_2` (set `USE_FLASH_ATTENTION=1`) and FSDP + CPU offload tradeoffs for larger context lengths
- Data: Expand from XLAM-60k to domain-specific tool suites to improve generalization

## Appendix: Key files and their roles
- `llm_finetune/setup_and_data/train.py` — Training script (QLoRA SFT)
- `llm_finetune/code/04-training-job.yaml` — 2-node PyTorchJob spec running torchrun
- `llm_finetune/code/pvc.yaml` — PV/PVC (2Ti RWX) for shared `/mnt/data`
- `llm_finetune/code/old/00-prepare-environment.yaml` — One-off job to pre-download model+dataset
- `llm_finetune/setup_and_data/0-setup.sh`, `1-data.sh`, `2-model.sh`, `8-pre-launch.sh`, `9-launch.sh` — Setup and launch utilities
- `llm_finetune/code/kube-funcs/*.sh` — Convenience scripts (create/delete jobs, exec, download adapters)
- `llm_finetune/logs/logs.txt` — Training logs excerpt used for the metrics summary
