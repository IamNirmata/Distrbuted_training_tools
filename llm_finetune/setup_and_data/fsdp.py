"""
Multi-node FSDP fine-tuning script for Meta Llama 3 using PyTorch DDP/FSDP.
This script is intended to be launched with `torchrun`.

It uses FSDP (Fully Sharded Data Parallelism) to shard the model, 
gradients, and optimizer state across all GPUs in all nodes.
This achieves the memory-saving goal of "tensor parallelism inside the node"
and the speed-up goal of "data parallel between nodes".
"""

import inspect
import json
import os
from typing import Tuple

import torch
import torch.distributed as dist  # Import torch distributed
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# --- Environment Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/data/models/Meta-Llama-3-8B-Instruct")
DATASET_PATH = os.environ.get("DATASET_PATH", "/mnt/data/datasets/xlam-function-calling-60k")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/data/output/llama-3-8b-function-calling-fsdp")

os.environ.setdefault("WANDB_PROJECT", "func_calls_llm")
os.environ.setdefault("WANDB_ENTITY", "iamnirmata-microsoft")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")

CUSTOM_PAD_TOKEN = "<|eot_id|>"


def _setup_ddp() -> Tuple[int, int, bool]:
    """Initializes the distributed process group and sets the device."""
    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires torch.distributed and CUDA.")

    # torchrun provides RANK, LOCAL_RANK, and WORLD_SIZE
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Pin the current process to a specific GPU
    torch.cuda.set_device(local_rank)
    
    is_main_process = (global_rank == 0)
    
    if is_main_process:
        print(f"Initialized DDP/FSDP: World Size {dist.get_world_size()}, Rank {global_rank}, Local Rank {local_rank}")
        
    return global_rank, local_rank, is_main_process


def _format_example(example: dict, eos_token: str) -> dict:
    """Convert a raw xLAM row into the single "text" field used for SFT."""
    try:
        query = example.get("query", "")
        tools_raw = example.get("tools", "[]")
        answers_raw = example.get("answers", "[]")

        try:
            tools = "\n".join(str(item) for item in json.loads(tools_raw))
        except (json.JSONDecodeError, TypeError):
            tools = str(tools_raw)

        try:
            answers = "\n".join(str(item) for item in json.loads(answers_raw))
        except (json.JSONDecodeError, TypeError):
            answers = str(answers_raw)

        text = (
            f"<user>{query}</user>\n\n"
            f"<tools>{tools}</tools>\n\n"
            f"<calls>{answers}</calls>"
            f"{eos_token}"
        )
        return {"text": text}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to format example: {exc}")
        return {"text": ""}


def _load_splits(path: str) -> Tuple[Dataset, Dataset]:
    data = load_from_disk(path)

    if isinstance(data, DatasetDict):
        train_split = data.get("train") or next(iter(data.values()))
        eval_split = data.get("validation") or data.get("test")
        if eval_split is None:
            split = train_split.train_test_split(test_size=0.1, seed=42)
            train_split, eval_split = split["train"], split["test"]
    else:
        split = data.train_test_split(test_size=0.1, seed=42)
        train_split, eval_split = split["train"], split["test"]

    return train_split, eval_split


def main() -> None:
    # --- DDP/FSDP Setup ---
    global_rank, local_rank, is_main_process = _setup_ddp()

    # --- Tokenizer Loading ---
    if is_main_process:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": CUSTOM_PAD_TOKEN})
        tokenizer.pad_token = CUSTOM_PAD_TOKEN
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(CUSTOM_PAD_TOKEN)
        if is_main_process:
            print(f"Added custom pad token: {CUSTOM_PAD_TOKEN}")
    tokenizer.padding_side = "right"

    # --- Dataset Loading & Formatting ---
    train_raw, eval_raw = _load_splits(DATASET_PATH)
    
    train_dataset = train_raw.map(
        lambda row: _format_example(row, tokenizer.eos_token),
        remove_columns=train_raw.column_names,
        desc="Formatting train split",
    )
    eval_dataset = eval_raw.map(
        lambda row: _format_example(row, tokenizer.eos_token),
        remove_columns=eval_raw.column_names,
        desc="Formatting eval split",
    )
    train_dataset = train_dataset.filter(lambda row: len(row["text"]) > 0)
    eval_dataset = eval_dataset.filter(lambda row: len(row["text"]) > 0)

    if is_main_process:
        print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")
        print("Example formatted prompt:\n", train_dataset[0]["text"][:400], "...", sep="")

    # --- Model Configuration ---
    # NOTE: FSDP + QLoRA requires a special `quant_storage_dtype`.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,  # Required for FSDP
    )

    model_kwargs = dict(
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, # Must match quant_storage_dtype
        trust_remote_code=True,
    )
    
    if os.environ.get("USE_FLASH_ATTENTION", "0") == "1":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main_process:
            print("Using flash_attention_2 kernels")
    elif is_main_process:
        print("Using standard attention implementation")

    # --- Model Loading and Dtype Correction ---
    if is_main_process:
        print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Run this first. It enables gradient checkpointing and 
    # casts LayerNorms/lm_head to float32 (which we will override).
    model = prepare_model_for_kbit_training(model)

    # --- THE FIX: Override dtypes ---
    # Manually cast the lm_head (which is not 4-bit) to bfloat16
    # to match the `torch_dtype` of the rest of the model.
    if hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.to(torch.bfloat16)

    # Manually cast all LayerNorms to bfloat16
    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.bfloat16)
    # --- END FIX ---
            
    model.config.use_cache = False

    # --- PEFT Configuration ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
    )

    # --- Training Configuration ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        remove_unused_columns=False,
        report_to="wandb",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,

        # === FSDP CHANGES ===
        # 1. Use a standard optimizer. `paged_adamw_8bit` is NOT compatible with FSDP.
        optim="adamw_torch",
        
        # 2. Enable FSDP
        fsdp="full_shard",  # "full" is equivalent to ZeRO-3
        fsdp_config={
            "fsdp_timeout": 1800,
            # This policy tells FSDP how to wrap LoRA layers correctly
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP", 
            "fsdp_peft_config": peft_config, # Pass PEFT config here for FSDP
            "activation_checkpointing": True, # Enable activation checkpointing
        },
        # === END FSDP CHANGES ===

        max_steps=1000,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=250,
        learning_rate=1e-4,
        
        # Use bf16 for FSDP training, fp16 is less stable
        bf16=True,
        fp16=False, 
        
        save_on_each_node=False,
        gradient_checkpointing=False, # Disable old method
        ddp_find_unused_parameters=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config, # Pass peft_config here for SFTTrainer
    )

    trainer_signature = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_signature.parameters:
        trainer_kwargs["max_seq_length"] = 2048
    if "dataset_text_field" in trainer_signature.parameters and "dataset_text_field" not in trainer_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    if is_main_process:
        print("Starting training...")
    trainer.train()

    dist.barrier()
    
    if is_main_process:
        print("Training finished. Saving adapters...")
        
    trainer.save_model(OUTPUT_DIR) # Only saves on rank 0

    if is_main_process:
        print("All done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()