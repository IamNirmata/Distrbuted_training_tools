"""
Simplified multi-node fine-tuning script for Meta Llama 3 using PyTorch DDP.
This script is intended to be launched with `torchrun`.
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
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/data/output/llama-3-8b-function-calling")

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
        print(f"Initialized DDP: World Size {dist.get_world_size()}, Rank {global_rank}, Local Rank {local_rank}")
        
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
        # If anything goes wrong keep the example but mark it empty so it is filtered.
        print(f"Failed to format example: {exc}")
        return {"text": ""}


def _load_splits(path: str) -> Tuple[Dataset, Dataset]:
    # This function is DDP-safe. `load_from_disk` is fine.
    # `train_test_split` is also fine as it's deterministic with `seed`.
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
    # --- DDP Setup ---
    # Replaces `Accelerator()`
    global_rank, local_rank, is_main_process = _setup_ddp()

    # --- Tokenizer Loading ---
    # No changes needed here, but we use `is_main_process` for logging.
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
    # The `datasets` library is DDP-aware. `.map()` and `.filter()`
    # will only run on the main process, and other processes will
    # load the cached results.
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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_kwargs = dict(
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # NOTE: Do NOT set `device_map` here. `SFTTrainer` will handle
        # device placement based on the DDP `local_rank`.
    )
    if os.environ.get("USE_FLASH_ATTENTION", "0") == "1":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main_process:
            print("Using flash_attention_2 kernels")
    elif is_main_process:
        print("Using standard attention implementation")

    # --- Model Loading ---
    # `torch.cuda.set_device(local_rank)` from `_setup_ddp` ensures
    # the model is loaded onto the correct GPU.
    if is_main_process:
        print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

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
    # `SFTConfig` (subclass of `TrainingArguments`) will automatically
    # detect the DDP environment from `torchrun` env vars.
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        remove_unused_columns=False,
        report_to="wandb",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        max_steps=1000,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=250,
        learning_rate=1e-4,
        bf16=False,
        fp16=True,
        save_on_each_node=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        # `local_rank` is set automatically by `TrainingArguments`
        # when it detects the `LOCAL_RANK` env var.
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # This dynamic argument checking remains the same
    trainer_signature = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_signature.parameters:
        trainer_kwargs["max_seq_length"] = 2048
    if "dataset_text_field" in trainer_signature.parameters and "dataset_text_field" not in trainer_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"

    # --- Trainer Initialization ---
    # `SFTTrainer` will handle wrapping the model with DDP
    # and setting up the DistributedSampler for data loading.
    trainer = SFTTrainer(**trainer_kwargs)

    # --- Training ---
    if is_main_process:
        print("Starting training...")
    trainer.train()

    # --- Synchronization and Saving ---
    # Replaces `accelerator.wait_for_everyone()`
    dist.barrier()
    
    if is_main_process:
        print("Training finished. Saving adapters...")
        
    # `trainer.save_model` is DDP-aware and will only save
    # on the main process (rank 0).
    trainer.save_model(OUTPUT_DIR)

    if is_main_process:
        print("All done!")

    # --- DDP Cleanup ---
    dist.destroy_process_group()


if __name__ == "__main__":
    main()