"""
Multi-node FSDP fine-tuning script for Meta Llama 3 using PyTorch DDP/FSDP.
Launch with `torchrun`.

This uses FSDP (Fully Sharded Data Parallelism) together with QLoRA (bitsandbytes 4-bit)
so that base weights remain quantized while LoRA adapters are trained and sharded.

Key points:
- Choose the compute dtype at load time via `dtype=...` in `from_pretrained`.
- Do NOT call `model.to(...)` on a bitsandbytes-quantized model after loading.
- FSDP wraps PEFT layers; base 4-bit weights are not fully sharded (expected for QLoRA).
"""

import inspect
import json
import os
from typing import Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# --- Environment Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/data/models/Meta-Llama-3-8B-Instruct")
DATASET_PATH = os.environ.get("DATASET_PATH", "/mnt/data/datasets/xlam-function-calling-60k")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/data/output/llama-3-8b-function-calling-fsdp")

os.environ.setdefault("WANDB_PROJECT", "func_calls_llm")
os.environ.setdefault("WANDB_ENTITY", "iamnirmata-microsoft")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")

# Llama 3 has <|eot_id|> token; we repurpose it as pad if tokenizer lacks one.
CUSTOM_PAD_TOKEN = "<|eot_id|>"


def _setup_ddp() -> Tuple[int, int, bool]:
    """Initializes the distributed process group and sets the device."""
    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires torch.distributed and CUDA.")

    # torchrun sets these env vars
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    # Initialize process group and pin GPU
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    is_main = (global_rank == 0)
    if is_main:
        print(
            f"Initialized DDP/FSDP: World Size {dist.get_world_size()}, "
            f"Rank {global_rank}, Local Rank {local_rank}",
            flush=True,
        )
    return global_rank, local_rank, is_main


def _format_example(example: dict, eos_token: str) -> dict:
    """Convert a raw xLAM row into the single 'text' field used for SFT."""
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
    except Exception as exc:  # defensive
        print(f"Failed to format example: {exc}", flush=True)
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
    global_rank, local_rank, is_main = _setup_ddp()

    # --- Tokenizer ---
    if is_main:
        print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": CUSTOM_PAD_TOKEN})
        tokenizer.pad_token = CUSTOM_PAD_TOKEN
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(CUSTOM_PAD_TOKEN)
        if is_main:
            print(f"Added custom pad token: {CUSTOM_PAD_TOKEN}", flush=True)
    tokenizer.padding_side = "right"

    # --- Dataset load & format ---
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

    if is_main:
        print(
            f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}",
            flush=True,
        )
        print("Example formatted prompt:\n", train_dataset[0]["text"][:400], "...\n", sep="", flush=True)

    # --- Model Configuration (QLoRA + FSDP) ---
    # Choose compute dtype at load time; do not call model.to(...) later.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # Keep bf16 quant storage when mixing FSDP and QLoRA on many GPUs
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model_kwargs = dict(
        quantization_config=bnb_config,
        dtype=torch.bfloat16,           # <— use dtype=..., not torch_dtype=...
        trust_remote_code=True,
    )

    if os.environ.get("USE_FLASH_ATTENTION", "0") == "1":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main:
            print("Using flash_attention_2 kernels", flush=True)
    elif is_main:
        print("Using standard attention implementation", flush=True)

    if is_main:
        print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

    # Resize embeddings if pad token was added
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing for memory saving
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Quick sanity prints on rank 0
    if is_main:
        print("is_loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", False), flush=True)
        try:
            lm_w = model.get_output_embeddings().weight
            print("lm_head weight dtype:", lm_w.dtype, flush=True)
        except Exception:
            pass

    # --- PEFT (LoRA) ---
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

    # --- Training / FSDP Config ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        remove_unused_columns=False,
        report_to="wandb",

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,

        # FSDP: full_shard (ZeRO-3-like) for PEFT params, base 4-bit weights remain quantized
        optim="adamw_torch",
        fsdp="full_shard",
        fsdp_config={
            "fsdp_timeout": 1800,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            # pass PEFT config so FSDP knows how to wrap LoRA layers
            "fsdp_peft_config": peft_config,
            "activation_checkpointing": True,
        },

        max_steps=1000,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=250,
        learning_rate=1e-4,

        # prefer bf16 with FSDP; fp16 off
        bf16=True,
        fp16=False,

        save_on_each_node=False,
        # Align with model.gradient_checkpointing_enable()
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer_signature = inspect.signature(SFTTrainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_signature.parameters:
        trainer_kwargs["max_seq_length"] = 2048
    if "dataset_text_field" in trainer_signature.parameters and "dataset_text_field" not in trainer_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    if is_main:
        print("Starting training...", flush=True)
    trainer.train()

    # sync all ranks before saving
    dist.barrier()

    if is_main:
        print("Training finished. Saving adapters...", flush=True)
        trainer.save_model(OUTPUT_DIR)
        print(f"All done! Saved to: {OUTPUT_DIR}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
