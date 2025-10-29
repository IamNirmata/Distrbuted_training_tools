"""Simplified multi-node fine-tuning script for Meta Llama 3."""

import inspect
import json
import os
from typing import Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/data/models/Meta-Llama-3-8B-Instruct")
DATASET_PATH = os.environ.get("DATASET_PATH", "/mnt/data/datasets/xlam-function-calling-60k")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/data/output/llama-3-8b-function-calling")

os.environ.setdefault("WANDB_PROJECT", "func_calls_llm")
os.environ.setdefault("WANDB_ENTITY", "iamnirmata-microsoft")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")

CUSTOM_PAD_TOKEN = "<|eot_id|>"


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
    accelerator = Accelerator()
    is_main_process = accelerator.is_local_main_process

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
    )
    if os.environ.get("USE_FLASH_ATTENTION", "0") == "1":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main_process:
            print("Using flash_attention_2 kernels")
    elif is_main_process:
        print("Using standard attention implementation")

    if is_main_process:
        print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

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

    if is_main_process:
        print("Starting training...")
    trainer.train()

    accelerator.wait_for_everyone()
    if is_main_process:
        print("Training finished. Saving adapters...")
    trainer.save_model(OUTPUT_DIR)

    if is_main_process:
        print("All done!")


if __name__ == "__main__":
    main()
