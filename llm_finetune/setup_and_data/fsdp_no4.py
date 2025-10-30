

import os, json, inspect
from typing import Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# -------------------- Env --------------------
MODEL_PATH   = os.environ.get("MODEL_PATH",   "/mnt/data/models/Meta-Llama-3-8B-Instruct")
DATASET_PATH = os.environ.get("DATASET_PATH", "/mnt/data/datasets/xlam-function-calling-60k")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR",   "/mnt/data/output/llama-3-8b-function-calling-fsdp-no4")

os.environ.setdefault("WANDB_PROJECT", "func_calls_llm")
os.environ.setdefault("WANDB_ENTITY",  "iamnirmata-microsoft")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")

def _setup_ddp() -> Tuple[int, int, bool]:
    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires torch.distributed + CUDA.")
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    is_main = (global_rank == 0)
    if is_main:
        print(f"Initialized DDP/FSDP: world={dist.get_world_size()} rank={global_rank} local_rank={local_rank}", flush=True)
    return global_rank, local_rank, is_main

def _format_example(example: dict, eos_token: str) -> dict:
    try:
        query      = example.get("query", "")
        tools_raw  = example.get("tools", "[]")
        answers_raw= example.get("answers", "[]")
        try:
            tools = "\n".join(str(x) for x in json.loads(tools_raw))
        except Exception:
            tools = str(tools_raw)
        try:
            answers = "\n".join(str(x) for x in json.loads(answers_raw))
        except Exception:
            answers = str(answers_raw)
        text = f"<user>{query}</user>\n\n<tools>{tools}</tools>\n\n<calls>{answers}</calls>{eos_token}"
        return {"text": text}
    except Exception as e:
        print(f"format error: {e}", flush=True)
        return {"text": ""}

def _load_splits(path: str) -> Tuple[Dataset, Dataset]:
    data = load_from_disk(path)
    if isinstance(data, DatasetDict):
        train_split = data.get("train") or next(iter(data.values()))
        eval_split  = data.get("validation") or data.get("test")
        if eval_split is None:
            split = train_split.train_test_split(test_size=0.1, seed=42)
            train_split, eval_split = split["train"], split["test"]
    else:
        split = data.train_test_split(test_size=0.1, seed=42)
        train_split, eval_split = split["train"], split["test"]
    return train_split, eval_split

def main():
    global_rank, local_rank, is_main = _setup_ddp()

    # --- Tokenizer ---
    if is_main: print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            if is_main: print(f"Using EOS as PAD: pad_token_id={tokenizer.pad_token_id}", flush=True)
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            tokenizer.pad_token    = "<|pad|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
            if is_main: print(f"Added PAD token <|pad|> id={tokenizer.pad_token_id}", flush=True)
    tokenizer.padding_side = "right"

    # --- Data ---
    train_raw, eval_raw = _load_splits(DATASET_PATH)
    train_ds = train_raw.map(lambda r: _format_example(r, tokenizer.eos_token),
                             remove_columns=train_raw.column_names, desc="Formatting train split").filter(lambda r: len(r["text"])>0)
    eval_ds  = eval_raw.map(lambda r: _format_example(r, tokenizer.eos_token),
                             remove_columns=eval_raw.column_names,  desc="Formatting eval split").filter(lambda r: len(r["text"])>0)
    if is_main:
        print(f"Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}", flush=True)
        print("Example formatted prompt:\n", train_ds[0]["text"][:400], "...\n", sep="", flush=True)

    # --- Base model (pure bf16) ---
    model_kwargs = dict(dtype=torch.bfloat16, trust_remote_code=True)
    if os.getenv("USE_FLASH_ATTENTION","0") == "1":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        if is_main: print("Using flash_attention_2", flush=True)
    else:
        if is_main: print("Using standard attention", flush=True)

    if is_main: print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # --- LoRA (pre-wrap) ---
    peft_cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    # --- Force uniform bf16 dtypes BEFORE FSDP ---
    n_p = n_b = 0
    for n,p in model.named_parameters():
        if p.dtype is not torch.bfloat16:
            p.data = p.data.to(torch.bfloat16); n_p += 1
    for b in model.buffers():
        if b.dtype is torch.float32:
            b.data = b.data.to(torch.bfloat16); n_b += 1
    if is_main: print(f"Casted params to bf16: {n_p}, buffers casted: {n_b}", flush=True)

    # --- Training / FSDP (minimal + compatible) ---
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        remove_unused_columns=False,
        report_to="wandb",

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,

        optim="adamw_torch",
        fsdp="full_shard",
        fsdp_config={
            "fsdp_timeout": 1800,
            "use_orig_params": True,
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "activation_checkpointing": True,  # use this instead of gradient_checkpointing in args
        },

        max_steps=1000,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=250,
        learning_rate=1e-4,

        bf16=True,
        fp16=False,

        save_on_each_node=False,
        gradient_checkpointing=False,  # keep False; we use activation_checkpointing above
        ddp_find_unused_parameters=False,
    )

    trainer_kwargs = dict(
        model=model,              # already PEFT-wrapped
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # no tokenizer kw (your TRL build doesn't accept it)
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "max_seq_length" in sig.parameters:
        trainer_kwargs["max_seq_length"] = 2048
    if "dataset_text_field" in sig.parameters and "dataset_text_field" not in trainer_kwargs:
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)

    if is_main: print("Starting training...", flush=True)
    trainer.train()

    dist.barrier()
    if is_main:
        print("Training finished. Saving adapters...", flush=True)
        trainer.save_model(OUTPUT_DIR)
        print(f"All done! Saved to: {OUTPUT_DIR}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
