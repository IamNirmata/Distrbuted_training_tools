import os
import torch
import wandb
import json
import multiprocessing
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

# ---
# This script is heavily adapted from 'hf_doc.md' to work in a
# distributed multi-node environment, launched by torchrun.
# ---

# Initialize accelerator. SFTTrainer will automatically use this.
# The env vars (RANK, WORLD_SIZE, etc.) are injected by the PyTorchJob
accelerator = Accelerator()

# --- 1. Configuration ---
# Model from hf_doc.md and user request
model_name = "/mnt/data/models/Meta-Llama-3-8B-Instruct"

# Dataset path (downloaded in step 1)
dataset_path = "/mnt/data/datasets/xlam-function-calling-60k"

# Output directory on shared storage
output_dir = "/mnt/data/output/llama-3-8b-function-calling"

# W&B Configuration from user request
os.environ["WANDB_PROJECT"] = "func_calls_llm"
os.environ["WANDB_ENTITY"] = "iamnirmata-microsoft"
os.environ["WANDB_LOG_MODEL"] = "checkpoint" # Log model checkpoints

# Llama 3's specific pad token (from hf_doc.md)
# It uses the end-of-turn token for padding
CUSTOM_PAD_TOKEN = "<|eot_id|>"

# --- 2. Load Tokenizer & Configure Model ---
if accelerator.is_main_process:
    print(f"--- Main Process (Rank {accelerator.process_index}) ---")
    print(f"Loading tokenizer for {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': CUSTOM_PAD_TOKEN})
    if accelerator.is_main_process:
        print(f"Added custom pad token: {CUSTOM_PAD_TOKEN}")

tokenizer.padding_side = "right" # Follows hf_doc.md's implied causal LM setup

# --- 3. Dataset Processing (from hf_doc.md) ---

def process_xlam_sample(row: dict) -> dict:
    """
    Process a single xLAM dataset sample into training format.
    Format:
    <user>[user query]</user>
    <tools>[tool definitions]</tools>
    <calls>[expected function calls]</calls>[EOS_TOKEN]
    """
    try:
        # Format user query
        formatted_query = f"<user>{row['query']}</user>\n\n"

        # Parse and format available tools
        try:
            parsed_tools = json.loads(row["tools"])
            tools_text = '\n'.join(str(tool) for tool in parsed_tools)
        except json.JSONDecodeError:
            tools_text = str(row["tools"])  # Fallback
        formatted_tools = f"<tools>{tools_text}</tools>\n\n"

        # Parse and format expected function calls
        try:
            parsed_answers = json.loads(row["answers"])
            answers_text = '\n'.join(str(answer) for answer in parsed_answers)
        except json.JSONDecodeError:
            answers_text = str(row["answers"])  # Fallback
        formatted_answers = f"<calls>{answers_text}</calls>"

        # Combine all parts with EOS token
        # Note: Llama-3's EOS is <|eot_id|>, which is also our pad token
        complete_text = formatted_query + formatted_tools + formatted_answers + tokenizer.eos_token

        row["text"] = complete_text
        return row
    except Exception as e:
        print(f"Error processing row: {e}")
        row["text"] = "" # Return empty text on error
        return row

# Load and process dataset
# We only want the main process to do the processing,
# then all other processes will load from disk.
if accelerator.is_main_process:
    print(f"Loading and processing dataset from {dataset_path}...")
    raw_dataset = load_from_disk(dataset_path)

    # Process all samples
    print("Processing dataset... (this may take a moment)")
    processed_dataset = raw_dataset.map(
        process_xlam_sample,
        num_proc=max(1, multiprocessing.cpu_count() // 2),
        desc="Processing xLAM samples"
    )
    
    # Filter out any failed rows
    processed_dataset = processed_dataset.filter(lambda x: len(x['text']) > 0)

    print(f"Processing complete. Total samples: {len(processed_dataset)}")
    print("Splitting dataset (90% train, 10% test)...")
    
    # Split into train/test
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Save to disk so other processes can load it
    split_dataset.save_to_disk(f"{output_dir}/processed_dataset")
    print(f"Processed dataset saved to {output_dir}/processed_dataset")

# Wait for main process to finish processing and saving
accelerator.wait_for_everyone()

# All processes load the pre-processed dataset
if accelerator.is_main_process:
    print("All processes loading processed dataset from disk...")
dataset_splits = load_from_disk(f"{output_dir}/processed_dataset")
train_dataset = dataset_splits["train"]
eval_dataset = dataset_splits["test"]

if accelerator.is_main_process:
    print(f"Loaded train samples: {len(train_dataset)}")
    print(f"Loaded test samples: {len(eval_dataset)}")
    print("--- Dataset Preview (Rank 0) ---")
    print(train_dataset[0]['text'])
    print("---------------------------------")


# --- 4. Model & QLoRA Setup (from hf_doc.md) ---

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bf16 for H100
    bnb_4bit_use_double_quant=True,
)

# if accelerator.is_main_process:
#     print("Loading base model with 4-bit quantization and flash_attention_2...")

# # Load model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map={"": accelerator.process_index}, # Map model to current device
#     attn_implementation="flash_attention_2", # Critical for H100
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )


if accelerator.is_main_process:
    print("Loading base model with 4-bit quantization and flash_attention_2...")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # REMOVE THE device_map ARGUMENT
    # device_map={"": accelerator.process_index}, # <-- DELETE THIS LINE
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)



# Resize token embeddings if we added a new pad token
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# PEFT (LoRA) config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    # Target modules from hf_doc.md for Llama
    target_modules=[
        'k_proj', 'q_proj', 'v_proj', 'o_proj',
        "gate_proj", "down_proj", "up_proj"
    ]
)

# --- 5. Training Arguments (SFTConfig from hf_doc.md) ---
training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,  # Fits well on H100
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 * 16 = 512
    
    optim="paged_adamw_8bit", # Use 8-bit optimizer for memory
    
    save_steps=250,
    logging_steps=10,
    learning_rate=1e-4,
    
    bf16=True, # Enable bfloat16 for H100
    fp16=False,
    
    max_steps=1000, # From hf_doc.md
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    
    # max_seq_length=2048, # From hf_doc.md
    
    # --- Distributed Training & Logging ---
    dataset_text_field="text",
    remove_unused_columns=False,
    
    # Report to W&B (will be picked up by main process)
    report_to="wandb",
    
    # Save checkpoints only on the main process
    save_on_each_node=False,
    
    # Evaluation
    evaluation_strategy="no",
    # eval_steps=100,
)

# --- 6. Initialize SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_arguments,
    max_seq_length=2048,
)

# --- 7. Train ---
if accelerator.is_main_process:
    print("--- Starting Training ---")
    
trainer.train()

if accelerator.is_main_process:
    print("--- Training Finished ---")

# --- 8. Save Final Model (only on main process) ---
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print(f"Saving final adapter model to {output_dir}")
    trainer.save_model(output_dir)
    print("Model saved successfully.")

print(f"Script finished on Rank {accelerator.process_index}.")
