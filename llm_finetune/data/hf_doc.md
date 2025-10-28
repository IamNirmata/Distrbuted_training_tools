# Fine-tuning LLMs for Function Calling with xLAM Dataset

_Authored by: [Behrooz Azarkhalili](https://github.com/behroozazarkhalili)_

This notebook demonstrates how to fine-tune language models for function calling capabilities using the **xLAM dataset** from Salesforce and **QLoRA** (Quantized Low-Rank Adaptation) technique. We'll work with popular models like Llama 3, Qwen2, Mistral, and others.

**What is Function Calling?**
Function calling enables language models to interact with external tools and APIs by generating structured function invocations. Instead of just generating text, the model learns to call specific functions with the right parameters based on user requests.

**What You'll Learn:**
- **Data Processing**: How to format the xLAM dataset for function calling training
- **Model Fine-tuning**: Using QLoRA for memory-efficient training on consumer GPUs
- **Evaluation**: Testing the fine-tuned models with example prompts
- **Multi-model Support**: Working with different model architectures

**Key Benefits:**
- **Memory Efficient**: QLoRA enables training on 16-24GB GPUs
- **Production Ready**: Modular code with proper error handling
- **Flexible Architecture**: Easy to adapt for different models and datasets
- **Universal Support**: Works with Llama, Qwen, Mistral, Gemma, Phi, and more

**Hardware Requirements:**
- **GPU**: 16GB+ VRAM (24GB recommended for larger models)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space for models and datasets

**Software Dependencies:**
The notebook will install required packages automatically, including:
- `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `accelerate`

*For detailed methodology and results, see: [Function Calling: Fine-tuning Llama 3 and Qwen2 on xLAM](https://newsletter.kaitchup.com/p/function-calling-fine-tuning-llama)*

```python
# Install required packages for function calling fine-tuning
# !uv pip install --upgrade bitsandbytes peft trl python-dotenv
```

## Basic Setup and Imports

Let's start with the essential imports and basic setup for our notebook.

```python
>>> import torch
>>> import os
>>> import warnings
>>> from typing import Dict, Any, Optional, Tuple

>>> # Set up GPU and suppress warnings for cleaner output
>>> os.environ["CUDA_VISIBLE_DEVICES"] = "0"
>>> warnings.filterwarnings("ignore")

>>> print(f"PyTorch version: {torch.__version__}")
>>> print(f"CUDA available: {torch.cuda.is_available()}")
>>> if torch.cuda.is_available():
...     print(f"GPU: {torch.cuda.get_device_name(0)}")
...     print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

<pre>
PyTorch version: 2.8.0+cu128
CUDA available: True
GPU: NVIDIA H100 NVL
VRAM: 100.0 GB
</pre>

## Hugging Face Authentication Setup

Next, we'll set up authentication with HuggingFace Hub. This allows us to download models and datasets, and optionally upload our fine-tuned models.

```python
>>> # Set up HuggingFace authentication
>>> from dotenv import load_dotenv
>>> from huggingface_hub import login

>>> # Load environment variables from .env file (optional)
>>> load_dotenv()

>>> # Authenticate with HuggingFace using token from .env file
>>> hf_token = os.getenv('hf_api_key')
>>> if hf_token:
...     login(token=hf_token)
...     print("✅ Successfully authenticated with HuggingFace!")
>>> else:
...     print("⚠️  Warning: HF_API_KEY not found in .env file")
...     print("   You can still run the notebook, but won't be able to upload models")
```

<pre>
✅ Successfully authenticated with HuggingFace!
</pre>

## Model Configuration Classes

We'll create two configuration classes to organize our settings:
1. **ModelConfig**: Stores model-specific settings like tokenizer configuration
2. **TrainingConfig**: Stores training parameters like learning rate and batch size

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model-specific settings."""
    model_name: str           # HuggingFace model identifier
    pad_token: str           # Padding token for the tokenizer
    pad_token_id: int        # Numerical ID for the padding token
    padding_side: str        # Side to add padding ('left' or 'right')
    eos_token: str          # End of sequence token
    eos_token_id: int       # End of sequence token ID
    vocab_size: int         # Vocabulary size
    model_type: str         # Model architecture type

@dataclass 
class TrainingConfig:
    """Configuration for training hyperparameters."""
    output_dir: str                    # Directory to save model checkpoints
    batch_size: int = 16              # Training batch size per device
    gradient_accumulation_steps: int = 8  # Steps to accumulate gradients
    learning_rate: float = 1e-4       # Learning rate for optimization
    max_steps: int = 1000             # Maximum training steps
    max_seq_length: int = 2048        # Maximum sequence length
    lora_r: int = 16                  # LoRA rank parameter
    lora_alpha: int = 16              # LoRA alpha scaling parameter
    lora_dropout: float = 0.05        # LoRA dropout rate
    save_steps: int = 250             # Steps between checkpoint saves
    logging_steps: int = 10           # Steps between log outputs
    warmup_ratio: float = 0.1         # Warmup ratio for learning rate
```

## Automatic Model Configuration

This function automatically detects the model's tokenizer settings and creates a proper configuration. It handles different model architectures (Llama, Qwen, Mistral, etc.) and their specific token requirements.

```python
from transformers import AutoTokenizer, AutoConfig

def auto_configure_model(model_name: str, custom_pad_token: str = None) -> ModelConfig:
    """
    Automatically configure any model by extracting information from its tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        custom_pad_token: Custom pad token if model doesn't have one
        
    Returns:
        ModelConfig: Complete model configuration
    """
    
    print(f"🔍 Loading model configuration: {model_name}")
    
    # Load tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_config = AutoConfig.from_pretrained(model_name)
    
    # Extract basic model info
    model_type = getattr(model_config, 'model_type', 'unknown')
    vocab_size = getattr(model_config, 'vocab_size', len(tokenizer.get_vocab()))
    
    print(f"📊 Model: {model_type}, vocab_size: {vocab_size:,}")
    
    # Get EOS token (required)
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    
    if eos_token is None:
        raise ValueError(f"Model '{model_name}' missing EOS token")
    
    # Get or set pad token
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id
    
    if pad_token is None:
        if custom_pad_token is None:
            raise ValueError(f"Model needs custom_pad_token. Use '<|eot_id|>' for Llama, '<|im_end|>' for Qwen")
        
        pad_token = custom_pad_token
        if pad_token in tokenizer.get_vocab():
            pad_token_id = tokenizer.get_vocab()[pad_token]
        else:
            tokenizer.add_special_tokens({'pad_token': pad_token})
            pad_token_id = tokenizer.pad_token_id
    
    print(f"✅ Configured - pad: '{pad_token}' (ID: {pad_token_id}), eos: '{eos_token}' (ID: {eos_token_id})")
    
    return ModelConfig(
        model_name=model_name,
        pad_token=pad_token,
        pad_token_id=pad_token_id,
        padding_side='left',  # Standard for causal LMs
        eos_token=eos_token,
        eos_token_id=eos_token_id,
        vocab_size=vocab_size,
        model_type=model_type
    )
```

```python
>>> def create_training_config(model_name: str, **kwargs) -> TrainingConfig:
...     """Create training configuration with automatic output directory."""
...     # Create clean directory name from model name
...     model_clean = model_name.split('/')[-1].replace('-', '_').replace('.', '_')
...     default_output_dir = f"./{model_clean}_xLAM"
    
...     config_dict = {'output_dir': default_output_dir, **kwargs}
...     return TrainingConfig(**config_dict)

... print("✅ Configuration system ready!")
... print("💡 Supports Llama, Qwen, Mistral, Gemma, Phi, and more")
```

<pre>
✅ Configuration system ready!
💡 Supports Llama, Qwen, Mistral, Gemma, Phi, and more
</pre>

## Hardware Detection and Setup

Let's detect our hardware capabilities and configure optimal settings. We'll check for bfloat16 support and set up the best attention mechanism for our GPU.

```python
def setup_hardware_config() -> Tuple[torch.dtype, str]:
    """
    Automatically detect and configure hardware-specific settings.
    
    Returns:
        Tuple[torch.dtype, str]: compute_dtype and attention_implementation
    """
    print("🔍 Detecting hardware capabilities...")
    
    if torch.cuda.is_bf16_supported():
        print("✅ bfloat16 supported - using optimal precision")
        print("📦 Installing FlashAttention for better performance...")
        
        # Install FlashAttention for supported hardware
        os.system('pip install flash_attn --no-build-isolation')
        
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
        
        print("🚀 Configuration: bfloat16 + FlashAttention 2")
    else:
        print("⚠️  bfloat16 not supported - using float16 fallback")
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'  # Scaled Dot Product Attention
        
        print("🔄 Configuration: float16 + SDPA")
    
    return compute_dtype, attn_implementation

# Configure hardware settings
compute_dtype, attn_implementation = setup_hardware_config()
```

## Tokenizer Setup Function

Now let's create a function to set up our tokenizer with the right configuration from our model settings.

```python
>>> from transformers import AutoTokenizer

>>> def setup_tokenizer(model_config: ModelConfig) -> AutoTokenizer:
...     """
...     Initialize and configure the tokenizer using model configuration.
    
...     Args:
...         model_config: Model configuration with all token information
        
...     Returns:
...         AutoTokenizer: Configured tokenizer with proper pad token settings
...     """
...     print(f"🔤 Loading tokenizer for {model_config.model_name}")
    
...     tokenizer = AutoTokenizer.from_pretrained(model_config.model_name, use_fast=True)
    
...     # Configure padding token using values from model_config
...     tokenizer.pad_token = model_config.pad_token
...     tokenizer.pad_token_id = model_config.pad_token_id
...     tokenizer.padding_side = model_config.padding_side
    
...     print(f"✅ Tokenizer configured - pad: '{model_config.pad_token}' (ID: {model_config.pad_token_id})")
    
...     return tokenizer

>>> print(f"📊 Hardware Configuration Complete:")
>>> print(f"   • Compute dtype: {compute_dtype}")
>>> print(f"   • Attention implementation: {attn_implementation}")
>>> print(f"   • Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

<pre>
📊 Hardware Configuration Complete:
   • Compute dtype: torch.bfloat16
   • Attention implementation: flash_attention_2
   • Device: NVIDIA H100 NVL
</pre>

## Dataset Processing

Now we'll work with the xLAM dataset from Salesforce. This dataset contains about 60,000 examples of function calling conversations that we'll use to train our model.

**Key Functions:**
- **`process_xlam_sample()`**: Converts a single dataset example into the training format with special tags (`<user>`, `<tools>`, `<calls>`) and EOS token
- **`load_and_process_xlam_dataset()`**: Loads the complete xLAM dataset (60K samples) from Hugging Face and processes all samples using multiprocessing for efficiency
- **`preview_dataset_sample()`**: Displays a formatted preview of a processed dataset sample for inspection with statistics

```python
import json
import multiprocessing
from datasets import load_dataset, Dataset

def process_xlam_sample(row: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Process a single xLAM dataset sample into training format.
    
    The format we create is:
    <user>[user query]</user>
    
    <tools>
    [tool definitions]
    </tools>
    
    <calls>
    [expected function calls]
    </calls>[EOS_TOKEN]
    """
    # Format user query
    formatted_query = f"<user>{row['query']}</user>\n\n"

    # Parse and format available tools
    try:
        parsed_tools = json.loads(row["tools"])
        tools_text = '\n'.join(str(tool) for tool in parsed_tools)
    except json.JSONDecodeError:
        tools_text = str(row["tools"])  # Fallback to raw string
    
    formatted_tools = f"<tools>{tools_text}</tools>\n\n"

    # Parse and format expected function calls
    try:
        parsed_answers = json.loads(row["answers"])
        answers_text = '\n'.join(str(answer) for answer in parsed_answers)
    except json.JSONDecodeError:
        answers_text = str(row["answers"])  # Fallback to raw string

    formatted_answers = f"<calls>{answers_text}</calls>"

    # Combine all parts with EOS token
    complete_text = formatted_query + formatted_tools + formatted_answers + tokenizer.eos_token

    # Update row with processed data
    row["query"] = formatted_query
    row["tools"] = formatted_tools
    row["answers"] = formatted_answers
    row["text"] = complete_text

    return row
```

```python
def load_and_process_xlam_dataset(tokenizer: AutoTokenizer, sample_size: Optional[int] = None) -> Dataset:
    """
    Load and process the complete xLAM dataset for function calling training.
    
    Args:
        tokenizer: Configured tokenizer for the model
        sample_size: Optional number of samples to use (None for full dataset)
        
    Returns:
        Dataset: Processed dataset ready for training
    """
    print("📊 Loading xLAM function calling dataset...")
    
    # Load the Salesforce xLAM dataset from Hugging Face
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    
    print(f"📋 Original dataset size: {len(dataset):,} samples")
    
    # Sample dataset if requested (useful for testing)
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
        print(f"🔬 Using sample size: {sample_size:,} samples")
    
    # Process all samples using multiprocessing for efficiency
    print("⚙️ Processing dataset samples into training format...")
    
    def process_batch(batch):
        """Process a batch of samples with the tokenizer."""
        processed_batch = []
        for i in range(len(batch['query'])):
            row = {
                'query': batch['query'][i],
                'tools': batch['tools'][i], 
                'answers': batch['answers'][i]
            }
            processed_row = process_xlam_sample(row, tokenizer)
            processed_batch.append(processed_row)
        
        # Convert to batch format
        return {
            'text': [item['text'] for item in processed_batch],
            'query': [item['query'] for item in processed_batch],
            'tools': [item['tools'] for item in processed_batch],
            'answers': [item['answers'] for item in processed_batch]
        }
    
    # Process the dataset
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,  # Process in batches for efficiency
        num_proc=min(4, multiprocessing.cpu_count()),  # Use multiple cores
        desc="Processing xLAM samples"
    )
    
    print("✅ Dataset processing complete!")
    print(f"📊 Final dataset size: {len(processed_dataset):,} samples")
    print(f"🔤 Average text length: {sum(len(text) for text in processed_dataset['text']) / len(processed_dataset):,.0f} characters")
    
    return processed_dataset
```

```python
def preview_dataset_sample(dataset: Dataset, index: int = 0) -> None:
    """
    Display a formatted preview of a dataset sample for inspection.
    
    Args:
        dataset: The processed dataset
        index: Index of the sample to preview (default: 0)
    """
    if index >= len(dataset):
        print(f"❌ Index {index} is out of range. Dataset has {len(dataset)} samples.")
        return
    
    sample = dataset[index]
    
    print(f"📋 Dataset Sample Preview (Index: {index})")
    print("=" * 80)
    
    print(f"\n🔍 Raw Components:")
    print(f"Query: {sample['query'][:200]}{'...' if len(sample['query']) > 200 else ''}")
    print(f"Tools: {sample['tools'][:200]}{'...' if len(sample['tools']) > 200 else ''}")
    print(f"Answers: {sample['answers'][:200]}{'...' if len(sample['answers']) > 200 else ''}")
    
    print(f"\n📝 Complete Training Text:")
    print("-" * 40)
    print(sample['text'])
    print("-" * 40)
    
    print(f"\n📊 Sample Statistics:")
    print(f"   • Text length: {len(sample['text']):,} characters")
    print(f"   • Estimated tokens: ~{len(sample['text']) // 4:,} tokens")
    
    print("\n✅ Preview complete!")
```

## Loading and Processing the Dataset

Now let's add functions to load the xLAM dataset and process it into the format our model needs for training.

```python
# Import QLoRA training components when we need them
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def create_qlora_model(model_config: ModelConfig, 
                       tokenizer: AutoTokenizer,
                       compute_dtype: torch.dtype, 
                       attn_implementation: str) -> AutoModelForCausalLM:
    """
    Create and configure a QLoRA-enabled model for efficient fine-tuning.
    
    QLoRA uses 4-bit quantization and low-rank adapters to enable
    fine-tuning large models on consumer GPUs.
    """
    print(f"🏗️  Creating QLoRA model: {model_config.model_name}")
    
    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",           # Use NF4 quantization
        bnb_4bit_compute_dtype=compute_dtype, # Computation data type
        bnb_4bit_use_double_quant=True,      # Double quantization for more memory savings
    )
    
    print("📦 Loading quantized model...")
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},                  # Load on first GPU
        attn_implementation=attn_implementation,
        torch_dtype=compute_dtype,
        trust_remote_code=True,              # Required for some models
    )
    
    # Prepare model for k-bit training (required for QLoRA)
    model = prepare_model_for_kbit_training(
        model, 
        gradient_checkpointing_kwargs={'use_reentrant': True}
    )
    
    # Configure tokenizer settings in model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable cache for training
    
    print("✅ QLoRA model prepared successfully!")
    print(f"💾 Model memory footprint: ~{model.get_memory_footprint() / 1e9:.1f} GB")
    
    return model
```

## QLoRA Training Setup

QLoRA (Quantized Low-Rank Adaptation) allows us to fine-tune large language models efficiently. It uses 4-bit quantization to reduce memory usage while maintaining training quality.

```python
def create_lora_config(training_config: TrainingConfig) -> LoraConfig:
    """
    Create LoRA configuration for parameter-efficient fine-tuning.
    
    LoRA (Low-Rank Adaptation) adds small trainable matrices to specific
    model layers while keeping the base model frozen.
    
    Args:
        training_config (TrainingConfig): Training configuration with LoRA parameters
        
    Returns:
        LoraConfig: Configured LoRA adapter settings
        
    LoRA Parameters:
        - r (rank): Dimensionality of adaptation matrices (higher = more capacity)
        - alpha: Scaling factor for LoRA weights
        - dropout: Regularization to prevent overfitting
        - target_modules: Which model layers to adapt
    """
    print("⚙️ Configuring LoRA adapters...")
    
    # Target modules for both Llama and Qwen architectures
    target_modules = [
        'k_proj', 'q_proj', 'v_proj', 'o_proj',  # Attention projections
        "gate_proj", "down_proj", "up_proj"       # Feed-forward projections
    ]
    
    lora_config = LoraConfig(
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        r=training_config.lora_r,
        bias="none",                             # Don't adapt bias terms
        task_type="CAUSAL_LM",                   # Causal language modeling
        target_modules=target_modules
    )
    
    print(f"🎯 LoRA targeting modules: {target_modules}")
    print(f"📊 LoRA parameters: r={training_config.lora_r}, alpha={training_config.lora_alpha}")
    
    return lora_config
```

## LoRA Configuration

LoRA (Low-Rank Adaptation) is the key technique that makes efficient fine-tuning possible. Instead of updating all model parameters, LoRA adds small trainable matrices to specific layers while keeping the base model frozen.

## Training Execution

Now we'll create the main training function that puts everything together. This function configures the training arguments and executes the fine-tuning process using TRL's SFTTrainer.

```python
def train_qlora_model(dataset: Dataset, 
                      model: AutoModelForCausalLM,
                      training_config: TrainingConfig,
                      compute_dtype: torch.dtype) -> SFTTrainer:
    """
    Execute QLoRA fine-tuning with comprehensive configuration and monitoring.
    
    Args:
        dataset (Dataset): Processed training dataset
        model (AutoModelForCausalLM): QLoRA-configured model
        training_config (TrainingConfig): Training hyperparameters
        compute_dtype (torch.dtype): Computation data type
        
    Returns:
        SFTTrainer: Trained model trainer
        
    Training Features:
        - Supervised fine-tuning with SFTTrainer
        - Memory-optimized settings for consumer GPUs
        - Comprehensive logging and checkpointing
        - Automatic mixed precision training
    """
    print("🚀 Starting QLoRA fine-tuning...")
    
    # Create LoRA configuration
    peft_config = create_lora_config(training_config)
    
    # Configure training arguments
    training_arguments = SFTConfig(
        output_dir=training_config.output_dir,
        optim="adamw_8bit",                      # 8-bit optimizer for memory efficiency
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        log_level="info",                        # Detailed logging
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        learning_rate=training_config.learning_rate,
        fp16=compute_dtype == torch.float16,     # Use FP16 if not using bfloat16
        bf16=compute_dtype == torch.bfloat16,    # Use bfloat16 if supported
        max_steps=training_config.max_steps,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type="linear",
        dataset_text_field="text",               # Field containing training text
        max_length=training_config.max_seq_length,
        remove_unused_columns=False,             # Keep all dataset columns
        
        # Additional stability and performance settings
        dataloader_drop_last=True,               # Drop incomplete batches
        gradient_checkpointing=True,             # Enable gradient checkpointing
        save_total_limit=3,                      # Keep only 3 most recent checkpoints
        load_best_model_at_end=False,            # Don't load best model (saves memory)
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
    )
    
    print(f"📊 Training configuration:")
    print(f"   • Dataset size: {len(dataset):,} samples")
    print(f"   • Batch size: {training_config.batch_size}")
    print(f"   • Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   • Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"   • Max steps: {training_config.max_steps:,}")
    print(f"   • Learning rate: {training_config.learning_rate}")
    print(f"   • Output directory: {training_config.output_dir}")
    
    # Start training
    print("\n🏁 Beginning training...")
    trainer.train()
    
    print("✅ Training completed successfully!")
    
    return trainer
```

## 🎯 Universal Model Selection

**Choose any model for fine-tuning!** This notebook supports a wide range of popular models. Simply uncomment the model you want to use or specify your own.

### 📋 Quick Model Selection
Uncomment one of these popular models or specify your own:

**Why Llama 3-8B-Instruct as default?**
- **Proven Performance**: Excellent function calling capabilities and instruction following
- **Optimal Size**: 8B parameters provide great balance between performance and resource usage

```python
>>> # 🎯 ONE-LINE MODEL CONFIGURATION 🎯
>>> # Just specify any Hugging Face model and its custom pad token - everything else is automatic!

>>> # === Simply change this line to use ANY model ===
>>> MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
>>> custom_pad_token = "<|eot_id|>"  
>>> # Use '<|eot_id|>' for Llama3+ models, '<|im_end|>' for Qwen2+ models, '</s>' for Mistral models, '<|end|>' for Phi3+ models

>>> # === Popular alternatives (uncomment to use) ===
>>> # MODEL_NAME = "Qwen/Qwen2-7B-Instruct"                # Qwen2 
>>> # MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"    # Mistral 
>>> # MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"      # Phi-3 Mini 
>>> # MODEL_NAME = "google/gemma-1.1-7b-it"                # Gemma 
>>> # MODEL_NAME = "your-custom-model/model-name"          # Any custom model

>>> print(f"🎯 Selected Model: {MODEL_NAME}")

>>> # 🚀 AUTOMATIC CONFIGURATION - No manual setup needed!
>>> print(f"\n🔧 Auto-configuring everything for {MODEL_NAME}...")

>>> # Extract ALL information automatically using transformers
>>> model_config = auto_configure_model(MODEL_NAME, custom_pad_token=custom_pad_token) 
>>> training_config = create_training_config(MODEL_NAME)

>>> print(f"\n🎉 Ready to fine-tune! Everything configured automatically:")
>>> print(f"   ✅ Model type: {model_config.model_type}")
>>> print(f"   ✅ Vocabulary: {model_config.vocab_size:,} tokens")
>>> print(f"   ✅ Pad token: '{model_config.pad_token}' (ID: {model_config.pad_token_id})")
>>> print(f"   ✅ Output dir: {training_config.output_dir}")

>>> print(f"\n🚀 Configuration complete for {MODEL_NAME}!")
```

<pre>
🎯 Selected Model: meta-llama/Meta-Llama-3-8B-Instruct

🔧 Auto-configuring everything for meta-llama/Meta-Llama-3-8B-Instruct...
🔍 Loading model configuration: meta-llama/Meta-Llama-3-8B-Instruct
📊 Model: llama, vocab_size: 128,256
✅ Configured - pad: '<|eot_id|>' (ID: 128009), eos: '<|eot_id|>' (ID: 128009)

🎉 Ready to fine-tune! Everything configured automatically:
   ✅ Model type: llama
   ✅ Vocabulary: 128,256 tokens
   ✅ Pad token: '<|eot_id|>' (ID: 128009)
   ✅ Output dir: ./Meta_Llama_3_8B_Instruct_xLAM

🚀 Configuration complete for meta-llama/Meta-Llama-3-8B-Instruct!
</pre>

```python
# Universal fine-tuning pipeline - works with any model!
print(f"🚀 Starting fine-tuning pipeline for {model_config.model_name}")

# Step 1: Setup tokenizer
print(f"\n📝 Setting up tokenizer...")
tokenizer = setup_tokenizer(model_config)

# Step 2: Load and process dataset
print(f"\n📊 Loading and processing xLAM dataset...")
dataset = load_and_process_xlam_dataset(tokenizer, sample_size=None)  # Set sample_size for testing

# Step 3: Preview dataset sample
print(f"\n👀 Dataset sample preview:")
preview_dataset_sample(dataset, index=0)

# Step 4: Create QLoRA model
print(f"\n🏗️  Creating QLoRA model...")
model = create_qlora_model(
    model_config, 
    tokenizer, 
    compute_dtype, 
    attn_implementation
)

# Step 5: Execute training
print(f"\n🎯 Starting training...")
trainer = train_qlora_model(
    dataset=dataset,
    model=model,
    training_config=training_config,
    compute_dtype=compute_dtype
)

print(f"\n🎉 Fine-tuning completed for {model_config.model_name.split('/')[-1]}!")
print(f"📁 Model saved to: {training_config.output_dir}")
print(f"🔍 To test the model, run the inference cells below")
```

## Model Loading for Inference

After training is complete, we need to load the trained model for inference. This function loads the base model with quantization and applies the trained LoRA adapters.

```python
# Import required components for inference
from peft import PeftModel

def load_trained_model(model_config: ModelConfig, 
                       adapter_path: str,
                       compute_dtype: torch.dtype,
                       attn_implementation: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a trained model with LoRA adapters for inference.
    
    This function loads the base model with quantization and applies the trained
    LoRA adapters for efficient inference. It's designed to work after training
    completion or for loading previously saved models.
    
    Args:
        model_config (ModelConfig): Configuration for the base model
        adapter_path (str): Path to the saved LoRA adapter
        compute_dtype (torch.dtype): Computation data type
        attn_implementation (str): Attention implementation
        
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
        
    Note:
        You may need to restart the notebook to free GPU memory before loading
        the model for inference, especially after training.
    """
    print(f"🔄 Loading trained model from {adapter_path}")
    
    # Configure quantization for inference
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load tokenizer with proper configuration
    tokenizer = setup_tokenizer(model_config)
    print(f"🔤 Tokenizer loaded for {model_config.model_name}")
    
    # Load base model
    print(f"📦 Loading base model {model_config.model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    
    # Load LoRA adapters
    print(f"🔗 Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Enable evaluation mode
    model.eval()
    
    print("✅ Model loaded successfully and ready for inference!")
    print(f"💾 Total memory usage: ~{model.get_memory_footprint() / 1e9:.1f} GB")
    
    return model, tokenizer
```

## Text Generation for Function Calls

Now let's create the function that generates responses from our fine-tuned model. This handles tokenization, generation parameters, and decoding.

```python
def generate_function_call(model: AutoModelForCausalLM,
                          tokenizer: AutoTokenizer, 
                          prompt: str,
                          max_new_tokens: int = 512,
                          temperature: float = 0.7,
                          do_sample: bool = True) -> str:
    """
    Generate a function call response using the fine-tuned model.
    
    Args:
        model (AutoModelForCausalLM): Fine-tuned model with LoRA adapters
        tokenizer (AutoTokenizer): Model tokenizer
        prompt (str): Input prompt for function calling
        max_new_tokens (int): Maximum tokens to generate
        temperature (float): Sampling temperature (only used when do_sample=True)
        do_sample (bool): Whether to use sampling
        
    Returns:
        str: Generated response with function calls
        
    Example Prompt Format:
        "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>"
    """
    print(f"🎯 Generating response for prompt...")
    print(f"📝 Input: {prompt}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate response with proper parameter handling
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    
    # Only add sampling parameters if do_sample=True
    if do_sample:
        generation_kwargs["temperature"] = temperature
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )
    
    # Decode result
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("✅ Generation completed!")
    print(f"📊 Generated {len(outputs[0]) - len(inputs['input_ids'][0])} new tokens")
    
    return result
```

## Testing Function Calling Capabilities

This function provides a comprehensive test suite to evaluate our fine-tuned model with different types of function calling scenarios.

```python
def test_function_calling_examples(model: AutoModelForCausalLM, 
                                  tokenizer: AutoTokenizer) -> None:
    """
    Test the model with various function calling examples.
    
    Args:
        model (AutoModelForCausalLM): Fine-tuned model
        tokenizer (AutoTokenizer): Model tokenizer
    """
    print("🧪 Testing function calling capabilities...")
    
    test_cases = [
        {
            "name": "Mathematical Function",
            "prompt": "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>"
        },
        {
            "name": "Weather Query", 
            "prompt": "<user>What's the weather like in New York today?</user>\n\n<tools>"
        },
        {
            "name": "Data Processing",
            "prompt": "<user>Calculate the average of these numbers: 10, 20, 30, 40, 50</user>\n\n<tools>"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        result = generate_function_call(
            model=model,
            tokenizer=tokenizer,
            prompt=test_case["prompt"],
            max_new_tokens=512,  # Adjust as needed
            temperature=0.7,
            do_sample=True  # Fixed: Use sampling with temperature
        )
        
        print(f"\n🔍 Complete Response:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    
    print("\n✅ All test cases completed!")
```

```python
# Load and test the trained model
# Note: You may need to restart the notebook to free memory before running this

print("🔄 Loading trained model for testing...")
print("⚠️  If you encounter memory issues, restart the notebook and run only this cell")

# Determine the adapter path based on the training configuration
adapter_path = f"{training_config.output_dir}/checkpoint-{training_config.max_steps}"

print(f"📁 Looking for adapter at: {adapter_path}")

# Load the trained model
trained_model, trained_tokenizer = load_trained_model(
    model_config=model_config,
    adapter_path=adapter_path,
    compute_dtype=compute_dtype,
    attn_implementation=attn_implementation
)

# Test with a single example
test_prompt = "<user>Check if the numbers 8 and 1233 are powers of two.</user>\n\n<tools>"
result = generate_function_call(trained_model, trained_tokenizer, test_prompt)

print(f"\n🎯 Test Result for {model_config.model_name.split('/')[-1]}:")
print("="*80)
print(result)
print("="*80)
```

```python
>>> # Run comprehensive testing suite for your trained model
>>> test_function_calling_examples(trained_model, trained_tokenizer)
```

<pre>
🧪 Testing function calling capabilities...

============================================================
Test Case 1: Mathematical Function
============================================================
🎯 Generating response for prompt...
📝 Input: <user>Check if the numbers 8 and 1233 are powers of two.</user>

<tools>
✅ Generation completed!
📊 Generated 90 new tokens

🔍 Complete Response:
----------------------------------------
<user>Check if the numbers 8 and 1233 are powers of two.</user>

<tools>{'name': 'is_power_of_two', 'description': 'Checks if a number is a power of two.', 'parameters': {'num': {'description': 'The number to check.', 'type': 'int'}}}</tools>

<calls>{'name': 'is_power_of_two', 'arguments': {'num': 8}}
{'name': 'is_power_of_two', 'arguments': {'num': 1233}}</calls>
----------------------------------------

============================================================
Test Case 2: Weather Query
============================================================
🎯 Generating response for prompt...
📝 Input: <user>What's the weather like in New York today?</user>

<tools>
✅ Generation completed!
📊 Generated 105 new tokens

🔍 Complete Response:
----------------------------------------
<user>What's the weather like in New York today?</user>

<tools>{'name':'realtime_weather_api', 'description': 'Fetches current weather information based on the provided query parameter.', 'parameters': {'q': {'description': 'Query parameter used to specify the location for which weather data is required. It can be in various formats such as:', 'type':'str', 'default': '53.1,-0.13'}}}</tools>

<calls>{'name':'realtime_weather_api', 'arguments': {'q': 'New York'}}</calls>
----------------------------------------

============================================================
Test Case 3: Data Processing
============================================================
🎯 Generating response for prompt...
📝 Input: <user>Calculate the average of these numbers: 10, 20, 30, 40, 50</user>

<tools>
✅ Generation completed!
📊 Generated 81 new tokens

🔍 Complete Response:
----------------------------------------
<user>Calculate the average of these numbers: 10, 20, 30, 40, 50</user>

<tools>{'name': 'average', 'description': 'Calculates the arithmetic mean of a list of numbers.', 'parameters': {'numbers': {'description': 'The list of numbers.', 'type': 'List[float]'}}}</tools>

<calls>{'name': 'average', 'arguments': {'numbers': [10, 20, 30, 40, 50]}}</calls>
----------------------------------------

✅ All test cases completed!
</pre>

## 🎉 Conclusion and Next Steps

---

### 📊 Summary

This notebook demonstrated a **complete, production-ready, universal pipeline** for fine-tuning language models for function calling capabilities using:

- **🎯 Universal Model Support**: Works with any model - just change the `MODEL_NAME` variable
- **🔧 Intelligent Configuration**: Automatic token detection using `auto_configure_model()`
- **⚡ QLoRA Efficiency**: Memory-efficient training on consumer GPUs (16-24GB)
- **📋 Comprehensive Testing**: Automated evaluation and interactive testing capabilities

### 🚀 Key Improvements Made

#### **Universal Compatibility**
- ✅ **Multi-Model Support**: Works with Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, Yi, and more
- ✅ **Smart Token Detection**: Automatically finds pad/EOS tokens from any model's tokenizer
- ✅ **Error Prevention**: Validates configurations and provides helpful error messages
- ✅ **Flexible Architecture**: Easy to add new models without code changes

#### **Code Quality**
- ✅ **Type Hints**: Full type annotations for better IDE support and error catching
- ✅ **Docstrings**: Comprehensive documentation for all functions
- ✅ **Error Handling**: Robust error handling with informative messages
- ✅ **Modular Design**: Clean separation of concerns and reusable components

#### **User Experience**  
- ✅ **One-Line Model Selection**: Simply change `MODEL_NAME` variable
- ✅ **Automatic Configuration**: Everything extracted from transformers automatically
- ✅ **Clear Progress Indicators**: Emojis and detailed logging throughout
- ✅ **Production Ready**: Code suitable for research and deployment

### 🔄 Next Steps and Extensions

#### **Model Improvements**
1. **Try Different Models**: Simply change the `MODEL_NAME` variable and re-run
2. **Hyperparameter Tuning**: Experiment with different LoRA ranks, learning rates
3. **Extended Training**: Try multi-epoch training for better convergence

#### **Evaluation Enhancements**
1. **Quantitative Metrics**: Add BLEU, ROUGE, or custom function calling accuracy
2. **Benchmark Datasets**: Test on additional function calling benchmarks
3. **Multi-Model Comparison**: Compare performance across different model families

#### **Deployment Options**
1. **Model Serving**: Deploy with FastAPI, TensorRT, or vLLM
2. **Integration**: Connect with real APIs and function execution environments
3. **Optimization**: Implement model quantization and pruning for production

#### **Additional Features**
1. **Multi-turn Conversations**: Extend to handle conversation context
2. **Tool Selection**: Improve tool selection and reasoning capabilities
3. **Error Recovery**: Add error handling and recovery mechanisms

### 📚 Resources and References

- **xLAM Dataset**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **QLoRA Paper**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Function Calling Guide**: [Complete methodology article](https://newsletter.kaitchup.com/p/function-calling-fine-tuning-llama)
- **PEFT Library**: [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)

### 🎖️ Achievement Unlocked

**🏆 Universal Function Calling Fine-tuning Master!**

You now have a production-ready system that can fine-tune virtually any open-source language model for function calling with just a single line change!

---

**Happy Fine-tuning! 🚀** Try different models, share your results, and contribute back to the community!

<EditOnGithub source="https://github.com/huggingface/cookbook/blob/main/notebooks/en/function_calling_fine_tuning_llms_on_xlam.md" />