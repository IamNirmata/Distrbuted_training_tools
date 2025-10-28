# Use the official NVIDIA PyTorch image, as specified in your original plan
FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /app

# Install dependencies for the XLAM example
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    deepspeed \
    # --- Libs required by the HF Cookbook ---
    peft \
    bitsandbytes \
    trl \
    # --- Performance for H100 ---
    # As recommended in hf_doc.md for H100s
    flash_attn --no-build-isolation \
    # --- Monitoring ---
    wandb \
    tensorboard \
    # --- Utils ---
    tqdm \
    huggingface_hub

# Copy all local files into the container
# This ensures the training script is inside the image
COPY 01-training-script.py /app/01-training-script.py

CMD ["/bin/bash"]
