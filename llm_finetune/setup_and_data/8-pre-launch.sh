cd /workspace/distrbuted_training_tools
git pull origin main
cd /workspace/distrbuted_training_tools/llm_finetune/setup_and_data
echo "Starting pre-launch setup and data scripts..."
echo "make sure the secrets are set by running: source ../../../secrets.sh"
bash 0-setup.sh
bash 1-data.sh
bash 2-model.sh



echo "Pre-launch setup and data scripts completed."

ls -lart /mnt/data/
ls -lart /mnt/data/models/
ls -lart /mnt/data/datasets/


"""
# Set environment variables for this node
export MASTER_ADDR="10.1.46.93"
export MASTER_PORT=12345
export WORLD_SIZE=16
export NODE_RANK=0

torchrun --nproc_per_node=8 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         fsdp.py

# Set environment variables for this node
export MASTER_ADDR="10.1.46.93"  # <-- IP of Node 0
export MASTER_PORT=12345
export WORLD_SIZE=16
export NODE_RANK=1

torchrun --nproc_per_node=8 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         fsdp.py





"""