cd /workspace/distrbuted_training_tools/llm_finetune/setup_and_data
echo "Starting pre-launch setup and data scripts..."
echo "make sure the secrets are set by running: source ../../../secrets.sh"
bash 0-setup.sh
bash 1-data.sh
bash 2-model.sh
