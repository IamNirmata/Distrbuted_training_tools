# Download the XLAM-60k dataset
echo "Downloading and saving XLAM-60k dataset to /mnt/data/datasets/..."
python -c "from datasets import load_dataset; \
dataset = load_dataset('Salesforce/xlam-function-calling-60k', split='train'); \
dataset.save_to_disk('/mnt/data/datasets/xlam-function-calling-60k')"