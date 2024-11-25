#!/bin/bash

# Configuration
BASE_MODEL="meta-llama/Llama-3.2-3B"
LORA_R=96
GPU_ID=0  # Specify which GPU to use

# Base directory for experiments
BASE_DIR="experiments/commonsense_reasoning"
MODEL_NAME=$(basename "$BASE_MODEL")

echo "=== Starting Training ==="
CUDA_VISIBLE_DEVICES=$GPU_ID python train_cr.py \
    --model $BASE_MODEL \
    --lora_r $LORA_R \

# Find the most recent run directory
RUN_DIR=$(ls -td "$BASE_DIR"/"$MODEL_NAME"/"$METHOD"/* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "Error: Could not find run directory"
    echo "Searching in: $BASE_DIR/$MODEL_NAME/$METHOD/"
    echo "Current directory contents:"
    ls -R "$BASE_DIR" 2>/dev/null || echo "Base directory not found"
    exit 1
fi
echo "Found run directory: $RUN_DIR"

# Get the final and merged model paths
FINAL_MODEL_PATH="$RUN_DIR/final_model"
MERGED_MODEL_PATH="$RUN_DIR/merged_model"

echo "=== Starting Merging ==="
if [ ! -d "$FINAL_MODEL_PATH" ]; then
    echo "Error: Final model not found at $FINAL_MODEL_PATH"
    exit 1
fi

MERGE_SCRIPT="utils.merge_adapter_to_base_model"

CUDA_VISIBLE_DEVICES=$GPU_ID python -m $MERGE_SCRIPT \
    --base_model $BASE_MODEL \
    --adapter "$FINAL_MODEL_PATH" \
    --output_path "$MERGED_MODEL_PATH"

echo "=== Starting Evaluation ==="
# Check if merged model exists
if [ ! -d "$MERGED_MODEL_PATH" ]; then
    echo "Error: Merged model not found at $MERGED_MODEL_PATH"
    exit 1
fi

declare -a datasets=(
    "ARC-Challenge"
    "ARC-Easy"
    "boolq"
    "hellaswag"
    "openbookqa"
    "piqa"
    "social_i_qa"
    "winogrande"
)

# Loop through datasets and evaluate
for dataset in "${datasets[@]}"; do
    echo "=== Evaluating on $dataset ==="
    
    # Convert dataset name to lowercase for file path if needed
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/commonsense_eval_2.py \
        --model "$MERGED_MODEL_PATH" \
        --dataset "$dataset" \
        --data_file "data/commonsense/$dataset_lower/test.json" \
        --batch_size 64 \
        --tensor_parallel_size 1 \
        --run_dir "$RUN_DIR"
done

# Clean up merged model directory
echo "=== Cleaning up merged model ==="
if [ -d "$MERGED_MODEL_PATH" ]; then
    echo "Removing merged model directory: $MERGED_MODEL_PATH"
    rm -rf "$MERGED_MODEL_PATH"
    if [ $? -eq 0 ]; then
        echo "Successfully removed merged model directory"
    else
        echo "Warning: Failed to remove merged model directory"
    fi
else
    echo "Merged model directory not found - nothing to clean up"
fi

echo "=== Pipeline Complete ==="
echo "Run directory: $RUN_DIR"
echo "Final model: $FINAL_MODEL_PATH"
echo "Merged model: $MERGED_MODEL_PATH"

# Save run information
echo "Saving run information..."
cat << EOF > "$RUN_DIR/run_info.txt"
Run completed at: $(date)
Base model: $BASE_MODEL
LoRA rank: $LORA_R
EOF