#!/bin/bash

# Configuration - MODIFY THESE AS NEEDED
GPU_ID=0  # Specify your GPU ID here


# Experiment configurations
TASKS=("cola" "mrpc" "rte")
LEARNING_RATES=(1e-4 5e-4 1e-3 5e-3)
RANK=24

# Function to run an experiment
run_experiment() {
    local task=$1
    local lr=$2
    
    echo "Running ${METHOD} experiment: Task=$task LR=$lr on GPU ${GPU_ID}"
    
    # Run the experiment
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 train_glue.py \
        --task "$task" \
        --lr "$lr" \
        --lora_r "$RANK" \
    
    # Small delay between experiments
    sleep 2
}

# Run experiments sequentially
for task in "${TASKS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "$task" "$lr" "$seed"
        done
    done
done

echo "All experiments completed! Logs are stored in ${LOGS_DIR}"