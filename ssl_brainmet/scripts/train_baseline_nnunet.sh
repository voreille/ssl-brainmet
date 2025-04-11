#!/bin/bash

# Default GPU ID
CUDA_DEVICE=0
FOLD_ID=0

export nnUNet_raw="/home/valentin/data/target/data/nnUnet_raw"
export nnUNet_preprocessed="/home/valentin/data/target/data/nnUNet_preprocessed"
export nnUNet_results="/home/valentin/data/target/data/nnUNet_results_baseline"

export nnUNet_compile=False

TASK=510

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "Running training for fold $FOLD_ID on CUDA device $CUDA_DEVICE..."
nnUNetv2_train $TASK 3d_fullres $FOLD_ID -device cuda -num_gpus 1 
# nnUNetv2_train $TASK 3d_fullres $FOLD_ID -device cuda -num_gpus 1

echo "All folds have been trained."