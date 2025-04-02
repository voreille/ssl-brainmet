#!/bin/bash

# Default GPU ID
CUDA_DEVICE=1
PRETRAINED_WEIGHTS=data/pretrained_weights/checkpoint.pt

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Loop through fold IDs 0 to 4
for FOLD_ID in {0..4}; do
    echo "Running training for fold $FOLD_ID on CUDA device $CUDA_DEVICE..."
    nnUNetv2_train 511 3d_fullres $FOLD_ID -device cuda -num_gpus 1 -pretrained_weights $PRETRAINED_WEIGHTS
done

echo "All folds have been trained."