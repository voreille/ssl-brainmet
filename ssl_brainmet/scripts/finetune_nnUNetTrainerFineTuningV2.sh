#!/bin/bash

# Default GPU ID
CUDA_DEVICE=0
FOLD_ID=0

export nnUNet_raw="/home/valentin/data/target/data/nnUnet_raw"
export nnUNet_preprocessed="/home/valentin/data/target/data/nnUNet_preprocessed"
export nnUNet_results="/home/valentin/data/target/data/nnUNet_results_finetuning_v2_exp2"

export nnUNet_compile=False

TASK=510

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "Running training for fold $FOLD_ID on CUDA device $CUDA_DEVICE..."
nnUNetv2_train_with_config $TASK 3d_fullres $FOLD_ID -device cuda -num_gpus 1 -tr nnUNetTrainerFineTuningV2 -trainer_config /home/valentin/workspaces/ssl-brainmet/ssl_brainmet/config/finetuning/nnUNetTrainerFineTuningV2.yaml