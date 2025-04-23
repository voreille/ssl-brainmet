#!/bin/bash

# Default GPU ID and Fold ID
CUDA_DEVICE=0
FOLD_ID=0

# Define paths for the raw and preprocessed data
export nnUNet_raw="/home/valentin/data/target/data/nnUnet_raw"
export nnUNet_preprocessed="/home/valentin/data/target/data/nnUNet_preprocessed"

# Define the path to your YAML configuration file
CONFIG_FILE="/home/valentin/workspaces/ssl-brainmet/ssl_brainmet/config/finetuning/nnUNetTrainerFineTuningV2_exp3.yaml"

echo "Using configuration file: $CONFIG_FILE"

# Parse the YAML file to extract the results directory.
nnUNet_results=$(./parse_yaml.py --file $CONFIG_FILE --key  metadata.output_path)
export nnUNet_results

echo "Storing results in $nnUNet_results"

# I add to add this otherwise nnUNet would not work
export nnUNet_compile=False

# Task ID
TASK=510

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "Running training for fold $FOLD_ID on CUDA device $CUDA_DEVICE..."
nnUNetv2_train_with_config $TASK 3d_fullres $FOLD_ID -device cuda -num_gpus 1 -tr nnUNetTrainerFineTuningV2 -trainer_config $CONFIG_FILE
