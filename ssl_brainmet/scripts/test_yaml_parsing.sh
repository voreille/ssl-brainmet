#!/bin/bash

# Define the path to your YAML configuration file
CONFIG_FILE="/home/valentin/workspaces/ssl-brainmet/ssl_brainmet/config/finetuning/nnUNetTrainerFineTuningV2_exp2.yaml"

# Define a variable to hold the extracted output path.
# This example extracts the key 'metadata.output_path'; if you're using the original keys, change accordingly.
OUTPUT_PATH=$(./parse_yaml.py --file $CONFIG_FILE --key metadata.output_path)

echo "Extracted output path: $OUTPUT_PATH"

# Continue with the rest of your script where you might use $OUTPUT_PATH...
