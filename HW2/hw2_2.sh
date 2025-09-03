#!/bin/bash

# Check if exactly three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: bash hw2_2.sh <noise_dir> <output_dir> <model_path>"
    exit 1
fi

# Assign arguments to variables for readability
NOISE_DIR=$1
OUTPUT_DIR=$2
MODEL_PATH=$3

# Run the Python script with the provided arguments
python3 p2_inference.py "$NOISE_DIR" "$OUTPUT_DIR" "$MODEL_PATH"
