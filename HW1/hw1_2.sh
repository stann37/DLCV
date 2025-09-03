#!/bin/bash

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Run the Python script with the provided arguments
python3 p2_inference.py "$1" "$2"