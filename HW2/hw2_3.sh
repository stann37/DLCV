#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: bash hw2_3.sh <json_input_path> <output_folder_path> <pretrained_model_weight>"
    exit 1
fi

# Run the Python script with the provided arguments
python p3_inference.py "$1" "$2" "$3"

exit 0