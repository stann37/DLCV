#!/bin/bash

# Check if all arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_csv> <path_to_image_folder> <path_to_output_csv>"
    exit 1
fi

# Run the Python script with the provided arguments 
python3 p1_inference.py "$1" "$2" "$3"