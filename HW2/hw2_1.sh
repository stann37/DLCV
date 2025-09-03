#!/bin/bash

# Check if the output directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: bash hw2_1.sh <output_directory>"
  exit 1
fi

# Run the Python script with the specified output directory
python3 p1_inference.py --output_folder "$1"
