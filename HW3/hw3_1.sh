#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_img_dir> <output_json>"
    exit 1
fi

# Run inference 
python3 p1_inference.py $1 $2