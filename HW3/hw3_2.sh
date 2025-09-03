#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <test_img_dir> <output_json> <decoder_weights>"
    exit 1
fi

# Run inference
python3 p2_inference.py $1 $2 $3