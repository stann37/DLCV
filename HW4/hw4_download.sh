#!/bin/bash

FILE_ID="1ITcJPyUlQPpaUq2lwJPW3JW8REfMAXyI"

echo "Downloading the model..."
gdown --id "$FILE_ID" -O best_model.zip || { echo "Download failed! Exiting..."; exit 1; }

echo "Unzipping the model..."
unzip -o best_model.zip || { echo "Unzipping failed! Exiting..."; exit 1; }

echo "Done! Model is downloaded and unzipped."