#!/bin/bash

# Function to download a file from Google Drive
download_file() {
    local file_id=$1
    local file_name=$2
    
    echo "Downloading file: $file_name"
    gdown "https://drive.google.com/uc?id=$file_id" -O "$file_name"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $file_name"
    else
        echo "Failed to download $file_name"
    fi
}

# Google Drive file IDs for the two .pth files
FILE_ID_1="17XsShB6e7wBTwbGHTxCGvAUiYxJsAjfl"
FILE_ID_2="1abMOuqqeMzqyo-2SWLzd-1gl74yT3sa-"

# Download each file
download_file "$FILE_ID_1" "checkpoint_p1.pth"
download_file "$FILE_ID_2" "checkpoint_p2.pth"

echo "Checkpoint download process complete."