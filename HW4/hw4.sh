#!/bin/bash
python3 gaussian-splatting/render.py -m best_model -s $1 --folder_path $2
# $1: path to the folder of split (e.g., */dataset/private_test)
# $2: path of the folder to put output images