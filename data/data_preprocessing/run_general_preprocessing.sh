#!/bin/bash

# Run preprocessing python script given a directory of treebanks.

for dir in data/sud-treebanks-v2.15/*/; do
    echo $dir
    python3 preprocessing.py --input "$dir"
    if [ $? -ne 0 ]; then
        echo "Error processing $dir"
    fi
done